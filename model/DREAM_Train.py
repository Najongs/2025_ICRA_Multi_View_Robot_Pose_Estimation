"""
DREAM 데이터셋으로 DINOv2 Pose Estimator 모델을 학습하는 스크립트.
torchrun을 사용한 분산 학습(DDP)을 지원합니다.

실행 예시 (GPU 3개 사용):
torchrun --nproc_per_node=3 DREAM_Train.py
"""

# ==============================================================================
# 0. 라이브러리 임포트
# ==============================================================================
import os
import glob
import json
import numpy as np
import random
import wandb
import threading
import time

import cv2
import math
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import kornia.augmentation as K

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# DDP 관련 라이브러리
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ==============================================================================
# 1. 상수 및 전역 변수 정의
# ==============================================================================
NUM_ANGLES = 7
NUM_JOINTS = 7
FEATURE_DIM = 768
HEATMAP_SIZE = (128, 128)
REQUIRED_KEYPOINTS = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


# ==============================================================================
# 2. 클래스 및 함수 정의
# ==============================================================================

# 헬퍼 함수: 가우시안 히트맵 생성
def create_gt_heatmap(keypoint_2d, HEATMAP_SIZE, sigma):
    # (이전에 제공된 create_gt_heatmap 함수 전체를 여기에 붙여넣으세요)
    H, W = HEATMAP_SIZE
    x, y = keypoint_2d
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist_sq = (xx - x)**2 + (yy - y)**2
    exponent = dist_sq / (2 * sigma**2)
    heatmap = np.exp(-exponent)
    heatmap[heatmap < np.finfo(float).eps * heatmap.max()] = 0
    return heatmap

# 데이터셋 클래스
class RobotPoseDataset(Dataset):
    def __init__(self, pairs, transform=None, heatmap_size=(128, 128), sigma=3.0):
        self.pairs = pairs
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.calib_lookup = {}
        DATA_PATHS = [
            '../dataset/DREAM_real/panda-3cam_azure',
            '../dataset/DREAM_real/panda-3cam_kinect360',
            '../dataset/DREAM_real/panda-3cam_realsense',
            '../dataset/DREAM_real/panda-orb',
        ]
        for base_path in DATA_PATHS:
            calib_path = os.path.join(base_path, '_camera_settings.json')
            try:
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
                intrinsics = calib_data['camera_settings'][0]['intrinsic_settings']
                fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                distortion_coeffs = np.zeros(5, dtype=np.float32)
                self.calib_lookup[base_path] = {"camera_matrix": camera_matrix, "distortion_coeffs": distortion_coeffs}
            except Exception as e:
                pass
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image_path = pair['image_path']
        try:
            calib_data = None
            for base_path, calib in self.calib_lookup.items():
                if image_path.startswith(base_path):
                    calib_data = calib
                    break
            if calib_data is None: return None, None, None
            camera_matrix = calib_data["camera_matrix"]
            dist_coeffs = calib_data["distortion_coeffs"]
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            undistorted_image_np = cv2.undistort(image_np, camera_matrix, dist_coeffs)
            original_h, original_w, _ = undistorted_image_np.shape
            undistorted_image = Image.fromarray(undistorted_image_np)
            image_tensor = self.transform(undistorted_image) if self.transform else transforms.ToTensor()(undistorted_image)
            gt_angles = torch.tensor(pair['joint_angles'], dtype=torch.float32)
            keypoints_2d = pair['keypoints_2d']
            num_keypoints = len(REQUIRED_KEYPOINTS)
            gt_heatmaps_np = np.zeros((num_keypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
            for i, name in enumerate(REQUIRED_KEYPOINTS):
                x, y = keypoints_2d[name]
                scaled_x = x * (self.heatmap_size[1] / original_w)
                scaled_y = y * (self.heatmap_size[0] / original_h)
                gt_heatmaps_np[i] = create_gt_heatmap((scaled_x, scaled_y), self.heatmap_size, self.sigma)
            gt_heatmaps = torch.from_numpy(gt_heatmaps_np)
            return image_tensor, gt_heatmaps, gt_angles
        except Exception as e:
            return None, None, None

        

class DINOv2Backbone(nn.Module):
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)

    def forward(self, image_tensor_batch): # 입력이 텐서 배치로 변경
        with torch.no_grad():
            features = self.model.forward_features(image_tensor_batch)
            patch_tokens = features[:, 1:, :]
        return patch_tokens

class JointAngleHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_angles=NUM_ANGLES, num_queries=4, nhead=8, num_decoder_layers=2):
        """
        어텐션을 사용하여 이미지 특징에서 핵심 정보를 추출하는 헤드.

        Args:
            input_dim (int): DINOv2 특징 벡터의 차원.
            num_angles (int): 예측할 관절 각도의 수.
            num_queries (int): 포즈 정보를 추출하기 위해 사용할 학습 가능한 쿼리의 수.
            nhead (int): Multi-head Attention의 헤드 수.
            num_decoder_layers (int): Transformer Decoder 레이어의 수.
        """
        super().__init__()
        
        # 1. "로봇 포즈에 대해 질문하는" 학습 가능한 쿼리 토큰 생성
        self.pose_queries = nn.Parameter(torch.randn(1, num_queries, input_dim))
        
        # 2. PyTorch의 표준 Transformer Decoder 레이어 사용
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=input_dim * 4, # 일반적인 설정
            dropout=0.1, 
            activation='gelu',
            batch_first=True  # (batch, seq, feature) 입력을 위함
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 3. 최종 각도 예측을 위한 MLP
        # 디코더를 거친 모든 쿼리 토큰의 정보를 사용
        self.angle_predictor = nn.Sequential(
            nn.LayerNorm(input_dim * num_queries),
            nn.Linear(input_dim * num_queries, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_angles)
        )

    def forward(self, fused_features):
        # fused_features: DINOv2의 패치 토큰들 (B, Num_Patches, Dim)
        # self.pose_queries: 학습 가능한 쿼리 (1, Num_Queries, Dim)
        
        # 배치 사이즈만큼 쿼리를 복제
        b = fused_features.size(0)
        queries = self.pose_queries.repeat(b, 1, 1)
        
        # Transformer Decoder 연산
        # 쿼리(queries)가 이미지 특징(fused_features)에 어텐션을 수행하여
        # 포즈와 관련된 정보로 자신의 값을 업데이트합니다.
        attn_output = self.transformer_decoder(tgt=queries, memory=fused_features)
        
        # 업데이트된 쿼리 토큰들을 하나로 펼쳐서 MLP에 전달
        output_flat = attn_output.flatten(start_dim=1)
        
        return self.angle_predictor(output_flat)

class TokenFuser(nn.Module):
    """
    ViT의 패치 토큰(1D 시퀀스)을 CNN이 사용하기 좋은 2D 특징 맵으로 변환하고 정제합니다.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        # x: (B, D, H, W) 형태로 reshape된 토큰 맵
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)

class LightCNNStem(nn.Module):
    """
    UNet의 인코더처럼 고해상도의 공간적 특징(shallow features)을 
    여러 스케일로 추출하기 위한 경량 CNN.
    """
    def __init__(self):
        super().__init__()
        # 간단한 CNN 블록 구성
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/2
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/4
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/8
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: 원본 이미지 텐서 배치 (B, 3, H, W)
        feat_4 = self.conv_block1(x)  # 1/4 스케일 특징
        feat_8 = self.conv_block2(feat_4) # 1/8 스케일 특징
        return feat_4, feat_8 # 다른 해상도의 특징들을 반환

class FusedUpsampleBlock(nn.Module):
    """
    업샘플링된 특징과 CNN 스템의 고해상도 특징(스킵 연결)을 융합하는 블록.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        
        # ✅ 해결책: skip_feature를 x의 크기에 강제로 맞춥니다.
        # ----------------------------------------------------------------------
        # 두 텐서의 높이와 너비가 다를 경우, skip_feature를 x의 크기로 리사이즈합니다.
        if x.shape[-2:] != skip_feature.shape[-2:]:
            skip_feature = F.interpolate(
                skip_feature, 
                size=x.shape[-2:], # target H, W
                mode='bilinear', 
                align_corners=False
            )
        # ----------------------------------------------------------------------
        
        # 이제 두 텐서의 크기가 같아졌으므로 안전하게 합칠 수 있습니다.
        fused = torch.cat([x, skip_feature], dim=1)
        return self.refine_conv(fused)
    
class UNetViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(128, 128)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)
        self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
        self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.heatmap_predictor = nn.Conv2d(64, num_joints, kernel_size=3, padding=1)

    def forward(self, dino_features, cnn_features):
        cnn_feat_4, cnn_feat_8 = cnn_features

        # 1. DINOv3 토큰을 표준 ViT 패치 개수인 196개로 잘라내고 2D 맵으로 변환
        num_patches_to_keep = 196
        dino_features_sliced = dino_features[:, :num_patches_to_keep, :]
        
        b, n, d = dino_features_sliced.shape
        h = w = int(n**0.5)
        x = dino_features_sliced.permute(0, 2, 1).reshape(b, d, h, w)

        x = self.token_fuser(x)

        # 2. 디코더 업샘플링 & 융합
        x = self.decoder_block1(x, cnn_feat_8)
        x = self.decoder_block2(x, cnn_feat_4)
        
        # 3. 최종 해상도로 업샘플링 및 예측
        x = self.final_upsample(x)
        heatmaps = self.heatmap_predictor(x)
        
        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
    
class DINOv2PoseEstimator(nn.Module):
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', num_joints=NUM_JOINTS, num_angles=NUM_ANGLES):
        super().__init__()
        self.backbone = DINOv2Backbone(model_name)
        feature_dim = self.backbone.model.embed_dim # timm 모델은 embed_dim 사용
        
        self.cnn_stem = LightCNNStem()
        # 헤드 생성 시 인자를 전달받아 사용
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim, num_joints=num_joints)
        self.angle_head = JointAngleHead(input_dim=feature_dim, num_angles=num_angles)

    def forward(self, image_tensor_batch):
        # 1. 두 경로로 병렬적으로 특징 추출
        dino_features = self.backbone(image_tensor_batch)      # 의미 정보
        cnn_stem_features = self.cnn_stem(image_tensor_batch) # 공간 정보
        
        # 2. 각 헤드에 필요한 특징 전달
        predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)
        predicted_angles = self.angle_head(dino_features)
        
        return predicted_heatmaps, predicted_angles

# ==============================================================================
# 2. 학습/검증/시각화 함수 정의
# ==============================================================================
# ==============================================================================
# 2. 데이터셋 시각화 (Dataset Visualization)
# ==============================================================================

def visualize_dataset_sample(dataset, config, num_samples=3):
    """데이터셋의 샘플을 시각화하여 GT가 올바른지 확인합니다."""
    print("\n--- Visualizing Dataset Samples ---")
    
    # 역정규화(Un-normalization)를 위한 값
    mean = np.array(config['mean'])
    std = np.array(config['std'])

    for i in range(num_samples):
        # 랜덤 샘플 선택
        idx = random.randint(0, len(dataset) - 1)
        image_tensor, gt_heatmaps, gt_angles = dataset[idx]
        
        # 1. 이미지 텐서를 시각화를 위한 Numpy 배열로 변환
        img_np = image_tensor.numpy().transpose((1, 2, 0))
        img_np = std * img_np + mean # 역정규화
        img_np = np.clip(img_np, 0, 1)

        # 2. GT 히트맵을 하나의 이미지로 결합
        composite_heatmap = torch.sum(gt_heatmaps, dim=0).numpy()
        
        # 3. GT 히트맵에서 키포인트 좌표 추출
        keypoints = []
        h, w = gt_heatmaps.shape[1:]
        for j in range(gt_heatmaps.shape[0]):
            heatmap = gt_heatmaps[j]
            max_val_idx = torch.argmax(heatmap)
            y, x = np.unravel_index(max_val_idx.numpy(), (h, w))
            keypoints.append([x, y])
        keypoints = np.array(keypoints)

        # 키포인트 좌표를 원본 이미지 크기에 맞게 스케일링
        img_h, img_w, _ = img_np.shape
        scaled_keypoints = keypoints.copy().astype(float)
        scaled_keypoints[:, 0] *= (img_w / w)
        scaled_keypoints[:, 1] *= (img_h / h)
        
        heatmap_resized = cv2.resize(composite_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        # 4. 시각화
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        
        # 원본 이미지
        axes[0].imshow(img_np)
        axes[0].set_title(f'Sample {idx+1}: Undistorted Image')
        axes[0].axis('off')
        
        # GT 히트맵
        axes[1].imshow(img_np, alpha=0.6)
        axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.4)
        axes[1].set_title('Ground Truth Heatmap Overlay')
        axes[1].axis('off')

        # GT 키포인트
        axes[2].imshow(img_np)
        axes[2].scatter(scaled_keypoints[:, 0], scaled_keypoints[:, 1], c='lime', s=40, edgecolors='black', linewidth=1)
        axes[2].set_title('Ground Truth Keypoints')
        axes[2].axis('off')

        plt.suptitle(f"GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_angles.numpy()]))
        plt.tight_layout()
        plt.show()

def visualize_predictions(model, dataset, device, config, epoch_num, num_samples=3):
    """
    Validation 데이터셋의 샘플에 대한 모델의 예측 결과를 Ground Truth와 함께 시각화합니다.
    (1행 4열 플롯으로 변경)
    """
    print(f"\n--- Visualizing Predictions for Epoch {epoch_num} ---")
    model.eval()  # 모델을 평가 모드로 설정
    
    mean = np.array(config['mean'])
    std = np.array(config['std'])

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image_tensor, gt_heatmaps, gt_angles = dataset[idx]
        
        # --- 모델 예측 수행 ---
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(device)
            pred_heatmaps_batch, pred_angles_batch = model(image_batch)
            
            pred_heatmaps = pred_heatmaps_batch[0].cpu()
            pred_angles = pred_angles_batch[0].cpu()

        # --- 시각화를 위한 데이터 준비 ---
        img_np = image_tensor.numpy().transpose((1, 2, 0))
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_h, img_w, _ = img_np.shape

        gt_composite_heatmap = torch.sum(gt_heatmaps, dim=0).numpy()
        gt_heatmap_resized = cv2.resize(gt_composite_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        
        pred_composite_heatmap = torch.sum(pred_heatmaps, dim=0).numpy()
        pred_heatmap_resized = cv2.resize(pred_composite_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # GT 키포인트 추출 및 스케일링
        gt_keypoints = []
        h, w = gt_heatmaps.shape[1:]
        for j in range(gt_heatmaps.shape[0]):
            y, x = np.unravel_index(torch.argmax(gt_heatmaps[j]).numpy(), (h, w))
            gt_keypoints.append([x * (img_w/w), y * (img_h/h)])
        gt_keypoints = np.array(gt_keypoints)
        
        # 예측 키포인트 추출 및 스케일링
        pred_keypoints = []
        for j in range(pred_heatmaps.shape[0]):
            y, x = np.unravel_index(torch.argmax(pred_heatmaps[j]).numpy(), (h, w))
            pred_keypoints.append([x * (img_w/w), y * (img_h/h)])
        pred_keypoints = np.array(pred_keypoints)

        # --- 1행 4열 서브플롯으로 GT와 예측 비교 시각화 ---
        fig, axes = plt.subplots(1, 4, figsize=(18, 5)) # ✅ figsize도 적절하게 조정

        # 1. GT 히트맵 오버레이
        axes[0].imshow(img_np, alpha=0.7)
        axes[0].imshow(gt_heatmap_resized, cmap='jet', alpha=0.3)
        axes[0].set_title('GT Heatmap')
        axes[0].axis('off')
        
        # 2. 예측 히트맵 오버레이
        axes[1].imshow(img_np, alpha=0.7)
        axes[1].imshow(pred_heatmap_resized, cmap='jet', alpha=0.3)
        axes[1].set_title('Pred Heatmap')
        axes[1].axis('off')

        # 3. GT 키포인트
        axes[2].imshow(img_np)
        axes[2].scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='lime', s=40, edgecolors='black', linewidth=1, label='GT')
        axes[2].set_title('GT Keypoints')
        axes[2].axis('off')
        
        # 4. 예측 키포인트
        axes[3].imshow(img_np)
        axes[3].scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c='red', s=40, marker='x', linewidth=1, label='Pred')
        axes[3].set_title('Pred Keypoints')
        axes[3].axis('off')

        # GT 각도와 예측 각도를 제목에 함께 표시
        gt_str = "GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        pred_str = "Pred Angles: " + ", ".join([f"{a:.2f}" for a in pred_angles.numpy()])
        plt.suptitle(f"Sample {idx+1} | Epoch {epoch_num}\n{gt_str}\n{pred_str}", fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # suptitle과 겹치지 않게 조정
        # plt.show()
    return fig

def log_predictions_to_wandb(model, images, gt_heatmaps, gt_angles, device, config, title):
    """
    주어진 데이터 배치를 사용하여 모델 예측을 시각화하고 wandb에 로깅합니다.
    """
    model.eval() # 평가 모드로 전환
    
    # 역정규화를 위한 값
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    
    log_images = []
    
    with torch.no_grad():
        # 입력된 이미지 배치 전체에 대해 예측 수행
        images_to_device = images.to(device)
        pred_heatmaps_batch, pred_angles_batch = model(images_to_device)

    # 배치 내 각 이미지에 대해 시각화 자료 생성 (최대 5개)
    for i in range(min(images.shape[0], 5)):
        img_tensor = images[i]
        
        # 텐서를 시각화용 Numpy 배열로 변환 및 역정규화
        img_np = img_tensor.cpu().numpy().transpose((1, 2, 0))
        img_np = std * img_np + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_h, img_w, _ = img_np.shape
        
        # BGR 변환 (OpenCV는 BGR 순서를 사용)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # --- GT 시각화 ---
        gt_hmap = gt_heatmaps[i]
        gt_ang = gt_angles[i]
        gt_composite_hmap = torch.sum(gt_hmap, dim=0).cpu().numpy()
        gt_heatmap_resized = cv2.resize(gt_composite_hmap, (img_w, img_h))
        gt_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
        gt_vis = cv2.addWeighted(gt_vis, 0.3, cv2.applyColorMap((gt_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.7, 0)

        # --- 예측 시각화 ---
        pred_hmap = pred_heatmaps_batch[i].cpu()
        pred_ang = pred_angles_batch[i].cpu()
        pred_composite_hmap = torch.sum(pred_hmap, dim=0).numpy()
        pred_heatmap_resized = cv2.resize(pred_composite_hmap, (img_w, img_h))
        pred_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)
        pred_vis = cv2.addWeighted(pred_vis, 0.3, cv2.applyColorMap((pred_heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.7, 0)

        # --- 텍스트 추가 ---
        gt_text = "GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_ang.numpy()])
        pred_text = "Pred Angles: " + ", ".join([f"{a:.2f}" for a in pred_ang.numpy()])
        cv2.putText(gt_vis, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(pred_vis, "Prediction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # --- 이미지 병합 ---
        comparison_image = cv2.hconcat([img_bgr, gt_vis, pred_vis])
        
        # wandb 로깅을 위해 PIL 이미지로 변환
        final_image = Image.fromarray(cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB))
        log_images.append(wandb.Image(final_image, caption=f"{gt_text}\n{pred_text}"))
        
    # wandb에 이미지 리스트 로깅
    wandb.log({title: log_images})
    model.train() # 모델을 다시 학습 모드로 전환
    
def train_one_epoch(model, loader, optimizer_kpt, optimizer_ang, crit_kpt, crit_ang, device, loss_weight_kpt=1.0, epoch_num=0):
    model.train()
    total_loss_kpt = 0
    total_loss_ang = 0
    first_batch = None # 첫 배치를 저장할 변수
    
    loop = tqdm(loader, desc=f"Epoch {epoch_num} [Train]")
    
    for i, (images, heatmaps, angles) in enumerate(loop):
        if i == 0: # ✅ 첫 번째 배치 저장
            first_batch = (images.cpu(), heatmaps.cpu(), angles.cpu())

        images, heatmaps, angles = images.to(device), heatmaps.to(device), angles.to(device)
        
        pred_heatmaps, pred_angles = model(images)
        
        # --- Keypoint Head 업데이트 ---
        optimizer_kpt.zero_grad()
        loss_kpt = crit_kpt(pred_heatmaps, heatmaps) * loss_weight_kpt
        loss_kpt.backward()
        optimizer_kpt.step()
        
        # --- Angle Head 업데이트 ---
        optimizer_ang.zero_grad()
        loss_ang = crit_ang(pred_angles, angles)
        loss_ang.backward()
        optimizer_ang.step()
        
        # 손실 기록
        total_loss_kpt += loss_kpt.item()
        total_loss_ang += loss_ang.item()
        
        # 진행률 표시줄에 각 손실 값을 업데이트
        loop.set_postfix(loss_ang=loss_ang.item(), loss_kpt=loss_kpt.item())
        
    # 평균 손실 반환
    avg_loss_kpt = total_loss_kpt / len(loader)
    avg_loss_ang = total_loss_ang / len(loader)
    return avg_loss_kpt, avg_loss_ang, first_batch 

def validate(model, loader, crit_kpt, crit_ang, device, loss_weight_kpt=1.0, epoch_num=0):
    model.eval()
    total_loss = 0
    first_batch = None # 첫 배치를 저장할 변수
    
    loop = tqdm(loader, desc=f"Epoch {epoch_num} [Validate]", leave=False)
    
    with torch.no_grad():
        for i, (images, heatmaps, angles) in enumerate(loop):
            if i == 0: # ✅ 첫 번째 배치 저장
                first_batch = (images.cpu(), heatmaps.cpu(), angles.cpu())
                
            images, heatmaps, angles = images.to(device), heatmaps.to(device), angles.to(device)
            
            pred_heatmaps, pred_angles = model(images)
            
            loss_kpt = crit_kpt(pred_heatmaps, heatmaps) * loss_weight_kpt
            loss_ang = crit_ang(pred_angles, angles)
            loss = loss_kpt + loss_ang
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    return total_loss / len(loader), first_batch

class RandomMasking(object):
    """
    PIL 이미지에 무작위 사각형 마스크를 적용하는 transform.
    """
    def __init__(self, num_masks=1, mask_size_ratio=(0.1, 0.3), mask_color='random'):
        assert isinstance(num_masks, int) and num_masks > 0
        assert isinstance(mask_size_ratio, tuple) and len(mask_size_ratio) == 2
        self.num_masks = num_masks
        self.mask_size_ratio = mask_size_ratio
        self.mask_color = mask_color

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 입력 이미지.
        Returns:
            PIL Image: 마스크가 적용된 이미지.
        """
        # PIL 이미지를 OpenCV가 다룰 수 있는 Numpy 배열로 변환 (RGB 순서 유지)
        img_np = np.array(img)
        h, w, _ = img_np.shape

        for _ in range(self.num_masks):
            # 마스크 크기 결정
            mask_w = int(w * random.uniform(self.mask_size_ratio[0], self.mask_size_ratio[1]))
            mask_h = int(h * random.uniform(self.mask_size_ratio[0], self.mask_size_ratio[1]))
            
            # 마스크 위치 결정
            x_start = random.randint(0, w - mask_w)
            y_start = random.randint(0, h - mask_h)
            
            # 마스크 색상 결정
            if self.mask_color == 'black':
                color = (0, 0, 0)
            elif self.mask_color == 'white':
                color = (255, 255, 255)
            else: # 'random'
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # 이미지에 마스크 적용
            img_np[y_start:y_start+mask_h, x_start:x_start+mask_w] = color
        
        # 다시 PIL 이미지로 변환하여 반환
        return Image.fromarray(img_np)
# ==============================================================================
# 3. DDP 및 컴포넌트 설정 함수
# ==============================================================================

def setup_ddp():
    """DDP 프로세스 그룹을 초기화하고 로컬 랭크를 반환합니다."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """DDP 프로세스 그룹을 정리합니다."""
    dist.destroy_process_group()

def setup_components(hyperparameters, dataset_pairs, rank, world_size):
    """학습에 필요한 모든 구성 요소를 준비합니다. (DDP 버전)"""
    model_name = hyperparameters['model_name']
    batch_size = hyperparameters['batch_size']
    val_split = hyperparameters['val_split']
    
    device = torch.device(f'cuda:{rank}')
    
    config = timm.create_model(model_name, pretrained=True,).default_cfg
    
    train_transform = transforms.Compose([
        transforms.Resize(config['input_size'][-2:]),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.1, 0.2), ratio=(0.3, 2.0)),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    # ✅ 검증 및 시각화용 Transform (증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize(config['input_size'][-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])
    
    indices = list(range(len(dataset_pairs)))
    train_size = int(len(indices) * (1 - val_split))
    
    # 모든 GPU가 동일한 분할을 사용하도록 시드를 고정합니다.
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = RobotPoseDataset(pairs=[dataset_pairs[i] for i in train_indices], transform=train_transform)
    val_dataset = RobotPoseDataset(pairs=[dataset_pairs[i] for i in val_indices], transform=val_transform)
    
    # --- DDP용 Sampler 설정 ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None, None)

    # shuffle=False (Sampler가 셔플을 담당)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
    
    model = DINOv2PoseEstimator(model_name).to(device)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    return model, train_loader, val_loader, config, val_transform, train_sampler


# ==============================================================================
# 4. 메인 실행 함수
# ==============================================================================
def main():
    """메인 학습 로직"""
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    
    # --- 하이퍼파라미터 ---
    hyperparameters = {
        'model_name': 'vit_base_patch14_dinov2.lvd142m',
        'batch_size': 240, # GPU당 배치사이즈 (전체 540 / GPU 3개)
        'num_epochs': 100,
        'val_split': 0.1,
        'loss_weight_kpt': 1000.0,
        'lr_kpt': 0.0001,
        'lr_ang': 0.0001,
    }

    # --- 데이터 로드 (모든 프로세스에서 실행, 로그는 rank 0에서만) ---
    if local_rank == 0:
        print("--- Loading and preparing dataset pairs ---")
    
    CSV_PATHS = [
        '../dataset/DREAM_real/panda-3cam_azure/panda-3cam_azure_matched_data.csv',
        '../dataset/DREAM_real/panda-3cam_kinect360/panda-3cam_kinect360_matched_data.csv',
        '../dataset/DREAM_real/panda-3cam_realsense/panda-3cam_realsense_matched_data.csv',
        '../dataset/DREAM_real/panda-orb/panda-orb_matched_data.csv',
    ]
    all_dfs = [pd.read_csv(path) for path in CSV_PATHS if os.path.exists(path)]
    if not all_dfs:
        if local_rank == 0: print("❌ ERROR: No CSV files were loaded.")
        return
    total_csv = pd.concat(all_dfs, ignore_index=True)
    dataset_pairs = [{'image_path': row['image_path'], 'joint_angles': [row[f'joint_{j}'] for j in range(1, NUM_ANGLES + 1)], 'keypoints_2d': {name: [row[f'kpt_{name}_proj_x'], row[f'kpt_{name}_proj_y']] for name in REQUIRED_KEYPOINTS}} for _, row in total_csv.iterrows()]
    
    if local_rank == 0:
        print(f"✅ All CSV files merged. Total pairs: {len(dataset_pairs)}")

    # --- 학습 컴포넌트 설정 ---
    model, train_loader, val_loader, config, val_transform, train_sampler = setup_components(
        hyperparameters, dataset_pairs, local_rank, world_size
    )
    
    model = DDP(model, device_ids=[local_rank])
    
    crit_kpt = nn.MSELoss()
    crit_ang = nn.SmoothL1Loss(beta=1.0)
    optimizer_kpt = torch.optim.AdamW(model.module.keypoint_head.parameters(), lr=hyperparameters['lr_kpt'])
    optimizer_ang = torch.optim.AdamW(model.module.angle_head.parameters(), lr=hyperparameters['lr_ang'])
    scheduler_kpt = CosineAnnealingLR(optimizer_kpt, T_max=hyperparameters['num_epochs'], eta_min=1e-6)
    scheduler_ang = CosineAnnealingLR(optimizer_ang, T_max=hyperparameters['num_epochs'], eta_min=1e-6)

    # --- WandB 및 학습 시작 (메인 프로세스에서만) ---
    if local_rank == 0:
        run = wandb.init(project="robot-pose-estimation", config=hyperparameters, name=f"DREAM_DDP_run_{time.strftime('%Y%m%d_%H%M%S')} BAEK")
        wandb.watch(model, log="all", log_freq=100)
        print("\n--- Starting Training ---")
        # visualize_dataset_sample(train_loader.dataset, config, num_samples=3)

    best_val_loss = float('inf')
    for epoch in range(hyperparameters['num_epochs']):
        train_sampler.set_epoch(epoch)
        
        # ✅ 반환값에 first_train_batch 추가
        train_loss_kpt, train_loss_ang, first_train_batch = train_one_epoch(
            model, train_loader, optimizer_kpt, optimizer_ang, crit_kpt, crit_ang, 
            torch.device(f'cuda:{local_rank}'), hyperparameters['loss_weight_kpt'], epoch+1
        )
        # ✅ 반환값에 first_val_batch 추가
        val_loss, first_val_batch = validate(
            model, val_loader, crit_kpt, crit_ang, 
            torch.device(f'cuda:{local_rank}'), hyperparameters['loss_weight_kpt'], epoch+1
        )
        
        scheduler_kpt.step()
        scheduler_ang.step()

        if local_rank == 0:
            current_lr_kpt = optimizer_kpt.param_groups[0]['lr']
            current_lr_ang = optimizer_ang.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{hyperparameters['num_epochs']} -> Train Losses [Kpt: {train_loss_kpt:.6f}, Ang: {train_loss_ang:.6f}], Val Loss: {val_loss:.6f}, LR [Kpt: {current_lr_kpt:.6f}, Ang: {current_lr_ang:.6f}]")
            
            # ✅ WandB에 숫자 메트릭 로깅
            wandb.log({
                "epoch": epoch + 1, 
                "train_loss_kpt": train_loss_kpt, 
                "train_loss_ang": train_loss_ang, 
                "avg_val_loss": val_loss, 
                "lr_kpt": current_lr_kpt, 
                "lr_ang": current_lr_ang
            })

            # ✅ WandB에 이미지 로깅
            # 검증 샘플은 매 에포크마다 로깅
            if first_val_batch[0] is not None:
                log_predictions_to_wandb(model.module, first_val_batch[0], first_val_batch[1], first_val_batch[2], 
                                         torch.device(f'cuda:{local_rank}'), config, "Validation Predictions")
            
            # 학습 샘플은 10 에포크마다 로깅 (너무 자주 로깅하는 것을 방지)
            if (epoch + 1) % 10 == 0 and first_train_batch[0] is not None:
                log_predictions_to_wandb(model.module, first_train_batch[0], first_train_batch[1], first_train_batch[2], 
                                         torch.device(f'cuda:{local_rank}'), config, "Train Predictions")

            # (기존 모델 저장 로직은 동일)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  -> 🎉 New best model saved with validation loss: {best_val_loss:.6f}")
                state_to_save = model.module.state_dict()
                save_thread = threading.Thread(target=torch.save, args=(state_to_save, 'best_pose_estimator_model.pth'))
                save_thread.start()

    if local_rank == 0:
        if 'save_thread' in locals() and save_thread.is_alive():
            save_thread.join()
        run.finish()
        
    cleanup_ddp()

if __name__ == '__main__':
    main()