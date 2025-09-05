import os
import glob
import json
import numpy as np
import random
import wandb
import threading

import cv2
import math
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm 

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR # 상단에 추가
import kornia.augmentation as K


from transformers import AutoModel
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

NUM_ANGLES = 7
NUM_JOINTS = 8
FEATURE_DIM = 768
HEATMAP_SIZE = (128, 128)

MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
MAX_VIEWS_PER_GROUP = 8

# ▼▼▼ [핵심 수정] 그룹핑 로직을 함수로 분리 ▼▼▼
def perform_grouping(df, tolerance, max_views):
    """주어진 tolerance 값으로 데이터프레임을 그룹핑합니다."""
    groups = []
    if not df.empty:
        current_views = []
        for _, row in df.iterrows():
            if not current_views:
                current_views.append(row)
                continue
            start_time = current_views[0]['robot_timestamp']
            if (row['robot_timestamp'] - start_time > tolerance) or (len(current_views) >= max_views):
                joint_angles = [current_views[0][f'position_fr3_joint{j}'] for j in range(1, NUM_ANGLES + 1)]
                image_paths = [{'image_path': view['image_path']} for view in current_views]
                groups.append({'views': image_paths, 'joint_angles': joint_angles})
                current_views = [row]
            else:
                current_views.append(row)
        if current_views:
            joint_angles = [current_views[0][f'position_fr3_joint{j}'] for j in range(1, NUM_ANGLES + 1)]
            image_paths = [{'image_path': view['image_path']} for view in current_views]
            groups.append({'views': image_paths, 'joint_angles': joint_angles})
    return groups

# ==============================================================================
# 헬퍼 함수 (Ground Truth 생성용)
# ==============================================================================

def create_gt_heatmap(keypoint_2d, HEATMAP_SIZE, sigma):
    """2D 좌표로부터 가우시안 히트맵을 생성합니다."""
    H, W = HEATMAP_SIZE
    x, y = keypoint_2d
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist_sq = (xx - x)**2 + (yy - y)**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    heatmap[heatmap < np.finfo(float).eps * heatmap.max()] = 0
    return heatmap

def get_modified_dh_matrix(a, d, alpha, theta):
    """Modified DH 파라미터로 변환 행렬 T를 계산합니다."""
    alpha_rad, theta_rad = math.radians(alpha), math.radians(theta)
    cos_th, sin_th = np.cos(theta_rad), np.sin(theta_rad)
    cos_al, sin_al = np.cos(alpha_rad), np.sin(alpha_rad)
    
    # Craig's Modified DH Convention
    T = np.array([
        [cos_th, -sin_th, 0, a],
        [sin_th * cos_al, cos_th * cos_al, -sin_al, -d * sin_al],
        [sin_th * sin_al, cos_th * sin_al,  cos_al,  d * cos_al],
        [0, 0, 0, 1]
    ])
    return T

def angle_to_joint_coordinate(joint_angles, selected_view):
    """[Forward Kinematics] 7개 관절 각도를 8개(베이스 포함)의 3D 공간 좌표로 변환합니다."""
    # Franka Research 3 로봇의 DH 파라미터 (단위: 미터, 도)
    fr3_dh_parameters = [
        {'a': 0,       'd': 0.333, 'alpha': 0,   'theta_offset': 0}, # Joint 1
        {'a': 0,       'd': 0,     'alpha': -90, 'theta_offset': 0}, # Joint 2
        {'a': 0,       'd': 0.316, 'alpha': 90,  'theta_offset': 0}, # Joint 3
        {'a': 0.0825,  'd': 0,     'alpha': 90,  'theta_offset': 0}, # Joint 4
        {'a': -0.0825, 'd': 0.384, 'alpha': -90, 'theta_offset': 0}, # Joint 5
        {'a': 0,       'd': 0,     'alpha': 90,  'theta_offset': 0}, # Joint 6
        {'a': 0.088,   'd': 0,     'alpha': 90,  'theta_offset': 0}, # Joint 7
        {'a': 0,       'd': 0.107, 'alpha': 0,   'theta_offset': 0}  # Flange (End-effector base)
    ]
    
    # 카메라 뷰에 따른 로봇 베이스의 좌표계 보정
    view_rotations = {
        'view1': R.from_euler('zyx', [90, 180, 0], degrees=True),
        'view2': R.from_euler('zyx', [90, 180, 0], degrees=True),
        'view3': R.from_euler('zyx', [90, 180, 0], degrees=True),
        'view4': R.from_euler('zyx', [90, 180, 0], degrees=True)
    }
    
    T_cumulative = np.eye(4)
    if selected_view in view_rotations:
        T_cumulative[:3, :3] = view_rotations[selected_view].as_matrix()

    # J0(베이스) 좌표는 원점
    joint_coords_3d = [np.array([0, 0, 0])] 
    
    origin_point = np.array([0, 0, 0, 1])
    # 각 관절 각도를 순서대로 적용하여 변환 행렬을 누적 곱셈
    for i, angle_rad in enumerate(joint_angles):
        params = fr3_dh_parameters[i]
        theta_deg = math.degrees(angle_rad) + params['theta_offset']
        T_i = get_modified_dh_matrix(params['a'], params['d'], params['alpha'], theta_deg)
        T_cumulative = T_cumulative @ T_i
        
        # 누적된 변환 행렬을 통해 현재 관절의 3D 좌표 계산
        joint_pos_3d = (T_cumulative @ origin_point)[:3]
        joint_coords_3d.append(joint_pos_3d)
        
    return np.array(joint_coords_3d, dtype=np.float32)

def joint_coordinate_to_pixel_plane(coords_3d, aruco_result, camera_matrix, dist_coeffs):
    """[3D-2D 투영] 3D 좌표를 ArUco 마커 기반 카메라 파라미터로 2D 픽셀 평면에 투영합니다."""
    # 카메라 외부 파라미터 (Extrinsics): 월드 좌표계 -> 카메라 좌표계 변환
    rvec = np.array([aruco_result['rvec_x'], aruco_result['rvec_y'], aruco_result['rvec_z']])
    tvec = np.array([aruco_result['tvec_x'], aruco_result['tvec_y'], aruco_result['tvec_z']])
    
    # OpenCV의 projectPoints 함수를 사용하여 3D 포인트를 2D 이미지 평면으로 투영
    pixel_coords, _ = cv2.projectPoints(coords_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return pixel_coords.reshape(-1, 2)

# ==============================================================================
# 2. 멀티뷰 데이터셋 클래스
# ==============================================================================
class RobotPoseDataset(Dataset):
    def __init__(self, groups, transform=None, HEATMAP_SIZE=(128, 128), sigma=5.0):
        self.groups = groups
        self.transform = transform
        self.heatmap_size = HEATMAP_SIZE
        self.sigma = sigma

        print("Loading and preprocessing metadata...")
        # ArUco와 Calibration 데이터를 미리 메모리에 로드하여 I/O 병목을 줄입니다.
        self.aruco_lookup = {}
        for pose_name in ['pose1', 'pose2']:
            aruco_path = f'../dataset/franka_research3/{pose_name}_aruco_pose_summary.json'
            with open(aruco_path, 'r') as f:
                for item in json.load(f):
                    self.aruco_lookup[f"{pose_name}_{item['view']}_{item['cam']}"] = item
        
        self.calib_lookup = {}
        calib_dir = "../dataset/franka_research3/franka_research3_calib_cam_from_conf"
        for calib_path in glob.glob(os.path.join(calib_dir, "*.json")):
            filename = os.path.basename(calib_path).replace("_calib.json", "")
            with open(calib_path, 'r') as f:
                self.calib_lookup[filename] = json.load(f)
        
        self.serial_to_view = {
            '41182735': "view1", '49429257': "view2",
            '44377151': "view3", '49045152': "view4"
        }
        print("✅ Metadata loaded.")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        first_image_path = group.get('views', [{}])[0].get('image_path', f"group_{idx}")
        try:
            gt_angles = torch.tensor(group['joint_angles'], dtype=torch.float32)
            image_dict, gt_heatmaps_dict = {}, {}

            for view_data in group['views']:
                image_path = view_data['image_path']
                
                # --- 경로 분석 ---
                filename = os.path.basename(image_path)
                parts = filename.split('_')
                serial_str, cam_type_str = parts[1], parts[2]
                selected_cam = cam_type_str + "cam"
                selected_view = self.serial_to_view[serial_str]
                
                # ▼▼▼ [핵심 버그 수정] view_key를 고유하게 만듭니다. ▼▼▼
                # (기존) view_key = serial_str 
                # (변경) view_key = f"{serial_str}_{cam_type_str}"  # 예: '41182735_left'
                view_key = f"{serial_str}_{cam_type_str}"

                # --- 이후 로직은 동일 ---
                calib_key = f"{selected_view}_{serial_str}_{selected_cam}"
                calib = self.calib_lookup[calib_key]
                camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
                dist_coeffs = np.array(calib["distortion_coeffs"], dtype=np.float32)
                pose_name = 'pose1' if 'pose1' in image_path else 'pose2'
                aruco_key = f"{pose_name}_{selected_view}_{selected_cam}"
                aruco_result = self.aruco_lookup[aruco_key]

                img_bgr = cv2.imread(image_path)
                if img_bgr is None: raise FileNotFoundError(f"Failed to read image")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                undistorted_np = cv2.undistort(img_rgb, camera_matrix, dist_coeffs)
                
                joint_coords_3d = angle_to_joint_coordinate(group['joint_angles'], selected_view)
                pixel_coords = joint_coordinate_to_pixel_plane(joint_coords_3d, aruco_result, camera_matrix, np.zeros_like(dist_coeffs))
                
                h, w, _ = undistorted_np.shape
                scaled_kpts = pixel_coords * [self.heatmap_size[1]/w, self.heatmap_size[0]/h]
                
                heatmaps_np = np.zeros((NUM_JOINTS, *self.heatmap_size), dtype=np.float32)
                for i in range(NUM_JOINTS):
                    heatmaps_np[i] = create_gt_heatmap(scaled_kpts[i], self.heatmap_size, self.sigma)
                
                image_dict[view_key] = self.transform(Image.fromarray(undistorted_np))
                gt_heatmaps_dict[view_key] = torch.from_numpy(heatmaps_np)

            return image_dict, gt_heatmaps_dict, gt_angles

        except Exception as e:
            # print(f"⚠️ Warning: Skipping group {idx} (e.g., {os.path.basename(first_image_path)}) due to error: {e}")
            return None, None, None

import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# 1. 시각화를 위한 Transform 및 데이터셋 준비 (기존과 동일)
# ==============================================================================



# ==============================================================================
# 2. 시각화 함수 정의 (수정)
# ==============================================================================

def visualize_samples_by_group_size(groups, transform, mean, std):
    """
    데이터셋에 존재하는 모든 그룹 크기(8, 7, 6...)에 대해
    각각 하나의 샘플을 시각화합니다.
    """
    print("\n--- Visualizing One Sample For Each Group Size ---")
    
    # 그룹 크기별로 데이터를 정리하기 위한 딕셔너리
    groups_by_size = {}
    for group in groups:
        size = len(group['views'])
        if size not in groups_by_size:
            groups_by_size[size] = []
        groups_by_size[size].append(group)

    # 그룹 크기가 큰 순서대로 (8, 7, 6...) 정렬
    sorted_sizes = sorted(groups_by_size.keys(), reverse=True)

    # 각 그룹 크기에 대해 반복
    for size in sorted_sizes:
        # 해당 크기의 그룹 중 하나를 랜덤으로 선택
        sample_group = random.choice(groups_by_size[size])
        
        # 임시 데이터셋을 만들어 __getitem__ 로직 활용
        temp_dataset = RobotPoseDataset(groups=[sample_group], transform=transform)
        image_dict, gt_heatmaps_dict, gt_angles = temp_dataset[0]

        if image_dict is None:
            print(f"Could not process sample for group size {size}. Skipping.")
            continue
            
        # --- 시각화 로직 (기존과 거의 동일) ---
        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6 * num_views, 10))
        if num_views == 1: axes = np.expand_dims(axes, axis=1)

        angle_str = ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        fig.suptitle(f"Sample for Group Size: {num_views} | GT Angles: [{angle_str}]", fontsize=16)

        for j, view_key in enumerate(image_dict.keys()):
            img_tensor = image_dict[view_key]
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = np.array(std) * img_np + np.array(mean)
            img_np = np.clip(img_np, 0, 1)
            H, W, _ = img_np.shape

            gt_heatmaps = gt_heatmaps_dict[view_key]
            composite_heatmap = torch.sum(gt_heatmaps, dim=0).numpy()
            heatmap_resized = cv2.resize(composite_heatmap, (W, H))

            keypoints = []
            h_map, w_map = gt_heatmaps.shape[1:]
            for k in range(gt_heatmaps.shape[0]):
                y, x = np.unravel_index(torch.argmax(gt_heatmaps[k]).numpy(), (h_map, w_map))
                keypoints.append([x * (W / w_map), y * (H / h_map)])
            keypoints = np.array(keypoints)

            ax = axes[0, j]
            ax.imshow(img_np, alpha=0.7)
            ax.imshow(heatmap_resized, cmap='jet', alpha=0.3)
            ax.set_title(f"View: {view_key} (Heatmap)")
            ax.axis('off')

            ax = axes[1, j]
            ax.imshow(img_np)
            ax.scatter(keypoints[:, 0], keypoints[:, 1], c='lime', s=40, edgecolors='black', linewidth=1)
            ax.set_title(f"View: {view_key} (Keypoints)")
            ax.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# ==============================================================================
# 3. 시각화 실행
# ==============================================================================
# 최종적으로 필터링된 dataset_groups를 사용하여 시각화 함수 호출

class DINOv3Backbone(nn.Module):
    """
    Hugging Face transformers 라이브러리를 사용하여 DINOv3 모델을 구성합니다.
    사전에 정규화된 이미지 텐서 배치를 입력받아 패치 토큰을 반환합니다.
    """
    def __init__(self, model_name=MODEL_NAME): # ViT-Base 모델을 기본값으로 사용
        super().__init__()
        # 사전 학습된 DINOv3 모델을 불러옵니다.
        self.model = AutoModel.from_pretrained(model_name)
        # ⚠️ 참고: 모델을 특정 장치(.to('cuda'))로 보내는 코드는
        # 메인 학습 스크립트에서 한 번에 처리하는 것이 좋습니다.

    def forward(self, image_tensor_batch):
        """
        Args:
            image_tensor_batch (torch.Tensor): (B, C, H, W) 형태의 정규화된 이미지 텐서
        """
        # 그래디언트 계산을 비활성화합니다.
        with torch.no_grad():
            # Hugging Face 모델은 'pixel_values'라는 키워드 인자를 기대합니다.
            outputs = self.model(pixel_values=image_tensor_batch)

        last_hidden_state = outputs.last_hidden_state
        
        # 클래스 토큰(CLS)을 제외한 패치 토큰들만 반환합니다.
        patch_tokens = last_hidden_state[:, 1:, :]
        
        return patch_tokens

class JointAngleHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_angles=NUM_ANGLES, num_queries=4, nhead=8, num_decoder_layers=2):
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

class MultiViewFusion(nn.Module):
    """
    Latent Query 기반의 Multi-view Fusion 모듈.
    """
    def __init__(self, feature_dim=FEATURE_DIM, num_heads=8, dropout=0.1, num_queries=16, num_layers=2):
        super().__init__()
        # 씬 전체의 정보를 요약할 학습 가능한 글로벌 쿼리
        self.global_queries = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        
        # Cross-Attention + Self-Attention으로 구성된 Transformer Decoder 레이어
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.fusion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, view_features: list):
        # 1. 모든 뷰의 토큰들을 시퀀스 차원에서 하나로 합침
        all_view_tokens = torch.cat(view_features, dim=1)
        b = all_view_tokens.size(0)
        
        # 2. 배치 사이즈만큼 글로벌 쿼리 복제
        queries = self.global_queries.repeat(b, 1, 1)
        
        # 3. Decoder를 통해 쿼리가 모든 뷰의 정보를 요약하도록 함
        # 쿼리가 Key/Value인 all_view_tokens에 Cross-Attention을 수행하고,
        # 이후 쿼리들끼리 Self-Attention을 수행하며 정보를 정제함
        fused_queries = self.fusion_decoder(tgt=queries, memory=all_view_tokens)
        
        return fused_queries

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
    def __init__(self, input_dim=768, num_joints=7, heatmap_size=(128, 128)):
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
    
class DINOv3PoseEstimator(nn.Module):
    """
    [전체 아키텍처 (Overall Architecture)]
    Multi-view 이미지들을 입력받아, 각 뷰의 특징을 추출하고 융합하여
    하나의 통합된 관절 각도(global pose)와 각 뷰에 대한 키포인트 히트맵(local keypoints)을 예측합니다.
    """
    def __init__(self, model_name=MODEL_NAME, num_joints=NUM_JOINTS, num_angles=NUM_ANGLES, max_views=10):
        super().__init__()
        
        # 1. 백본: 고차원 의미 정보 추출
        self.backbone = DINOv3Backbone(model_name)
        feature_dim = self.backbone.model.config.hidden_size
        
        # ▼▼▼ [수정 1] 뷰 임베딩을 유연하게 처리하기 위한 변경 ▼▼▼
        # 최대 처리 가능한 카메라(뷰) 개수를 기반으로 임베딩 레이어 생성
        self.view_embeddings = nn.Embedding(max_views, feature_dim)
        
        # forward 시점에 뷰 이름/시리얼과 인덱스를 동적으로 매핑할 딕셔너리
        self.view_to_idx = {} 
        self.next_view_idx = 0

        # 3. CNN 스템: 저차원 공간 정보 추출
        self.cnn_stem = LightCNNStem()
        
        # 4. 퓨전 모듈: 모든 뷰의 정보를 하나의 전역 요약 정보로 압축
        self.fusion_module = MultiViewFusion(feature_dim=feature_dim)
        
        # 5. 헤드 (예측기)
        self.angle_head = JointAngleHead(input_dim=feature_dim, num_angles=num_angles, num_queries=16)
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim, num_joints=num_joints)
        self.keypoint_enricher = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=8, dim_feedforward=feature_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )

    def forward(self, multi_view_images: dict):
        all_dino_features_with_embed = []
        all_cnn_features = {}
        view_keys_ordered = list(multi_view_images.keys())

        # --- Step 1: 각 뷰에 대한 병렬 특징 추출 ---
        for view_key in view_keys_ordered:
            view_tensor = multi_view_images[view_key]
            dino_features = self.backbone(view_tensor)
            
            if view_key not in self.view_to_idx:
                if self.next_view_idx >= self.view_embeddings.num_embeddings:
                    raise ValueError(f"Exceeded maximum number of views ({self.view_embeddings.num_embeddings}).")
                self.view_to_idx[view_key] = self.next_view_idx
                self.next_view_idx += 1
            
            view_idx = self.view_to_idx[view_key]
            
            embedding = self.view_embeddings(
                torch.tensor([view_idx], device=dino_features.device)
            ).unsqueeze(0)
            all_dino_features_with_embed.append(dino_features + embedding)
            
            all_cnn_features[view_key] = self.cnn_stem(view_tensor)

        # --- Step 2: Multi-view 정보 융합 ---
        # Latent Query를 통해 모든 뷰의 DINO 특징을 'fused_queries'라는 전역 정보로 요약
        fused_queries = self.fusion_module(all_dino_features_with_embed)
        
        # --- Step 3: 관절 각도 예측 ---
        # 요약된 전역 정보로부터 직접 관절 각도를 예측
        predicted_angles = self.angle_head(fused_queries)
        
        # --- Step 4: 키포인트 히트맵 예측 ---
        predicted_heatmaps_dict = {}
        for i, view_name in enumerate(view_keys_ordered):
            enriched_tokens = self.keypoint_enricher(
                tgt=all_dino_features_with_embed[i], 
                memory=fused_queries
            )
            heatmap = self.keypoint_head(enriched_tokens, all_cnn_features[view_name])
            predicted_heatmaps_dict[view_name] = heatmap
        
        return predicted_heatmaps_dict, predicted_angles

# ==============================================================================
# Cell 5: 학습/검증용 시각화 함수
# ==============================================================================

def visualize_dataset_sample(dataset, mean, std, results_dir, num_samples=1):
    os.makedirs(results_dir, exist_ok=True)
    """데이터셋의 GT 샘플을 시각화하여 데이터 파이프라인을 검증합니다."""
    print("\n--- Visualizing Dataset Samples ---")
    # (이전 Cell 3에서 사용했던 visualize_final_groups 함수와 거의 동일한 로직)
    for i in range(num_samples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None: break
        
        image_dict, gt_heatmaps_dict, gt_angles = sample
        num_views = len(image_dict)
        fig, axes = plt.subplots(1, num_views, figsize=(6 * num_views, 6))
        if num_views == 1: axes = [axes]

        angle_str = ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        fig.suptitle(f"Sample Group {idx} | GT Angles: [{angle_str}]", fontsize=16)

        for j, view_key in enumerate(image_dict.keys()):
            img_tensor = image_dict[view_key]
            img_np = (img_tensor.numpy().transpose(1, 2, 0) * np.array(std)) + np.array(mean)
            img_np = np.clip(img_np, 0, 1)
            H, W, _ = img_np.shape

            gt_heatmaps = gt_heatmaps_dict[view_key]
            heatmap_resized = cv2.resize(torch.sum(gt_heatmaps, dim=0).numpy(), (W, H))
            
            axes[j].imshow(img_np, alpha=0.7)
            axes[j].imshow(heatmap_resized, cmap='jet', alpha=0.3)
            axes[j].set_title(f"View: {view_key} (GT Heatmap)")
            axes[j].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"gt_sample_{idx}_{int(time.time())}.png"
        plt.savefig(os.path.join(results_dir, filename))
        print(f"  -> Saved GT sample visualization to {os.path.join(results_dir, filename)}")
        plt.close() # 메모리 해제

def visualize_predictions(model, dataset, device, mean, std, epoch_num, results_dir, num_samples=1):
    """검증 데이터셋 샘플에 대한 모델의 예측 결과를 GT와 함께 시각화합니다."""
    print(f"\n--- Visualizing Predictions for Epoch {epoch_num} ---")
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    figures = []
    
    for i in range(num_samples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None: break

        image_dict, gt_heatmaps_dict, gt_angles = sample

        with torch.no_grad():
            input_batch = {k: v.unsqueeze(0).to(device) for k, v in image_dict.items()}
            pred_heatmaps_dict, pred_angles_batch = model(input_batch)
            pred_angles = pred_angles_batch[0].cpu()

        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6 * num_views, 10))
        if num_views == 1: axes = np.expand_dims(axes, axis=1)

        gt_str = "GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        pred_str = "Pred Angles: " + ", ".join([f"{a:.2f}" for a in pred_angles.numpy()])
        fig.suptitle(f"Sample {idx} | Epoch {epoch_num}\n{gt_str}\n{pred_str}", fontsize=12)

        for j, view_key in enumerate(image_dict.keys()):
            img_tensor = image_dict[view_key]
            img_np = (img_tensor.numpy().transpose(1, 2, 0) * np.array(std)) + np.array(mean)
            img_np = np.clip(img_np, 0, 1)
            H, W, _ = img_np.shape
            
            # GT Heatmap
            gt_heatmap = torch.sum(gt_heatmaps_dict[view_key], dim=0).numpy()
            axes[0, j].imshow(img_np, alpha=0.7)
            axes[0, j].imshow(cv2.resize(gt_heatmap, (W, H)), cmap='jet', alpha=0.3)
            axes[0, j].set_title(f"View: {view_key} (GT)")
            axes[0, j].axis('off')

            # Predicted Heatmap
            pred_heatmap = torch.sum(pred_heatmaps_dict[view_key][0].cpu(), dim=0).numpy()
            axes[1, j].imshow(img_np, alpha=0.7)
            axes[1, j].imshow(cv2.resize(pred_heatmap, (W, H)), cmap='jet', alpha=0.3)
            axes[1, j].set_title(f"View: {view_key} (Pred)")
            axes[1, j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        figures.append(fig)

    for i, fig in enumerate(figures):
        filename = f"prediction_epoch_{epoch_num}_sample_{idx}_{i}.png"
        fig.savefig(os.path.join(results_dir, filename))
        print(f"  -> Saved prediction visualization to {os.path.join(results_dir, filename)}")
        # wandb 로깅을 위해 figure 객체는 그대로 반환
    return figures

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

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==============================================================================
# Cell 6: 학습 및 검증 루프 정의
# ==============================================================================

def train_one_epoch(model, loader, optimizers, criteria, device, loss_weight_kpt, epoch_num):
    model.train()
    total_loss_kpt, total_loss_ang = 0.0, 0.0
    optimizer_kpt, optimizer_ang = optimizers['kpt'], optimizers['ang']
    crit_kpt, crit_ang = criteria['kpt'], criteria['ang']

    loop = tqdm(loader, desc=f"Epoch {epoch_num} [Train]")
    for batch in loop:
        image_dict, gt_heatmaps_dict, gt_angles = batch

        # ---- (A) 랭크 간 '유효 배치' 여부 동기화 ----
        has_data_local = int(image_dict is not None)
        has_data_all = torch.tensor(has_data_local, device=device)
        import torch.distributed as dist
        dist.all_reduce(has_data_all, op=dist.ReduceOp.SUM)
        has_any_rank_data = int(has_data_all.item())

        if not has_any_rank_data:
            # 모든 랭크가 빈 배치이면 전체 스킵
            continue

        if image_dict is None:
            # 이 랭크만 빈 배치인 경우: 더미 0-loss backward로 통신 타이밍 맞춤
            optimizer_kpt.zero_grad(set_to_none=True)
            optimizer_ang.zero_grad(set_to_none=True)
            # 모든 파라미터에 대해 '정의된(=0) grad'를 만들기 위해 합계를 0배 함
            dummy = None
            for p in model.parameters():
                if p.requires_grad:
                    dummy = (p.sum() if dummy is None else dummy + p.sum())
            if dummy is None:
                # 이론상 발생X: 학습 파라미터가 하나도 없는 경우
                dummy = torch.zeros((), device=device, requires_grad=True)
            (dummy * 0.0).backward()
            optimizer_kpt.step()
            optimizer_ang.step()
            loop.set_postfix(loss_kpt='skip', loss_ang='skip')
            continue

        # ---- (B) 정상 경로: 단일 backward ----
        images_gpu   = {k: v.to(device) for k, v in image_dict.items()}
        heatmaps_gpu = {k: v.to(device) for k, v in gt_heatmaps_dict.items()}
        angles_gpu   = gt_angles.to(device)

        pred_heatmaps_dict, pred_angles = model(images_gpu)

        loss_ang = crit_ang(pred_angles, angles_gpu)

        real_view_keys = list(pred_heatmaps_dict.keys())
        if not real_view_keys:
            # 극히 드물게 키 없음 → 더미 backward로 정렬
            optimizer_kpt.zero_grad(set_to_none=True)
            optimizer_ang.zero_grad(set_to_none=True)
            dummy = None
            for p in model.parameters():
                if p.requires_grad:
                    dummy = (p.sum() if dummy is None else dummy + p.sum())
            (dummy * 0.0).backward()
            optimizer_kpt.step()
            optimizer_ang.step()
            loop.set_postfix(loss_kpt='skip2', loss_ang='skip2')
            continue

        loss_kpt_views = [crit_kpt(pred_heatmaps_dict[k], heatmaps_gpu[k]) for k in real_view_keys]
        loss_kpt = torch.stack(loss_kpt_views).mean() * loss_weight_kpt

        total_loss = loss_kpt + loss_ang

        optimizer_kpt.zero_grad(set_to_none=True)
        optimizer_ang.zero_grad(set_to_none=True)
        total_loss.backward()   # ← 한 번만!
        optimizer_kpt.step()
        optimizer_ang.step()

        total_loss_kpt += loss_kpt.item()
        total_loss_ang += loss_ang.item()
        loop.set_postfix(loss_kpt=loss_kpt.item(), loss_ang=loss_ang.item())

    return total_loss_kpt / len(loader), total_loss_ang / len(loader)



def validate(model, loader, criteria, device, loss_weight_kpt, epoch_num):
    model.eval() # 모델을 평가 모드로 설정
    total_val_loss = 0.0
    crit_kpt, crit_ang = criteria['kpt'], criteria['ang']
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch_num} [Validate]", leave=False):
            image_dict, gt_heatmaps_dict, gt_angles = batch
            if image_dict is None: continue

            images_gpu = {k: v.to(device) for k, v in image_dict.items()}
            heatmaps_gpu = {k: v.to(device) for k, v in gt_heatmaps_dict.items()}
            angles_gpu = gt_angles.to(device)
            
            pred_heatmaps_dict, pred_angles = model(images_gpu)
            
            loss_ang = crit_ang(pred_angles, angles_gpu)
            
            real_view_keys = [k for k in pred_heatmaps_dict if not k.startswith('dummy')]
            if not real_view_keys: continue
                
            loss_kpt_views = [crit_kpt(pred_heatmaps_dict[k], heatmaps_gpu[k]) for k in real_view_keys]
            loss_kpt = (torch.stack(loss_kpt_views).mean()) * loss_weight_kpt
            
            total_loss = loss_kpt + loss_ang
            total_val_loss += total_loss.item()
            
    return total_val_loss / len(loader)

# ==============================================================================
# Cell 7: 학습 환경 설정 (Setup) 함수
# ==============================================================================
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

# ▼▼▼ DDP 필수 라이브러리 ▼▼▼
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ==============================================================================
# Setup / Teardown 함수
# ==============================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank

def cleanup_ddp():
    dist.destroy_process_group()

def setup(hyperparameters, dataset_groups, rank, world_size):
    print(f"--- [Rank {rank}] Setting up environment ---")
    device = torch.device(f'cuda:{rank}')
    
    processor = AutoImageProcessor.from_pretrained(hyperparameters['model_name'])
    mean, std = processor.image_mean, processor.image_std
    resize_size, crop_size = 512, 512
    
    train_transform = transforms.Compose([
        transforms.Resize(resize_size), transforms.CenterCrop(crop_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_size), transforms.CenterCrop(crop_size),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)
    ])
    
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset_groups)).tolist()
    train_size = int(len(dataset_groups) * (1 - hyperparameters['val_split']))
    train_groups = [dataset_groups[i] for i in indices[:train_size]]
    val_groups = [dataset_groups[i] for i in indices[train_size:]]
    
    train_dataset = RobotPoseDataset(groups=train_groups, transform=train_transform)
    val_dataset = RobotPoseDataset(groups=val_groups, transform=val_transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch: return None, None, None
        image_dicts, heatmap_dicts, angles_list = zip(*batch)
        all_keys = set().union(*[d.keys() for d in image_dicts])
        sample_img = list(image_dicts[0].values())[0]
        dummy_img = torch.zeros_like(sample_img)
        sample_hmap = list(heatmap_dicts[0].values())[0]
        dummy_hmap = torch.zeros_like(sample_hmap)
        standardized_images, standardized_heatmaps = [], []
        for i in range(len(batch)):
            new_img_dict = {key: image_dicts[i].get(key, dummy_img) for key in all_keys}
            new_hmap_dict = {key: heatmap_dicts[i].get(key, dummy_hmap) for key in all_keys}
            standardized_images.append(new_img_dict)
            standardized_heatmaps.append(new_hmap_dict)
        images = torch.utils.data.dataloader.default_collate(standardized_images)
        heatmaps = torch.utils.data.dataloader.default_collate(standardized_heatmaps)
        angles = torch.stack(angles_list)
        return images, heatmaps, angles

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], num_workers=8, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], num_workers=8, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)
    
    model = DINOv3PoseEstimator(hyperparameters['model_name']).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    criteria = {'kpt': nn.MSELoss(), 'ang': nn.SmoothL1Loss(beta=1.0)}
    
    m = model.module
    params_kpt = list(m.cnn_stem.parameters()) + list(m.view_embeddings.parameters()) + list(m.fusion_module.parameters()) + list(m.keypoint_enricher.parameters()) + list(m.keypoint_head.parameters())
    params_ang = list(m.angle_head.parameters())
    
    optimizers = { 'kpt': optim.AdamW(params_kpt, lr=hyperparameters['lr_kpt']), 'ang': optim.AdamW(params_ang, lr=hyperparameters['lr_ang']) }
    schedulers = { 'kpt': CosineAnnealingLR(optimizers['kpt'], T_max=hyperparameters['num_epochs']), 'ang': CosineAnnealingLR(optimizers['ang'], T_max=hyperparameters['num_epochs']) }
    
    if rank == 0: print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val.")
    return model, train_loader, val_loader, criteria, optimizers, schedulers, device, mean, std, train_sampler

# ==============================================================================
# Cell 8: 메인 실행부
# ==============================================================================
import time

def main():
    rank = setup_ddp()
    world_size = dist.get_world_size()

    # --- 🖥️ GPU 설정 확인 (내부에서 CUDA_VISIBLE_DEVICES 설정 X) ---
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if torch.cuda.is_available():
        if rank == 0:
            print(f"✅ 사용 가능한 GPU: {torch.cuda.device_count()}개")
    else:
        if rank == 0:
            print("⚠️ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")

    # --- 📄 CSV 로드 (rank==0에서만 디스크 I/O) ---
    TOTAL_CSV_PATH = '../dataset/franka_research3/fr3_matched_joint_angle.csv'
    if rank == 0:
        print(f"\nLoading data from {TOTAL_CSV_PATH}...")
        total_csv = pd.read_csv(TOTAL_CSV_PATH)
        total_csv.sort_values('robot_timestamp', inplace=True, ignore_index=True)
        print("✅ CSV file loaded and sorted successfully.")
    else:
        total_csv = None  # placeholder

    # --- 모든 랭크에 CSV 브로드캐스트 ---
    obj_list = [total_csv]
    dist.broadcast_object_list(obj_list, src=0)
    total_csv = obj_list[0]

    # --- TIME_TOLERANCE 그리드 서치 (출력은 rank==0만) ---
    tolerance_candidates = np.round(np.arange(0.05, 0.101, 0.01), 2)
    best_tolerance_recommendation = 0
    max_full_groups = 0

    if rank == 0:
        print(f"\nStarting Grid Search for TIME_TOLERANCE in range: {list(tolerance_candidates)}")
    for tolerance in tolerance_candidates:
        temp_groups = perform_grouping(total_csv, tolerance, MAX_VIEWS_PER_GROUP)
        view_counts = [len(g['views']) for g in temp_groups]
        distribution = pd.Series(view_counts).value_counts().sort_index(ascending=False)

        if rank == 0:
            print("-" * 50)
            print(f"Testing Tolerance: {tolerance:.2f} seconds...")
            print(f"  -> Total groups created: {len(temp_groups)}")
            print("  -> View count distribution:")
            print(distribution.to_string())

        current_full_groups = distribution.get(8, 0)
        if current_full_groups > max_full_groups:
            max_full_groups = current_full_groups
            best_tolerance_recommendation = tolerance

    if rank == 0:
        print("-" * 50)
        print(f"\n🏆 Grid Search Recommendation: TIME_TOLERANCE = {best_tolerance_recommendation} (produced {max_full_groups} full groups)")

    # --- 최종 tolerance 적용 및 그룹 생성 ---
    final_tolerance = 0.07
    if rank == 0:
        print(f"\nFinal TIME_TOLERANCE set to: {final_tolerance}")
    dataset_groups = perform_grouping(total_csv, final_tolerance, MAX_VIEWS_PER_GROUP)
    if rank == 0:
        print(f"Total {len(dataset_groups)} groups created before filtering.")

    # --- 1뷰 그룹 제거 ---
    groups_before_filtering = len(dataset_groups)
    dataset_groups = [group for group in dataset_groups if len(group['views']) > 1]
    if rank == 0:
        print(f"ℹ️ Removed {groups_before_filtering - len(dataset_groups)} groups with only 1 view.")
        print(f"\n✅ Final Total Groups: {len(dataset_groups)}")
        total_images_in_groups = sum(len(g['views']) for g in dataset_groups)
        print(f"✅ Final Total Images to be used: {total_images_in_groups}")
        if dataset_groups:
            view_counts = [len(g['views']) for g in dataset_groups]
            print(f"\n--- Final View count distribution ---")
            print(pd.Series(view_counts).value_counts().sort_index(ascending=False))

    # --- 하이퍼파라미터 & 경로 ---
    hyperparameters = {
        'model_name': MODEL_NAME, 'batch_size': 18, 'num_epochs': 100, 'val_split': 0.1,
        'loss_weight_kpt': 100.0, 'lr_kpt': 1e-4, 'lr_ang': 1e-4,
    }
    RESULTS_DIR = "results_ddp"
    CHECKPOINT_PATH = 'multiview_checkpoint_ddp.pth'
    BEST_MODEL_PATH = 'best_multiview_model_ddp.pth'

    if rank == 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print("--- Data Preparation ---")

    # --- dataset_groups 객체 브로드캐스트 (파일 대신 broadcast_object_list 사용) ---
    obj_list = [dataset_groups]
    dist.broadcast_object_list(obj_list, src=0)
    dataset_groups = obj_list[0]

    # --- DINOv3 Processor 로드 (모든 랭크 동일) ---
    if rank == 0:
        print("Loading DINOv3 Processor for transformation config...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dino_mean = processor.image_mean
    dino_std = processor.image_std
    try:
        crop_size = processor.crop_size['height']
        resize_size = processor.size['shortest_edge']
    except (TypeError, KeyError):
        if rank == 0:
            print(f"Resized the image to 224x224")
        resize_size = crop_size = 224

    vis_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=dino_mean, std=dino_std)
    ])

    # --- 시각화는 rank==0에서만 (plt.show 사용하지 말고 저장만) ---
    if rank == 0:
        visualize_samples_by_group_size(dataset_groups, transform=vis_transform, mean=dino_mean, std=dino_std)

    dist.barrier()  # 모든 프로세스 동기화

    # --- 학습 세팅 ---
    model, train_loader, val_loader, criteria, optimizers, schedulers, device, mean, std, train_sampler = setup(
        hyperparameters, dataset_groups, rank, world_size
    )

    # visualize_* 함수 내부에서 사용할 전역 결과 경로 주입 (해당 함수가 results_dir를 전역 참조할 때 대비)
    global results_dir
    results_dir = RESULTS_DIR

    # --- wandb (rank==0 한정) ---
    if rank == 0:
        run = wandb.init(project="multiview-ddp-final", config=hyperparameters,
                         name=f"run_ddp_{time.strftime('%Y%m%d_%H%M%S')}")
        wandb.watch(model, log="all", log_freq=100)
    else:
        run = None

    start_epoch, best_val_loss = 0, float('inf')

    # --- 체크포인트 로드 ---
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=lambda storage, loc: storage.cuda(rank))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        # (옵티마이저/스케줄러 상태 복원 필요 시 여기에 추가)
        if rank == 0:
            print(f"✅ Resuming training from checkpoint.")

    # --- 학습 루프 ---
    if rank == 0:
        print("\n--- Starting Training ---")
    for epoch in range(start_epoch, hyperparameters['num_epochs']):
        train_sampler.set_epoch(epoch)

        train_loss_kpt, train_loss_ang = train_one_epoch(
            model, train_loader, optimizers, criteria, device, hyperparameters['loss_weight_kpt'], epoch + 1
        )
        val_loss = validate(
            model, val_loader, criteria, device, hyperparameters['loss_weight_kpt'], epoch + 1
        )
        schedulers['kpt'].step(); schedulers['ang'].step()

        if rank == 0:
            current_lr_kpt = optimizers['kpt'].param_groups[0]['lr']
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_kpt": train_loss_kpt,
                "train_loss_ang": train_loss_ang,
                "avg_val_loss": val_loss,
                "lr_kpt": current_lr_kpt
            })
            print(f"Epoch {epoch+1} -> Val Loss: {val_loss:.6f} | LR_kpt: {current_lr_kpt:.6f}")

            # DataParallel 분기 불필요 (DDP 사용 중)지만 안전하게 유지
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"🎉 New best model saved with validation loss: {best_val_loss:.6f}")
                torch.save(state_to_save, BEST_MODEL_PATH)

                # 주의: visualize_predictions 시그니처가 (model, dataset, device, mean, std, epoch_num, num_samples=1) 인 경우
                # 아래 호출에서 RESULTS_DIR 인자를 제거하세요. (현재 코드엔 RESULTS_DIR를 전역 results_dir로 주입)
                figs = visualize_predictions(model, val_loader.dataset, device, mean, std, epoch + 1, RESULTS_DIR, 1)
                wandb.log({"validation_predictions": [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    plt.close(fig)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': state_to_save,
                'optimizer_kpt_state_dict': optimizers['kpt'].state_dict(),
                'optimizer_ang_state_dict': optimizers['ang'].state_dict(),
                'scheduler_kpt_state_dict': schedulers['kpt'].state_dict(),
                'scheduler_ang_state_dict': schedulers['ang'].state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)

    cleanup_ddp()

    if rank == 0:
        print("\n--- Training Finished ---")
        if run is not None:
            run.finish()

if __name__ == '__main__':
    main()
