import os
import glob
import json
import numpy as np
import random
import wandb
import time
import cv2
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

# DDP 관련 라이브러리
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ======================= DDP 설정 함수 =======================
def setup_ddp():
    """DDP 프로세스 그룹을 초기화합니다."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    """DDP 프로세스 그룹을 정리합니다."""
    dist.destroy_process_group()

# ======================= 데이터셋 클래스 및 함수 =======================
def create_gt_heatmap(keypoint_2d, heatmap_size, sigma):
    H, W = heatmap_size
    x, y = keypoint_2d
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist_sq = (xx - x)**2 + (yy - y)**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    heatmap[heatmap < np.finfo(float).eps * heatmap.max()] = 0
    return heatmap

def _scale_points(points_xy, from_size, to_size):
    Wf, Hf = from_size
    Wt, Ht = to_size
    out = np.empty_like(points_xy, dtype=np.float32)
    out[:, 0] = points_xy[:, 0] * (Wt / float(Wf))
    out[:, 1] = points_xy[:, 1] * (Ht / float(Hf))
    return out

class RobotPoseDataset(Dataset):
    def __init__(self, json_files, transform, sigma=2.0):
        self.json_files = json_files
        self.transform = transform
        self.sigma = sigma

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        
        image_path = sample['meta']['image_path']
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_pil = Image.fromarray(img_rgb)
        image_tensor = self.transform(img_pil)

        Ht, Wt = (224, 224)
        keypoints = sample["objects"][0]["keypoints"]
        joint_num = len(keypoints)
        kpts_2d_orig = np.array([kp["projected_location"] for kp in keypoints])
        kpts_on_heatmap = _scale_points(kpts_2d_orig, from_size=(w, h), to_size=(Wt, Ht))
        
        heatmaps_np = np.zeros((joint_num, Ht, Wt), dtype=np.float32)
        for i in range(joint_num):
            heatmaps_np[i] = create_gt_heatmap(kpts_on_heatmap[i], (Ht, Wt), self.sigma)
        gt_heatmaps = torch.from_numpy(heatmaps_np)

        angles = [angle["position"] for angle in sample['sim_state']["joints"]]
        gt_angles = torch.tensor(angles, dtype=torch.float32)
        
        return image_tensor, gt_heatmaps, gt_angles

###----------- 관절개수 패딩 관련 -------------###
def robot_collate_fn(batch):
    images, heatmaps, angles = zip(*batch)
    images = torch.stack(images, 0)

    MAX_JOINTS = 9
    MAX_ANGLES = 9

    heatmaps_padded = torch.zeros(len(heatmaps), MAX_JOINTS, heatmaps[0].shape[1], heatmaps[0].shape[2])
    angles_padded = torch.zeros(len(angles), MAX_ANGLES)
    
    lengths = []
    for i, (h, a) in enumerate(zip(heatmaps, angles)):
        num_joints = h.shape[0]
        lengths.append(num_joints)
        heatmaps_padded[i, :num_joints, :, :] = h
        angles_padded[i, :a.shape[0]] = a

    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, heatmaps_padded, angles_padded, lengths

# ======================= 모델 정의 =======================
# 모델 클래스에서 사용할 상수들을 미리 정의
MODEL_NAME = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
FEATURE_DIM = 512
NUM_ANGLES = 9
NUM_JOINTS = 9

class DINOv3Backbone(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, image_tensor_batch):
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor_batch)
        tokens = outputs.last_hidden_state
        num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
        patch_tokens = tokens[:, 1 + num_reg :, :]  # (B, N_patches, D)
        return patch_tokens

class JointAngleHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_angles=NUM_ANGLES, num_queries=4, nhead=8, num_decoder_layers=2):
        super().__init__()
        
        self.pose_queries = nn.Parameter(torch.randn(1, num_queries, input_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=input_dim * 4, # 일반적인 설정
            dropout=0.1, 
            activation='gelu',
            batch_first=True  # (batch, seq, feature) 입력을 위함
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
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
        b = fused_features.size(0)
        queries = self.pose_queries.repeat(b, 1, 1)
        attn_output = self.transformer_decoder(tgt=queries, memory=fused_features)
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
        
        # skip_feature를 x의 크기에 강제로 맞춥니다.
        # ----------------------------------------------------------------------
        # 두 텐서의 높이와 너비가 다를 경우, skip_feature를 x의 크기로 리사이즈합니다.
        if x.shape[-2:] != skip_feature.shape[-2:]:
            skip_feature = F.interpolate(
                skip_feature, 
                size=x.shape[-2:], # target H, W
                mode='bilinear', 
                align_corners=False
            )
        fused = torch.cat([x, skip_feature], dim=1)
        return self.refine_conv(fused)

class UNetViTKeypointHead(nn.Module):
    def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(224, 224)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)
        self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
        self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.heatmap_predictor = nn.Conv2d(64, num_joints, kernel_size=3, padding=1)

    def forward(self, dino_features, cnn_features):
        cnn_feat_4, cnn_feat_8 = cnn_features
        num_patches_to_keep = 196
        dino_features_sliced = dino_features[:, :num_patches_to_keep, :]
        
        b, n, d = dino_features_sliced.shape
        h = w = int(n**0.5)
        x = dino_features_sliced.permute(0, 2, 1).reshape(b, d, h, w)

        x = self.token_fuser(x)
        x = self.decoder_block1(x, cnn_feat_8)
        x = self.decoder_block2(x, cnn_feat_4)
        x = self.final_upsample(x)
        heatmaps = self.heatmap_predictor(x)
        
        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
        
class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name=MODEL_NAME):
        super().__init__()
        self.backbone = DINOv3Backbone(dino_model_name)
        feature_dim = self.backbone.model.config.hidden_size
        self.cnn_stem = LightCNNStem() # 경량 CNN 스템 추가
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim) # 새로운 UNet-ViT 헤드
        self.angle_head = JointAngleHead(input_dim=feature_dim)      # 기존 Attention 헤드

    def forward(self, image_tensor_batch):
        dino_features = self.backbone(image_tensor_batch)      # 의미 정보
        cnn_stem_features = self.cnn_stem(image_tensor_batch) # 공간 정보
        predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)
        predicted_angles = self.angle_head(dino_features)
        
        return predicted_heatmaps, predicted_angles

def compute_masked_loss(pred_heatmaps, gt_heatmaps, pred_angles, gt_angles, lengths, 
                        loss_fn_h, loss_fn_a, weight_h, weight_a):
    """패딩을 제외한 실제 데이터에 대해서만 가중합 손실을 계산합니다."""
    device = pred_heatmaps.device
    
    # 1. 마스크 생성
    mask_angle = torch.arange(gt_angles.shape[1], device=device)[None, :] < lengths[:, None]
    mask_heatmap = mask_angle[:, :, None, None].expand_as(gt_heatmaps)
    
    # 2. 마스킹된 손실 계산
    loss_h = (loss_fn_h(pred_heatmaps, gt_heatmaps) * mask_heatmap).sum() / mask_heatmap.sum()
    loss_a = (loss_fn_a(pred_angles, gt_angles) * mask_angle).sum() / mask_angle.sum()
    
    # 3. 최종 가중합 손실 반환
    total_loss = (weight_h * loss_h) + (weight_a * loss_a)
    return total_loss

# ======================= 메인 학습 함수 =======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank, local_rank, world_size = setup_ddp()

    # --- 하이퍼파라미터 ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64  # GPU당 배치 사이즈
    EPOCHS = 100
    VAL_RATIO = 0.1
    WANDB_PROJECT = "DINOv3_Pose_Estimation_DDP"
    
    if rank == 0:
        wandb.init(project=WANDB_PROJECT, config={
            "learning_rate": LEARNING_RATE, "total_batch_size": BATCH_SIZE * world_size,
            "epochs": EPOCHS, "world_size": world_size
        })

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if rank == 0: print("Loading dataset files...")
    json_files = glob.glob("../dataset/Converted_dataset/**/*.json", recursive=True)
    if rank == 0: print(f"Found {len(json_files)} files.")

    full_dataset = RobotPoseDataset(json_files, transform)
    train_size = int(len(full_dataset) * (1 - VAL_RATIO))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                              pin_memory=True, collate_fn=robot_collate_fn, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                            pin_memory=True, collate_fn=robot_collate_fn, sampler=val_sampler)

    model = DINOv3PoseEstimator().to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_fn_heatmap = nn.MSELoss(reduction='none')
    loss_fn_angle = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    heatmap_loss_weight = 1.0
    angle_loss_weight = 0.5
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=(rank != 0))
        for batch in pbar:
            if batch[0] is None: continue
            images, gt_heatmaps, gt_angles, lengths = batch
            images, gt_heatmaps, gt_angles, lengths = \
                images.to(local_rank), gt_heatmaps.to(local_rank), gt_angles.to(local_rank), lengths.to(local_rank)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                pred_heatmaps, pred_angles = model(images)
                total_loss = compute_masked_loss(pred_heatmaps, gt_heatmaps, pred_angles, gt_angles, lengths,
                                                 loss_fn_heatmap, loss_fn_angle, heatmap_loss_weight, angle_loss_weight)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 모든 GPU의 loss를 더해서 train_loss에 누적
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_loss += total_loss.item() / world_size

            if rank == 0:
                pbar.set_postfix(loss=total_loss.item() / world_size)
        
        # --- 검증 루프 (수정된 부분) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # tqdm도 메인 프로세스에서만 보이도록 수정
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", disable=(rank != 0))
            for batch in val_pbar:
                if batch[0] is None: continue
                images, gt_heatmaps, gt_angles, lengths = batch
                images, gt_heatmaps, gt_angles, lengths = \
                    images.to(local_rank), gt_heatmaps.to(local_rank), gt_angles.to(local_rank), lengths.to(local_rank)
                
                with torch.cuda.amp.autocast():
                    pred_heatmaps, pred_angles = model(images)
                    total_loss = compute_masked_loss(pred_heatmaps, gt_heatmaps, pred_angles, gt_angles, lengths,
                                                     loss_fn_heatmap, loss_fn_angle, heatmap_loss_weight, angle_loss_weight)
                
                # 모든 GPU의 loss를 더해서 val_loss에 누적
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                val_loss += total_loss.item() / world_size
        
        scheduler.step()
        
        # --- 로깅 및 모델 저장 (메인 프로세스에서만) ---
        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })

            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs("checkpoints", exist_ok=True)
                # DDP 모델 저장 시에는 model.module을 저장해야 함
                model_to_save = model.module.state_dict()
                torch.save(model_to_save, f"checkpoints/best_model_epoch_{epoch+1}.pth")
                print(f"✨ New best model saved with validation loss: {best_val_loss:.6f}")

    if rank == 0:
        wandb.finish()
    cleanup_ddp()

if __name__ == '__main__':
    main()