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
import argparse
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, SiglipVisionModel
from scipy.spatial.transform import Rotation as R

# DDP 관련 라이브러리
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# [추가] Kornia 라이브러리
import kornia

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ======================= DDP 설정 함수 =======================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
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
            return self.__getitem__((idx + 1) % len(self))
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_pil = Image.fromarray(img_rgb)
        image_tensor = self.transform(img_pil)

        Ht, Wt = (224, 224)
        keypoints = sample["objects"][0]["keypoints"]
        joint_num = len(keypoints)
        kpts_2d_orig = np.array([kp["projected_location"] for kp in keypoints], dtype=np.float32)
        kpts_on_heatmap = _scale_points(kpts_2d_orig, from_size=(w, h), to_size=(Wt, Ht))
        
        heatmaps_np = np.zeros((joint_num, Ht, Wt), dtype=np.float32)
        for i in range(joint_num):
            heatmaps_np[i] = create_gt_heatmap(kpts_on_heatmap[i], (Ht, Wt), self.sigma)
        gt_heatmaps = torch.from_numpy(heatmaps_np)

        angles = [angle["position"] for angle in sample['sim_state']["joints"]]
        gt_angles = torch.tensor(angles, dtype=torch.float32)
        
        gt_class = sample['objects'][0]['class']
        gt_3d_points = torch.tensor([kp["location"] for kp in keypoints])
        K = torch.tensor(sample['meta']['K'], dtype=torch.float32)
        dist = torch.tensor(sample['meta']['dist_coeffs'], dtype=torch.float32)
            
        return image_tensor, gt_heatmaps, gt_angles, gt_class, gt_3d_points, K, dist

def robot_collate_fn(batch):
    image_tensor, gt_heatmaps, gt_angles, gt_class, gt_3d_points, K_list, dist_list = zip(*batch)
    image_tensors = torch.stack(image_tensor, 0)
    
    MAX_JOINTS = 7
    MAX_ANGLES = 9
    MAX_POINTS = 7

    heatmaps_padded = torch.zeros(len(gt_heatmaps), MAX_JOINTS, gt_heatmaps[0].shape[1], gt_heatmaps[0].shape[2])
    angles_padded   = torch.zeros(len(gt_angles), MAX_ANGLES)
    points_padded   = torch.zeros(len(gt_3d_points), MAX_POINTS, 3)

    joint_lengths = torch.zeros(len(gt_heatmaps), dtype=torch.long)
    angle_lengths = torch.zeros(len(gt_angles), dtype=torch.long)
    point_lengths = torch.zeros(len(gt_3d_points), dtype=torch.long)

    for i, (h, a, p) in enumerate(zip(gt_heatmaps, gt_angles, gt_3d_points)):
        joint_num, angle_num, point_num = h.shape[0], a.shape[0], p.shape[0]

        heatmaps_padded[i, :joint_num, :, :] = h
        angles_padded[i, :angle_num] = a
        points_padded[i, :point_num] = p

        joint_lengths[i], angle_lengths[i], point_lengths[i] = joint_num, angle_num, point_num

    K = torch.stack(K_list, 0)
    dist = torch.stack(dist_list, 0)
    
    return image_tensors, heatmaps_padded, angles_padded, gt_class, points_padded, K, dist, joint_lengths, angle_lengths, point_lengths

# ======================= 모델 아키텍처 (변경 없음) =======================
# ... (DINOv3Backbone, JointAngleHead 등 이전과 동일한 모델 코드) ...
FEATURE_DIM = 512
NUM_ANGLES = 9
NUM_JOINTS = 7

class DINOv3Backbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        if "siglip" in model_name:
            self.model = SiglipVisionModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)
    def forward(self, image_tensor_batch):
        with torch.no_grad():
            if "siglip" in self.model_name:
                outputs = self.model(pixel_values=image_tensor_batch, interpolate_pos_encoding=True)
                tokens = outputs.last_hidden_state
                patch_tokens = tokens[:, 1:, :]
            else:
                outputs = self.model(pixel_values=image_tensor_batch)
                tokens = outputs.last_hidden_state
                num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
                patch_tokens = tokens[:, 1 + num_reg :, :]
            return patch_tokens

class JointAngleHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_angles=NUM_ANGLES, num_queries=4, nhead=8, num_decoder_layers=2):
        super().__init__()
        self.pose_queries = nn.Parameter(torch.randn(1, num_queries, input_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=input_dim * 4,
                                                   dropout=0.1, activation='gelu', batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.angle_predictor = nn.Sequential(
            nn.LayerNorm(input_dim * num_queries),
            nn.Linear(input_dim * num_queries, 512), nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(),
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        projected = self.projection(x)
        refined = self.refine_blocks(projected)
        residual = self.residual_conv(x)
        return torch.nn.functional.gelu(refined + residual)

class LightCNNStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU()
        )
    def forward(self, x):
        feat_4 = self.conv_block1(x)
        feat_8 = self.conv_block2(feat_4)
        return feat_4, feat_8

class FusedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU()
        )
    def forward(self, x, skip_feature):
        x = self.upsample(x)
        if x.shape[-2:] != skip_feature.shape[-2:]:
            skip_feature = F.interpolate(skip_feature, size=x.shape[-2:], mode='bilinear', align_corners=False)
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
        b, n, d = dino_features.shape
        h = w = int(math.sqrt(n))
        if h * w != n:
            n_new = h * w
            dino_features = dino_features[:, :n_new, :]
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        x = self.token_fuser(x)
        x = self.decoder_block1(x, cnn_feat_8)
        x = self.decoder_block2(x, cnn_feat_4)
        x = self.final_upsample(x)
        heatmaps = self.heatmap_predictor(x)
        return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
        
class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, ablation_mode=None):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.backbone = DINOv3Backbone(dino_model_name)
        
        if "siglip" in self.dino_model_name:
            feature_dim = self.backbone.model.config.hidden_size
        else:
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size
        
        self.cnn_stem = LightCNNStem()
        self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim)
        self.angle_head = JointAngleHead(input_dim=feature_dim)
        
    def forward(self, image_tensor_batch):
        dino_features = self.backbone(image_tensor_batch)
        cnn_stem_features = self.cnn_stem(image_tensor_batch)
        predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)
        predicted_angles = self.angle_head(dino_features)
        return predicted_heatmaps, predicted_angles

# ======================= [수정] PyTorch 기반 기구학 (Differentiable Kinematics) =======================
def get_dh_matrix_torch(a, d, alpha, theta, B, device):
    alpha_rad = torch.deg2rad(alpha)
    theta_rad = torch.deg2rad(theta)
    
    cos_a, sin_a = torch.cos(alpha_rad), torch.sin(alpha_rad)
    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)

    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)
    
    T = torch.stack([
        torch.stack([cos_t, -sin_t * cos_a,  sin_t * sin_a, a * cos_t], dim=-1),
        torch.stack([sin_t,  cos_t * cos_a, -cos_t * sin_a, a * sin_t], dim=-1),
        torch.stack([zeros, sin_a.expand(B), cos_a.expand(B), d.expand(B)], dim=-1),
        torch.stack([zeros, zeros, zeros, ones], dim=-1)
    ], dim=-2)
    return T

class RobotKinematicsTorch:
    def __init__(self, robot_name, device):
        self.name = robot_name
        self.device = device
        self.dh_params_list = self._get_dh_params_list()
        self.num_joints = len(self.dh_params_list)
        self.dh_tensors = self._load_dh_tensors()
        self.base_correction = torch.eye(4, device=device) # 필요시 오버라이드

    def _get_dh_params_list(self):
        if self.name == "Meca500":
             return [
                {'alpha': -90, 'a': 0,     'd': 0.135, 'theta_offset': 0},
                {'alpha': 0,   'a': 0.135, 'd': 0,     'theta_offset': -90},
                {'alpha': -90, 'a': 0.038, 'd': 0,     'theta_offset': 0},
                {'alpha': 90,  'a': 0,     'd': 0.120, 'theta_offset': 0},
                {'alpha': -90, 'a': 0,     'd': 0,     'theta_offset': 0},
                {'alpha': 0,   'a': 0,     'd': 0.070, 'theta_offset': 0}
            ]
        # [참고] 다른 로봇들도 여기에 추가
        else:
            raise ValueError(f"Unknown robot for Torch Kinematics: {self.name}")

    def _load_dh_tensors(self):
        return {
            'a': torch.tensor([p['a'] for p in self.dh_params_list], device=self.device),
            'd': torch.tensor([p['d'] for p in self.dh_params_list], device=self.device),
            'alpha': torch.tensor([p['alpha'] for p in self.dh_params_list], device=self.device),
            'theta_offset': torch.tensor([p['theta_offset'] for p in self.dh_params_list], device=self.device)
        }

    def forward_kinematics(self, joint_angles_rad):
        B = joint_angles_rad.shape[0]
        thetas_deg = torch.rad2deg(joint_angles_rad[:, :self.num_joints]) + self.dh_tensors['theta_offset']
        
        T_cumulative = self.base_correction.unsqueeze(0).repeat(B, 1, 1)
        base_point = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).view(1, 4, 1).repeat(B, 1, 1)
        
        joint_coords = T_cumulative[:, :3, 3].unsqueeze(1) # (B, 1, 3), 베이스 좌표

        for i in range(self.num_joints):
            T_i = get_dh_matrix_torch(self.dh_tensors['a'][i], self.dh_tensors['d'][i],
                                      self.dh_tensors['alpha'][i], thetas_deg[:, i], B, self.device)
            T_cumulative = torch.bmm(T_cumulative, T_i)
            new_pos = torch.bmm(T_cumulative, base_point)[:, :3, 0].unsqueeze(1)
            joint_coords = torch.cat([joint_coords, new_pos], dim=1)
            
        return joint_coords

# ======================= [수정] Kornia 기반 PnP 및 좌표 변환 함수 =======================
def extract_2d_points_from_heatmap(heatmaps):
    B, J, H, W = heatmaps.shape
    max_indices = torch.argmax(heatmaps.view(B, J, -1), dim=2)
    y_coords = (max_indices / W).float()
    x_coords = (max_indices % W).float()
    return torch.stack([x_coords, y_coords], dim=2)

def solve_pnp_and_transform_torch(joints_3d_robot, kpts2d, K, valid_mask):
    B, J, _ = joints_3d_robot.shape
    joints_3d_cam_batch = torch.zeros_like(joints_3d_robot)
    
    for i in range(B):
        mask = valid_mask[i]
        if mask.sum() < 4: continue # PnP는 최소 4개의 점이 필요

        obj_pts = joints_3d_robot[i, mask]
        img_pts = kpts2d[i, mask]
        
        try:
            R_mat, t_vec = kornia.geometry.solve_pnp_dlt(obj_pts.unsqueeze(0), img_pts.unsqueeze(0), K[i].unsqueeze(0))
            
            X = torch.transpose(joints_3d_robot[i], 0, 1) # (3, J)
            Y = torch.matmul(R_mat[0], X) + t_vec[0]      # (3, J)
            joints_3d_cam_batch[i] = torch.transpose(Y, 0, 1) # (J, 3)
        except:
            continue # PnP 실패 시 0으로 유지
            
    return joints_3d_cam_batch

# ======================= [수정] 손실 함수 =======================
def compute_masked_loss(pred_heatmaps, gt_heatmaps, pred_angles, gt_angles, pred_3d, gt_3d,
                        joint_lengths, angle_lengths, point_lengths,
                        loss_fn_h, loss_fn_a, loss_fn_3D,
                        weight_h=1.0, weight_a=1.0, weight_3d=1.0):
    device = pred_heatmaps.device
    B, J_max, H, W = gt_heatmaps.shape
    A_max = gt_angles.shape[1]
    P_max = gt_3d.shape[1]
    
    mask_h = (torch.arange(J_max, device=device)[None, :] < joint_lengths[:, None]).unsqueeze(-1).unsqueeze(-1)
    loss_h = loss_fn_h(pred_heatmaps * mask_h, gt_heatmaps * mask_h)

    mask_a = (torch.arange(A_max, device=device)[None, :] < angle_lengths[:, None])
    loss_a = loss_fn_a(pred_angles[mask_a], gt_angles[mask_a])

    mask_p = (torch.arange(P_max, device=device)[None, :] < point_lengths[:, None]).unsqueeze(-1)
    loss_3d = loss_fn_3D(pred_3d * mask_p, gt_3d * mask_p)

    total_loss = weight_h * loss_h + weight_a * loss_a + weight_3d * loss_3d

    return total_loss, {
        'loss_h': loss_h.detach(), 'loss_a': loss_a.detach(), 'loss_3d': loss_3d.detach()
    }

def save_checkpoints(checkpoint_data, best_model_state_dict, checkpoint_dir, is_best):
    torch.save(checkpoint_data, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
    if is_best:
        torch.save(best_model_state_dict, os.path.join(checkpoint_dir, "best_model.pth"))
        
# ======================= 메인 학습 함수 =======================
def main(args): 
    rank, local_rank, world_size = setup_ddp()
    save_thread = None
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 32 # [수정] GPU 메모리 사용량이 늘어나므로 배치 사이즈 감소
    EPOCHS = 100
    VAL_RATIO = 0.1

    ablation_mode = args.ablation_mode
    WANDB_PROJECT = f"DINOv3_Ablation_Differentiable_{ablation_mode}"
    CHECKPOINT_DIR = f"checkpoints_differentiable_{ablation_mode}"
    LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    
    MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m' # 단순화를 위해 모델 고정

    model = DINOv3PoseEstimator(dino_model_name=MODEL_NAME, ablation_mode=ablation_mode).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    loss_fn_h  = nn.MSELoss()
    loss_fn_a  = nn.SmoothL1Loss()
    loss_fn_3D = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)
    scaler = torch.cuda.amp.GradScaler()
    
    # [추가] PyTorch 기반 로봇 기구학 객체 생성
    kinematics_solvers = {
        "Meca500": RobotKinematicsTorch("Meca500", local_rank)
        # 다른 로봇들도 여기에 추가
    }
    
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(LATEST_CHECKPOINT_PATH):
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=f'cuda:{local_rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if rank == 0: print(f"✅ 체크포인트 로드 완료. {start_epoch} 에포크부터 재개합니다.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Heatmap 크기와 통일
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if rank == 0: print("Loading dataset...")
    json_files = glob.glob("../dataset/Converted_dataset/Meca500/**/*.json", recursive=True) # Meca500만 테스트
    
    full_dataset = RobotPoseDataset(json_files, transform)
    train_size = int(len(full_dataset) * (1 - VAL_RATIO))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True, collate_fn=robot_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True, collate_fn=robot_collate_fn)
    
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        wandb.init(project=WANDB_PROJECT, name=f"run_{ablation_mode}", resume="allow")

    weight_h, weight_a, weight_3d = 3.0, 1.0, 2.0

    for epoch in range(start_epoch, EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=(rank != 0))
        for batch in pbar:
            images, gt_heatmaps, gt_angles, gt_class, gt_3d, K, _, joint_lengths, angle_lengths, point_lengths = batch
            
            images, gt_heatmaps, gt_angles, gt_3d, K = \
                images.to(local_rank), gt_heatmaps.to(local_rank), gt_angles.to(local_rank), gt_3d.to(local_rank), K.to(local_rank)
            joint_lengths, angle_lengths, point_lengths = \
                joint_lengths.to(local_rank), angle_lengths.to(local_rank), point_lengths.to(local_rank)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred_heatmaps, pred_angles = model(images)
                
                # --- [수정] Differentiable 3D Prediction ---
                # 현재는 배치 내 모든 로봇이 같다고 가정 (Meca500)
                kin_solver = kinematics_solvers[gt_class[0]] 
                
                # 1. Forward Kinematics (PyTorch)
                joints_3d_robot = kin_solver.forward_kinematics(pred_angles)
                
                # 2. Extract 2D points (GT 사용, pred_heatmaps로 교체 가능)
                kpts2d = extract_2d_points_from_heatmap(gt_heatmaps)
                
                # 3. PnP & Transform (Kornia)
                valid_mask = (torch.arange(MAX_JOINTS, device=local_rank)[None, :] < point_lengths[:, None])
                pred_3d = solve_pnp_and_transform_torch(joints_3d_robot, kpts2d, K, valid_mask)
                # ---------------------------------------------
                
                total_loss, loss_dict = compute_masked_loss(
                    pred_heatmaps, gt_heatmaps, pred_angles, gt_angles, pred_3d, gt_3d,
                    joint_lengths, angle_lengths, point_lengths,
                    loss_fn_h, loss_fn_a, loss_fn_3D, weight_h, weight_a, weight_3d
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0: pbar.set_postfix(loss=total_loss.item())

        # 검증 루프 (학습 루프와 동일한 로직, no_grad 컨텍스트)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=(rank != 0))
            for batch in val_pbar:
                images, gt_heatmaps, gt_angles, gt_class, gt_3d, K, _, joint_lengths, angle_lengths, point_lengths = batch
                images, gt_heatmaps, gt_angles, gt_3d, K = \
                    images.to(local_rank), gt_heatmaps.to(local_rank), gt_angles.to(local_rank), gt_3d.to(local_rank), K.to(local_rank)
                joint_lengths, angle_lengths, point_lengths = \
                    joint_lengths.to(local_rank), angle_lengths.to(local_rank), point_lengths.to(local_rank)

                with torch.cuda.amp.autocast():
                    pred_heatmaps, pred_angles = model(images)
                    kin_solver = kinematics_solvers[gt_class[0]] 
                    joints_3d_robot = kin_solver.forward_kinematics(pred_angles)
                    kpts2d = extract_2d_points_from_heatmap(gt_heatmaps)
                    valid_mask = (torch.arange(MAX_JOINTS, device=local_rank)[None, :] < point_lengths[:, None])
                    pred_3d = solve_pnp_and_transform_torch(joints_3d_robot, kpts2d, K, valid_mask)
                    
                    total_loss, loss_dict = compute_masked_loss(
                        pred_heatmaps, gt_heatmaps, pred_angles, gt_angles, pred_3d, gt_3d,
                        joint_lengths, angle_lengths, point_lengths,
                        loss_fn_h, loss_fn_a, loss_fn_3D, weight_h, weight_a, weight_3d
                    )
                val_loss += total_loss
        
        avg_val_loss = val_loss / len(val_loader)
        dist.all_reduce(avg_val_loss, op=dist.ReduceOp.AVG)
        scheduler.step()
        
        if rank == 0:
            wandb.log({ "val_loss": avg_val_loss.item(), **loss_dict })
            print(f"Epoch {epoch+1} -> Val Loss: {avg_val_loss.item():.6f}")
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"✨ New best model saved!")
            
            # 체크포인트 저장 (간소화)
            checkpoint_data = {'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                               'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'best_val_loss': best_val_loss}
            torch.save(checkpoint_data, LATEST_CHECKPOINT_PATH)
            if is_best:
                torch.save(model.module.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    if rank == 0: wandb.finish()
    cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DINOv3 Differentiable Pose Estimation")
    parser.add_argument('--ablation_mode', type=str, default='combined', help="Select the ablation mode")
    args = parser.parse_args()
    main(args)