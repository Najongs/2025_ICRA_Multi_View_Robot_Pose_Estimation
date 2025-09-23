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
import argparse # 인자 파싱을 위해 추가
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, SiglipVisionModel

# DDP 관련 라이브러리
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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

        # Ht, Wt = (224, 224)
        # keypoints = sample["objects"][0]["keypoints"]
        # joint_num = len(keypoints)
        # kpts_2d_orig = np.array([kp["projected_location"] for kp in keypoints])
        # kpts_on_heatmap = _scale_points(kpts_2d_orig, from_size=(w, h), to_size=(Wt, Ht))
        
        # heatmaps_np = np.zeros((joint_num, Ht, Wt), dtype=np.float32)
        # for i in range(joint_num):
        #     heatmaps_np[i] = create_gt_heatmap(kpts_on_heatmap[i], (Ht, Wt), self.sigma)
        # gt_heatmaps = torch.from_numpy(heatmaps_np)

        angles = [angle["position"] for angle in sample['sim_state']["joints"]]
        gt_angles = torch.tensor(angles, dtype=torch.float32)
        
        return image_tensor, gt_angles

###----------- 관절개수 패딩 관련 (수정) -------------###
def robot_collate_fn(batch):
    images, angles = zip(*batch)
    images = torch.stack(images, 0)

    MAX_ANGLES = 9 # UR5e + Gripper 고려

    # 각도 텐서만 패딩 처리
    angles_padded = torch.zeros(len(angles), MAX_ANGLES)
    for i, a in enumerate(angles):
        # a.shape[0]이 MAX_ANGLES를 넘지 않도록 안전장치 추가
        num_angles_to_copy = min(a.shape[0], MAX_ANGLES)
        angles_padded[i, :num_angles_to_copy] = a[:num_angles_to_copy]

    return images, angles_padded

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
                outputs = self.model(
                    pixel_values=image_tensor_batch,
                    interpolate_pos_encoding=True 
                )
                tokens = outputs.last_hidden_state
                patch_tokens = tokens[:, 1:, :]
            else: # DINOv3 계열
                outputs = self.model(pixel_values=image_tensor_batch)
                tokens = outputs.last_hidden_state
                num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
                patch_tokens = tokens[:, 1 + num_reg :, :]
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

# class TokenFuser(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.refine_blocks = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#         self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#     def forward(self, x):
#         projected = self.projection(x)
#         refined = self.refine_blocks(projected)
#         residual = self.residual_conv(x)
#         return torch.nn.functional.gelu(refined + residual)

# class LightCNNStem(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/2
#             nn.BatchNorm2d(16),
#             nn.GELU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/4
#             nn.BatchNorm2d(32),
#             nn.GELU()
#         )
#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 해상도 1/8
#             nn.BatchNorm2d(64),
#             nn.GELU()
#         )
        
#     def forward(self, x):
#         feat_4 = self.conv_block1(x)  # 1/4 스케일 특징
#         feat_8 = self.conv_block2(feat_4) # 1/8 스케일 특징
#         return feat_4, feat_8 # 다른 해상도의 특징들을 반환

# class FusedUpsampleBlock(nn.Module):
#     def __init__(self, in_channels, skip_channels, out_channels):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.refine_conv = nn.Sequential(
#             nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU()
#         )

#     def forward(self, x, skip_feature):
#         x = self.upsample(x)

#         if x.shape[-2:] != skip_feature.shape[-2:]:
#             skip_feature = F.interpolate(
#                 skip_feature, 
#                 size=x.shape[-2:], # target H, W
#                 mode='bilinear', 
#                 align_corners=False
#             )

#         fused = torch.cat([x, skip_feature], dim=1)
#         return self.refine_conv(fused)

# class UNetViTKeypointHead(nn.Module):
#     def __init__(self, input_dim=768, num_joints=NUM_JOINTS, heatmap_size=(224, 224)):
#         super().__init__()
#         self.heatmap_size = heatmap_size
#         self.token_fuser = TokenFuser(input_dim, 256)
#         self.decoder_block1 = FusedUpsampleBlock(in_channels=256, skip_channels=64, out_channels=128)
#         self.decoder_block2 = FusedUpsampleBlock(in_channels=128, skip_channels=32, out_channels=64)
#         self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.heatmap_predictor = nn.Conv2d(64, num_joints, kernel_size=3, padding=1)

#     def forward(self, dino_features, cnn_features):
#         cnn_feat_4, cnn_feat_8 = cnn_features
#         b, n, d = dino_features.shape
#         h = w = int(math.sqrt(n))

#         if h * w != n:
#             n_new = h * w
#             dino_features = dino_features[:, :n_new, :]
#         x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        
#         x = self.token_fuser(x)
#         x = self.decoder_block1(x, cnn_feat_8)
#         x = self.decoder_block2(x, cnn_feat_4)
#         x = self.final_upsample(x)
#         heatmaps = self.heatmap_predictor(x)
        
#         return F.interpolate(heatmaps, size=self.heatmap_size, mode='bilinear', align_corners=False)
        
class DINOv3PoseEstimator(nn.Module):
    def __init__(self, dino_model_name, ablation_mode=None):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.dino_model_name = dino_model_name
        self.backbone = DINOv3Backbone(dino_model_name)
        
        if "siglip" in self.dino_model_name:
            config = self.backbone.model.config
            feature_dim = config.hidden_size
        else: # DINOv3 계열
            config = self.backbone.model.config
            feature_dim = config.hidden_sizes[-1] if "conv" in self.dino_model_name else config.hidden_size
        
        # self.cnn_stem = LightCNNStem()
        # self.keypoint_head = UNetViTKeypointHead(input_dim=feature_dim)
        self.angle_head = JointAngleHead(input_dim=feature_dim)      # 기존 Attention 헤드
        
    def forward(self, image_tensor_batch):
        dino_features = self.backbone(image_tensor_batch) # 항상 3D 텐서
        # cnn_stem_features = self.cnn_stem(image_tensor_batch)

        if self.ablation_mode == 'cnn_only':
            dino_features = torch.zeros_like(dino_features)
        # elif 'dino_only' in self.ablation_mode or 'siglip_only' in self.ablation_mode:
        #     cnn_stem_features = [torch.zeros_like(feat) for feat in cnn_stem_features]
        
        # predicted_heatmaps = self.keypoint_head(dino_features, cnn_stem_features)
        predicted_angles = self.angle_head(dino_features)
        
        return predicted_angles

def compute_masked_loss(pred_angles, gt_angles, loss_fn_a, weight_a):
    # device = pred_heatmaps.device
    
    # 1. 마스크 생성
    # mask_angle = torch.arange(gt_angles.shape[1], device=device)[None, :] < lengths[:, None]
    # mask_heatmap = mask_angle[:, :, None, None].expand_as(gt_heatmaps)
    
    # 2. 마스킹된 손실 계산
    # loss_h = (loss_fn_h(pred_heatmaps, gt_heatmaps) * mask_heatmap).sum() / mask_heatmap.sum()
    loss_a = loss_fn_a(pred_angles, gt_angles) # * mask_angle).sum() / mask_angle.sum()
    
    # 3. 최종 가중합 손실 반환
    total_loss = weight_a * loss_a
    return total_loss

# main 함수 바깥, 다른 함수들과 같은 레벨에 추가합니다.
def save_checkpoints(checkpoint_data, best_model_state_dict, checkpoint_dir, is_best):
    torch.save(checkpoint_data, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
    if is_best:
        torch.save(best_model_state_dict, os.path.join(checkpoint_dir, "best_model.pth"))

# ======================= 메인 학습 함수 =======================
def main(args): # args 인자를 받도록 수정
    # torch.backends.cudnn.benchmark = True
    rank, local_rank, world_size = setup_ddp()

    save_thread = None
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 198
    EPOCHS = 100
    VAL_RATIO = 0.1

    # --- 1. Ablation 모드에 따른 경로 및 Wandb 설정 ---
    ablation_mode = args.ablation_mode
    WANDB_PROJECT = f"DINOv3_Ablation_{ablation_mode}"
    CHECKPOINT_DIR = f"checkpoints_{ablation_mode}"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    
    # 모드에 따라 모델 이름 결정
    if 'vit' in ablation_mode:
        MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    elif 'conv' in ablation_mode:
        MODEL_NAME = 'facebook/dinov3-convnext-base-pretrain-lvd1689m'
    elif 'siglip2' in ablation_mode:
        MODEL_NAME = 'google/siglip2-base-patch16-224'
    elif 'siglip' in ablation_mode:
        MODEL_NAME = 'google/siglip-base-patch16-224'
    else: # cnn_only
        MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'

    start_epoch = 0
    best_val_loss = float('inf')
    
    model = DINOv3PoseEstimator(dino_model_name=MODEL_NAME, ablation_mode=ablation_mode).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # --- 체크포인트 불러오기 (모델 가중치만) ---
    loss_fn_heatmap = nn.MSELoss(reduction='none')
    loss_fn_angle = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    heatmap_loss_weight = 1.0
    angle_loss_weight = 1.0 # 사용 안 함
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    
    if os.path.exists(LATEST_CHECKPOINT_PATH):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=map_location)

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        if rank == 0:
            print(f"✅ 체크포인트({LATEST_CHECKPOINT_PATH})를 성공적으로 불러왔습니다.")
            print(f"   - {start_epoch - 1} 에포크까지 학습 완료됨. {start_epoch} 에포크부터 학습을 재개합니다.")
            print(f"   - 현재까지 Best Val Loss: {best_val_loss:.6f}")
            
    elif os.path.exists(CHECKPOINT_PATH):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)
        
        # 체크포인트가 딕셔너리 형태인지, state_dict 자체인지 확인하여 가중치를 추출
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # 파일 자체가 state_dict인 경우
            
        # DDP 모델은 model.module로 내부 모델에 접근하여 가중치를 로드
        model.module.load_state_dict(state_dict)
        if rank == 0:
            print(f"✅ 체크포인트에서 모델 가중치를 성공적으로 불러왔습니다: {CHECKPOINT_PATH}")
            
    else:
        if rank == 0:
            print(f"ℹ️ 체크포인트 파일({LATEST_CHECKPOINT_PATH})이 없으므로, 처음부터 학습을 시작합니다.")


    transform = transforms.Compose([
        transforms.Resize((512, 512)), # DINOv3는 518x518도 많이 사용
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
                              pin_memory=True, collate_fn=robot_collate_fn, sampler=train_sampler, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                            pin_memory=True, collate_fn=robot_collate_fn, sampler=val_sampler, persistent_workers=True)
    
    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        wandb.init(project="DINOv3_Ablation_Study", name=f"run_{ablation_mode}", config={
            "learning_rate": LEARNING_RATE, "total_batch_size": BATCH_SIZE * world_size,
            "epochs": EPOCHS, "world_size": world_size, "ablation_mode": ablation_mode
        }, resume="allow") # wandb에서 이어하기 허용
    
    if rank == 0:
        wandb.init(project=WANDB_PROJECT, name=f"run_{ablation_mode}", config={
            "learning_rate": LEARNING_RATE, "total_batch_size": BATCH_SIZE * world_size,
            "epochs": EPOCHS, "world_size": world_size, "ablation_mode": ablation_mode
        }, resume="allow") # wandb에서 이어하기 허용

    for epoch in range(start_epoch, EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=(rank != 0))
        for batch in pbar:
            if batch[0] is None: continue
            # images, gt_angles만 반환됨
            images, gt_angles = batch 
            images = images.to(local_rank) # 이미지를 GPU로 보냅니다.
            gt_angles = gt_angles.to(local_rank)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                pred_angles = model(images)
                total_loss = loss_fn_angle(pred_angles, gt_angles)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            train_loss += total_loss.item() / world_size

            if rank == 0:
                pbar.set_postfix(loss=total_loss.item() / world_size)
        
        # --- 검증 루프 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", disable=(rank != 0))
            for batch in pbar:
                if batch[0] is None: continue
                # images, gt_angles만 반환됨
                images, gt_angles = batch 
                images = images.to(local_rank) # 이미지를 GPU로 보냅니다.
                gt_angles = gt_angles.to(local_rank)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    pred_angles = model(images)
                    total_loss = loss_fn_angle(pred_angles, gt_angles)
                
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                val_loss += total_loss.item() / world_size
        
        scheduler.step()
        
        if rank == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })

            print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # --- 수정된 저장 로직 ---
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"✨ New best model saved for '{ablation_mode}' with val_loss: {best_val_loss:.6f}")

            if save_thread is not None and save_thread.is_alive():
                print(f"Epoch {epoch+1}: Previous save is still running. Skipping this save.")
            else:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                
                save_thread = threading.Thread(
                    target=save_checkpoints,
                    args=(checkpoint_data, model.module.state_dict(), CHECKPOINT_DIR, is_best)
                )
                save_thread.start()

    if rank == 0:
        wandb.finish()
    cleanup_ddp()

if __name__ == '__main__':
    # --- 커맨드 라인 인자 파서 설정 ---
    parser = argparse.ArgumentParser(description="DINOv3 Pose Estimation Ablation Study")
    parser.add_argument(
        '--ablation_mode', 
        type=str, 
        default='dino_only_joint', 
        choices=['dino_only_joint', 'dino_conv_only_joint', 'siglip2_only_joint'],
        help="Select the ablation mode: 'dino_only_joint', 'dino_conv_only_joint',  or 'siglip2_only_joint'"
    )
    args = parser.parse_args()
    
    main(args)
