# main.py
"""
DREAM 데이터셋으로 DINOv3 Pose Estimator 모델을 학습하는 스크립트.
torchrun을 사용한 분산 학습(DDP)을 지원합니다.

예시 (GPU 3개):
torchrun --nproc_per_node=3 fr5_main.py
"""

import os
# NCCL env: deprecated → TORCH_NCCL_ASYNC_ERROR_HANDLING 로 교체
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


import time
import random
import json
import glob
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader
from torchvision import transforms

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
from transformers import AutoImageProcessor

# === our modules ===
from fr5_utils import (
    MODEL_NAME, NUM_ANGLES, NUM_JOINTS, HEATMAP_SIZE, MAX_VIEWS_PER_GROUP,
    perform_grouping,
)
from fr5_dataset import RobotPoseDataset
from fr5_models import (
    DINOv3PoseEstimator, FR5FK,
    weighed_kpt_loss,                     # ② 엔트로피 가중 키포인트 손실
    VonMisesAngleLoss, CosineAngleLoss,  # ① 각도 손실 구성요소
    make_angle_loss                      # ① von Mises + cosine 합성
)
from fr5_vis import (
    visualize_samples_by_group_size,
    visualize_dataset_sample,
    visualize_predictions,
)

# ------------------------------------------------
# 경로 유틸
# ------------------------------------------------
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))               # 현재 파일 폴더
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../.."))    # 프로젝트 루트(두 단계 위)
DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset", "Fr5")

def _join_ds(*parts):
    """dataset/fr5 기준으로 절대경로 생성"""
    return os.path.abspath(os.path.join(DATASET_ROOT, *parts))

def _proj_path(*parts):
    """프로젝트 루트 기준 절대경로 생성(결과물/체크포인트 저장용)"""
    return os.path.abspath(os.path.join(_PROJECT_ROOT, *parts))


# ================================================================
# AMP 유틸 (CPU에서도 안전하게 동작하도록 No-Op Scaler)
# ================================================================
class _NoOpScaler:
    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def unscale_(self, optimizer): pass


# ================================================================
# Train / Validate
# ================================================================
def train_one_epoch(model, loader, optimizers, criteria, device, loss_weight_kpt, epoch_num, param_sets, scalers):
    """
    AMP + 두 경로 분리 업데이트:
      (A) angle  : with autocast + GradScaler + no_sync (통신 최소화)
      (B) keypoint: with autocast + GradScaler + 동기 backward (allreduce 1회)
    grad 마스킹/clip, OOM 가드 유지
    """
    import torch
    import torch.distributed as dist
    import math

    scaler_kpt, scaler_ang = scalers['kpt'], scalers['ang']

    model.train()
    total_loss_kpt, total_loss_ang = 0.0, 0.0
    num_effective_batches = 0

    optimizer_kpt, optimizer_ang = optimizers['kpt'], optimizers['ang']
    crit_ang = criteria['ang']
    fk = criteria['fk']
    lambda_fk = criteria.get('lambda_fk', 0.0)

    m = model.module if hasattr(model, 'module') else model
    kpt_ids = param_sets['kpt']
    ang_ids = param_sets['ang']

    def _dummy_sync_step():
        optimizer_kpt.zero_grad(set_to_none=True)
        optimizer_ang.zero_grad(set_to_none=True)
        dummy = None
        for p in model.parameters():
            if p.requires_grad:
                dummy = (p.sum() if dummy is None else dummy + p.sum())
        if dummy is None:
            dummy = torch.zeros((), device=device, requires_grad=True)
        (dummy * 0.0).backward()
        optimizer_kpt.step()
        optimizer_ang.step()

    loop = tqdm(loader, desc=f"Epoch {epoch_num} [Train]")

    for batch in loop:
        image_dict, gt_heatmaps_dict, gt_angles = batch

        # rank 간 유효 배치 동기화
        has_data_local = int(image_dict is not None)
        has_data_all = torch.tensor(has_data_local, device=device)
        dist.all_reduce(has_data_all, op=dist.ReduceOp.SUM)
        has_any_rank_data = int(has_data_all.item())
        if not has_any_rank_data:
            loop.set_postfix(loss_kpt='skip_all', loss_ang='skip_all')
            continue

        if image_dict is None:
            _dummy_sync_step()
            loop.set_postfix(loss_kpt='skip', loss_ang='skip')
            continue

        try:
            images_gpu   = {k: v.to(device, non_blocking=True) for k, v in image_dict.items()}
            heatmaps_gpu = {k: v.to(device, non_blocking=True) for k, v in gt_heatmaps_dict.items()}
            angles_gpu   = gt_angles.to(device, non_blocking=True)  # (B, A) deg

            # =========================
            # (A) ANGLE 경로  — no_sync
            # =========================
            optimizer_ang.zero_grad(set_to_none=True)
            optimizer_kpt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, pred_angles_A = model(images_gpu)                     # (B,A,2)
                loss_ang = crit_ang(pred_angles_A, angles_gpu)

                pred_rad = torch.atan2(pred_angles_A[..., 0], pred_angles_A[..., 1])
                pred_deg = pred_rad * (180.0 / math.pi)
                pred_pts = fk(pred_deg)                                  # (B,7(or 9),3)
                gt_pts   = fk(angles_gpu)
                loss_fk  = F.smooth_l1_loss(pred_pts, gt_pts, beta=2.0)

                loss_ang_total = loss_ang + lambda_fk * loss_fk

            # 첫 backward는 통신 생략
            with model.no_sync():
                scaler_ang.scale(loss_ang_total).backward()

            # unscale → 마스킹 → clip → step
            scaler_ang.unscale_(optimizer_ang)
            for p in m.parameters():
                if p.grad is None:
                    continue
                if id(p) not in ang_ids:
                    p.grad.detach_()
                    p.grad.zero_()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            scaler_ang.step(optimizer_ang)
            scaler_ang.update()

            # =========================
            # (B) KEYPOINT 경로 — 동기 backward (allreduce 발생)
            # =========================
            optimizer_kpt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred_heatmaps_B, _ = model(images_gpu)
                loss_kpt = weighed_kpt_loss(pred_heatmaps_B, heatmaps_gpu, m.softarg) * loss_weight_kpt

            if not torch.isfinite(loss_kpt):
                _dummy_sync_step()
                loop.set_postfix(loss_kpt='nan_guard', loss_ang=f"{loss_ang_total.item():.4f}")
                continue

            # 두 번째 backward는 통신 동기화(기본)
            scaler_kpt.scale(loss_kpt).backward()

            scaler_kpt.unscale_(optimizer_kpt)
            for p in m.parameters():
                if p.grad is None:
                    continue
                if id(p) not in kpt_ids:
                    p.grad.detach_()
                    p.grad.zero_()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            scaler_kpt.step(optimizer_kpt)
            scaler_kpt.update()

            total_loss_kpt += float(loss_kpt.item())
            total_loss_ang += float(loss_ang_total.item())
            num_effective_batches += 1

            loop.set_postfix(
                loss_kpt=f"{loss_kpt.item():.4f}",
                loss_ang=f"{loss_ang_total.item():.4f}",
                fk=f"{loss_fk.item():.4f}"
            )

        except RuntimeError as e:
            msg = str(e).lower()
            if 'out of memory' in msg or 'cublas' in msg or 'illegal memory' in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _dummy_sync_step()
                loop.set_postfix(loss_kpt='oom_skip', loss_ang='oom_skip')
                continue
            _dummy_sync_step()
            loop.set_postfix(loss_kpt='err_skip', loss_ang='err_skip')
            continue

    denom = max(1, num_effective_batches)
    return total_loss_kpt / denom, total_loss_ang / denom


def validate(model, loader, criteria, device, loss_weight_kpt, epoch_num, amp_enabled=True):
    """
    검증 루프 + 지표:
    - Loss: kpt/ang 분리, total = kpt*weight + (ang + lambda_fk*fk)
    - Metric: angle MAE(deg), heatmap argmax L2-px (128x128), (옵션) FK position MAE
    - AMP 지원(autocast)으로 추론 가속
    """
    import torch
    import torch.distributed as dist
    import math

    model.eval()
    fk = criteria['fk']
    angle_crit = criteria['ang']
    lambda_fk = criteria.get('lambda_fk', 0.0)

    total_val_loss = 0.0
    total_val_kpt  = 0.0
    total_val_ang  = 0.0
    total_ang_mae  = 0.0
    total_kpt_px   = 0.0
    total_fk_mae   = 0.0
    num_effective  = 0

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0

    def _batch_heatmap_l2_px(pred_hm, gt_hm):
        B, J, H, W = gt_hm.shape
        gt_idx = gt_hm.view(B, J, -1).argmax(dim=-1)
        pr_idx = pred_hm.view(B, J, -1).argmax(dim=-1)
        gt_y = gt_idx // W; gt_x = gt_idx % W
        pr_y = pr_idx // W; pr_x = pr_idx % W
        dx = (pr_x - gt_x).float(); dy = (pr_y - gt_y).float()
        return torch.sqrt(dx*dx + dy*dy).mean().item()

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Epoch {epoch_num} [Validate]", leave=False) if rank == 0 else loader
        for batch in loop:
            image_dict, gt_heatmaps_dict, gt_angles = batch

            has_data_local = int(image_dict is not None)
            has_data_all = torch.tensor(has_data_local, device=device)
            if dist.is_initialized():
                dist.all_reduce(has_data_all, op=dist.ReduceOp.SUM)
            has_any_rank_data = int(has_data_all.item())
            if not has_any_rank_data:
                if rank == 0 and isinstance(loop, tqdm): loop.set_postfix_str("skip_all")
                continue
            if image_dict is None:
                if rank == 0 and isinstance(loop, tqdm): loop.set_postfix_str("skip")
                continue

            images_gpu   = {k: v.to(device, non_blocking=True) for k, v in image_dict.items()}
            heatmaps_gpu = {k: v.to(device, non_blocking=True) for k, v in gt_heatmaps_dict.items()}
            angles_gpu   = gt_angles.to(device, non_blocking=True)  # (B,A) deg

            # AMP 추론
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp_enabled):
                pred_heatmaps, pred_angles = model(images_gpu)  # pred_angles: (B,A,2)

                # 키포인트 손실 (학습과 동일한 함수)
                softarg_ref = getattr(model.module if hasattr(model, "module") else model, "softarg")
                loss_kpt = weighed_kpt_loss(pred_heatmaps, heatmaps_gpu, softarg_ref) * loss_weight_kpt

                # 각도 손실
                loss_ang = angle_crit(pred_angles, angles_gpu)

                # FK 보조 항
                pred_deg = torch.atan2(pred_angles[..., 0], pred_angles[..., 1]) * (180.0 / math.pi)
                gt_deg   = angles_gpu
                pred_pts = fk(pred_deg)
                gt_pts   = fk(gt_deg)
                fk_loss  = F.smooth_l1_loss(pred_pts, gt_pts, beta=2.0)

                loss_ang_total = loss_ang + (lambda_fk * fk_loss)

            if (not torch.isfinite(loss_kpt)) or (not torch.isfinite(loss_ang_total)):
                if rank == 0 and isinstance(loop, tqdm): loop.set_postfix_str("nan_guard")
                continue

            total = (loss_kpt + loss_ang_total).item()
            total_val_loss += total
            total_val_kpt  += float(loss_kpt.item())
            total_val_ang  += float(loss_ang_total.item())
            num_effective  += 1

            # --- 각도 MAE(deg): 원형오차 ---
            def vector_to_deg(vec: torch.Tensor):
                rad = torch.atan2(vec[..., 0], vec[..., 1])
                return rad * 180.0 / math.pi

            pred_deg_for_mae = vector_to_deg(pred_angles)
            diff = (pred_deg_for_mae - angles_gpu + 180.0) % 360.0 - 180.0
            ang_mae = torch.mean(torch.abs(diff)).item()

            # --- 히트맵 argmax L2 픽셀오차(128x128 기준) ---
            real_view_keys = list(pred_heatmaps.keys())
            per_view_err = []
            for k in real_view_keys:
                per_view_err.append(_batch_heatmap_l2_px(pred_heatmaps[k], heatmaps_gpu[k]))
            kpt_px_err = (sum(per_view_err) / len(per_view_err)) if per_view_err else 0.0

            # --- (선택) FK 위치 MAE(m) ---
            fk_mae = torch.mean(torch.linalg.norm(pred_pts - gt_pts, dim=-1)).item()

            total_ang_mae += ang_mae
            total_kpt_px  += kpt_px_err
            total_fk_mae  += fk_mae

            if rank == 0 and isinstance(loop, tqdm):
                loop.set_postfix(
                    loss_total=f"{total:.4f}",
                    loss_kpt=f"{float(loss_kpt):.4f}",
                    loss_ang=f"{float(loss_ang_total):.4f}",
                    ang_MAE_deg=f"{ang_mae:.3f}",
                    kpt_L2px=f"{kpt_px_err:.2f}",
                    fk_MAE_m=f"{fk_mae:.4f}"
                )

    denom = max(1, num_effective)
    avg_total = total_val_loss / denom
    avg_kpt   = total_val_kpt  / denom
    avg_ang   = total_val_ang  / denom
    avg_ang_mae_deg  = total_ang_mae / denom
    avg_kpt_l2px_128 = total_kpt_px  / denom
    avg_fk_mae_m     = total_fk_mae  / denom

    if rank == 0:
        print(f"[Validate/Epoch {epoch_num}] "
              f"avg_total={avg_total:.6f} | avg_kpt={avg_kpt:.6f} | avg_ang={avg_ang:.6f} | "
              f"ang_MAE_deg={avg_ang_mae_deg:.3f} | kpt_L2px_128={avg_kpt_l2px_128:.2f} | "
              f"fk_pos_MAE_m={avg_fk_mae_m:.4f}")

    return avg_total, avg_kpt, avg_ang, avg_ang_mae_deg, avg_kpt_l2px_128



# ================================================================
# DDP setup/teardown
# ================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank

def cleanup_ddp():
    dist.destroy_process_group()


# ================================================================
# Experiment setup (datasets, loaders, model, opt, sched)
# ================================================================
def setup(hyperparameters, dataset_groups, rank, world_size):
    print(f"--- [Rank {rank}] Setting up environment ---")
    device = torch.device(f'cuda:{rank}')

    # DINO 계열 표준(ImageNet) 정규화 직접 지정
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    resize_size = 224  # 현재 파이프라인은 resize-only

    def build_base_transform(mean, std, resize_size=224):
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),  # 224x224로 강제 워핑
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def build_strong_transform(mean, std, resize_size=224):
        # 8번: 색/노이즈는 독립, 기하 왜곡은 작게 유지(멀티뷰 정합 보호)
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),  # crop 제거
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    base_transform   = build_base_transform(mean, std, resize_size)
    strong_transform = build_strong_transform(mean, std, resize_size)

    torch.manual_seed(42)
    indices = torch.randperm(len(dataset_groups)).tolist()
    train_size = int(len(dataset_groups) * (1 - hyperparameters['val_split']))
    train_groups = [dataset_groups[i] for i in indices[:train_size]]
    val_groups   = [dataset_groups[i] for i in indices[train_size:]]

    # 초기에는 "기본 전처리"로 시작
    train_dataset = RobotPoseDataset(groups=train_groups, transform=base_transform)
    val_dataset   = RobotPoseDataset(groups=val_groups,   transform=base_transform)

    # 전체 view-key 수집
    def collect_view_keys_from_groups(groups):
        keys = set()
        for g in groups:
            for v in g['views']:
                p = v['image_path']
                fname = os.path.basename(p)
                parts = fname.split('_')
                if len(parts) < 4:
                    continue
                serial = parts[1]
                cam = parts[2]
                keys.add(f"{serial}_{cam}")
        return sorted(keys)

    all_view_keys = collect_view_keys_from_groups(dataset_groups)
    if not all_view_keys:
        raise RuntimeError("No valid view keys found in datasets.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    # collate 준비용 템플릿
    template_sample = None
    for _ in range(512):
        idx = random.randint(0, len(train_dataset)-1)
        s = train_dataset[idx]
        if s[0] is not None:
            template_sample = s
            break
    if template_sample is None:
        raise RuntimeError("Could not find a valid sample to build dummy batches.")

    tmpl_img_dict, tmpl_hmap_dict, _ = template_sample
    tmpl_all_keys = sorted(set(tmpl_img_dict.keys()))
    sample_img   = list(tmpl_img_dict.values())[0]
    sample_hmap  = list(tmpl_hmap_dict.values())[0]
    dummy_img    = torch.zeros_like(sample_img)
    dummy_hmap   = torch.zeros_like(sample_hmap)
    dummy_angles = torch.zeros(NUM_ANGLES, dtype=torch.float32)

    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            image_dict  = {k: dummy_img.clone()  for k in tmpl_all_keys}
            hmap_dict   = {k: dummy_hmap.clone() for k in tmpl_all_keys}
            angles      = dummy_angles.clone()
            images      = {k: torch.stack([v]) for k, v in image_dict.items()}
            heatmaps    = {k: torch.stack([v]) for k, v in hmap_dict.items()}
            angles      = angles.unsqueeze(0)
            return images, heatmaps, angles

        image_dicts, heatmap_dicts, angles_list = zip(*batch)
        all_keys = sorted(set().union(*[d.keys() for d in image_dicts]))

        std_images, std_heatmaps = [], []
        for i in range(len(batch)):
            new_img  = {key: image_dicts[i].get(key,  dummy_img)   for key in all_keys}
            new_hmap = {key: heatmap_dicts[i].get(key, dummy_hmap) for key in all_keys}
            std_images.append(new_img); std_heatmaps.append(new_hmap)

        images   = torch.utils.data.dataloader.default_collate(std_images)
        heatmaps = torch.utils.data.dataloader.default_collate(std_heatmaps)
        angles   = torch.stack(angles_list)
        return images, heatmaps, angles

    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters['batch_size'], num_workers=8,
        collate_fn=collate_fn, pin_memory=True, sampler=train_sampler,
        drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=hyperparameters['batch_size'], num_workers=8,
        collate_fn=collate_fn, pin_memory=True, sampler=val_sampler,
        drop_last=False, persistent_workers=True
    )

    # Model + DDP
    model = DINOv3PoseEstimator(
        model_name=hyperparameters['model_name'],
        known_view_keys=all_view_keys
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True, gradient_as_bucket_view=False)

    # ① 각도 손실: von Mises + cosine
    angle_loss = make_angle_loss(NUM_ANGLES, vm_weight=0.5, cos_weight=0.5)

    # FK 모듈    # --- Angle loss: von Mises + Cosine ---
    angle_loss = make_angle_loss(NUM_ANGLES, vm_weight=0.5, cos_weight=0.5)
    # κ 파라미터를 rank 디바이스로 이동
    if hasattr(angle_loss, "vm"):
        angle_loss.vm = angle_loss.vm.to(device)

    fk = FR5FK(device)
    criteria = {
        'kpt': nn.MSELoss(),   # 실제 kpt 손실은 weighed_kpt_loss에서 사용
        'ang': angle_loss,
        'fk': fk
    }
    criteria['lambda_fk'] = hyperparameters.get('lambda_fk', 0.5)

    m = model.module
    params_shared = list(m.view_embeddings.parameters()) + list(m.fusion.parameters())

    # ✅ angle 경로: κ 파라미터까지 포함해 학습
    params_ang = (
        list(m.ang_head.parameters())
        + params_shared
        + list(m.kp_token_enc.parameters())
        + list(m.cnn_token_enc.parameters())
        + (list(angle_loss.vm.parameters()) if hasattr(angle_loss, "vm") else [])
    )

    # kpt 경로
    params_kpt = (
        list(m.cnn_stem.parameters())
        + params_shared
        + list(m.kpt_enricher.parameters())
        + list(m.kpt_head.parameters())
    )

    optimizers = {
        'kpt': torch.optim.AdamW(params_kpt, lr=hyperparameters['lr_kpt']),
        'ang': torch.optim.AdamW(params_ang, lr=hyperparameters['lr_ang'])
    }

    # 🔥 Warmup(5) + Cosine
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
    warmup_epochs = 5
    total_epochs = hyperparameters['num_epochs']
    assert total_epochs > warmup_epochs, "num_epochs는 warmup_epochs보다 커야 합니다."

    warm_kpt = LinearLR(optimizers['kpt'], start_factor=0.2, total_iters=warmup_epochs)
    warm_ang = LinearLR(optimizers['ang'], start_factor=0.2, total_iters=warmup_epochs)

    cosine_kpt = CosineAnnealingLR(optimizers['kpt'], T_max=total_epochs - warmup_epochs)
    cosine_ang = CosineAnnealingLR(optimizers['ang'], T_max=total_epochs - warmup_epochs)

    schedulers = {
        'kpt': SequentialLR(optimizers['kpt'], schedulers=[warm_kpt, cosine_kpt], milestones=[warmup_epochs]),
        'ang': SequentialLR(optimizers['ang'], schedulers=[warm_ang, cosine_ang], milestones=[warmup_epochs]),
    }

    if rank == 0:
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val.")
        print(f"Warmup epochs: {warmup_epochs}")

    param_sets = {
        'kpt': set(id(p) for p in params_kpt),
        'ang': set(id(p) for p in params_ang),
    }

    # strong_transform을 반환해서 메인 루프에서 스위치 가능하게
    return model, train_loader, val_loader, criteria, optimizers, schedulers, device, mean, std, train_sampler, param_sets, strong_transform


# ================================================================
# Main
# ================================================================
def main():
    rank = setup_ddp()
    world_size = dist.get_world_size()

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if torch.cuda.is_available():
        if rank == 0:
            print(f"✅ 사용 가능한 GPU: {torch.cuda.device_count()}개")
    else:
        if rank == 0:
            print("⚠️ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")

    # ---------- CSV 절대경로 ----------
    TOTAL_CSV_PATH = _join_ds("fr5_matched_joint_angle.csv")
    if rank == 0:
        print(f"\nLoading data from {TOTAL_CSV_PATH}...")
        total_csv = pd.read_csv(TOTAL_CSV_PATH)
        total_csv.sort_values('joint_timestamp', inplace=True, ignore_index=True)
        print("✅ CSV file loaded and sorted successfully.")
    else:
        total_csv = None

    # 브로드캐스트
    obj_list = [total_csv]
    dist.broadcast_object_list(obj_list, src=0)
    total_csv = obj_list[0]

    # ---------- TIME_TOLERANCE grid-search ----------
    tolerance_candidates = np.round(np.arange(0.01, 0.101, 0.01), 2)
    best_tolerance_recommendation, max_full_groups = 0, 0

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

    # ---------- 최종 tolerance ----------
    final_tolerance = 0.0
    if rank == 0:
        print(f"\nFinal TIME_TOLERANCE set to: {final_tolerance}")
    dataset_groups = perform_grouping(total_csv, final_tolerance, MAX_VIEWS_PER_GROUP)
    if rank == 0:
        print(f"Total {len(dataset_groups)} groups created before filtering.")

    # 1뷰 그룹 제거
    groups_before_filtering = len(dataset_groups)
    dataset_groups = [g for g in dataset_groups if len(g['views']) > 1]
    if rank == 0:
        print(f"ℹ️ Removed {groups_before_filtering - len(dataset_groups)} groups with only 1 view.")
        print(f"\n✅ Final Total Groups: {len(dataset_groups)}")
        total_images_in_groups = sum(len(g['views']) for g in dataset_groups)
        print(f"✅ Final Total Images to be used: {total_images_in_groups}")
        if dataset_groups:
            view_counts = [len(g['views']) for g in dataset_groups]
            print(f"\n--- Final View count distribution ---")
            print(pd.Series(view_counts).value_counts().sort_index(ascending=False))

    # ---------- 하이퍼파라미터 & 파일 경로 ----------
    hyperparameters = {
        'model_name': MODEL_NAME,
        'batch_size': 72,
        'num_epochs': 100,
        'val_split': 0.05,
        'loss_weight_kpt': 1000.0,  # ② 도입 후 20~50 스윕 권장
        'lr_kpt': 1e-5,
        'lr_ang': 1e-5,
        'lambda_fk': 0.5,   # FK 보조 손실 가중치
    }

    RESULTS_DIR      = os.path.join(_CUR_DIR, "results_ddp")
    CHECKPOINT_PATH  = os.path.join(_CUR_DIR, "multiview_checkpoint_ddp.pth")
    BEST_MODEL_PATH  = os.path.join(_CUR_DIR, "best_multiview_model_ddp.pth")
    FINETUNE_WEIGHTS = os.path.join(_CUR_DIR, "No1_best_multiview_model_ddp.pth")  # 있으면 사용

    if rank == 0:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"--- Data Preparation (results -> {RESULTS_DIR}) ---")

    # 그룹 브로드캐스트
    obj_list = [dataset_groups]
    dist.broadcast_object_list(obj_list, src=0)
    dataset_groups = obj_list[0]

    # ---------- 시각화용 processor ----------
    if rank == 0:
        print("Loading DINOv3 Processor for transformation config...")
    dino_mean = [0.485, 0.456, 0.406]
    dino_std  = [0.229, 0.224, 0.225]
    resize_size = 224

    vis_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),  # crop 금지
        transforms.ToTensor(),
        transforms.Normalize(mean=dino_mean, std=dino_std),
    ])

    # 샘플 시각화 (rank0)
    if rank == 0:
        visualize_samples_by_group_size(dataset_groups, transform=vis_transform, mean=dino_mean, std=dino_std)

    dist.barrier()

    # ---------- 세팅 ----------
    (model, train_loader, val_loader, criteria,
     optimizers, schedulers, device, mean, std, train_sampler, param_sets, strong_transform) = setup(
        hyperparameters, dataset_groups, rank, world_size
    )

    global results_dir
    results_dir = RESULTS_DIR

        # ---------- AMP Grad Scaler ----------
    scalers = {
        'kpt': torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available()),
        'ang': torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available()),
    }

    
    # ---------- wandb ----------
    if rank == 0:
        run = wandb.init(project="multiview-fr5-ddp-final", config=hyperparameters,
                         name=f"run_ddp_{time.strftime('%Y%m%d_%H%M%S')}")
        wandb.watch(model, log="parameters", log_freq=100, log_graph=False)
    else:
        run = None

    start_epoch, best_val_loss = 0, float('inf')

    # ---------- (선택) 파인튜닝 가중치 로드 → 브로드캐스트 ----------
    def _safe_load_state_dict(path, device, rank):
        if not os.path.isfile(path):
            return None
        if rank == 0:
            print(f"🔁 Loading fine-tune weights from: {path}")
        try:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage.cuda(rank), weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage.cuda(rank))
        state = ckpt.get('model_state_dict', ckpt)
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        return state

    state_to_broadcast = None
    if rank == 0:
        state_to_broadcast = _safe_load_state_dict(FINETUNE_WEIGHTS, device, rank)
    obj_list = [state_to_broadcast]
    dist.broadcast_object_list(obj_list, src=0)
    finetune_state = obj_list[0]

    if finetune_state is not None:
        msg = model.module.load_state_dict(finetune_state, strict=False)
        if rank == 0:
            missing = getattr(msg, 'missing_keys', [])
            unexpected = getattr(msg, 'unexpected_keys', [])
            print("✅ Fine-tune weights loaded with strict=False.")
            if missing:
                print(f"   Missing keys   ({len(missing)}): {missing[:20]}{' ...' if len(missing)>20 else ''}")
            if unexpected:
                print(f"   Unexpected keys({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected)>20 else ''}")
    else:
        if rank == 0:
            print("ℹ️ No fine-tune weights found; training from scratch configuration.")

    # ---------- 학습 루프 ----------
        # ---------- 학습 루프 ----------
    if rank == 0:
        print("\n--- Starting Training ---")
    switch_epoch = hyperparameters['num_epochs'] // 3

    # SoftArgmax β, CNN token dropout 스케줄 파라미터
    beta0, beta1 = 1.0, 3.0           # soft-argmax 온도: 초반 부드럽게 → 후반 샤프하게
    base_token_drop = 0.10            # 초기 토큰 드롭 비율(에폭이 지날수록 감소)

    for epoch in range(start_epoch, hyperparameters['num_epochs']):
        # --- 스케줄 값 계산 ---
        progress = epoch / max(1, hyperparameters['num_epochs'] - 1)
        m = model.module if hasattr(model, "module") else model

        # 8) SoftArgmax β 스케줄링
        m.softarg.beta = float(beta0 + (beta1 - beta0) * progress)

        # 8) CNN 토큰 드롭아웃 스케줄링 (forward에서 drop_prob_scheduled 사용)
        m.drop_prob_scheduled = max(0.0, base_token_drop * (1.0 - progress))

        # 증강 전환(강증강)
        if epoch == switch_epoch:
            if rank == 0:
                print(f"[Augment] Switching to strong augmentation at epoch {epoch}.")
            train_loader.dataset.transform = strong_transform

        train_sampler.set_epoch(epoch)

        # === 한 에폭 학습 ===
        train_loss_kpt, train_loss_ang = train_one_epoch(
            model, train_loader, optimizers, criteria, device,
            hyperparameters['loss_weight_kpt'], epoch + 1, param_sets, scalers
        )

        # === 검증 ===
        (val_loss, val_kpt, val_ang,
         val_ang_mae, val_kpt_px) = validate(
            model, val_loader, criteria, device,
            hyperparameters['loss_weight_kpt'], epoch + 1,
            amp_enabled=torch.cuda.is_available()
        )

        # 스케줄러 스텝
        schedulers['kpt'].step(); schedulers['ang'].step()

        # --- rank==0 로깅/저장 ---
        if rank == 0:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss_kpt": train_loss_kpt,
                "train_loss_ang": train_loss_ang,
                "avg_val_loss": val_loss,
                "val_kpt_loss": val_kpt,
                "val_ang_loss": val_ang,
                "val_angle_MAE_deg": val_ang_mae,
                "val_kpt_L2px_128": val_kpt_px,
                "lr_kpt": optimizers['kpt'].param_groups[0]['lr'],
                "lr_ang": optimizers['ang'].param_groups[0]['lr'],
                # 8) 모니터링: 현재 softarg β / token drop
                "softarg_beta": m.softarg.beta,
                "cnn_token_drop_sched": m.drop_prob_scheduled,
            }
            # 7) (선택) von Mises κ 로깅
            if hasattr(criteria['ang'], 'vm'):
                with torch.no_grad():
                    kappa = criteria['ang'].vm.log_kappa.exp().detach().cpu().numpy()
                # joint별로 너무 많으면 평균만 기록
                log_dict["kappa_mean"] = float(kappa.mean())

            if run is not None:
                wandb.log(log_dict)

            lr_kpt = optimizers['kpt'].param_groups[0]['lr']
            lr_ang = optimizers['ang'].param_groups[0]['lr']
            print(
                f"Epoch {epoch+1} -> "
                f"Val Total: {val_loss:.6f} | ValKPT: {val_kpt:.6f} | ValANG: {val_ang:.6f} | "
                f"MAE(deg): {val_ang_mae:.3f} | KPT_L2px(128): {val_kpt_px:.2f} | "
                f"LR_kpt: {lr_kpt:.6f} | LR_ang: {lr_ang:.6f} | "
                f"beta: {m.softarg.beta:.2f} | drop: {m.drop_prob_scheduled:.3f}"
            )

            # -------------------------------
            # ✅ 모델 저장(베스트) + 시각화
            # -------------------------------
            state_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

            did_best_visualize = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"🎉 New best model saved with validation loss: {best_val_loss:.6f}")
                torch.save(state_to_save, BEST_MODEL_PATH)

                figs = visualize_predictions(
                    model, val_loader.dataset, device, mean, std,
                    epoch + 1, results_dir=RESULTS_DIR, num_samples=1
                )
                if run is not None:
                    wandb.log({"validation_predictions": [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                did_best_visualize = True

            # --------------------------------------------------------
            # 🆕 매 5 에폭마다 시각화 저장 (베스트가 아니어도 강제 저장)
            # --------------------------------------------------------
            if ((epoch + 1) % 5 == 0) and (not did_best_visualize):
                print(f"🖼️ Periodic visualization at epoch {epoch+1} (every 5 epochs).")
                figs = visualize_predictions(
                    model, val_loader.dataset, device, mean, std,
                    epoch + 1, results_dir=RESULTS_DIR, num_samples=1
                )
                if run is not None:
                    wandb.log({f"periodic_predictions/epoch_{epoch+1}": [wandb.Image(fig) for fig in figs]})
                for fig in figs:
                    import matplotlib.pyplot as plt
                    plt.close(fig)

            # 체크포인트 저장(항상)
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
