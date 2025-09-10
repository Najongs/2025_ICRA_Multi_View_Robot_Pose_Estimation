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
import itertools

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
from fr5_utils import (MODEL_NAME, NUM_ANGLES, NUM_JOINTS, HEATMAP_SIZE, MAX_VIEWS_PER_GROUP,perform_grouping)
from fr5_dataset import RobotPoseDataset
from fr5_models import DINOv3PoseEstimator
from fr5_train_val import (VonMisesAngleLoss, CosineAngleLoss, make_angle_loss, weighed_kpt_loss, FR5FK, train_one_epoch, validate)
from fr5_vis import (visualize_samples_by_group_size, visualize_dataset_sample, visualize_predictions)

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
# DDP setup/teardown
# ================================================================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank
def cleanup_ddp(): dist.destroy_process_group()

# ================================================================
# Experiment Setup (수정된 최종 버전)
# ================================================================
def setup(hyperparameters, dataset_groups, rank, world_size):
    print(f"--- [Rank {rank}] Setting up environment ---")
    device = torch.device(f'cuda:{rank}')

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    resize_size = 224

    def build_base_transform(mean, std, resize_size=224):
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def build_strong_transform(mean, std, resize_size=224):
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
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

    # 2개 뷰 이상을 가진 그룹만 학습에 사용
    train_groups_filtered = [g for g in train_groups if len(g.get('views', [])) >= 2]
    val_groups_filtered = [g for g in val_groups if len(g.get('views', [])) >= 2]
    
    train_dataset = RobotPoseDataset(groups=train_groups_filtered, transform=base_transform)
    val_dataset   = RobotPoseDataset(groups=val_groups_filtered,   transform=base_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    # ▼▼▼ [핵심 수정] KeyError 해결을 위한 collate_fn ▼▼▼
    def collate_fn(batch):
        # 1. 데이터셋 로딩 중 에러가 발생한 샘플(None)을 걸러냅니다.
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return None, None, None

        image_dicts, heatmap_dicts, angles_list = zip(*batch)
        
        # 2. 현재 배치에 포함된 모든 뷰의 키(key)를 수집합니다.
        # 예: {'cam_A', 'cam_B', 'cam_C'}
        all_keys_in_batch = sorted(list(set(itertools.chain.from_iterable(d.keys() for d in image_dicts))))

        # 3. 빈 자리를 채울 더미 텐서를 하나 만듭니다.
        sample_img_tensor = next(iter(image_dicts[0].values()))
        sample_hmap_tensor = next(iter(heatmap_dicts[0].values()))
        dummy_img = torch.zeros_like(sample_img_tensor)
        dummy_hmap = torch.zeros_like(sample_hmap_tensor)

        # 4. 각 샘플을 순회하며, all_keys_in_batch 기준으로 없는 뷰는 더미 텐서로 채워줍니다.
        padded_images = []
        padded_heatmaps = []
        for i in range(len(batch)):
            # .get(key, dummy_img)는 딕셔너리에 key가 없으면 dummy_img를 대신 사용하라는 의미입니다.
            padded_img_dict = {key: image_dicts[i].get(key, dummy_img) for key in all_keys_in_batch}
            padded_hmap_dict = {key: heatmap_dicts[i].get(key, dummy_hmap) for key in all_keys_in_batch}
            padded_images.append(padded_img_dict)
            padded_heatmaps.append(padded_hmap_dict)

        # 5. 이제 모든 샘플이 동일한 키 구조를 가지므로, 에러 없이 배치를 만들 수 있습니다.
        images_collated = torch.utils.data.dataloader.default_collate(padded_images)
        heatmaps_collated = torch.utils.data.dataloader.default_collate(padded_heatmaps)
        angles_collated = torch.stack(angles_list)

        return images_collated, heatmaps_collated, angles_collated

    # DataLoader에 수정된 collate_fn 적용
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], num_workers=8, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters['batch_size'], num_workers=8, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)

    # known_view_keys 없이 모델 생성
    model = DINOv3PoseEstimator(model_name=hyperparameters['model_name']).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # (Optimizer, Scheduler, Criteria 등 나머지 설정은 이전 답변과 동일하게 유지)
    angle_loss = make_angle_loss(NUM_ANGLES, vm_weight=0.5, cos_weight=0.5)
    if hasattr(angle_loss, "vm"): angle_loss.vm = angle_loss.vm.to(device)
    fk = FR5FK(device)
    criteria = {'ang': angle_loss, 'fk': fk}
    criteria['lambda_fk'] = hyperparameters.get('lambda_fk', 0.5)

    m = model.module
    params_shared = list(m.view_embeddings.parameters()) + list(m.fusion.parameters())
    params_ang = list(m.ang_head.parameters()) + params_shared + list(m.kp_token_enc.parameters()) + list(m.cnn_token_enc.parameters()) + (list(angle_loss.vm.parameters()) if hasattr(angle_loss, "vm") else [])
    params_kpt = list(m.cnn_stem.parameters()) + params_shared + list(m.kpt_enricher.parameters()) + list(m.kpt_head.parameters())

    optimizers = {
        'kpt': torch.optim.AdamW(params_kpt, lr=hyperparameters['lr_kpt']),
        'ang': torch.optim.AdamW(params_ang, lr=hyperparameters['lr_ang'])
    }
    
    warmup_epochs = 5
    total_epochs = hyperparameters['num_epochs']
    schedulers = {
        'kpt': SequentialLR(optimizers['kpt'], [LinearLR(optimizers['kpt'], 0.2, 1, total_iters=warmup_epochs), CosineAnnealingLR(optimizers['kpt'], T_max=total_epochs-warmup_epochs)], [warmup_epochs]),
        'ang': SequentialLR(optimizers['ang'], [LinearLR(optimizers['ang'], 0.2, 1, total_iters=warmup_epochs), CosineAnnealingLR(optimizers['ang'], T_max=total_epochs-warmup_epochs)], [warmup_epochs]),
    }

    param_sets = {
        'kpt': set(id(p) for p in params_kpt),
        'ang': set(id(p) for p in params_ang)
    }
    
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
        'batch_size': 72, # GPU 메모리에 따라 조절
        'num_epochs': 100,
        'val_split': 0.05,
        'loss_weight_kpt': 100.0,
        'lr_kpt': 1e-4,
        'lr_ang': 1e-4,
        'lr_backbone': 1e-7,   # <<< Add this line for fine-tuning
        'lambda_fk': 0.5,      # FK 보조 손실 가중치
    }

    RESULTS_DIR      = os.path.join(_CUR_DIR, "results_ddp")
    CHECKPOINT_PATH  = os.path.join(_CUR_DIR, "multiview_checkpoint_ddp.pth")
    BEST_MODEL_PATH  = os.path.join(_CUR_DIR, "best_multiview_model_ddp.pth")
    FINETUNE_WEIGHTS = os.path.join(_CUR_DIR, "best_multiview_model_ddp.pth")  # 있으면 사용

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
        'kpt': torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()),
        'ang': torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
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
    switch_epoch = hyperparameters['num_epochs']*2 // 3

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
