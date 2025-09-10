# main.py
"""
통합 DREAM-robot 학습 스크립트 (FR3 / FR5 / MECA500)
- torchrun DDP 지원
- utils/dataset 통합본 기반
- 로봇별 결과/체크포인트 폴더 분리 저장
예시:
torchrun --nproc_per_node=3 main.py --robot fr5
"""

import os, time, random, math, json, glob, argparse
import numpy as np
import pandas as pd
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 시각화 (선택)
try:
    from vis import visualize_dataset_sample, visualize_predictions
    _HAS_VIS = True
except Exception:
    _HAS_VIS = False

# === our modules (통합본) ===
from dataset import (
    build_items_from_csv,
    build_unified_dataset_from_csv,   # 필요시 직접 dataset만 만들 때
    UnifiedRobotPoseDataset,
    SPECS,                            # dataset별 스펙/경로 anchor
)
from setup import setup               # 앞서 제공한 통합 setup 함수
from models import DINOv3PoseEstimator  # 네 모델 경로에 맞게
from train_val import train_one_epoch, validate   # 네가 쓰는 학습/검증 루틴
# (주의) loss는 setup()에서 make_angle_loss 호출

# ------------------------------------------------
# DDP
# ------------------------------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank

def cleanup_ddp():
    dist.destroy_process_group()

# ------------------------------------------------
# 경로 유틸
# ------------------------------------------------
def _get_project_root():
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(_cur_dir, "../.."))

def _result_root(robot):
    return os.path.join(_get_project_root(), "results", robot)

def _make_run_dirs(robot):
    root = _result_root(robot)
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root, ts)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "best_path": os.path.join(run_dir, f"best_{robot}.pth"),
        "ckpt_path": os.path.join(run_dir, f"ckpt_{robot}.pth"),
        "vis_dir": os.path.join(run_dir, "vis"),
    }

# ------------------------------------------------
# AMP 유틸 (CPU에서도 안전하게 동작하도록 No-Op Scaler)
# ------------------------------------------------
class _NoOpScaler:
    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def unscale_(self, optimizer): pass

# ------------------------------------------------
# argparse
# ------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", type=str, default="fr5", choices=["fr3", "fr5", "meca500"],
                    help="학습할 로봇 타입 선택")
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV 파일명 (데이터셋 루트 기준). 기본값은 로봇별 권장 파일명 사용")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=72)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--do-grid", action="store_true", help="TIME_TOLERANCE grid-search 수행")
    ap.add_argument("--final-tol", type=float, default=None,
                    help="grid-search 건너뛸 때 사용할 TIME_TOLERANCE")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--name", type=str, default=None, help="wandb run name")
    return ap.parse_args()

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    # NCCL async 오류 처리
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    args = parse_args()
    rank = setup_ddp()
    world_size = dist.get_world_size()
    robot = args.robot.lower()

    # 디폴트 CSV 이름 매핑
    default_csv = {
        "fr5":     "fr5_matched_joint_angle.csv",
        "fr3":     "fr3_matched_joint_angle.csv",
        "meca500": "Meca_insertion_matched_joint_angle.csv",
    }
    csv_filename = args.csv or default_csv[robot]

    # 공통 하이퍼파라미터
    hparams = {
        'model_name': "facebook/dinov3-vitb16-pretrain-lvd1689m",
        'batch_size': args.batch,
        'num_epochs': args.epochs,
        'val_split': args.val_split,
        'loss_weight_kpt': 100.0,
        'lr_kpt': 1e-4,
        'lr_ang': 1e-4,
        'lr_backbone': 1e-7,
        'lambda_fk': 0.5,
        'input_size': 224,
        'heatmap_size': (128, 128),
        'sigma': 5.0,
        'num_workers': 8,
        'warmup_epochs': 5,
    }

    # 결과/체크포인트 경로
    paths = _make_run_dirs(robot)
    if rank == 0:
        os.makedirs(paths["vis_dir"], exist_ok=True)
        print(f"[{robot.upper()}] Results -> {paths['run_dir']}")

    # ------------------------------------------------
    # CSV → items(groups or pairs)
    # ------------------------------------------------
    # grid 후보군 로봇별 기본값
    grid_cands = None
    if args.do_grid:
        if robot == "fr5":
            grid_cands = np.round(np.arange(0.01, 0.101, 0.01), 2)
        elif robot == "fr3":
            grid_cands = np.round(np.arange(0.05, 0.101, 0.01), 2)
        else:
            grid_cands = None  # meca500은 보통 싱글 페어

    if rank == 0:
        print(f"\n[CSV] Loading/building items for {robot} from {csv_filename} ...")
    items = build_items_from_csv(
        dataset_type=robot,
        csv_filename=csv_filename,
        max_views_per_group=8,
        do_grid_search=bool(args.do_grid and robot in ("fr3", "fr5")),
        final_tolerance=args.final_tol,
        grid_candidates=grid_cands,
        drop_single_view_groups=True,
        rank=rank,
    )

    # 브로드캐스트 (DDP)
    obj = [items]
    dist.broadcast_object_list(obj, src=0)
    items = obj[0]

    # (선택) rank0 샘플 간단 출력
    if rank == 0 and len(items) > 0:
        n_groups = sum(1 for it in items if 'views' in it)
        n_pairs  = len(items) - n_groups
        print(f"Items: groups={n_groups}, pairs={n_pairs}, total={len(items)}")

    # ------------------------------------------------
    # Setup (dataset/dataloader/model/opt/sched/loss)
    # ------------------------------------------------
    model, train_loader, val_loader, criteria, optimizers, schedulers, device, mean, std, train_sampler, param_sets, strong_transform = \
        setup(
            dataset_type=robot,
            dataset_items=items,
            hyperparameters=hparams,
            rank=rank,
            world_size=world_size,
            model_cls=DINOv3PoseEstimator,
            extra_model_kwargs={'model_name': hparams['model_name']},
        )

    # AMP Grad Scaler
    scalers = {
        'kpt': torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()),
        'ang': torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()),
    }

    # ------------------------------------------------
    # wandb
    # ------------------------------------------------
    run = None
    if args.wandb and rank == 0:
        import wandb
        run = wandb.init(
            project=f"multiview-{robot}",
            name=args.name or f"{robot}_ddp_{time.strftime('%Y%m%d_%H%M%S')}",
            config=hparams
        )
        wandb.watch(model, log="parameters", log_freq=100, log_graph=False)

    # ------------------------------------------------
    # (선택) 파인튜닝 가중치 로드(있으면)
    # ------------------------------------------------
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

    # 예: 같은 로봇의 직전 best를 기본 파인튜닝 소스로 쓸 수 있음
    finetune_candidate = None  # 원하면 경로 지정
    state_to_bcast = None
    if rank == 0 and finetune_candidate:
        state_to_bcast = _safe_load_state_dict(finetune_candidate, device, rank)
    obj = [state_to_bcast]
    dist.broadcast_object_list(obj, src=0)
    finetune_state = obj[0]

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
    elif rank == 0:
        print("ℹ️ No fine-tune weights; training from scratch.")

    # ------------------------------------------------
    # 학습 루프
    # ------------------------------------------------
    if rank == 0:
        print("\n--- Starting Training ---")
    start_epoch, best_val_loss = 0, float('inf')

    # SoftArgmax β, CNN token dropout 스케줄 파라미터(모델이 제공하면 적용)
    beta0, beta1 = 1.0, 3.0
    base_token_drop = 0.10

    for epoch in range(start_epoch, hparams['num_epochs']):
        progress = epoch / max(1, hparams['num_epochs'] - 1)
        m = model.module if hasattr(model, "module") else model

        # softarg β 스케줄
        if hasattr(m, "softarg") and hasattr(m.softarg, "beta"):
            m.softarg.beta = float(beta0 + (beta1 - beta0) * progress)

        # CNN 토큰 드롭 스케줄
        if hasattr(m, "drop_prob_scheduled"):
            m.drop_prob_scheduled = max(0.0, base_token_drop * (1.0 - progress))

        # 강증강 전환
        switch_epoch = hparams['num_epochs'] * 2 // 3
        if epoch == switch_epoch:
            if rank == 0:
                print(f"[Augment] Switching to strong augmentation at epoch {epoch}.")
            train_loader.dataset.transform = strong_transform

        train_sampler.set_epoch(epoch)

        # === train ===
        train_loss_kpt, train_loss_ang = train_one_epoch(
            model, train_loader, optimizers, criteria, device,
            hparams['loss_weight_kpt'], epoch + 1, param_sets, scalers
        )

        # === validate ===
        (val_loss, val_kpt, val_ang,
         val_ang_mae, val_kpt_px) = validate(
            model, val_loader, criteria, device,
            hparams['loss_weight_kpt'], epoch + 1,
            amp_enabled=torch.cuda.is_available()
        )

        # 스케줄러
        schedulers['kpt'].step()
        schedulers['ang'].step()

        # rank0 로깅/저장
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
            }
            if hasattr(m, "softarg") and hasattr(m.softarg, "beta"):
                log_dict["softarg_beta"] = m.softarg.beta
            if hasattr(m, "drop_prob_scheduled"):
                log_dict["cnn_token_drop_sched"] = m.drop_prob_scheduled

            # (선택) von Mises κ 로깅
            if 'ang' in criteria and hasattr(criteria['ang'], 'vm'):
                with torch.no_grad():
                    kappa = criteria['ang'].vm.log_kappa.exp().detach().cpu().numpy()
                log_dict["kappa_mean"] = float(kappa.mean())

            if args.wandb and 'wandb' in globals():
                wandb.log(log_dict)

            print(
                f"[{robot.upper()}][Epoch {epoch+1}] "
                f"ValTotal: {val_loss:.6f} | ValKPT: {val_kpt:.6f} | ValANG: {val_ang:.6f} | "
                f"MAE(deg): {val_ang_mae:.3f} | KPT_L2px(128): {val_kpt_px:.2f} | "
                f"LR_kpt: {log_dict['lr_kpt']:.6f} | LR_ang: {log_dict['lr_ang']:.6f} | "
                f"beta: {log_dict.get('softarg_beta','-')} | drop: {log_dict.get('cnn_token_drop_sched','-')}"
            )

            # 베스트 저장 + (선택) 시각화
            state_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

            did_best_visualize = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"🎉 New best model saved with validation loss: {best_val_loss:.6f}")
                torch.save(state_to_save, paths["best_path"])

                if _HAS_VIS:
                    figs = visualize_predictions(
                        model, val_loader.dataset, device, mean, std,
                        epoch + 1, results_dir=paths["vis_dir"], num_samples=1
                    )
                    if args.wandb and 'wandb' in globals():
                        wandb.log({"validation_predictions": [wandb.Image(fig) for fig in figs]})
                    import matplotlib.pyplot as plt
                    for fig in figs: plt.close(fig)
                did_best_visualize = True

            # 매 5 에폭마다 시각화
            if _HAS_VIS and ((epoch + 1) % 5 == 0) and (not did_best_visualize):
                print(f"🖼️ Periodic visualization at epoch {epoch+1} (every 5 epochs).")
                figs = visualize_predictions(
                    model, val_loader.dataset, device, mean, std,
                    epoch + 1, results_dir=paths["vis_dir"], num_samples=1
                )
                if args.wandb and 'wandb' in globals():
                    wandb.log({f"periodic_predictions/epoch_{epoch+1}": [wandb.Image(fig) for fig in figs]})
                import matplotlib.pyplot as plt
                for fig in figs: plt.close(fig)

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
            torch.save(checkpoint, paths["ckpt_path"])

    cleanup_ddp()
    if rank == 0:
        print("\n--- Training Finished ---")
        if args.wandb and 'wandb' in globals():
            wandb.finish()

if __name__ == "__main__":
    main()
