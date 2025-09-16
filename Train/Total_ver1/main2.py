"""
통합 DREAM-robot 학습 스크립트 (FR3 / FR5 / MECA500)
- torchrun DDP 지원
- utils/dataset 통합본 기반
- 로봇/백본/퓨전 별 결과 폴더 분리 저장
예시:
  torchrun --nproc_per_node=3 main.py --robot fr5
"""

import os, time, argparse
import numpy as np
import pandas as pd

import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler

# 시각화 (선택)
try:
    from vis import visualize_dataset_sample, visualize_predictions, visualize_dataset_samples
    _HAS_VIS = True
except Exception:
    _HAS_VIS = False

# === our modules (통합본) ===
import dataset as ds_mod                          # ✅ 모듈 핸들
from dataset import build_items_from_csv, SPECS   # 함수/상수만 직접 import
from setup import setup
from models import DINOv3PoseEstimator, ModelCfg
from train_val import train_one_epoch, evaluate
from utils import get_spec, set_globals_for       # ✅ 스펙 단일 소스

class _ModelForVis(torch.nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
    @torch.no_grad()
    def forward(self, inp):
        out = self.wrapped(inp, as_dict=True, return_3d=False)
        hm = out["heatmaps"]
        angles = out.get("angles", out.get("coords_3d", None))
        return hm, angles

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

def _result_root(robot, model_tag, fusion):
    return os.path.join(_get_project_root(), "results", robot, model_tag, fusion)

def _make_run_dirs(robot, model_tag, fusion):
    root = _result_root(robot, model_tag, fusion)
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root, ts)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "best_path": os.path.join(run_dir, "best.pth"),
        "ckpt_path": os.path.join(run_dir, "checkpoint.pth"),
        "vis_dir": os.path.join(run_dir, "vis"),
    }

# ------------------------------------------------
# AMP 유틸
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
    ap.add_argument("--robot", type=str, default="fr5",
                    choices=["fr3", "fr5", "meca500"])
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=72)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--do-grid", action="store_true")
    ap.add_argument("--final-tol", type=float, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--name", type=str, default=None)

    ap.add_argument("--viz-start", type=int, default=8,
                    help="시작 시 GT 디버그 시각화 샘플 개수(0이면 비활성)")
    ap.add_argument("--viz-dir", type=str, default="viz_start",
                    help="시작 시각화 저장 폴더명(런 디렉토리 내부)")
    # backbone / fusion
    ap.add_argument("--model-id", type=str,
                    default="facebook/dinov3-convnext-large-pretrain-lvd1689m")
    ap.add_argument("--fusion", type=str, default="auto",
                    choices=["auto", "early", "middle", "late"])
    ap.add_argument("--fusion-default-multi", type=str, default="late",
                    choices=["early", "middle", "late"])
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--feature-dim", type=int, default=768)
    return ap.parse_args()

# ------------------------------------------------
# Dataset patcher
# ------------------------------------------------
def _patch_dataset_class(robot: str):
    """
    setup() 내부에서 ds_mod.UnifiedRobotPoseDataset를 인스턴스화할 때
    로봇별 전용 Dataset 클래스로 교체해서 GT 파이프라인을 보정한다.
    """
    if robot == "fr5":
        from fr5_dataset import FR5_RobotPoseDataset
        ds_mod.UnifiedRobotPoseDataset = FR5_RobotPoseDataset
        print("[PATCH] Using FR5_RobotPoseDataset for UnifiedRobotPoseDataset")
    elif robot == "fr3":
        from fr3_dataset import FR3_RobotPoseDataset
        ds_mod.UnifiedRobotPoseDataset = FR3_RobotPoseDataset
        print("[PATCH] Using FR3_RobotPoseDataset for UnifiedRobotPoseDataset")
    else:
        print("[PATCH] Using default UnifiedRobotPoseDataset (no patch)")

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    args = parse_args()
    rank = setup_ddp()
    world_size = dist.get_world_size()
    robot = args.robot.lower()

    # ✅ 로봇 스펙을 전역 상수로 동기화(레거시 호환) + spec 객체 획득
    set_globals_for(robot)
    spec = get_spec(robot)
    if rank == 0:
        print(f"[CFG] robot={robot} -> NUM_ANGLES={spec.num_angles}, NUM_JOINTS={spec.num_joints}, "
              f"HEATMAP_SIZE={spec.heatmap_size}, MAX_VIEWS={spec.max_views_per_group}")

    # ✅ 로봇별 전용 Dataset으로 클래스 교체
    _patch_dataset_class(robot)

    default_csv = {
        "fr5":     "fr5_matched_joint_angle.csv",
        "fr3":     "fr3_matched_joint_angle.csv",
        "meca500": "Meca_insertion_matched_joint_angle.csv",
    }
    csv_filename = args.csv or default_csv[robot]

    hparams = {
        'batch_size': args.batch,
        'num_epochs': args.epochs,
        'val_split': args.val_split,
        'loss_weight_kpt': 100.0,
        'lr_kpt': 1e-4,
        'lr_ang': 1e-4,
        'lr_backbone': 1e-7,
        'lambda_fk': 0.5,
        'input_size': 224,
        # ✅ spec 반영
        'heatmap_size': spec.heatmap_size,
        'sigma': 5.0,
        'num_workers': 8,
        'warmup_epochs': 5,
    }

    model_tag = "cnx" if "convnext" in args.model_id.lower() else ("vit" if "vit" in args.model_id.lower() else "hf")
    fusion_tag = args.fusion
    paths = _make_run_dirs(robot, model_tag, fusion_tag)
    if rank == 0:
        os.makedirs(paths["vis_dir"], exist_ok=True)
        print(f"[{robot.upper()}] Results -> {paths['run_dir']}")

    # CSV → items
    grid_cands = None
    if args.do_grid:
        if robot == "fr5":
            grid_cands = np.round(np.arange(0.01, 0.101, 0.01), 2)
        elif robot == "fr3":
            grid_cands = np.round(np.arange(0.05, 0.101, 0.01), 2)

    if rank == 0:
        print(f"\n[CSV] Loading/building items for {robot} from {csv_filename} ...")
    items = build_items_from_csv(
        dataset_type=robot,
        csv_filename=csv_filename,
        # ✅ spec 반영
        max_views_per_group=spec.max_views_per_group,
        do_grid_search=bool(args.do_grid and robot in ("fr3", "fr5")),
        final_tolerance=args.final_tol,
        grid_candidates=grid_cands,
        drop_single_view_groups=True,
        rank=rank,
    )

    obj = [items]
    dist.broadcast_object_list(obj, src=0)
    items = obj[0]

    if rank == 0 and len(items) > 0:
        n_groups = sum(1 for it in items if 'views' in it)
        n_pairs  = len(items) - n_groups
        print(f"Items: groups={n_groups}, pairs={n_pairs}, total={len(items)}")

    # Model Config (스펙 반영)
    cfg = ModelCfg(
        MODEL_NAME=args.model_id,
        NUM_ANGLES=spec.num_angles,
        NUM_JOINTS=spec.num_joints,
        FEATURE_DIM=args.feature_dim,
        HEATMAP_SIZE=hparams['heatmap_size'],
        MAX_VIEWS_PER_GROUP=spec.max_views_per_group,
        FUSION=args.fusion,
        DEFAULT_FUSION_FOR_MULTI=args.fusion_default_multi,
        FREEZE_BACKBONE=args.freeze_backbone,
    )

    # === setup 호출로 train_loader/val_loader/mean/std 확보 ===
    model, train_loader, val_loader, criteria, optimizers, schedulers, device, mean, std, train_sampler, param_sets, strong_transform = \
        setup(
            dataset_type=robot,
            dataset_items=items,
            hyperparameters=hparams,
            rank=rank,
            world_size=world_size,
            model_cls=DINOv3PoseEstimator,
            extra_model_kwargs={'cfg': cfg},
        )

    # === ✅ 시작 시 GT 시각화: setup 이후, rank0만 ===
    if rank == 0 and args.viz_start > 0:
        viz_dir = os.path.join(paths["run_dir"], args.viz_dir)
        ds_for_viz = getattr(train_loader, "dataset", None)
        if ds_for_viz is None:
            print("[GT-VIZ] train_loader.dataset이 없습니다. 시각화를 건너뜁니다.")
        else:
            try:
                visualize_dataset_samples(
                    ds_for_viz,
                    save_dir=viz_dir,
                    num_samples=args.viz_start,
                    mean=mean, std=std,
                    input_size=hparams['input_size']
                )
            except Exception as e:
                print(f"[GT-VIZ] 시각화 실패: {e}")

    scalers = {
        'kpt': GradScaler("cuda", enabled=torch.cuda.is_available()),
        'ang': GradScaler("cuda", enabled=torch.cuda.is_available()),
    }

    run = None
    if args.wandb and rank == 0:
        import wandb
        run_name = args.name or f"{robot}-{model_tag}-{fusion_tag}-{time.strftime('%Y%m%d_%H%M%S')}"
        run = wandb.init(
            project=f"multiview-{robot}",
            name=run_name,
            config={**hparams,
                    "model_id": args.model_id,
                    "fusion": args.fusion,
                    "fusion_default_multi": args.fusion_default_multi,
                    "freeze_backbone": args.freeze_backbone,
                    "feature_dim": args.feature_dim}
        )
        wandb.watch(model, log="parameters", log_freq=100, log_graph=False)

    # (선택) FT 가중치 로드
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

    finetune_state = None
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

    # --- Training Loop ---
    if rank == 0:
        print("\n--- Starting Training ---")
    start_epoch, best_val_loss = 0, float('inf')
    beta0, beta1 = 1.0, 3.0
    base_token_drop = 0.10
    switch_epoch = hparams['num_epochs'] * 2 // 3

    for epoch in range(start_epoch, hparams['num_epochs']):
        progress = epoch / max(1, hparams['num_epochs'] - 1)
        m = model.module if hasattr(model, "module") else model

        if hasattr(m, "softarg") and hasattr(m.softarg, "beta"):
            m.softarg.beta = float(beta0 + (beta1 - beta0) * progress)
        if hasattr(m, "drop_prob_scheduled"):
            m.drop_prob_scheduled = max(0.0, base_token_drop * (1.0 - progress))
        if epoch == switch_epoch:
            if rank == 0:
                print(f"[Augment] Switching to strong augmentation at epoch {epoch}.")
            train_loader.dataset.transform = strong_transform

        train_sampler.set_epoch(epoch)

        train_loss_kpt, train_loss_ang = train_one_epoch(
            model, train_loader, optimizers, criteria,
            m.device if hasattr(m, "device") else torch.device(f"cuda:{rank}"),
            hparams['loss_weight_kpt'], epoch + 1, param_sets, scalers
        )

        (val_loss, val_kpt, val_ang, val_ang_mae, val_kpt_px) = evaluate(
            model, val_loader, criteria,
            m.device if hasattr(m, "device") else torch.device(f"cuda:{rank}"),
            hparams['loss_weight_kpt'], epoch + 1,
            amp_enabled=torch.cuda.is_available()
        )

        schedulers['kpt'].step()
        schedulers['ang'].step()

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

            if 'ang' in criteria and hasattr(criteria['ang'], 'vm'):
                with torch.no_grad():
                    kappa = criteria['ang'].vm.log_kappa.exp().detach().cpu().numpy()
                log_dict["kappa_mean"] = float(kappa.mean())

            if args.wandb and 'wandb' in globals():
                wandb.log(log_dict)

            print(
                f"[{robot.upper()}][{model_tag}|{fusion_tag}] Epoch {epoch+1} "
                f"ValTotal: {val_loss:.6f} | ValKPT: {val_kpt:.6f} | ValANG: {val_ang:.6f} | "
                f"MAE(deg): {val_ang_mae:.3f} | KPT_L2px(128): {val_kpt_px:.2f} | "
                f"LR_kpt: {log_dict['lr_kpt']:.6f} | LR_ang: {log_dict['lr_ang']:.6f} | "
                f"beta: {log_dict.get('softarg_beta','-')} | drop: {log_dict.get('cnn_token_drop_sched','-')}"
            )

            state_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

            did_best_visualize = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"🎉 New best model saved with validation loss: {best_val_loss:.6f}")
                torch.save(state_to_save, paths["best_path"])

                if _HAS_VIS:
                    model_for_vis = _ModelForVis(model.module if hasattr(model, "module") else model)
                    figs = visualize_predictions(
                        model_for_vis, val_loader.dataset, device, mean, std,
                        epoch + 1, results_dir=paths["vis_dir"], num_samples=1
                    )
                    if args.wandb and 'wandb' in globals():
                        wandb.log({"validation_predictions": [wandb.Image(fig) for fig in figs]})
                    import matplotlib.pyplot as plt
                    for fig in figs: plt.close(fig)
                did_best_visualize = True

            if _HAS_VIS and ((epoch + 1) % 5 == 0) and (not did_best_visualize):
                print(f"🖼️ Periodic visualization at epoch {epoch+1} (every 5 epochs).")
                model_for_vis = _ModelForVis(model.module if hasattr(model, "module") else model)
                figs = visualize_predictions(
                    model_for_vis, val_loader.dataset, device, mean, std,
                    epoch + 1, results_dir=paths["vis_dir"], num_samples=1
                )
                if args.wandb and 'wandb' in globals():
                    wandb.log({f"periodic_predictions/epoch_{epoch+1}": [wandb.Image(fig) for fig in figs]})
                import matplotlib.pyplot as plt
                for fig in figs: plt.close(fig)

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
