
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from fr5_utils import (
    MODEL_NAME, NUM_ANGLES, NUM_JOINTS, HEATMAP_SIZE, FEATURE_DIM
)

# ... (All other classes like VonMisesAngleLoss, FR5FK, DINOv3Backbone, etc., remain unchanged) ...
# ====== Angle Losses: von Mises NLL + cosine distance ======
class VonMisesAngleLoss(nn.Module):
    """
    pred_vec: (B,A,2) 단위벡터(sin,cos)
    gt_deg:   (B,A)   각도(도)
    learnable kappa(조인트별)로 원형 분포 우도 최대화 학습.
    """
    def __init__(self, num_angles, learn_kappa=True, init_kappa=2.0):
        super().__init__()
        init = torch.ones(num_angles) * init_kappa
        if learn_kappa:
            self.log_kappa = nn.Parameter(init.log())
        else:
            self.register_buffer("log_kappa", init.log())

    def forward(self, pred_vec, gt_deg):
        gt_rad = gt_deg * math.pi / 180.0
        gt_vec = torch.stack([torch.sin(gt_rad), torch.cos(gt_rad)], dim=-1)  # (B,A,2)
        cos_delta = (pred_vec * gt_vec).sum(dim=-1).clamp(-1+1e-6, 1-1e-6)   # (B,A)
        kappa = self.log_kappa.exp().view(1, -1)                              # (1,A)
        nll = -(kappa * cos_delta)                                           # 상수항 제외
        return nll.mean()

class CosineAngleLoss(nn.Module):
    """단순 코사인 거리: 1 - cos(Δθ)"""
    def forward(self, pred_vec, gt_deg):
        gt_rad = gt_deg * math.pi / 180.0
        gt_vec = torch.stack([torch.sin(gt_rad), torch.cos(gt_rad)], dim=-1)
        cos_delta = (pred_vec * gt_vec).sum(dim=-1).clamp(-1+1e-6, 1-1e-6)
        return (1.0 - cos_delta).mean()

def make_angle_loss(num_angles, vm_weight=0.5, cos_weight=0.5):
    vm = VonMisesAngleLoss(num_angles, learn_kappa=True, init_kappa=2.0)
    cos = CosineAngleLoss()
    def _loss(pred_vec, gt_deg):
        return vm_weight * vm(pred_vec, gt_deg) + cos_weight * cos(pred_vec, gt_deg)
    # 두 모듈을 속성으로 보관(DDP에서 파라미터 노출 위해)
    _loss.vm = vm
    _loss.cos = cos
    return _loss

def weighed_kpt_loss(pred_hm_dict, heatmaps_gpu, softarg_module):
    losses = []
    for k, pred in pred_hm_dict.items():
        gt = heatmaps_gpu[k]
        base = F.mse_loss(pred, gt, reduction='none')
        with torch.no_grad():
            _, _, ent = softarg_module(pred.detach())
            w = (1.0 - ent / (math.log(pred.shape[-1]*pred.shape[-2]) + 1e-6)).clamp(0.1, 1.0)
            w = w.unsqueeze(-1).unsqueeze(-1)
        loss = (base * w).mean()
        losses.append(loss)
    return torch.stack(losses).mean()

# ==== Torch FK (Modified DH, differentiable) ====
class FR5FK(nn.Module):
    def __init__(self, device):
        super().__init__()
        a     = torch.tensor([ 0.000, -0.425, -0.395, 0.000, 0.000, 0.000], device=device)
        d     = torch.tensor([ 0.152,  0.000,  0.000, 0.102, 0.102, 0.100], device=device)
        alpha = torch.tensor([90.0,    0.0,    0.0,  90.0, -90.0,  0.0],    device=device)
        self.register_buffer('a', a); self.register_buffer('d', d); self.register_buffer('alpha', alpha)

    @staticmethod
    def _dh(a, d, alpha_deg, theta_deg):
        alpha = torch.deg2rad(alpha_deg); th = torch.deg2rad(theta_deg)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        ct, st = torch.cos(th),    torch.sin(th)
        zeros = torch.zeros_like(ct); ones = torch.ones_like(ct)
        row0 = torch.stack([ct, -st*ca,  st*sa, a*ct], dim=-1)
        row1 = torch.stack([st,  ct*ca, -ct*sa, a*st], dim=-1)
        row2 = torch.stack([0*ct,  sa,     ca,     d], dim=-1)
        row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        return torch.stack([row0, row1, row2, row3], dim=-2)

    def forward(self, joint_deg):
        B = joint_deg.shape[0]
        theta = joint_deg[..., :6]
        T = torch.eye(4, device=joint_deg.device).unsqueeze(0).repeat(B,1,1)
        pts = [T[..., :3, 3]]
        for i in range(6):
            Ti = self._dh(self.a[i].expand(B), self.d[i].expand(B), self.alpha[i].expand(B), theta[:, i])
            T = T @ Ti
            pts.append(T[..., :3, 3])
        return torch.stack(pts, dim=1)

def train_one_epoch(model, loader, optimizers, criteria, device, loss_weight_kpt, epoch_num, param_sets, scalers):
    """
    DINOv3 백본이 동결된 상태에 맞는 두 경로 분리 학습 로직입니다.
    (Angle 경로 backward -> Keypoint 경로 backward)
    """
    model.train()
    total_loss_kpt, total_loss_ang = 0.0, 0.0
    num_effective_batches = 0

    # ▼▼▼ [수정] 백본 관련 optimizer와 scaler를 모두 제거합니다. ▼▼▼
    optimizer_kpt, optimizer_ang = optimizers['kpt'], optimizers['ang']
    scaler_kpt, scaler_ang = scalers['kpt'], scalers['ang']
    
    crit_ang, fk = criteria['ang'], criteria['fk']
    lambda_fk = criteria.get('lambda_fk', 0.0)
    m = model.module if hasattr(model, 'module') else model
    kpt_ids, ang_ids = param_sets['kpt'], param_sets['ang']

    loop = tqdm(loader, desc=f"Epoch {epoch_num} [Train]")

    for batch in loop:
        image_dict, gt_heatmaps_dict, gt_angles = batch
        if image_dict is None:
            # 데이터셋에서 에러가 발생한 경우 건너뜁니다.
            continue

        images_gpu   = {k: v.to(device, non_blocking=True) for k, v in image_dict.items()}
        heatmaps_gpu = {k: v.to(device, non_blocking=True) for k, v in gt_heatmaps_dict.items()}
        angles_gpu   = gt_angles.to(device, non_blocking=True)

        # --- (A) Angle 경로 ---
        optimizer_ang.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _, pred_angles = model(images_gpu)
            loss_ang_main = crit_ang(pred_angles, angles_gpu)
            pred_rad = torch.atan2(pred_angles[..., 0], pred_angles[..., 1])
            pred_deg = pred_rad * (180.0 / math.pi)
            loss_fk  = F.smooth_l1_loss(fk(pred_deg), fk(angles_gpu), beta=2.0)
            loss_ang_total = loss_ang_main + lambda_fk * loss_fk
        
        # Angle 경로에 대한 그래디언트 계산 및 업데이트
        scaler_ang.scale(loss_ang_total).backward()
        scaler_ang.unscale_(optimizer_ang)
        # Angle 경로에 속하지 않는 파라미터의 그래디언트를 0으로 만듭니다.
        for p in m.parameters():
            if p.grad is not None and id(p) not in ang_ids:
                p.grad.zero_()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if id(p) in ang_ids], max_norm=1.0)
        scaler_ang.step(optimizer_ang)
        scaler_ang.update()
        
        # --- (B) Keypoint 경로 ---
        optimizer_kpt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            pred_heatmaps, _ = model(images_gpu)
            loss_kpt_total = weighed_kpt_loss(pred_heatmaps, heatmaps_gpu, m.softarg) * loss_weight_kpt
        
        # Keypoint 경로에 대한 그래디언트 계산 및 업데이트
        scaler_kpt.scale(loss_kpt_total).backward()
        scaler_kpt.unscale_(optimizer_kpt)
        # Keypoint 경로에 속하지 않는 파라미터의 그래디언트를 0으로 만듭니다.
        for p in m.parameters():
            if p.grad is not None and id(p) not in kpt_ids:
                p.grad.zero_()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if id(p) in kpt_ids], max_norm=1.0)
        scaler_kpt.step(optimizer_kpt)
        scaler_kpt.update()

        total_loss_kpt += loss_kpt_total.item()
        total_loss_ang += loss_ang_total.item()
        num_effective_batches += 1
        loop.set_postfix(loss_kpt=f"{loss_kpt_total.item():.4f}", loss_ang=f"{loss_ang_total.item():.4f}")

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