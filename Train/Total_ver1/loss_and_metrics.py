# loss_and_metrics.py
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# ---- Angle utilities
# =========================

def _deg_to_rad(x: torch.Tensor) -> torch.Tensor:
    return x * math.pi / 180.0

def _rad_to_deg(x: torch.Tensor) -> torch.Tensor:
    return x * 180.0 / math.pi

def anglevec_to_angle_rad(vec: torch.Tensor) -> torch.Tensor:
    """
    vec: (..., 2) as [sin, cos]
    return: angle in rad, same leading shape (...)
    """
    s = vec[..., 0]
    c = vec[..., 1]
    return torch.atan2(s, c)

def anglevec_to_angle_deg(vec: torch.Tensor) -> torch.Tensor:
    return _rad_to_deg(anglevec_to_angle_rad(vec))

def wrap_angle_diff_deg(diff: torch.Tensor) -> torch.Tensor:
    """
    wrap degree difference to [-180, 180]
    """
    return (diff + 180.0) % 360.0 - 180.0

def build_gt_angle_vec(gt_angles: torch.Tensor, unit: str = "deg") -> torch.Tensor:
    """
    gt_angles: (B, N) in deg or rad
    return: (B, N, 2) as [sin, cos]
    """
    if unit == "deg":
        ang = _deg_to_rad(gt_angles)
    elif unit == "rad":
        ang = gt_angles
    else:
        raise ValueError("unit must be 'deg' or 'rad'")
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)

# =========================
# ---- Angle losses
# =========================

class VonMisesAngleLoss(nn.Module):
    """
    -log p(θ | μ, κ) = -κ cos(θ-μ) + log I0(κ)
    예측은 벡터 [sinμ, cosμ] 로 주어짐.
    per-joint learnable κ (logκ 파라미터) 옵션 제공.
    """
    def __init__(self, num_angles: int, learn_kappa: bool = True, init_log_kappa: float = 1.0):
        super().__init__()
        self.num_angles = num_angles
        if learn_kappa:
            self.log_kappa = nn.Parameter(torch.full((num_angles,), init_log_kappa))
        else:
            # 학습하지 않는다면 buffer로 고정
            self.register_buffer("log_kappa", torch.full((num_angles,), init_log_kappa))
        # torch.i0 사용 (I0)
        self._eps = 1e-9

    def forward(self, pred_vec: torch.Tensor, gt_vec: torch.Tensor) -> torch.Tensor:
        """
        pred_vec: (B, N, 2) [sin, cos], unit-length로 가정
        gt_vec:   (B, N, 2) [sin, cos]
        """
        # cos(θ-μ) = dot(u_pred, u_gt)
        dot = (pred_vec * gt_vec).sum(dim=-1)  # (B, N)
        kappa = self.log_kappa.exp()  # (N,)
        # broadast to (B, N)
        kappa = kappa.unsqueeze(0).expand_as(dot)

        # -κ * cos(θ-μ) + log I0(κ)
        # torch.i0는 elementwise
        loss = -kappa * dot + torch.i0(kappa + self._eps).log()
        return loss.mean()


class CosineAngleLoss(nn.Module):
    """
    1 - cos(θ-μ) = 1 - dot([sin,cos]_pred, [sin,cos]_gt)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_vec: torch.Tensor, gt_vec: torch.Tensor) -> torch.Tensor:
        dot = (pred_vec * gt_vec).sum(dim=-1)  # (B, N)
        return (1.0 - dot).mean()


class AngleLossCombo(nn.Module):
    """
    VonMises + Cosine 합성. setup에서 .vm.to(device) 를 호출할 수 있도록 .vm 보유.
    """
    def __init__(self, num_angles: int, vm_weight: float = 0.5, cos_weight: float = 0.5, learn_kappa: bool = True):
        super().__init__()
        self.vm_weight = float(vm_weight)
        self.cos_weight = float(cos_weight)
        self.vm = VonMisesAngleLoss(num_angles=num_angles, learn_kappa=learn_kappa)
        self.cos = CosineAngleLoss()

    def forward(self,
                pred_angle_vec: torch.Tensor,   # (B, N, 2) [sin, cos]
                gt_angles: torch.Tensor,        # (B, N)
                gt_unit: str = "deg") -> torch.Tensor:
        gt_vec = build_gt_angle_vec(gt_angles, unit=gt_unit)  # (B, N, 2)
        loss_vm  = self.vm(pred_angle_vec, gt_vec) if self.vm_weight > 0 else pred_angle_vec.new_tensor(0.0)
        loss_cos = self.cos(pred_angle_vec, gt_vec) if self.cos_weight > 0 else pred_angle_vec.new_tensor(0.0)
        return self.vm_weight * loss_vm + self.cos_weight * loss_cos


def make_angle_loss(num_angles: int,
                    vm_weight: float = 0.5,
                    cos_weight: float = 0.5,
                    learn_kappa: bool = True) -> AngleLossCombo:
    """
    setup.py에서 import 하는 팩토리 함수
    """
    return AngleLossCombo(num_angles=num_angles, vm_weight=vm_weight, cos_weight=cos_weight, learn_kappa=learn_kappa)

# =========================
# ---- Heatmap loss
# =========================

def mse_heatmap_loss(pred_hm: torch.Tensor, gt_hm: torch.Tensor) -> torch.Tensor:
    """
    pred_hm, gt_hm: (B, J, H, W)
    """
    return F.mse_loss(pred_hm, gt_hm)

def dict_mse_heatmap_loss(pred_hm_dict: Dict[str, torch.Tensor],
                          gt_hm_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    뷰별 heatmap dict에 대해 평균 MSE
    """
    losses = []
    for k in gt_hm_dict.keys():
        if k not in pred_hm_dict:
            continue
        losses.append(mse_heatmap_loss(pred_hm_dict[k], gt_hm_dict[k]))
    if len(losses) == 0:
        # 예외적으로 키가 안 맞을 때 0 반환 (학습이 진행되도록)
        return next(iter(gt_hm_dict.values())).new_tensor(0.0)
    return torch.stack(losses).mean()

# =========================
# ---- Multi-task wrapper
# =========================

class MultiTaskPoseLoss(nn.Module):
    """
    총손실 = hm_weight * HeatmapMSE + angle_weight * AngleLoss (+ fk_weight * FKaux if 주입)
    FK aux는 외부에서 criteria['fk'](pred_angles or pred_3d, gt_3d)를 통해 별도로 계산한 값을
    forward 인자로 넘기도록 훅을 두었음.
    """
    def __init__(self,
                 angle_loss: AngleLossCombo,
                 hm_weight: float = 1.0,
                 angle_weight: float = 1.0,
                 fk_weight: float = 0.0,
                 gt_unit: str = "deg"):
        super().__init__()
        self.angle_loss = angle_loss
        self.hm_weight = float(hm_weight)
        self.angle_weight = float(angle_weight)
        self.fk_weight = float(fk_weight)
        self.gt_unit = gt_unit

    def forward(self,
                pred_hm_dict: Dict[str, torch.Tensor],  # each: (B,J,H,W)
                gt_hm_dict: Dict[str, torch.Tensor],    # each: (B,J,H,W)
                pred_angle_vec: torch.Tensor,           # (B,N,2) [sin, cos]
                gt_angles: torch.Tensor,                # (B,N)
                fk_aux_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        loss_kpt = dict_mse_heatmap_loss(pred_hm_dict, gt_hm_dict)
        loss_ang = self.angle_loss(pred_angle_vec, gt_angles, gt_unit=self.gt_unit)

        total = self.hm_weight * loss_kpt + self.angle_weight * loss_ang
        if fk_aux_loss is not None and self.fk_weight > 0:
            total = total + self.fk_weight * fk_aux_loss

        return {
            "total": total,
            "kpt": loss_kpt.detach(),
            "ang": loss_ang.detach(),
            "fk": fk_aux_loss.detach() if (fk_aux_loss is not None) else torch.tensor(0.0, device=total.device),
        }

# =========================
# ---- Metrics
# =========================

def _argmax_uv(hm: torch.Tensor) -> torch.Tensor:
    """
    hm: (B, J, H, W)
    return: (B, J, 2) as (x, y) in heatmap coords
    """
    B, J, H, W = hm.shape
    idx = hm.view(B, J, -1).argmax(dim=-1)  # (B,J)
    y = (idx // W).float()
    x = (idx %  W).float()
    return torch.stack([x, y], dim=-1)

def _dict_argmax_uv(hm_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: _argmax_uv(v) for k, v in hm_dict.items()}

@torch.no_grad()
def metrics_from_batch(pred_hm_dict: Dict[str, torch.Tensor],
                       gt_hm_dict: Dict[str, torch.Tensor],
                       pred_angle_vec: torch.Tensor,  # (B,N,2)
                       gt_angles: torch.Tensor,       # (B,N) in deg by default
                       gt_unit: str = "deg") -> Dict[str, float]:
    """
    반환:
      - 'val_kpt_L2px_128': (heatmap 해상도 기준) L2(px)
      - 'val_angle_MAE_deg': 각도 MAE(deg, 원형거리)
    """
    # 2D keypoint L2(px) on heatmap grid
    l2_all = []
    for k in gt_hm_dict.keys():
        if k not in pred_hm_dict:
            continue
        uv_pd = _argmax_uv(pred_hm_dict[k])  # (B,J,2)
        uv_gt = _argmax_uv(gt_hm_dict[k])
        l2 = torch.linalg.norm(uv_pd - uv_gt, dim=-1)  # (B,J)
        l2_all.append(l2)
    if len(l2_all) > 0:
        l2_px = torch.cat(l2_all, dim=1).mean().item()  # 평균
    else:
        l2_px = float("nan")

    # Angle MAE (deg)
    pred_deg = anglevec_to_angle_deg(pred_angle_vec)  # (B,N)
    if gt_unit == "rad":
        gt_deg = _rad_to_deg(gt_angles)
    else:
        gt_deg = gt_angles
    diff = wrap_angle_diff_deg(pred_deg - gt_deg).abs().mean().item()

    return {
        "val_kpt_L2px_128": l2_px,
        "val_angle_MAE_deg": diff,
    }
