# loss_and_metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------
# Angle helpers
# ---------------------------
def angles_to_sin_cos(angles: torch.Tensor, unit: str = "deg") -> torch.Tensor:
    """
    angles: (B, A)  # A = num_angles
    return: (B, A, 2) with [sin, cos]
    """
    if unit == "deg":
        rad = angles * (math.pi / 180.0)
    elif unit == "rad":
        rad = angles
    else:
        raise ValueError("unit must be 'deg' or 'rad'")
    return torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)  # (B,A,2)

def angular_error_deg(pred_sin_cos: torch.Tensor, target_sin_cos: torch.Tensor) -> torch.Tensor:
    """
    pred_sin_cos, target_sin_cos: (B,A,2) with [sin, cos]
    return: (B,A) in degrees
    """
    # 각도 차: Δθ = atan2( sinΔ, cosΔ ), 여기서 sinΔ = sinθp cosθt - cosθp sinθt, cosΔ = cosθp cosθt + sinθp sinθt
    sp, cp = pred_sin_cos[..., 0], pred_sin_cos[..., 1]
    st, ct = target_sin_cos[..., 0], target_sin_cos[..., 1]
    sin_delta = sp * ct - cp * st
    cos_delta = cp * ct + sp * st
    delta = torch.atan2(sin_delta, cos_delta)  # (-pi, pi]
    return torch.abs(delta) * (180.0 / math.pi)

# ---------------------------
# Multi-task loss
# ---------------------------
class MultiTaskPoseLoss(nn.Module):
    """
    total = λ_hm * L_hm + λ_ang * L_ang + λ_unit * L_unit
    - L_hm: per-view heatmap L2
    - L_ang: per-angle [sin,cos] L2
    - L_unit: (||v||-1)^2 for predicted angle vectors
    """
    def __init__(self,
                 lambda_hm: float = 1.0,
                 lambda_ang: float = 1.0,
                 lambda_unit: float = 0.05,
                 angle_unit: str = "deg"):
        super().__init__()
        self.lambda_hm = lambda_hm
        self.lambda_ang = lambda_ang
        self.lambda_unit = lambda_unit
        assert angle_unit in ("deg", "rad")
        self.angle_unit = angle_unit
        self.mse = nn.MSELoss(reduction="mean")

    def heatmap_loss(self, pred_hm_dict, gt_hm_dict):
        """
        pred_hm_dict[vk]: (B,J,H,W)
        gt_hm_dict[vk]:   (B,J,H,W) or (J,H,W) when B=1
        """
        total, cnt = 0.0, 0
        for vk in pred_hm_dict.keys():
            pred = pred_hm_dict[vk]
            gt = gt_hm_dict[vk]
            if gt.dim() == 3:
                gt = gt.unsqueeze(0)
            total = total + self.mse(pred, gt)
            cnt += 1
        return total / max(cnt, 1)

    def angle_loss(self, pred_angles, gt_angles):
        """
        pred_angles: (B,A,2) [sin, cos]
        gt_angles:   (B,A)   numeric angles in deg/rad
        """
        target_sc = angles_to_sin_cos(gt_angles, unit=self.angle_unit)
        return self.mse(pred_angles, target_sc)

    def unit_norm_loss(self, pred_angles):
        """
        pred_angles: (B,A,2)
        """
        norms = torch.linalg.norm(pred_angles, dim=-1)  # (B,A)
        return torch.mean((norms - 1.0) ** 2)

    def forward(self, pred_hm_dict, gt_hm_dict, pred_angles, gt_angles):
        l_hm   = self.heatmap_loss(pred_hm_dict, gt_hm_dict)
        l_ang  = self.angle_loss(pred_angles, gt_angles)
        l_unit = self.unit_norm_loss(pred_angles)
        total  = self.lambda_hm * l_hm + self.lambda_ang * l_ang + self.lambda_unit * l_unit
        return total, {"L_hm": l_hm.detach(), "L_ang": l_ang.detach(), "L_unit": l_unit.detach(), "L_total": total.detach()}

# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def metrics_from_batch(pred_hm_dict, gt_hm_dict, pred_angles, gt_angles, angle_unit="deg"):
    """
    간단 메트릭:
      - heatmap MSE (per-view 평균)
      - mean angular error (deg)
    """
    mse = nn.MSELoss(reduction="mean")
    # heatmap mse
    hm_vals, n = 0.0, 0
    for vk in pred_hm_dict.keys():
        pred = pred_hm_dict[vk]
        gt = gt_hm_dict[vk]
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)
        hm_vals += mse(pred, gt).item()
        n += 1
    hm_mse = hm_vals / max(n, 1)

    # angle error
    target_sc = angles_to_sin_cos(gt_angles, unit=angle_unit)
    # pred_angles: (B,A,2), target_sc: (B,A,2)
    ang_err = angular_error_deg(pred_angles, target_sc).mean().item()

    return {"hm_mse": hm_mse, "angle_err_deg": ang_err}
