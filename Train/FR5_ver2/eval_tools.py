# eval_tools.py
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import resize_intrinsics  # (K, sx, sy) -> K'

# -------------------------------------------------------------------
# Soft-Argmax 2D (heatmap -> (x,y) expectation) + confidence
# -------------------------------------------------------------------
class SoftArgmax2D(nn.Module):
    """
    입력:  (B,J,H,W) 또는 (B,V,J,H,W)  (logit 형태 권장)
    출력:  uv  -> (B,J,2) 또는 (B,V,J,2)   (좌표계: heatmap pixel index)
           conf-> (B,J)   또는 (B,V,J)     (최대 확률값)
    """
    def __init__(self, temperature: float = 1.0, eps: float = 1e-9):
        super().__init__()
        self.tau = float(temperature)
        self.eps = eps

    def _softargmax(self, hm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hm: (B,J,H,W)
        return: uv(B,J,2), conf(B,J)
        """
        B, J, H, W = hm.shape
        # softmax over HW
        logits = hm.reshape(B, J, -1) / max(self.tau, 1e-6)
        prob = F.softmax(logits, dim=-1)  # (B,J,HW)

        # expected coordinates
        ys = torch.arange(H, device=hm.device, dtype=hm.dtype)
        xs = torch.arange(W, device=hm.device, dtype=hm.dtype)
        yy = ys[None, None, :, None].expand(B, J, H, W)  # (B,J,H,W)
        xx = xs[None, None, None, :].expand(B, J, H, W)  # (B,J,H,W)

        prob2d = prob.view(B, J, H, W)
        ex = (prob2d * xx).sum(dim=(-1, -2))  # (B,J)
        ey = (prob2d * yy).sum(dim=(-1, -2))  # (B,J)
        uv = torch.stack([ex, ey], dim=-1)    # (B,J,2)

        # confidence = max probability on map
        conf, _ = prob.max(dim=-1)           # (B,J)
        return uv, conf

    def forward(self, hm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if hm.dim() == 4:           # (B,J,H,W)
            return self._softargmax(hm)
        elif hm.dim() == 5:         # (B,V,J,H,W)
            B, V, J, H, W = hm.shape
            uv_list, cf_list = [], []
            for v in range(V):
                uv_v, cf_v = self._softargmax(hm[:, v])
                uv_list.append(uv_v)  # (B,J,2)
                cf_list.append(cf_v)  # (B,J)
            uv = torch.stack(uv_list, dim=1)   # (B,V,J,2)
            cf = torch.stack(cf_list, dim=1)   # (B,V,J)
            return uv, cf
        else:
            raise ValueError(f"Heatmap must be 4D or 5D, got shape={tuple(hm.shape)}")


# -------------------------------------------------------------------
# 간단 2D/3D metrics
# -------------------------------------------------------------------
@torch.no_grad()
def pck_2d(pred_uv: torch.Tensor, gt_uv: torch.Tensor, thr_px: float) -> float:
    """
    pred_uv, gt_uv: (B,V,J,2) 또는 (B,J,2)   (동일 좌표계)
    thr_px: threshold in pixels (해당 좌표계 픽셀 단위)
    """
    if pred_uv.dim() == 4:
        err = torch.linalg.norm(pred_uv - gt_uv, dim=-1)  # (B,V,J)
    else:
        err = torch.linalg.norm(pred_uv - gt_uv, dim=-1)  # (B,J)
    return (err <= thr_px).float().mean().item()

@torch.no_grad()
def mpjpe_3d(pred_X: torch.Tensor, gt_X: torch.Tensor) -> float:
    """
    pred_X, gt_X: (B,J,3)  (단위 일치 필수: mm or m)
    """
    mask = torch.isfinite(pred_X).all(dim=-1) & torch.isfinite(gt_X).all(dim=-1)  # (B,J)
    diff = (pred_X - gt_X)[mask]
    if diff.numel() == 0:
        return float("nan")
    return torch.linalg.norm(diff, dim=-1).mean().item()


# -------------------------------------------------------------------
# DLT Triangulation (픽셀 좌표계 + 해당 좌표계의 K, Rt 필요)
# -------------------------------------------------------------------
@torch.no_grad()
def triangulate_dlt(
    uv_list: List[torch.Tensor],           # list of (B,J,2) per view  (픽셀 좌표)
    K_list:  List[torch.Tensor],           # list of (3,3) or (B,3,3)
    Rt_list: List[torch.Tensor],           # list of (3,4) or (B,3,4)
    valid_mask: Optional[torch.Tensor] = None  # (B,V,J) 1/0
) -> torch.Tensor:
    """
    return: X (B,J,3)  (Rt 정의에 따라 world/base 좌표)
    """
    V = len(uv_list)
    B, J, _ = uv_list[0].shape
    X = torch.zeros(B, J, 3, device=uv_list[0].device, dtype=uv_list[0].dtype)

    for b in range(B):
        for j in range(J):
            A_rows = []
            for v in range(V):
                if valid_mask is not None and valid_mask[b, v, j] == 0:
                    continue
                K = K_list[v] if K_list[v].dim() == 2 else K_list[v][b]
                Rt = Rt_list[v] if Rt_list[v].dim() == 2 else Rt_list[v][b]
                P = K @ Rt  # (3,4)
                u, vpx = uv_list[v][b, j]
                A_rows.append(u * P[2, :] - P[0, :])
                A_rows.append(vpx * P[2, :] - P[1, :])

            if len(A_rows) < 4:  # < 2 views
                X[b, j] = torch.tensor([float('nan')] * 3, device=X.device)
                continue

            A = torch.stack(A_rows, dim=0)  # (2*n, 4)
            try:
                U, S, Vh = torch.linalg.svd(A)
                Xh = Vh[-1]
                X[b, j] = Xh[:3] / (Xh[3] + 1e-9)
            except RuntimeError:
                X[b, j] = torch.tensor([float('nan')] * 3, device=X.device)
    return X


# -------------------------------------------------------------------
# Intrinsics rescale helper
# -------------------------------------------------------------------
def rescale_K_for_target_size(K: torch.Tensor,
                              from_size: Tuple[int, int],
                              to_size: Tuple[int, int]) -> torch.Tensor:
    """
    K: (3,3) or (B,3,3)
    from_size: (W_from, H_from)  (예: undist 이미지 크기)
    to_size:   (W_to,   H_to)    (예: heatmap 크기)
    """
    sx = to_size[0] / float(from_size[0])
    sy = to_size[1] / float(from_size[1])
    if K.dim() == 2:
        Kp = resize_intrinsics(K.cpu().numpy(), sx, sy)
        return torch.from_numpy(Kp).to(K.device, dtype=K.dtype)
    else:
        out = []
        for b in range(K.shape[0]):
            Kp = resize_intrinsics(K[b].cpu().numpy(), sx, sy)
            out.append(torch.from_numpy(Kp))
        return torch.stack(out, dim=0).to(K.device, dtype=K.dtype)


# -------------------------------------------------------------------
# Eval Harness (우리 모델 API: images_dict -> (pred_hm_dict, pred_angles))
# -------------------------------------------------------------------
class EvalHarness:
    """
    - model(images_dict) -> (pred_hm_dict, pred_angles)
      * pred_hm_dict[vk]: (B,J,H,W)
    - decoder: SoftArgmax2D

    NOTE
    ----
    * 이 하니스는 '뷰 dict' 출력에 맞춰 동작.
    * uv는 기본적으로 heatmap 좌표계(H,W)로 반환.
      - 픽셀 좌표(원본/undist)로 삼각측량하려면 uv와 K를 같은 좌표계로 맞춰야 함.
      - 방법 A: uv(heatmap) -> 원본픽셀로 스케일업
      - 방법 B: K(원본) -> heatmap 크기로 스케일다운(rescale_K_for_target_size)
    """
    def __init__(self, model: nn.Module, decoder: Optional[SoftArgmax2D] = None):
        self.model = model.eval()
        self.dec = decoder or SoftArgmax2D()

    @torch.no_grad()
    def infer_2d_from_dict(self, images_dict: Dict[str, torch.Tensor]):
        """
        images_dict: {vk: (B,C,H,W)}
        return:
          uv_dict:   {vk: (B,J,2)} in heatmap coords
          conf_dict: {vk: (B,J)}
        """
        pred_hm_dict, _ = self.model(images_dict)  # pred_hm_dict[vk]: (B,J,H,W)
        uv_dict, conf_dict = {}, {}
        for vk, Hm in pred_hm_dict.items():
            uv, cf = self.dec(Hm)      # (B,J,2), (B,J)
            uv_dict[vk] = uv
            conf_dict[vk] = cf
        return uv_dict, conf_dict

    @torch.no_grad()
    def eval_singleview(self,
                        images_dict: Dict[str, torch.Tensor],
                        gt_uv_hm: Dict[str, torch.Tensor],
                        thr_px: float = 3.0) -> Dict[str, float]:
        """
        gt_uv_hm[vk]: (B,J,2)  (heatmap 좌표계)
        """
        uv_dict, _ = self.infer_2d_from_dict(images_dict)
        accs = []
        for vk in gt_uv_hm.keys():
            accs.append(pck_2d(uv_dict[vk], gt_uv_hm[vk], thr_px))
        return {f"PCK@{thr_px}px": float(sum(accs) / max(len(accs), 1))}

    @torch.no_grad()
    def eval_multiview_3d(self,
                          images_dict: Dict[str, torch.Tensor],
                          Ks: Dict[str, torch.Tensor],         # {vk: (3,3)} for 원본 undist
                          Rts: Dict[str, torch.Tensor],        # {vk: (3,4)}
                          orig_size_hw: Dict[str, Tuple[int,int]],  # {vk: (H,W)} of undist img
                          heatmap_size: Tuple[int,int],        # (Hh,Wh)  (= our GT heatmap)
                          gt_X: Optional[torch.Tensor] = None,
                          scale_intrinsics: bool = True):
        """
        - uv는 heatmap 좌표에서 soft-argmax로 추정
        - triangulate_dlt에 넣기 전, K를 heatmap 좌표계로 스케일 or uv를 원본픽셀로 스케일
        여기선 기본으로 **K를 heatmap 크기로 스케일**해서 사용(scale_intrinsics=True).
        """
        uv_dict, conf_dict = self.infer_2d_from_dict(images_dict)  # uv in heatmap coords

        # dict -> view-정렬 list
        view_keys = sorted(uv_dict.keys())
        uv_list = [uv_dict[k] for k in view_keys]  # list of (B,J,2)
        if scale_intrinsics:
            K_list = []
            for k in view_keys:
                H0, W0 = orig_size_hw[k]
                K_heat = rescale_K_for_target_size(Ks[k], (W0, H0), (heatmap_size[1], heatmap_size[0]))
                K_list.append(K_heat)
        else:
            # 원한다면 uv를 원본픽셀로 스케일업해서 사용 가능(여기선 K 그대로)
            # uv_heat -> uv_pix: (x * W0/Wt, y * H0/Ht)
            raise NotImplementedError("Set scale_intrinsics=True or implement uv upscaling here.")

        Rt_list = [Rts[k] for k in view_keys]

        X = triangulate_dlt(uv_list, K_list, Rt_list, valid_mask=None)  # (B,J,3)

        out = {"triangulated_3D": X, "conf": {k: conf_dict[k] for k in view_keys}}
        if gt_X is not None:
            out["MPJPE"] = mpjpe_3d(X, gt_X)
        return out
