import torch

class EvalHarness:
    def __init__(self, model: nn.Module, decoder: SoftArgmax2D):
        self.model = model.eval()
        self.dec = decoder

    @torch.no_grad()
    def infer_2d(self, images):
        """
        images: (B,V,C,H,W) or (B,C,H,W)
        return: uv (B,V,J,2), conf (B,V,J)
        """
        out = self.model(images)  # {"heatmaps": (B,V,J,Hh,Wh)}
        Hm = out["heatmaps"]
        uv, conf = self.dec(Hm)   # handles both 4D/5D
        # unify to (B,V,J,2)
        if uv.dim()==3:  # (B,J,2) -> (B,1,J,2)
            uv = uv.unsqueeze(1); conf = conf.unsqueeze(1)
        return uv, conf

    @torch.no_grad()
    def eval_singleview(self, images, gt_uv, thr_px: float = 3.0):
        """
        images: (B,1,C,H,W) or (B,C,H,W)
        gt_uv:  (B,1,J,2) or (B,J,2)
        """
        pred_uv, _ = self.infer_2d(images)      # (B,1,J,2)
        if gt_uv.dim()==3: gt_uv = gt_uv.unsqueeze(1)
        acc = pck_2d(pred_uv, gt_uv, thr_px)
        return {"PCK@{}px".format(thr_px): acc}

    @torch.no_grad()
    def eval_multiview_3d(self, images, Ks, Rts, gt_X=None, valid_views=None, thr_px: float = 3.0):
        """
        images: (B,V,C,H,W), Ks: list of (3,3) or (B,3,3), Rts: list of (3,4) or (B,3,4)
        gt_X: (B,J,3) optional
        """
        pred_uv, conf = self.infer_2d(images)   # (B,V,J,2)
        B, V, J, _ = pred_uv.shape
        if valid_views is None:
            valid_views = torch.ones(B, V, J, device=pred_uv.device, dtype=torch.long)
        uv_list = [pred_uv[:, v] for v in range(V)]
        X = triangulate_dlt(uv_list, Ks, Rts, valid_mask=valid_views)  # (B,J,3)
        metrics = {"triangulated_3D": X}

        # Optional: 2D PCK 평균(멀티뷰)
        # gt_uv_mv: 필요 시 넣어 평균 PCK도 계산 가능
        if gt_X is not None:
            metrics["MPJPE"] = mpjpe_3d(X, gt_X)
        return metrics

def triangulate_dlt(uv_list, K_list, Rt_list, valid_mask=None):
    """
    uv_list:  list of (B, J, 2) per view  (픽셀)
    K_list:   list of (3,3) intrinsics per view (또는 (B,3,3))
    Rt_list:  list of (3,4) extrinsics [R|t] per view (또는 (B,3,4))
    valid_mask: (B, V, J) 1/0

    return: X (B, J, 3) in world(or robot base) coords depending on Rt
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
                K = K_list[v] if K_list[v].dim()==2 else K_list[v][b]
                Rt = Rt_list[v] if Rt_list[v].dim()==2 else Rt_list[v][b]
                P = K @ Rt  # (3,4)
                u, vpx = uv_list[v][b, j]
                # x cross (P X)=0 → 2 rows
                A_rows.append(u * P[2,:] - P[0,:])
                A_rows.append(vpx * P[2,:] - P[1,:])

            if len(A_rows) < 4:  # <2 views
                X[b, j] = torch.tensor([float('nan')]*3, device=X.device)
                continue

            A = torch.stack(A_rows, dim=0)  # (2*n, 4)
            # SVD 최소해
            try:
                U,S,Vh = torch.linalg.svd(A)
                Xh = Vh[-1]  # (4,)
                X[b, j] = (Xh[:3] / (Xh[3] + 1e-9))
            except RuntimeError:
                X[b, j] = torch.tensor([float('nan')]*3, device=X.device)
    return X  # (B,J,3)

# metrics
def pck_2d(pred_uv, gt_uv, thr: float):
    """
    pred_uv, gt_uv: (B, V, J, 2) or (B, J, 2)
    thr: pixels
    """
    if pred_uv.dim()==4:
        err = torch.linalg.norm(pred_uv - gt_uv, dim=-1)  # (B,V,J)
        acc = (err <= thr).float().mean().item()
    else:
        err = torch.linalg.norm(pred_uv - gt_uv, dim=-1)  # (B,J)
        acc = (err <= thr).float().mean().item()
    return acc

def mpjpe_3d(pred_X, gt_X):
    """
    pred_X, gt_X: (B, J, 3)
    """
    mask = torch.isfinite(pred_X).all(dim=-1) & torch.isfinite(gt_X).all(dim=-1)  # (B,J)
    diff = (pred_X - gt_X)[mask]
    if diff.numel()==0: return float("nan")
    return torch.linalg.norm(diff, dim=-1).mean().item()  # in same units as inputs (mm 추천)

