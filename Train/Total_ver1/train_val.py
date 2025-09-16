# train_val.py
import torch
import torch.nn.functional as F
from torch.amp import autocast

def _stack_by_view_keys(tensor_dict, keys):
    """dict(view_key -> (B,...)) -> (B,V,...)  (keys 순서를 따름)"""
    return torch.stack([tensor_dict[k] for k in keys], dim=1)

def _gt_valid_mask(gt_hm):
    """
    gt_hm: (B,V,J,H,W)
    return valid_mask: (B,V,1,1,1)  # True(=1)면 유효
    """
    with torch.no_grad():
        valid = (gt_hm.abs().sum(dim=(2,3,4)) > 0)  # (B,V)
        return valid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()

def _kpt_l2px_from_softargmax(pred_heat, gt_heat):
    """
    pred_heat: (B,V,J,H,W)
    gt_heat:   (B,V,J,H,W)
    return mean L2 distance on heatmap grid (float)
    """
    B,V,J,H,W = pred_heat.shape
    # Soft-argmax (간단 버전)
    # (모델 안에도 softarg 모듈이 있지만, 여기선 독립적으로 수행)
    ph = pred_heat.view(B*V, J, H*W)
    ph = torch.softmax(ph, dim=-1)
    u = torch.linspace(0, W-1, W, device=pred_heat.device)
    v = torch.linspace(0, H-1, H, device=pred_heat.device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")  # (H,W)
    uu = uu.reshape(-1); vv = vv.reshape(-1)
    pu = (ph * uu).sum(dim=-1)  # (B*V,J)
    pv = (ph * vv).sum(dim=-1)

    # GT argmax
    gh = gt_heat.view(B*V, J, H, W)
    gt_idx = gh.view(B*V, J, -1).argmax(dim=-1)             # (B*V,J)
    gvy = (gt_idx // W).float()
    gux = (gt_idx %  W).float()

    du = (pu - gux); dv = (pv - gvy)
    d  = torch.sqrt(du*du + dv*dv)                          # (B*V,J)
    return d.mean().item()

import os

def _bjhw_or_bvjhw(t, name):
    """(B,V,J,H,W) or (B,J,H,W)를 강제 점검. 기타 형태면 에러."""
    if t.ndim == 5:
        B,V,J,H,W = t.shape
        # J/H/W가 헷갈리는 꼬임 방지를 위해 H/W는 16~1024, J는 보통 6~32 범위라 가정
        # 여기서는 (B,V,J,H,W)라고 가정하고 assert로 고정
        assert H >= 8 and W >= 8, f"[{name}] Expect H/W as spatial, got {t.shape}"
        return t  # (B,V,J,H,W)
    elif t.ndim == 4:
        B,J,H,W = t.shape
        assert H >= 8 and W >= 8, f"[{name}] Expect (B,J,H,W), got {t.shape}"
        return t.unsqueeze(1)  # (B,1,J,H,W)로 승격
    else:
        raise AssertionError(f"[{name}] Unexpected shape {t.shape} (ndim={t.ndim})")

def _align_joints(pred_hm, gt_hm, strict=True):
    """
    pred_hm, gt_hm: (B,V,J,H,W)
    strict=True면 J 다르면 바로 Assert. False면 공통 min(J)로 잘라 임시 언블록.
    """
    Bp,Vp,Jp,Hp,Wp = pred_hm.shape
    Bg,Vg,Jg,Hg,Wg = gt_hm.shape
    assert (Bp==Bg) and (Vp==Vg) and (Hp==Hg) and (Wp==Wg), \
        f"[align] Shape mismatch pred={pred_hm.shape} gt={gt_hm.shape}"

    if Jp == Jg:
        return pred_hm, gt_hm

    if strict:
        raise AssertionError(f"[align] NUM_JOINTS mismatch: pred={Jp}, gt={Jg}")

    J = min(Jp, Jg)
    if Jp != J: pred_hm = pred_hm[:, :, :J]
    if Jg != J: gt_hm   = gt_hm[:, :, :J]
    return pred_hm, gt_hm


def train_one_epoch(model, loader, optimizers, criteria, device,
                    loss_weight_kpt, epoch_idx, param_sets, scalers):
    """
    optimizers: {'kpt': opt, 'ang': opt} 또는 단일 optimizer
    scalers:    {'kpt': scaler, 'ang': scaler} 또는 단일 scaler
    """
    model.train()
    use_dict_opt = isinstance(optimizers, dict)
    opt_kpt = optimizers['kpt'] if use_dict_opt else optimizers
    opt_ang = optimizers.get('ang', None) if use_dict_opt else None

    scaler = scalers['kpt'] if isinstance(scalers, dict) else scalers

    lambda_fk = float(criteria.get('lambda_fk', 0.0)) if isinstance(criteria, dict) else 0.0

    running_kpt = 0.0
    running_ang = 0.0
    n_batches   = 0

    for batch in loader:
        if batch is None:
            continue
        image_dict, gt_heatmaps_dict, gt_angles = batch
        if image_dict is None:
            continue

        # 공통 키 순서 확보
        view_keys = list(image_dict.keys())

        # to(device)
        images = {k: image_dict[k].to(device, non_blocking=True) for k in view_keys}
        gt_hm  = {k: gt_heatmaps_dict[k].to(device, non_blocking=True) for k in view_keys}

        gt_hm_t = _stack_by_view_keys(gt_hm, view_keys)  # (B,V,J,H,W)
        valid_mask = _gt_valid_mask(gt_hm_t)             # (B,V,1,1,1)

        if use_dict_opt:
            opt_kpt.zero_grad(set_to_none=True)
            if opt_ang is not None:
                opt_ang.zero_grad(set_to_none=True)
        else:
            optimizers.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            # 모델 호출: dict를 그대로 넣으면 내부에서 (B,V,...)로 스택해서 처리
            out = model(images)  # {"heatmaps": (B,V,J,H,W), ...}
            pred_hm = out["heatmaps"]

            # === 규격화 & 검증 ===
            pred_hm = _bjhw_or_bvjhw(pred_hm, "pred_hm")   # (B,V,J,H,W)
            gt_hm_t = _bjhw_or_bvjhw(gt_hm_t, "gt_hm_t")   # (B,V,J,H,W)

            # rank 포함 디버그(1회만 찍고 싶으면 조건 달아라)
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[DEBUG] pred {tuple(pred_hm.shape)} | gt {tuple(gt_hm_t.shape)}")

            # 관절 채널 정렬: 개발 중엔 strict=True로 조기 실패 유도
            # 일단 ‘돌려보기’가 필요하면 strict=False로 바꿔 임시 언블록
            strict_align = True
            try:
                pred_hm, gt_hm_t = _align_joints(pred_hm, gt_hm_t, strict=strict_align)
            except AssertionError as e:
                # 임시 언블록 경로 (원인 파악용 로그 남기고 클램프)
                print(f"[WARN] {e} → fallback to min(J) clamp")
                pred_hm, gt_hm_t = _align_joints(pred_hm, gt_hm_t, strict=False)

            # === KPT loss ===
            diff = (pred_hm - gt_hm_t) ** 2

            diff = diff * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)
            loss_kpt = diff.sum() / denom

            # Aux/triangulation loss (있으면)
            loss_aux = out.get("loss_aux", None)
            if loss_aux is not None:
                loss_total = loss_weight_kpt * loss_kpt + lambda_fk * loss_aux
            else:
                loss_total = loss_weight_kpt * loss_kpt

        # backward + step (하나의 scaler로 두 optimizer step)
        scaler.scale(loss_total).backward()
        if use_dict_opt:
            scaler.step(opt_kpt)
            if opt_ang is not None:
                scaler.step(opt_ang)
        else:
            scaler.step(optimizers)
        scaler.update()

        running_kpt += float(loss_kpt.detach().cpu().item())
        running_ang += float(loss_aux.detach().cpu().item()) if (loss_aux is not None) else 0.0
        n_batches   += 1

    if n_batches == 0:
        return 0.0, 0.0

    return running_kpt / n_batches, running_ang / n_batches


@torch.no_grad()
def evaluate(model, loader, criteria, device, loss_weight_kpt, epoch_idx, amp_enabled=True):
    model.eval()
    lambda_fk = float(criteria.get('lambda_fk', 0.0)) if isinstance(criteria, dict) else 0.0

    tot_loss = 0.0
    tot_kpt  = 0.0
    tot_ang  = 0.0
    tot_px   = 0.0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue
        image_dict, gt_heatmaps_dict, gt_angles = batch
        if image_dict is None:
            continue

        view_keys = list(image_dict.keys())
        images = {k: image_dict[k].to(device, non_blocking=True) for k in view_keys}
        gt_hm  = {k: gt_heatmaps_dict[k].to(device, non_blocking=True) for k in view_keys}

        gt_hm_t = _stack_by_view_keys(gt_hm, view_keys)  # (B,V,J,H,W)
        valid_mask = _gt_valid_mask(gt_hm_t)

        with autocast("cuda", enabled=amp_enabled and torch.cuda.is_available()):
            out = model(images)
            pred_hm = out["heatmaps"]

            diff = (pred_hm - gt_hm_t) ** 2
            diff = diff * valid_mask
            denom = valid_mask.sum().clamp_min(1.0)
            loss_kpt = diff.sum() / denom

            loss_aux = out.get("loss_aux", None)
            loss_total = loss_weight_kpt * loss_kpt + (lambda_fk * loss_aux if loss_aux is not None else 0.0)

            # 픽셀 L2 (heatmap grid 기준)
            l2px = _kpt_l2px_from_softargmax(pred_hm, gt_hm_t)

        tot_loss += float(loss_total.detach().cpu().item())
        tot_kpt  += float(loss_kpt.detach().cpu().item())
        tot_ang  += float(loss_aux.detach().cpu().item()) if (loss_aux is not None) else 0.0
        tot_px   += float(l2px)
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    val_loss   = tot_loss / n_batches
    val_kpt    = tot_kpt  / n_batches
    val_ang    = tot_ang  / n_batches
    val_ang_mae = 0.0   # 각도 회귀를 안 쓰는 구조라면 0으로 둠 (필요 시 추가)
    val_kpt_px = tot_px  / n_batches

    return val_loss, val_kpt, val_ang, val_ang_mae, val_kpt_px
