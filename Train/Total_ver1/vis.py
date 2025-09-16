# vis.py
import os, random, time, math

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 통합 데이터셋
from dataset import UnifiedRobotPoseDataset

def _to_uint8(img_t: torch.Tensor, mean=None, std=None):
    """
    (3,H,W) tensor -> (H,W,3) uint8
    mean/std가 있으면 역정규화, 없으면 [0,1] 가정
    """
    img = img_t.detach().cpu().float()
    if mean is not None and std is not None:
        m = torch.tensor(mean, dtype=img.dtype, device=img.device).view(3,1,1)
        s = torch.tensor(std,  dtype=img.dtype, device=img.device).view(3,1,1)
        img = img * s + m
    img = img.clamp(0,1)
    img = (img * 255.0).round().to(torch.uint8)
    return img.permute(1,2,0).numpy()  # (H,W,3)

def _peaks_from_heatmaps(hm: torch.Tensor):
    """
    hm: (J, Ht, Wt) -> peaks_xy: (J,2)  (x=W, y=H) in heatmap coords
    """
    J, Ht, Wt = hm.shape
    hm_np = hm.detach().cpu().numpy()
    flat_idx = hm_np.reshape(J, -1).argmax(axis=1)             # (J,)
    ys = (flat_idx // Wt).astype(np.float32)                   # (J,)
    xs = (flat_idx %  Wt).astype(np.float32)
    peaks = np.stack([xs, ys], axis=1)                         # (J,2)
    return peaks

def _scale_xy(points_xy, from_size, to_size):
    Wf, Hf = from_size
    Wt, Ht = to_size
    out = np.empty_like(points_xy, dtype=np.float32)
    out[:,0] = points_xy[:,0] * (Wt / float(Wf))
    out[:,1] = points_xy[:,1] * (Ht / float(Hf))
    return out

def draw_points(img_rgb: np.ndarray, pts_xy: np.ndarray, radius=3):
    """
    img_rgb: (H,W,3) uint8, pts_xy: (J,2) in pixel coords of img
    """
    out = img_rgb.copy()
    H, W = out.shape[:2]
    for j, (x, y) in enumerate(pts_xy):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(out, (xi, yi), radius, (0,255,0), -1)
            cv2.putText(out, str(j), (xi+4, yi-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1, cv2.LINE_AA)
    return out

def visualize_dataset_samples(dataset,
                              save_dir: str,
                              num_samples: int = 8,
                              mean=None, std=None,
                              input_size: int = 224):
    """
    dataset[idx] -> (image_dict, heatmaps_dict, gt_angles)
    각 샘플에서 뷰별로:
      - 이미지(입력과 동일한 224x224) 추출
      - 히트맵 argmax로 키포인트 좌표 추출 (Ht,Wt)
      - (Ht,Wt) -> (IN,IN)로 스케일 → 이미지에 점 찍기
    결과를 PNG로 저장
    """
    os.makedirs(save_dir, exist_ok=True)
    Ht, Wt = None, None

    printed_shape_info = False

    cnt = 0
    for idx in range(len(dataset)):
        if cnt >= num_samples: break
        item = dataset[idx]
        if item is None or item[0] is None:  # skip None
            continue
        image_dict, hm_dict, gt_angles = item

        # 정렬된 뷰 순서로 보기 좋게
        keys = sorted(image_dict.keys())

        # 뷰가 많으면 한 장당 일정 개수씩 그리드로 저장
        cols = min(4, max(1, len(keys)))
        rows = math.ceil(len(keys) / cols)

        fig_w = 4 * cols
        fig_h = 4 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        for vi, k in enumerate(keys):
            r, c = divmod(vi, cols)
            ax = axes[r, c]

            img_t = image_dict[k]            # (3, IN, IN)
            hm_t  = hm_dict[k]               # (J, Ht, Wt)

            if not printed_shape_info:
                print(f"[GT-VIZ] sample idx={idx}, view={k}, img={tuple(img_t.shape)}, hm={tuple(hm_t.shape)}")
                printed_shape_info = True

            img_u8 = _to_uint8(img_t, mean, std)  # (IN,IN,3) uint8
            J, Ht, Wt = hm_t.shape
            peaks_hm = _peaks_from_heatmaps(hm_t) # (J,2) in heatmap coords
            peaks_in = _scale_xy(peaks_hm, from_size=(Wt, Ht), to_size=(img_u8.shape[1], img_u8.shape[0]))

            overlay = draw_points(img_u8, peaks_in, radius=3)

            ax.imshow(overlay)
            ax.set_title(f"{k} | J={J}")
            ax.axis('off')

        # 빈 서브플롯 비활성화
        for vi in range(len(keys), rows*cols):
            r, c = divmod(vi, cols)
            axes[r, c].axis('off')

        out_path = os.path.join(save_dir, f"gt_check_idx{idx}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        cnt += 1

    print(f"[GT-VIZ] Saved {cnt} sample visualizations to: {save_dir}")


def _run_and_unpack(model, inp):
    """
    모델 출력이 (pred_hm_dict, pred_angles) 튜플이든, dict이든 모두 지원.
    dict일 때는:
      - heatmaps: (B,V,J,H,W) 또는 dict(view_key -> (B,J,H,W))
      - angles/coords_3d가 없으면 None 반환
    """
    out = model(inp)

    # 구(舊) 방식: (pred_hm_dict, pred_angles)
    if isinstance(out, tuple) and len(out) == 2:
        return out

    # 신(新) 방식: dict
    if isinstance(out, dict):
        # heatmaps 꺼내기
        hm = out.get("heatmaps", None)
        if hm is None:
            raise ValueError("model output dict has no 'heatmaps' key")

        # dict-of-views면 그대로
        if isinstance(hm, dict):
            pred_hm_dict = hm
        else:
            # 텐서면 (B,V,J,H,W) 또는 (B,J,H,W)
            if hm.dim() == 5:
                B, V, J, H, W = hm.shape
                keys = out.get("view_keys", [f"v{i}" for i in range(V)])
                pred_hm_dict = { keys[i]: hm[:, i].contiguous() for i in range(V) }
            elif hm.dim() == 4:
                pred_hm_dict = { "v0": hm.contiguous() }
            else:
                raise ValueError(f"unexpected heatmaps shape: {tuple(hm.shape)}")

        # 각도/3D 좌표 후보 (없으면 None)
        pred_angles_b = out.get("angles", None)
        if pred_angles_b is None:
            pred_angles_b = out.get("coords_3d", None)

        return pred_hm_dict, pred_angles_b

    raise ValueError(f"Unexpected model output type: {type(out)}")

# ---------------------------
# 기본 유틸
# ---------------------------
def vector_to_deg(vec_np: np.ndarray) -> np.ndarray:
    """
    vec_np: (num_angles, 2) numpy array [sin, cos]
    return: (num_angles,) numpy array in degrees
    """
    rad = np.arctan2(vec_np[:, 0], vec_np[:, 1])   # sin, cos 순서 주의!
    deg = np.degrees(rad)
    return deg

def _denorm_img(img_chw: torch.Tensor, mean, std) -> np.ndarray:
    """
    img_chw: (3,H,W) float [0..1] (이미 transform으로 [0,1] 정규화/표준화가 들어갔다면 mean/std로 역정규화)
    mean, std: 시각화용 역정규화 파라미터(list/tuple/np)
    return: (H,W,3) float [0..1]
    """
    img = img_chw.numpy().transpose(1,2,0)
    img = np.array(std) * img + np.array(mean)
    img = np.clip(img, 0, 1)
    return img

def _sum_heat(hm: torch.Tensor) -> np.ndarray:
    """
    hm: (J,H,W) torch -> (H,W) np
    """
    return torch.sum(hm, dim=0).cpu().numpy()

def _extract_kpts_from_heatmap(hm: torch.Tensor, out_wh) -> np.ndarray:
    """
    hm: (J,Hm,Wm) torch
    out_wh: (W_out, H_out)
    returns: (J,2) np float32 in (x,y) on output size
    """
    Hm, Wm = hm.shape[1:]
    H_out, W_out = out_wh[1], out_wh[0]
    kpts = []
    for k in range(hm.shape[0]):
        argmax = torch.argmax(hm[k]).item()
        y, x = divmod(argmax, Wm)
        kpts.append([x * (W_out / Wm), y * (H_out / Hm)])
    return np.array(kpts, dtype=np.float32)

# ---------------------------
# 1) 그룹 사이즈별 샘플 시각화
# ---------------------------
def visualize_samples_by_group_size(dataset_type: str,
                                    groups_or_pairs,
                                    transform,
                                    mean, std,
                                    heatmap_size=(128,128),
                                    sigma=5.0,
                                    input_size=224,
                                    robot_fk_unit=None):
    """
    dataset_type: 'fr3' | 'fr5' | 'meca500'
    groups_or_pairs: build_items_from_csv(...) 결과 리스트(멀티뷰 그룹 or 싱글 페어 혼재 가능)
    transform, mean, std: 시각화용
    """
    print("\n--- Visualizing One Sample For Each Group Size ---")
    # 멀티뷰 그룹만 묶고, 싱글 페어는 '1'로 취급
    by_size = {}
    for it in groups_or_pairs:
        n = len(it["views"]) if "views" in it else 1
        by_size.setdefault(n, []).append(it)

    for size in sorted(by_size.keys(), reverse=True):
        sample_item = random.choice(by_size[size])
        temp = UnifiedRobotPoseDataset(
            dataset_type=dataset_type,
            items=[sample_item],
            transform=transform,
            heatmap_size=heatmap_size,
            sigma=sigma,
            input_size=input_size,
            robot=dataset_type,                 # 로봇 FK는 기본 dataset_type로
            robot_fk_unit=robot_fk_unit,        # None이면 스펙 default 사용
        )
        image_dict, gt_heatmaps_dict, gt_angles = temp[0]
        if image_dict is None:
            print(f"Could not process sample for group size {size}. Skipping.")
            continue

        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6*num_views, 10))
        if num_views == 1:
            axes = np.expand_dims(axes, 1)

        angle_str = ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        fig.suptitle(f"Sample for Group Size: {num_views} | GT Angles: [{angle_str}]", fontsize=16)

        for j, vk in enumerate(image_dict.keys()):
            # 역정규화된 이미지
            img = _denorm_img(image_dict[vk], mean, std)
            H, W, _ = img.shape

            # heatmap overlay
            gt_hm = gt_heatmaps_dict[vk]
            heat = _sum_heat(gt_hm)
            heat = cv2.resize(heat, (W, H))

            ax = axes[0, j]
            ax.imshow(img, alpha=0.7)
            ax.imshow(heat, cmap='jet', alpha=0.3)
            ax.set_title(f"View: {vk} (Heatmap)"); ax.axis('off')

            # keypoints overlay
            pts = _extract_kpts_from_heatmap(gt_hm, out_wh=(W, H))
            ax = axes[1, j]
            ax.imshow(img)
            ax.scatter(pts[:,0], pts[:,1], c='lime', s=40, edgecolors='black', linewidth=1)
            ax.set_title(f"View: {vk} (Keypoints)"); ax.axis('off')

        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.show()

# ---------------------------
# 2) 데이터셋에서 임의 샘플 시각화 & 저장
# ---------------------------
def visualize_dataset_sample(dataset,
                             mean, std,
                             results_dir,
                             num_samples=1):
    os.makedirs(results_dir, exist_ok=True)
    print("\n--- Visualizing Dataset Samples ---")
    for _ in range(num_samples):
        # None 샘플 스킵
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None:
                break

        image_dict, gt_heatmaps_dict, gt_angles = sample
        num_views = len(image_dict)
        fig, axes = plt.subplots(1, num_views, figsize=(6*num_views, 6))
        if num_views == 1:
            axes = [axes]

        angle_str = ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        fig.suptitle(f"Sample Group {idx} | GT Angles: [{angle_str}]", fontsize=16)

        for j, vk in enumerate(image_dict.keys()):
            img = _denorm_img(image_dict[vk], mean, std)
            H, W, _ = img.shape
            heat = _sum_heat(gt_heatmaps_dict[vk])
            heat = cv2.resize(heat, (W, H))
            axes[j].imshow(img, alpha=0.7)
            axes[j].imshow(heat, cmap='jet', alpha=0.3)
            axes[j].set_title(f"View: {vk} (GT Heatmap)")
            axes[j].axis('off')

        plt.tight_layout(rect=[0,0.03,1,0.95])
        fn = f"gt_sample_{idx}_{int(time.time())}.png"
        path = os.path.join(results_dir, fn)
        plt.savefig(path)
        print(f"  -> Saved GT sample visualization to {path}")
        plt.close()

# ---------------------------
# 3) 예측 결과 시각화
# ---------------------------
def visualize_predictions(model,
                          dataset,
                          device,
                          mean, std,
                          epoch_num,
                          results_dir,
                          num_samples=1):
    """
    - 모델 출력이 dict이고 angles가 없을 수 있음 → 각도 섹션은 N/A 처리
    - DDP(model.module) / single model 모두 지원
    - 저장하면서 fig 객체를 반환하여 main.py에서 wandb 로깅 가능
    """
    print(f"\n--- Visualizing Predictions for Epoch {epoch_num} ---")
    os.makedirs(results_dir, exist_ok=True)

    # DDP 안전 호출
    m = model.module if hasattr(model, "module") else model
    m.eval()

    figs = []
    for _ in range(num_samples):
        # None 샘플 스킵
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None:
                break

        image_dict, gt_heatmaps_dict, gt_angles = sample

        with torch.no_grad():
            # per-view batch=1
            inp = {k: v.unsqueeze(0).to(device) for k, v in image_dict.items()}
            pred_hm_dict, pred_extra = _run_and_unpack(m, inp)  # pred_hm_dict[vk]: (B,J,H,W)

            # 각도 텐서가 "있고 (B, A, 2)" 형태일 때만 표시
            pred_angles = None
            if isinstance(pred_extra, torch.Tensor):
                # 보통 (B, num_angles, 2) 이어야 함
                t0 = pred_extra[0] if pred_extra.dim() >= 2 else None
                if t0 is not None and t0.dim() == 2 and t0.shape[-1] == 2:
                    pred_angles = t0.detach().cpu()

        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6*num_views, 10))
        if num_views == 1:
            axes = np.expand_dims(axes, 1)

        # 제목(각도 없으면 N/A)
        gt_str = "GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        if pred_angles is not None:
            pred_vec = pred_angles.numpy()            # (num_angles, 2) [sin, cos]
            pred_deg = vector_to_deg(pred_vec)        # (num_angles,)
            pd_str = "Pred Angles: " + ", ".join([f"{a:.2f}" for a in pred_deg])
        else:
            pd_str = "Pred Angles: N/A"
        fig.suptitle(f"Sample {idx} | Epoch {epoch_num}\n{gt_str}\n{pd_str}", fontsize=12)

        for j, vk in enumerate(image_dict.keys()):
            img = _denorm_img(image_dict[vk], mean, std)
            H, W, _ = img.shape

            gt_heat = _sum_heat(gt_heatmaps_dict[vk])
            pd_heat = _sum_heat(pred_hm_dict[vk][0].cpu())

            axes[0, j].imshow(img, alpha=0.7)
            axes[0, j].imshow(cv2.resize(gt_heat, (W, H)), cmap='jet', alpha=0.3)
            axes[0, j].set_title(f"View: {vk} (GT)"); axes[0, j].axis('off')

            axes[1, j].imshow(img, alpha=0.7)
            axes[1, j].imshow(cv2.resize(pd_heat, (W, H)), cmap='jet', alpha=0.3)
            axes[1, j].set_title(f"View: {vk} (Pred)"); axes[1, j].axis('off')

        plt.tight_layout(rect=[0,0,1,0.92])

        # 저장 + figs 반환(메인에서 wandb 로깅/close)
        fn = f"prediction_epoch_{epoch_num}_sample_{idx}_{int(time.time())}.png"
        path = os.path.join(results_dir, fn)
        fig.savefig(path)
        print(f"  -> Saved prediction visualization to {path}")

        figs.append(fig)

    return figs
