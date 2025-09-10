# vis.py
import os, random, time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 통합 데이터셋
from dataset import UnifiedRobotPoseDataset

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
    print(f"\n--- Visualizing Predictions for Epoch {epoch_num} ---")
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    for i in range(num_samples):
        # None 샘플 스킵
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None:
                break

        image_dict, gt_heatmaps_dict, gt_angles = sample

        with torch.no_grad():
            inp = {k: v.unsqueeze(0).to(device) for k, v in image_dict.items()}  # per-view batch=1
            pred_hm_dict, pred_angles_b = model(inp)  # pred_hm_dict[vk]: (B,J,H,W)
            pred_angles = pred_angles_b[0].cpu()      # (num_angles, 2) [sin, cos]

        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6*num_views, 10))
        if num_views == 1:
            axes = np.expand_dims(axes, 1)

        gt_str = "GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        pred_vec = pred_angles.cpu().numpy()   # (num_angles, 2)
        pred_deg = vector_to_deg(pred_vec)     # (num_angles,)
        pd_str = "Pred Angles: " + ", ".join([f"{a:.2f}" for a in pred_deg])
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
        fn = f"prediction_epoch_{epoch_num}_sample_{idx}_{int(time.time())}.png"
        path = os.path.join(results_dir, fn)
        fig.savefig(path)
        print(f"  -> Saved prediction visualization to {path}")
        plt.close(fig)
