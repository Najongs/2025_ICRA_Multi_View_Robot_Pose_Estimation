# vis.py
import os, random, time, math
import numpy as np
import torch
import cv2

# 👇 GUI 없는 백엔드 강제 (반드시 pyplot import 전에)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from franka_research3_dataset import RobotPoseDataset


def denorm(img_chw, mean, std):
    """
    img_chw: (C,H,W) torch.Tensor or np.ndarray in [0,1] normalized by (x-mean)/std
    return: (H,W,C) np.ndarray in [0,1]
    """
    if torch.is_tensor(img_chw):
        img = img_chw.detach().cpu().numpy().transpose(1,2,0)
    else:
        img = img_chw.transpose(1,2,0)
    mean = np.array(mean, dtype=np.float32)
    std  = np.array(std,  dtype=np.float32)
    img = img * std + mean      # inverse of Normalize
    return np.clip(img, 0.0, 1.0)


def vector_to_deg(vec_np):
    rad = np.arctan2(vec_np[:, 0], vec_np[:, 1])   # [sin, cos]
    deg = np.degrees(rad)
    return deg


def visualize_samples_by_group_size(groups, transform, mean, std, results_dir="results_vis", sizes=(8,6,4,2)):
    """
    그룹 크기별로 딱 1개씩만 시각화해서 results_dir에 저장.
    GUI 창은 띄우지 않음.
    """
    print("\n--- Visualizing One Sample For Each Group Size ---")
    os.makedirs(results_dir, exist_ok=True)

    # 그룹 크기 → 샘플 하나 매핑
    picked = {}
    for g in groups:
        s = len(g['views'])
        if s in sizes and s not in picked:
            picked[s] = g
        if len(picked) == len(sizes):
            break

    if not picked:
        print("No groups selected for visualization.")
        return

    for size in sorted(picked.keys(), reverse=True):
        sample_group = picked[size]
        temp = RobotPoseDataset(groups=[sample_group], transform=transform)
        image_dict, gt_heatmaps_dict, gt_angles = temp[0]
        if image_dict is None:
            print(f"Could not process sample for group size {size}. Skipping.")
            continue

        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6*num_views, 10))
        if num_views == 1:
            axes = np.expand_dims(axes, 1)  # axes[0,0], axes[1,0] 접근 가능하게

        angle_str = ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        fig.suptitle(f"Sample for Group Size: {num_views} | GT Angles: [{angle_str}]", fontsize=16)

        for j, vk in enumerate(image_dict.keys()):
            # 역정규화
            img = denorm(image_dict[vk], mean, std)
            H, W, _ = img.shape

            # GT heatmaps 합성
            gt_hm = gt_heatmaps_dict[vk]  # (J,Hm,Wm)
            heat = torch.sum(gt_hm, dim=0).numpy()
            heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)

            # 상단: 히트맵 오버레이
            ax = axes[0, j]
            ax.imshow(img, alpha=0.7)
            ax.imshow(heat, cmap='jet', alpha=0.3)
            ax.set_title(f"View: {vk} (Heatmap)")
            ax.axis('off')

            # 하단: argmax 키포인트
            pts = []
            h_map, w_map = gt_hm.shape[1:]
            for k in range(gt_hm.shape[0]):
                idx = torch.argmax(gt_hm[k]).item()
                y, x = divmod(idx, w_map)
                pts.append([x*(W/w_map), y*(H/h_map)])
            pts = np.array(pts)

            ax = axes[1, j]
            ax.imshow(img)
            ax.scatter(pts[:,0], pts[:,1], c='lime', s=40, edgecolors='black', linewidth=1)
            ax.set_title(f"View: {vk} (Keypoints)")
            ax.axis('off')

        plt.tight_layout(rect=[0,0.03,1,0.95])

        # ✅ show() 대신 저장
        out_path = os.path.join(results_dir, f"sample_group_size_{size}.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved visualization: {out_path}")


def visualize_dataset_sample(dataset, mean, std, results_dir, num_samples=1):
    os.makedirs(results_dir, exist_ok=True)
    print("\n--- Visualizing Dataset Samples ---")
    for _ in range(num_samples):
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
            img = denorm(image_dict[vk], mean, std)
            H, W, _ = img.shape
            heat = torch.sum(gt_heatmaps_dict[vk], dim=0).numpy()
            heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
            axes[j].imshow(img, alpha=0.7)
            axes[j].imshow(heat, cmap='jet', alpha=0.3)
            axes[j].set_title(f"View: {vk} (GT Heatmap)")
            axes[j].axis('off')

        plt.tight_layout(rect=[0,0.03,1,0.95])
        fn = f"gt_sample_{idx}_{int(time.time())}.png"
        out_path = os.path.join(results_dir, fn)
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"  -> Saved GT sample visualization to {out_path}")
        plt.close()


def visualize_predictions(model, dataset, device, mean, std, epoch_num, results_dir, num_samples=1):
    print(f"\n--- Visualizing Predictions for Epoch {epoch_num} ---")
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    saved_paths = []

    for _ in range(num_samples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None:
                break

        image_dict, gt_heatmaps_dict, gt_angles = sample
        with torch.no_grad():
            inp = {k: v.unsqueeze(0).to(device) for k, v in image_dict.items()}
            pred_hm_dict, pred_angles_b = model(inp)
            pred_angles = pred_angles_b[0].cpu()

        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6*num_views, 10))
        if num_views == 1:
            axes = np.expand_dims(axes, 1)

        gt_str = "GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        pred_vec = pred_angles.numpy()   # (A,2)
        pred_deg = vector_to_deg(pred_vec)
        pd_str = "Pred Angles: " + ", ".join([f"{a:.2f}" for a in pred_deg])
        fig.suptitle(f"Sample {idx} | Epoch {epoch_num}\n{gt_str}\n{pd_str}", fontsize=12)

        for j, vk in enumerate(image_dict.keys()):
            img = denorm(image_dict[vk], mean, std)
            H, W, _ = img.shape

            gt_heat = torch.sum(gt_heatmaps_dict[vk], dim=0).numpy()
            pd_heat = torch.sum(pred_hm_dict[vk][0].cpu(), dim=0).numpy()

            axes[0, j].imshow(img, alpha=0.7)
            axes[0, j].imshow(cv2.resize(gt_heat, (W, H), interpolation=cv2.INTER_LINEAR), cmap='jet', alpha=0.3)
            axes[0, j].set_title(f"View: {vk} (GT)")
            axes[0, j].axis('off')

            axes[1, j].imshow(img, alpha=0.7)
            axes[1, j].imshow(cv2.resize(pd_heat, (W, H), interpolation=cv2.INTER_LINEAR), cmap='jet', alpha=0.3)
            axes[1, j].set_title(f"View: {vk} (Pred)")
            axes[1, j].axis('off')

        plt.tight_layout(rect=[0,0,1,0.92])
        out_path = os.path.join(results_dir, f"prediction_epoch_{epoch_num}_sample_{idx}.png")
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved prediction visualization to {out_path}")
        saved_paths.append(out_path)

    return saved_paths
