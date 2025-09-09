# vis.py
import os, random, time, math
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from fr5_dataset import RobotPoseDataset

def vector_to_deg(vec_np):
    """
    vec_np: (num_angles, 2) numpy array [sin, cos]
    return: (num_angles,) numpy array in degrees
    """
    rad = np.arctan2(vec_np[:, 0], vec_np[:, 1])   # sin, cos 순서 주의!
    deg = np.degrees(rad)
    return deg

def visualize_samples_by_group_size(groups, transform, mean, std):
    print("\n--- Visualizing One Sample For Each Group Size ---")
    by_size = {}
    for g in groups:
        by_size.setdefault(len(g['views']), []).append(g)
    for size in sorted(by_size.keys(), reverse=True):
        sample_group = random.choice(by_size[size])
        temp = RobotPoseDataset(groups=[sample_group], transform=transform)
        image_dict, gt_heatmaps_dict, gt_angles = temp[0]
        if image_dict is None:
            print(f"Could not process sample for group size {size}. Skipping.")
            continue
        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6*num_views, 10))
        if num_views == 1: axes = np.expand_dims(axes, 1)
        angle_str = ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        fig.suptitle(f"Sample for Group Size: {num_views} | GT Angles: [{angle_str}]", fontsize=16)
        for j, vk in enumerate(image_dict.keys()):
            img = image_dict[vk].numpy().transpose(1,2,0)
            img = np.array(std)*img + np.array(mean)
            img = np.clip(img, 0, 1)
            H, W, _ = img.shape
            gt_hm = gt_heatmaps_dict[vk]
            heat = torch.sum(gt_hm, dim=0).numpy()
            heat = cv2.resize(heat, (W, H))
            # heatmap
            ax = axes[0, j]
            ax.imshow(img, alpha=0.7)
            ax.imshow(heat, cmap='jet', alpha=0.3)
            ax.set_title(f"View: {vk} (Heatmap)"); ax.axis('off')
            # keypoints
            pts = []
            h_map, w_map = gt_hm.shape[1:]
            for k in range(gt_hm.shape[0]):
                y, x = np.unravel_index(torch.argmax(gt_hm[k]).numpy(), (h_map, w_map))
                pts.append([x*(W/w_map), y*(H/h_map)])
            pts = np.array(pts)
            ax = axes[1, j]
            ax.imshow(img)
            ax.scatter(pts[:,0], pts[:,1], c='lime', s=40, edgecolors='black', linewidth=1)
            ax.set_title(f"View: {vk} (Keypoints)"); ax.axis('off')
        plt.tight_layout(rect=[0,0.03,1,0.95]); plt.show()

def visualize_dataset_sample(dataset, mean, std, results_dir, num_samples=1):
    os.makedirs(results_dir, exist_ok=True)
    print("\n--- Visualizing Dataset Samples ---")
    for _ in range(num_samples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None: break
        image_dict, gt_heatmaps_dict, gt_angles = sample
        num_views = len(image_dict)
        fig, axes = plt.subplots(1, num_views, figsize=(6*num_views, 6))
        if num_views == 1: axes = [axes]
        angle_str = ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        fig.suptitle(f"Sample Group {idx} | GT Angles: [{angle_str}]", fontsize=16)
        for j, vk in enumerate(image_dict.keys()):
            img = image_dict[vk].numpy().transpose(1,2,0)
            img = np.array(std)*img + np.array(mean)
            img = np.clip(img, 0, 1)
            H, W, _ = img.shape
            heat = torch.sum(gt_heatmaps_dict[vk], dim=0).numpy()
            heat = cv2.resize(heat, (W, H))
            axes[j].imshow(img, alpha=0.7)
            axes[j].imshow(heat, cmap='jet', alpha=0.3)
            axes[j].set_title(f"View: {vk} (GT Heatmap)")
            axes[j].axis('off')
        plt.tight_layout(rect=[0,0.03,1,0.95])
        fn = f"gt_sample_{idx}_{int(time.time())}.png"
        plt.savefig(os.path.join(results_dir, fn))
        print(f"  -> Saved GT sample visualization to {os.path.join(results_dir, fn)}")
        plt.close()

def visualize_predictions(model, dataset, device, mean, std, epoch_num, results_dir, num_samples=1):
    print(f"\n--- Visualizing Predictions for Epoch {epoch_num} ---")
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    figures = []
    for i in range(num_samples):
        while True:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            if sample[0] is not None: break
        image_dict, gt_heatmaps_dict, gt_angles = sample
        with torch.no_grad():
            inp = {k: v.unsqueeze(0).to(device) for k, v in image_dict.items()}
            pred_hm_dict, pred_angles_b = model(inp)
            pred_angles = pred_angles_b[0].cpu()
        num_views = len(image_dict)
        fig, axes = plt.subplots(2, num_views, figsize=(6*num_views, 10))
        if num_views == 1: axes = np.expand_dims(axes, 1)
        gt_str = "GT Angles: " + ", ".join([f"{a:.2f}" for a in gt_angles.numpy()])
        pred_vec = pred_angles.cpu().numpy()   # (num_angles, 2)
        pred_deg = vector_to_deg(pred_vec)     # (num_angles,)
        pd_str = "Pred Angles: " + ", ".join([f"{a:.2f}" for a in pred_deg])
        fig.suptitle(f"Sample {idx} | Epoch {epoch_num}\n{gt_str}\n{pd_str}", fontsize=12)
        for j, vk in enumerate(image_dict.keys()):
            img = image_dict[vk].numpy().transpose(1,2,0)
            img = np.array(std)*img + np.array(mean)
            img = np.clip(img, 0, 1)
            H, W, _ = img.shape
            gt_heat = torch.sum(gt_heatmaps_dict[vk], dim=0).numpy()
            pd_heat = torch.sum(pred_hm_dict[vk][0].cpu(), dim=0).numpy()
            axes[0, j].imshow(img, alpha=0.7)
            axes[0, j].imshow(cv2.resize(gt_heat, (W, H)), cmap='jet', alpha=0.3)
            axes[0, j].set_title(f"View: {vk} (GT)"); axes[0, j].axis('off')
            axes[1, j].imshow(img, alpha=0.7)
            axes[1, j].imshow(cv2.resize(pd_heat, (W, H)), cmap='jet', alpha=0.3)
            axes[1, j].set_title(f"View: {vk} (Pred)"); axes[1, j].axis('off')
        plt.tight_layout(rect=[0,0,1,0.92])
        figures.append(fig)
    for i, fig in enumerate(figures):
        fn = f"prediction_epoch_{epoch_num}_sample_{idx}_{i}.png"
        fig.savefig(os.path.join(results_dir, fn))
        print(f"  -> Saved prediction visualization to {os.path.join(results_dir, fn)}")
    return figures
