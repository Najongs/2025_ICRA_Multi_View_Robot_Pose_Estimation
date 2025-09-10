# train_val.py
import os, time, math
from typing import Dict
import torch
from torch.utils.data import DataLoader

from dataset import (
    UnifiedRobotPoseDataset,
    build_unified_dataset_from_csv,
    collate_skip_none,
)
from loss_and_metrics import MultiTaskPoseLoss, metrics_from_batch

def move_batch_to_device(images_dict, heatmaps_dict, angles, device):
    images_dict = {k: v.to(device, non_blocking=True) for k, v in images_dict.items()}
    heatmaps_dict = {k: v.to(device, non_blocking=True) for k, v in heatmaps_dict.items()}
    angles = angles.to(device, non_blocking=True)
    return images_dict, heatmaps_dict, angles

def train_one_epoch(model,
                    loader: DataLoader,
                    optimizer,
                    criterion: MultiTaskPoseLoss,
                    device,
                    scaler: torch.cuda.amp.GradScaler = None,
                    log_interval: int = 100,
                    accum_steps: int = 1,
                    angle_unit: str = "deg"):
    model.train()
    running = {"L_total": 0.0, "L_hm": 0.0, "L_ang": 0.0, "L_unit": 0.0}
    n_seen = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if batch is None:  # collate가 다 걸러낸 경우
            continue
        images_list, heatmaps_list, angles_b = batch
        # batch_size=1을 가정 (ragged-view라 안전)
        images_dict = images_list[0]
        heatmaps_dict = heatmaps_list[0]
        angles = angles_b[0].unsqueeze(0)  # (1,A)

        images_dict, heatmaps_dict, angles = move_batch_to_device(images_dict, heatmaps_dict, angles, device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            pred_hm_dict, pred_angles_b = model(images_dict)  # pred_angles_b: (B,A,2)
            # 안전장치: B=1 강제
            if pred_angles_b.dim() == 2:
                pred_angles_b = pred_angles_b.unsqueeze(0)

            loss, parts = criterion(pred_hm_dict, heatmaps_dict, pred_angles_b, angles)
            # grad accumulation
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if ((step + 1) % accum_steps) == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # log
        for k in running.keys():
            running[k] += parts[k].item()
        n_seen += 1

        if (step + 1) % log_interval == 0:
            avg = {k: v / max(n_seen, 1) for k, v in running.items()}
            print(f"[train] step {step+1}/{len(loader)} "
                  f"L_total={avg['L_total']:.4f} | hm={avg['L_hm']:.4f} | ang={avg['L_ang']:.4f} | unit={avg['L_unit']:.4f}")

    avg = {k: v / max(n_seen, 1) for k, v in running.items()}
    return avg

@torch.no_grad()
def evaluate(model,
             loader: DataLoader,
             criterion: MultiTaskPoseLoss,
             device,
             angle_unit: str = "deg"):
    model.eval()
    running_loss = {"L_total": 0.0, "L_hm": 0.0, "L_ang": 0.0, "L_unit": 0.0}
    running_metrics = {"hm_mse": 0.0, "angle_err_deg": 0.0}
    n_seen = 0

    for batch in loader:
        if batch is None:
            continue
        images_list, heatmaps_list, angles_b = batch
        images_dict = images_list[0]
        heatmaps_dict = heatmaps_list[0]
        angles = angles_b[0].unsqueeze(0)

        images_dict, heatmaps_dict, angles = move_batch_to_device(images_dict, heatmaps_dict, angles, device)

        pred_hm_dict, pred_angles_b = model(images_dict)
        if pred_angles_b.dim() == 2:
            pred_angles_b = pred_angles_b.unsqueeze(0)

        loss, parts = criterion(pred_hm_dict, heatmaps_dict, pred_angles_b, angles)
        for k in running_loss.keys():
            running_loss[k] += parts[k].item()

        m = metrics_from_batch(pred_hm_dict, heatmaps_dict, pred_angles_b, angles, angle_unit=angle_unit)
        for k in running_metrics.keys():
            running_metrics[k] += m[k]
        n_seen += 1

    avg_loss = {k: v / max(n_seen, 1) for k, v in running_loss.items()}
    avg_metrics = {k: v / max(n_seen, 1) for k, v in running_metrics.items()}
    return avg_loss, avg_metrics
