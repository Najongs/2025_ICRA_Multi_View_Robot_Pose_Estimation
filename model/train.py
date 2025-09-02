import os
import glob
import json
import numpy as np
import random
import time
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# 다른 모듈에서 클래스와 함수 임포트
from dataset import RobotPoseDataset
from models import DINOv2PoseEstimator
from visualize import visualize_dataset_sample, visualize_predictions

# ==============================================================================
# 2. 학습 환경 설정 함수
# ==============================================================================

def setup(hyperparameters, dataset_pairs):
    """학습에 필요한 모든 구성 요소를 준비합니다."""
    print("--- Setting up the environment ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Transform 정의
    model_name = hyperparameters['model_name']
    dino_model_for_config = timm.create_model(model_name, pretrained=True)
    config = dino_model_for_config.default_cfg
    transform = transforms.Compose([
        transforms.Resize(config['input_size'][-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])
    print(f"Image transform created for input size: {config['input_size'][-2:]}")

    # 2. Dataset 및 DataLoader 준비
    full_dataset = RobotPoseDataset(pairs=dataset_pairs, transform=transform, sigma=hyperparameters['heatmap_sigma'])
    val_split = hyperparameters['val_split']
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = hyperparameters['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 3. 모델, 손실 함수, 옵티마이저, 스케줄러 준비
    model = DINOv2PoseEstimator(model_name)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
        
    crit_kpt = nn.MSEloss()
    crit_ang = nn.SmoothL1Loss(beta=1.0)

    model_to_access = model.module if isinstance(model, nn.DataParallel) else model
    optimizer_kpt = torch.optim.AdamW(model_to_access.keypoint_head.parameters(), lr=hyperparameters['lr_kpt'])
    optimizer_ang = torch.optim.AdamW(model_to_access.angle_head.parameters(), lr=hyperparameters['lr_ang'])

    num_epochs = hyperparameters['num_epochs']
    scheduler_kpt = CosineAnnealingLR(optimizer_kpt, T_max=num_epochs, eta_min=1e-6)
    scheduler_ang = CosineAnnealingLR(optimizer_ang, T_max=num_epochs, eta_min=1e-6)
    
    print(f"Model, losses, and optimizers are ready. Using device: {device}")
    
    return model, train_loader, val_loader, crit_kpt, crit_ang, optimizer_kpt, optimizer_ang, scheduler_kpt, scheduler_ang, device, config

# ==============================================================================
# 3. 학습 및 검증 루프
# ==============================================================================

def train_one_epoch(model, loader, optimizer_kpt, optimizer_ang, crit_kpt, crit_ang, device, loss_weight_kpt, epoch_num):
    model.train()
    total_loss_kpt, total_loss_ang = 0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch_num} [Train]")
    
    for images, heatmaps, angles in loop:
        images, heatmaps, angles = images.to(device), heatmaps.to(device), angles.to(device)
        pred_heatmaps, pred_angles = model(images)
        
        # Keypoint Head 업데이트
        optimizer_kpt.zero_grad()
        loss_kpt = crit_kpt(pred_heatmaps, heatmaps) * loss_weight_kpt
        loss_kpt.backward(retain_graph=True) # Angle head의 backward를 위해 그래프 유지
        optimizer_kpt.step()
        
        # Angle Head 업데이트
        optimizer_ang.zero_grad()
        loss_ang = crit_ang(pred_angles, angles)
        loss_ang.backward()
        optimizer_ang.step()
        
        total_loss_kpt += loss_kpt.item()
        total_loss_ang += loss_ang.item()
        loop.set_postfix(loss_kpt=loss_kpt.item(), loss_ang=loss_ang.item())
        
    avg_loss_kpt = total_loss_kpt / len(loader)
    avg_loss_ang = total_loss_ang / len(loader)
    return avg_loss_kpt, avg_loss_ang

def validate(model, loader, crit_kpt, crit_ang, device, loss_weight_kpt, epoch_num):
    model.eval()
    total_loss_kpt, total_loss_ang = 0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch_num} [Validate]", leave=False)
    
    with torch.no_grad():
        for images, heatmaps, angles in loop:
            images, heatmaps, angles = images.to(device), heatmaps.to(device), angles.to(device)
            pred_heatmaps, pred_angles = model(images)
            
            loss_kpt = crit_kpt(pred_heatmaps, heatmaps)
            loss_ang = crit_ang(pred_angles, angles)
            
            total_loss_kpt += loss_kpt.item()
            total_loss_ang += loss_ang.item()
            loop.set_postfix(val_kpt=loss_kpt.item(), val_ang=loss_ang.item())
            
    avg_loss_kpt = total_loss_kpt / len(loader)
    avg_loss_ang = total_loss_ang / len(loader)
    # 검증 시에는 두 손실의 가중합을 최종 성능 지표로 사용
    avg_total_loss = (avg_loss_kpt * loss_weight_kpt) + avg_loss_ang
    return avg_total_loss, avg_loss_kpt, avg_loss_ang

# ==============================================================================
# 4. 메인 실행부
# ==============================================================================

if __name__ == '__main__':
    # --- 데이터 페어링 ---
    # 이 부분은 dataset.py의 create_dataset_pairs() 함수로 분리하는 것이 더 깔끔합니다.
    # 여기서는 스크립트의 독립 실행을 위해 포함합니다.
    # ... (dataset_pairs 생성 로직) ...

    # --- 하이퍼파라미터 및 경로 설정 ---
    hyperparameters = {
        'model_name': 'vit_base_patch14_dinov2.lvd142m',
        'batch_size': 150,
        'num_epochs': 100,
        'val_split': 0.1,
        'loss_weight_kpt': 1.0, 
        'lr_kpt': 0.005,
        'lr_ang': 0.005,
        'heatmap_sigma': 2.0
    }
    CHECKPOINT_PATH = 'checkpoint.pth'
    BEST_MODEL_PATH = 'best_pose_estimator_model.pth'

    # --- W&B 초기화 ---
    run = wandb.init(project="robot-pose-estimation", config=hyperparameters)
    
    # --- 학습 환경 설정 ---
    model, train_loader, val_loader, crit_kpt, crit_ang, optimizer_kpt, optimizer_ang, scheduler_kpt, scheduler_ang, device, config = setup(
        hyperparameters, dataset_pairs
    )
    wandb.watch(model, log="all", log_freq=100)
    
    # --- 체크포인트 로드 ---
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Resuming training from '{CHECKPOINT_PATH}'.")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer_kpt.load_state_dict(checkpoint['optimizer_kpt_state_dict'])
        optimizer_ang.load_state_dict(checkpoint['optimizer_ang_state_dict'])
        scheduler_kpt.load_state_dict(checkpoint['scheduler_kpt_state_dict'])
        scheduler_ang.load_state_dict(checkpoint['scheduler_ang_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"   -> Resumed from epoch {start_epoch}, with best validation loss: {best_val_loss:.6f}")
    else:
        print("ℹ️ No checkpoint found. Starting training from scratch.")

    # --- 학습 전 데이터 시각화 ---
    visualize_dataset_sample(train_loader.dataset.dataset, config, num_samples=3)

    # --- 학습 시작 ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, hyperparameters['num_epochs']):
        train_loss_kpt, train_loss_ang = train_one_epoch(
            model, train_loader, optimizer_kpt, optimizer_ang, crit_kpt, crit_ang, device, 
            loss_weight_kpt=hyperparameters['loss_weight_kpt'], epoch_num=epoch+1
        )
        val_loss, val_loss_kpt, val_loss_ang = validate(
            model, val_loader, crit_kpt, crit_ang, device, 
            loss_weight_kpt=hyperparameters['loss_weight_kpt'], epoch_num=epoch+1
        )
        
        scheduler_kpt.step()
        scheduler_ang.step()

        current_lr_kpt = optimizer_kpt.param_groups[0]['lr']
        current_lr_ang = optimizer_ang.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{hyperparameters['num_epochs']} Summary -> Train [Kpt: {train_loss_kpt:.6f}, Ang: {train_loss_ang:.6f}], Val [Total: {val_loss:.6f}, Kpt: {val_loss_kpt:.6f}, Ang: {val_loss_ang:.6f}]")
        print(f"  -> LRs [Kpt: {current_lr_kpt:.6f}, Ang: {current_lr_ang:.6f}]")
        
        # W&B 로깅
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_kpt": train_loss_kpt,
            "train_loss_ang": train_loss_ang,
            "val_loss_total": val_loss,
            "val_loss_kpt": val_loss_kpt,
            "val_loss_ang": val_loss_ang,
            "lr_kpt": current_lr_kpt,
            "lr_ang": current_lr_ang
        })

        # 체크포인트 및 최고 성능 모델 저장
        state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint = {
            'epoch': epoch + 1, 'model_state_dict': state_to_save,
            'optimizer_kpt_state_dict': optimizer_kpt.state_dict(),
            'optimizer_ang_state_dict': optimizer_ang.state_dict(),
            'scheduler_kpt_state_dict': scheduler_kpt.state_dict(),
            'scheduler_ang_state_dict': scheduler_ang.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state_to_save, BEST_MODEL_PATH)
            print(f"  -> 🎉 New best model saved with validation loss: {best_val_loss:.6f}")
            prediction_fig = visualize_predictions(
                model, val_loader.dataset, device, config, epoch_num=epoch + 1, num_samples=3
            )
            wandb.log({"validation_predictions": wandb.Image(prediction_fig)})
            plt.close(prediction_fig)

    print("\n--- Training Finished ---")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    
    run.finish()