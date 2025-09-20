#!/bin/bash

echo "Starting Ablation Study in SEQUENTIAL mode with delays..."

# # --- 실험 1: dino_only ---
# echo "[$(date)] --- Running experiment: dino_only ---"
# torchrun --nproc_per_node=4 3nd_Single_view_train_CNN_DINO_ablation.py --ablation_mode dino_only
# echo "[$(date)] --- Finished: dino_only."

# # --- 5초 대기 ---
# echo "Waiting for 5 seconds before the next run..."
# sleep 5

# # --- 실험 2: cnn_only ---
# echo "[$(date)] --- Running experiment: cnn_only ---"
# torchrun --nproc_per_node=4 3nd_Single_view_train_CNN_DINO_ablation.py --ablation_mode cnn_only
# echo "[$(date)] --- Finished: cnn_only."

# # --- 5초 대기 ---
# echo "Waiting for 5 seconds before the next run..."
# sleep 5

# # --- 실험 3: combined ---
# echo "[$(date)] --- Running experiment: combined ---"
# torchrun --nproc_per_node=4 3nd_Single_view_train_CNN_DINO_ablation.py --ablation_mode combined
# echo "[$(date)] --- Finished: combined."

# # --- 5초 대기 ---
# echo "Waiting for 5 seconds before the next run..."
# sleep 5

# --- 실험 4: dino_conv_only ---
# echo "[$(date)] --- Running experiment: dino_conv_only ---"
# torchrun --nproc_per_node=4 6th_Single_view_train_ablation_SigLIP.py --ablation_mode dino_conv_only
# echo "[$(date)] --- Finished: dino_conv_only."

# # --- 5초 대기 ---
# echo "Waiting for 5 seconds before the next run..."
# sleep 5

# # --- 실험 5: combined_conv ---
# echo "[$(date)] --- Running experiment: combined_conv ---"
# torchrun --nproc_per_node=4 6th_Single_view_train_ablation_SigLIP.py --ablation_mode combined_conv
# echo "[$(date)] --- Finished: combined_conv."

# # --- 5초 대기 ---
# echo "Waiting for 5 seconds before the next run..."
# sleep 5

# # --- 실험 6: siglip_only ---
# echo "[$(date)] --- Running experiment: siglip_only ---"
# torchrun --nproc_per_node=4 6th_Single_view_train_ablation_SigLIP.py --ablation_mode siglip_only
# echo "[$(date)] --- Finished: siglip_only."

# # --- 5초 대기 ---
# echo "Waiting for 5 seconds before the next run..."
# sleep 5

# # --- 실험 7: siglip_combined ---
# echo "[$(date)] --- Running experiment: siglip_combined ---"
# torchrun --nproc_per_node=4 6th_Single_view_train_ablation_SigLIP.py --ablation_mode siglip_combined
# echo "[$(date)] --- Finished: siglip_combined."

# # --- 5초 대기 ---
# echo "Waiting for 5 seconds before the next run..."
# sleep 5

# --- 실험 8: siglip2_only ---
echo "[$(date)] --- Running experiment: siglip2_only ---"
torchrun --nproc_per_node=4 6th_Single_view_train_ablation_SigLIP.py --ablation_mode siglip2_only
echo "[$(date)] --- Finished: siglip2_only."

# --- 5초 대기 ---
echo "Waiting for 5 seconds before the next run..."
sleep 5

# --- 실험 9: siglip2_combined ---
echo "[$(date)] --- Running experiment: siglip2_combined ---"
torchrun --nproc_per_node=4 6th_Single_view_train_ablation_SigLIP.py --ablation_mode siglip2_combined
echo "[$(date)] --- Finished: siglip2_combined."

# --- 5초 대기 ---
echo "Waiting for 5 seconds before the next run..."
sleep 5


# 순차 실행에서는 백그라운드 작업이 없으므로 'wait' 명령어는 필요 없습니다.

echo "[$(date)] All ablation experiments have finished sequentially."