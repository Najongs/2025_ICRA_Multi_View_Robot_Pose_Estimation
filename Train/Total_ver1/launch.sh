#!/usr/bin/env bash
set -euo pipefail

# 사용법:
#   ./launch.sh fr5 3         # fr5만, GPU 3개
#   ./launch.sh all 3         # fr5→fr3→meca500 순차, GPU 3개
#   ./launch.sh fr3           # fr3만, 기본 GPU=3

ROBOT=${1:-fr3}        # fr5 | fr3 | meca500 | all
NP=${2:-3}             # GPU 개수

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 백본 2종: CNX 먼저 → ViT
MODELS=(
  "facebook/dinov3-convnext-base-pretrain-lvd1689m"
  "facebook/dinov3-vitb16-pretrain-lvd1689m"
)

# fusion 3종
FUSIONS=("early" "middle" "late")

# 공통 하이퍼파라미터(원하면 수정)
EPOCHS=100
BATCH=72
VAL_SPLIT=0.10

run_robot () {
  local robot="$1"
  for model_id in "${MODELS[@]}"; do
    for fusion in "${FUSIONS[@]}"; do
      echo "=== Launching ${robot} | ${model_id} | fusion=${fusion} | GPUs=${NP} ==="
      torchrun --nproc_per_node="${NP}" main2.py \
        --robot "${robot}" \
        --model-id "${model_id}" \
        --fusion "${fusion}" \
        --epochs "${EPOCHS}" \
        --batch "${BATCH}" \
        --val-split "${VAL_SPLIT}" \
        --do-grid \
        --wandb
      echo
    done
  done
}

if [[ "${ROBOT}" == "all" ]]; then
  for rb in fr5 fr3 meca500; do
    run_robot "${rb}"
  done
else
  run_robot "${ROBOT}"
fi
