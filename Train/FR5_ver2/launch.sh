#!/usr/bin/env bash
set -e

# 사용법:
# 1) 단일 로봇
#   ./launch.sh fr5
# 2) 전부 순차 실행
#   ./launch.sh all

ROBOT=${1:-fr5}   # 기본 fr5
NP=3              # GPU 3개

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

run_one () {
  local robot="$1"
  echo "=== Launching training for ${robot} with ${NP} GPUs ==="
  torchrun --nproc_per_node=${NP} main.py --robot ${robot} --epochs 100 --batch 72 --val-split 0.1 --do-grid --wandb
}

if [ "${ROBOT}" == "all" ]; then
  run_one fr5
  run_one fr3
  run_one meca500
else
  run_one "${ROBOT}"
fi
