#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TXT_PATH="${TXT_PATH:-/path/to/train.txt}"
VAL_TXT_PATH="${VAL_TXT_PATH:-/path/to/val.txt}"
DATA_ROOT="${DATA_ROOT:-/path/to/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/phase1}"
DEVICE="${DEVICE:-cuda}"
ITERATIONS="${ITERATIONS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
TEMPERATURE="${TEMPERATURE:-2.0}"
PATCH_SIZE="${PATCH_SIZE:-128}"
PE_DIM="${PE_DIM:-16}"
RAW_SUFFIX="${RAW_SUFFIX:-.npy}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"

python "${REPO_ROOT}/train_phase1.py" \
  --txt_path "${TXT_PATH}" \
  --val_txt_path "${VAL_TXT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --iterations "${ITERATIONS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --temperature "${TEMPERATURE}" \
  --patch_size "${PATCH_SIZE}" \
  --pe_dim "${PE_DIM}" \
  --raw_suffix "${RAW_SUFFIX}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}"
