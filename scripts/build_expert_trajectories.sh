#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TXT_PATH="${TXT_PATH:-/path/to/train.txt}"
DATA_ROOT="${DATA_ROOT:-/path/to/raw}"
POLICY_CKPT="${POLICY_CKPT:-${REPO_ROOT}/checkpoints/phase1/best_model.pth}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/artifacts/expert_trajectories.json}"
ORIGINAL_JSON="${ORIGINAL_JSON:-${REPO_ROOT}/artifacts/original_trajectories.json}"
DEVICE="${DEVICE:-cuda}"
MAX_STEPS="${MAX_STEPS:-4}"
M_DIVISOR="${M_DIVISOR:-5}"
ALGOS="${ALGOS:-1,2,3}"
PATCH_SIZE="${PATCH_SIZE:-128}"
PE_DIM="${PE_DIM:-16}"
RAW_SUFFIX="${RAW_SUFFIX:-.npy}"

mkdir -p "$(dirname "${OUTPUT_JSON}")"

python "${REPO_ROOT}/trajectory_builder.py" \
  --txt_path "${TXT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --policy_ckpt "${POLICY_CKPT}" \
  --output "${OUTPUT_JSON}" \
  --save_original_json "${ORIGINAL_JSON}" \
  --device "${DEVICE}" \
  --max_steps "${MAX_STEPS}" \
  --m "${M_DIVISOR}" \
  --algos "${ALGOS}" \
  --patch_size "${PATCH_SIZE}" \
  --pe_dim "${PE_DIM}" \
  --raw_suffix "${RAW_SUFFIX}"
