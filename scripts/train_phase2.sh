#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TXT_PATH="${TXT_PATH:-/path/to/train.txt}"
VAL_TXT_PATH="${VAL_TXT_PATH:-/path/to/val.txt}"
DATA_ROOT="${DATA_ROOT:-/path/to/raw}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${REPO_ROOT}/checkpoints/phase1/best_model.pth}"
EXPERT_JSON="${EXPERT_JSON:-${REPO_ROOT}/artifacts/expert_trajectories.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/phase2}"
DEVICE="${DEVICE:-cuda}"
UPDATES="${UPDATES:-100}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2048}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
EXPERT_BATCH_TRAJECTORIES="${EXPERT_BATCH_TRAJECTORIES:-32}"
LR="${LR:-1e-5}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_EPS="${CLIP_EPS:-0.2}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
VALUE_COEF="${VALUE_COEF:-0.5}"
EXPERT_LAMBDA="${EXPERT_LAMBDA:-1e-3}"
FH_PENALTY="${FH_PENALTY:-"-1.5"}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"
PE_DIM="${PE_DIM:-16}"
RAW_SUFFIX="${RAW_SUFFIX:-.npy}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

python "${REPO_ROOT}/train_phase2.py" \
  --txt_path "${TXT_PATH}" \
  --val_txt_path "${VAL_TXT_PATH}" \
  --data_root "${DATA_ROOT}" \
  --pretrained "${PRETRAINED_CKPT}" \
  --expert_json "${EXPERT_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --updates "${UPDATES}" \
  --rollout_steps "${ROLLOUT_STEPS}" \
  --ppo_epochs "${PPO_EPOCHS}" \
  --mini_batch_size "${MINI_BATCH_SIZE}" \
  --expert_batch_trajectories "${EXPERT_BATCH_TRAJECTORIES}" \
  --lr "${LR}" \
  --gamma "${GAMMA}" \
  --gae_lambda "${GAE_LAMBDA}" \
  --clip_eps "${CLIP_EPS}" \
  --entropy_coef "${ENTROPY_COEF}" \
  --value_coef "${VALUE_COEF}" \
  --expert_lambda "${EXPERT_LAMBDA}" \
  --fh_penalty "${FH_PENALTY}" \
  --max_env_steps "${MAX_ENV_STEPS}" \
  --patch_size "${PATCH_SIZE}" \
  --pe_dim "${PE_DIM}" \
  --raw_suffix "${RAW_SUFFIX}" \
  --num_workers "${NUM_WORKERS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --save_interval "${SAVE_INTERVAL}"
