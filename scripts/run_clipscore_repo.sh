#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DATA_DIR="${TEST_DATA_DIR:-$ROOT_DIR/test_data}"
DESCRIPTION_FILE="${DESCRIPTION_FILE:-$TEST_DATA_DIR/description.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/metrics_rerun}"
MAX_FRAMES="${MAX_FRAMES:-73}"
FRAME_INTERVAL="${FRAME_INTERVAL:-1}"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-base-patch32}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

DEVICE="${DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
  if python3 - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

mkdir -p "$OUTPUT_DIR"

echo "ROOT_DIR=$ROOT_DIR"
echo "TEST_DATA_DIR=$TEST_DATA_DIR"
echo "DESCRIPTION_FILE=$DESCRIPTION_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DEVICE=$DEVICE"
echo "MAX_FRAMES=$MAX_FRAMES"
echo "FRAME_INTERVAL=$FRAME_INTERVAL"

CMD=(
  python3 "$ROOT_DIR/code/video_clip_evaluator.py"
  --test_data_dir "$TEST_DATA_DIR"
  --description_file "$DESCRIPTION_FILE"
  --max_frames "$MAX_FRAMES"
  --frame_interval "$FRAME_INTERVAL"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --model_name "$MODEL_NAME"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  CMD+=(--local_files_only)
fi

MPLCONFIGDIR=/tmp/matplotlib "${CMD[@]}"
