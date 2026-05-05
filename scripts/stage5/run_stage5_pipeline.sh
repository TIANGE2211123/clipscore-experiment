#!/usr/bin/env bash
# Stage 5 / Stage C — DoTA (Detection of Traffic Anomaly) label-bias audit.
#
# Inputs:
#   tmp/dota/dataset/metadata_val.json   (cloned from MoonBlvd/Detection-of-Traffic-Anomaly)
#   tmp/dota_frames/<clip_id>/{000000..000119}.jpg   (extracted from gdrive split-zip mirror)
#
# Outputs:
#   outputs/stage5/manifest/dota_candidates.csv
#   outputs/stage5/videos/<clip_id>.mp4
#   outputs/stage5/frames/<clip_id>/frame_{00..07}.png
#   outputs/stage5/prompts/scenario_descriptions.json
#   outputs/stage5/clipscore/per_video.csv
#   outputs/stage5/clipscore/per_frame.json
#   outputs/stage5/per_video_pairs.csv
#   outputs/stage5/frame_level/*.csv
#   outputs/stage5/cross_dataset_comparison.csv
#   outputs/stage5/REPORT.md
#   outputs/stage5/figures/*.png
#
set -euo pipefail

ROOT="/home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace"
REPO="$ROOT/clipscore-experiment"
STAGE5="$ROOT/outputs/stage5"

META_JSON="$ROOT/tmp/dota/dataset/metadata_val.json"
FRAMES_ROOT="${FRAMES_ROOT:-$ROOT/tmp/dota_frames}"
SAMPLE_SIZE="${SAMPLE_SIZE:-10}"
SEED="${SEED:-42}"
FPS="${FPS:-10}"
PY="${PY:-python3}"

mkdir -p "$STAGE5"/{manifest,videos,frames,prompts,clipscore,figures,frame_level}

if [[ ! -f "$META_JSON" ]]; then
  echo "[stage5] ERROR: metadata not found at $META_JSON — did you clone MoonBlvd/Detection-of-Traffic-Anomaly into tmp/dota?" >&2
  exit 1
fi
if [[ ! -d "$FRAMES_ROOT" ]]; then
  echo "[stage5] ERROR: $FRAMES_ROOT does not exist — extract the DoTA gdrive split-zip bundle there first." >&2
  exit 1
fi

echo "[stage5] === Step 1: build stratified DoTA manifest ==="
MANIFEST="$STAGE5/manifest/dota_candidates.csv"
if [[ ! -f "$MANIFEST" ]]; then
  "$PY" "$REPO/scripts/stage5/build_dota_manifest.py" \
    --metadata-json "$META_JSON" \
    --frames-root "$FRAMES_ROOT" \
    --output-csv "$MANIFEST" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED"
fi

echo "[stage5] === Step 2: render clip MP4s + extract keyframes ==="
"$PY" "$REPO/scripts/stage5/render_dota_videos.py" \
  --manifest-csv "$MANIFEST" \
  --frames-root "$FRAMES_ROOT" \
  --videos-dir "$STAGE5/videos" \
  --keyframes-dir "$STAGE5/frames" \
  --fps "$FPS" \
  --num-keyframes 8

echo "[stage5] === Step 3: Gemini 3-way descriptions ==="
"$PY" "$REPO/scripts/stage5/describe_v2x_with_gemini.py" \
  --manifest-csv "$MANIFEST" \
  --frames-dir "$STAGE5/frames" \
  --output-json "$STAGE5/prompts/scenario_descriptions.json"

echo "[stage5] === Step 4: CLIP ViT-B/32 scoring ==="
"$PY" "$REPO/scripts/stage5/clipscore_v2x.py" \
  --descriptions "$STAGE5/prompts/scenario_descriptions.json" \
  --frames-dir "$STAGE5/frames" \
  --output-csv "$STAGE5/clipscore/per_video.csv" \
  --output-json "$STAGE5/clipscore/per_frame.json"

echo "[stage5] === Step 5: paired statistical analysis ==="
"$PY" "$REPO/scripts/stage5/analyze_v2x_label_bias.py" \
  --scores-csv "$STAGE5/clipscore/per_video.csv" \
  --scores-json "$STAGE5/clipscore/per_frame.json" \
  --output-dir "$STAGE5"

echo "[stage5] === Step 6: cross-dataset comparison (Crash-1500 vs DoTA) ==="
"$PY" "$REPO/scripts/stage5/build_cross_dataset_table.py" \
  --stage3-dir "$ROOT/outputs/stage3" \
  --stage5-dir "$STAGE5" \
  --output "$STAGE5/cross_dataset_comparison.csv"

echo "[stage5] DONE. See $STAGE5/REPORT.md"
