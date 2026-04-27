#!/usr/bin/env bash
# Stage 4 / step E — generate 60 videos (10 src × 3 labels × 2 prompt sets)
# with generate_video_sdk.js. Resumable: skips any output that already exists.
set -u
cd /home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace

SDK=/home/node/.claude/skills/generate-video/scripts/generate_video_sdk.js
MODEL=google/veo-3.1-fast-generate-preview
OUT=outputs/stage4/generated
LOG=outputs/stage4/generation.log
mkdir -p "$OUT"

python3 - <<'PY' > /tmp/tasks.tsv
import csv, json
from pathlib import Path
sets = json.load(open("outputs/stage4/prompts/prompt_sets.json"))
for vid, s in sets.items():
    for ptype in ("P_label", "P_scenario"):
        for lbl in ("safe", "near_crash", "crash"):
            prompt = s[ptype][lbl].replace("\t", " ").replace("\n", " ")
            print(f"{vid}\t{ptype}\t{lbl}\t{prompt}")
PY

total=$(wc -l < /tmp/tasks.tsv)
i=0
while IFS=$'\t' read -r vid ptype lbl prompt; do
  i=$((i+1))
  dir="$OUT/$ptype/$lbl"
  mkdir -p "$dir"
  out="$dir/${vid}.mp4"
  if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null)" -gt 100000 ]; then
    echo "[$i/$total] skip $ptype/$lbl/$vid (exists $(stat -c%s "$out") bytes)" | tee -a "$LOG"
    continue
  fi
  echo "[$i/$total] gen $ptype/$lbl/$vid" | tee -a "$LOG"
  echo "  prompt: ${prompt:0:120}..." | tee -a "$LOG"
  t0=$(date +%s)
  node "$SDK" "$prompt" \
    --model "$MODEL" \
    --duration 6 \
    --aspect-ratio 16:9 \
    --output "$out" \
    --timeout 600 > /tmp/genlog 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ] || [ ! -f "$out" ]; then
    echo "  FAIL rc=$rc dt=${dt}s" | tee -a "$LOG"
    tail -5 /tmp/genlog | tee -a "$LOG"
  else
    echo "  ok ${dt}s $(stat -c%s "$out") bytes" | tee -a "$LOG"
  fi
done < /tmp/tasks.tsv

echo "done." | tee -a "$LOG"
