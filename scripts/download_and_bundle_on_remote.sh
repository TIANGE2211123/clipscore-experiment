#!/usr/bin/env bash
# Script to run on AutoDL after transferring descriptions
# This downloads videos and generates queue bundle

set -euo pipefail

echo "=== Euro NCAP: Download videos + Generate queue bundle ==="
echo ""

# Step 1: Download videos
echo "[1/2] Downloading 100 Euro NCAP videos from YouTube..."
python3 scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 \
    --sample-size 100 \
    --download \
    --download-dir test_data/euroncap_source/videos

echo ""
echo "[1/2] ✓ Videos downloaded"
echo ""

# Step 2: Generate queue bundle
echo "[2/2] Generating FramePack queue bundle..."
python3 scripts/prepare_framepack_queue_bundle.py \
    --manifest-csv test_data/euroncap_source/euroncap_candidates.csv \
    --descriptions-json test_data/euroncap_source/classified_descriptions.json \
    --video-dir test_data/euroncap_source/videos \
    --output-dir test_data/euroncap_source \
    --dataset-name euroncap

echo ""
echo "[2/2] ✓ Queue bundle generated"
echo ""
echo "=== Complete! ==="
echo ""
echo "Output files:"
echo "  - test_data/euroncap_source/queue_seed.json"
echo "  - test_data/euroncap_source/queue_images/"
echo "  - test_data/euroncap_source/job_manifest.csv"
