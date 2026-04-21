#!/usr/bin/env bash
# Full dataset pipeline orchestrator for ClipScore experiment
# This script ties together all three stages:
#   1. Source → Candidate manifest
#   2. Candidate → Descriptions (3 types)
#   3. Videos + Descriptions → FramePack queue bundle

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# ============================================================================
# STAGE 1: Generate Candidate Manifests
# ============================================================================

echo "=== STAGE 1: Generating candidate manifests ==="

# 1a. Euro NCAP (100 samples from official YouTube channel)
if [[ ! -f test_data/euroncap_source/euroncap_candidates.csv ]]; then
    echo "[1a] Extracting Euro NCAP candidates..."
    python3 scripts/export_euroncap_channel_manifest.py \
        --output-dir test_data/euroncap_source \
        --max-entries 120 \
        --sample-size 100
else
    echo "[1a] Euro NCAP candidates already exist ($(wc -l < test_data/euroncap_source/euroncap_candidates.csv) rows)"
fi

# 1b. V2X-Seq-SPD (100 samples from official example metadata)
if [[ ! -f test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv ]]; then
    echo "[1b] Building V2X-Seq-SPD candidates..."
    # NOTE: This requires V2X-Seq-SPD-Example.zip downloaded first
    # If not present, download from: https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6
    if [[ -f test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip ]]; then
        python3 scripts/build_v2x_seq_proxy_manifest.py \
            --metadata-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
            --output-dir test_data/v2x_seq_proxy_source \
            --sample-size 100 \
            --seed 42
    else
        echo "  ⚠️  V2X-Seq-SPD-Example.zip not found. Skipping V2X manifest generation."
        echo "     Download from: https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6"
    fi
else
    echo "[1b] V2X-Seq-SPD candidates already exist ($(wc -l < test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv) rows)"
fi

# ============================================================================
# STAGE 2: Generate Three Types of Descriptions
# ============================================================================

echo ""
echo "=== STAGE 2: Generating classified descriptions ==="

for source_dir in test_data/euroncap_source test_data/v2x_seq_proxy_source; do
    if [[ -d "$source_dir" ]]; then
        manifest_csv="$source_dir/$(basename "$source_dir" _source)_candidates.csv"
        if [[ -f "$manifest_csv" ]]; then
            dataset_name=$(basename "$source_dir" _source)
            output_csv="$source_dir/classified_descriptions.csv"
            output_json="$source_dir/classified_descriptions.json"
            echo "[2] Processing $manifest_csv..."
            python3 scripts/generate_dataset_descriptions.py \
                --input-csv "$manifest_csv" \
                --output-json "$output_json" \
                --output-csv "$output_csv" \
                --dataset-name "$dataset_name" \
                --seed 42
            echo "    → Generated: $output_csv"
        fi
    fi
done

# ============================================================================
# STAGE 3: Prepare FramePack Queue Bundle
# ============================================================================

echo ""
echo "=== STAGE 3: Preparing FramePack queue bundles ==="

for source_dir in test_data/euroncap_source test_data/v2x_seq_proxy_source; do
    if [[ -d "$source_dir" ]]; then
        dataset_name=$(basename "$source_dir" _source)
        manifest_csv="$source_dir/${dataset_name}_candidates.csv"
        descriptions_json="$source_dir/classified_descriptions.json"
        if [[ -f "$manifest_csv" ]] && [[ -f "$descriptions_json" ]]; then
            echo "[3] Building queue bundle for $dataset_name..."
            python3 scripts/prepare_framepack_queue_bundle.py \
                --manifest-csv "$manifest_csv" \
                --descriptions-json "$descriptions_json" \
                --video-dir "$source_dir/videos" \
                --output-dir "$source_dir" \
                --dataset-name "$dataset_name"
            echo "    → Generated: $source_dir/queue_seed.json"
            echo "    → Generated: $source_dir/queue_images/"
        else
            echo "  ⚠️  Manifest or descriptions not found for $source_dir. Skipping queue bundle."
        fi
    fi
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=== Pipeline Summary ==="
echo ""

for dataset in euroncap v2x_seq_proxy; do
    source_dir="test_data/${dataset}_source"
    if [[ -d "$source_dir" ]]; then
        echo "[$dataset]"
        [[ -f "$source_dir/${dataset}_candidates.csv" ]] && \
            echo "  ✓ Candidates:    $source_dir/${dataset}_candidates.csv ($(tail -n +2 "$source_dir/${dataset}_candidates.csv" | wc -l | xargs) samples)"
        [[ -f "$source_dir/classified_descriptions.csv" ]] && \
            echo "  ✓ Descriptions:  $source_dir/classified_descriptions.csv"
        [[ -f "$source_dir/queue_seed.json" ]] && \
            echo "  ✓ Queue bundle:  $source_dir/queue_seed.json + queue_images/"
        echo ""
    fi
done

echo "Pipeline complete. Next steps:"
echo "  1. Download videos (if not already done)"
echo "  2. Upload queue bundles to AutoDL for FramePack inference"
