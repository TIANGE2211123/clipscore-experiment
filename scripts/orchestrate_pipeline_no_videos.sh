#!/usr/bin/env bash
# Pipeline without video download - generates manifests and descriptions only
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=== STAGE 1: Generating candidate manifests ==="

# Euro NCAP
if [[ ! -f test_data/euroncap_source/euroncap_candidates.csv ]]; then
    echo "[1a] Extracting Euro NCAP candidates..."
    python3 scripts/export_euroncap_channel_manifest.py \
        --output-dir test_data/euroncap_source \
        --max-entries 120 \
        --sample-size 100
else
    echo "[1a] Euro NCAP candidates already exist ($(wc -l < test_data/euroncap_source/euroncap_candidates.csv) rows)"
fi

# V2X-Seq-SPD
if [[ ! -f test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv ]]; then
    echo "[1b] Building V2X-Seq-SPD candidates..."
    if [[ -f test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip ]]; then
        python3 scripts/build_v2x_seq_proxy_manifest.py \
            --metadata-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
            --output-dir test_data/v2x_seq_proxy_source \
            --sample-size 100 \
            --seed 42
    else
        echo "  ⚠️  V2X-Seq-SPD-Example.zip not found. Skipping."
    fi
else
    echo "[1b] V2X-Seq-SPD candidates already exist ($(wc -l < test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv) rows)"
fi

echo ""
echo "=== STAGE 2: Generating classified descriptions ==="

for source_dir in test_data/euroncap_source test_data/v2x_seq_proxy_source; do
    if [[ -d "$source_dir" ]]; then
        dataset_name=$(basename "$source_dir" _source)
        manifest_csv="$source_dir/${dataset_name}_candidates.csv"
        if [[ -f "$manifest_csv" ]]; then
            output_csv="$source_dir/classified_descriptions.csv"
            output_json="$source_dir/classified_descriptions.json"
            echo "[2] Processing $manifest_csv..."
            python3 scripts/generate_dataset_descriptions.py \
                --input-csv "$manifest_csv" \
                --output-json "$output_json" \
                --output-csv "$output_csv" \
                --dataset-name "$dataset_name" \
                --seed 42
            echo "    ✓ Generated: $output_csv"
        fi
    fi
done

echo ""
echo "=== Pipeline Summary ==="
echo ""
for dataset in euroncap v2x_seq_proxy; do
    source_dir="test_data/${dataset}_source"
    if [[ -d "$source_dir" ]]; then
        echo "[$dataset]"
        [[ -f "$source_dir/${dataset}_candidates.csv" ]] && \
            echo "  ✓ Candidates:    $(tail -n +2 "$source_dir/${dataset}_candidates.csv" | wc -l | xargs) samples"
        [[ -f "$source_dir/classified_descriptions.csv" ]] && \
            echo "  ✓ Descriptions:  $(tail -n +2 "$source_dir/classified_descriptions.csv" | wc -l | xargs) samples × 3 types"
        echo ""
    fi
done

echo "✅ Manifests and descriptions ready!"
echo ""
echo "Next steps:"
echo "  1. Download videos (optional - can be done on AutoDL)"
echo "  2. Run queue bundle prep once videos are available"
