# ClipScore Pipeline - Quick Start

**TL;DR:** Run everything in one command once V2X data arrives.

---

## One-Command Full Pipeline

```bash
cd /Users/Lenovo/Desktop/学业/实验ClipScore
./scripts/orchestrate_full_pipeline.sh
```

**What it does:**
1. ✅ Euro NCAP: candidates → descriptions → queue bundle
2. ⏳ V2X-Seq-SPD: candidates → render videos → descriptions → queue bundle

---

## Current Status (2026-04-19)

### ✅ Euro NCAP (Complete)
- 100 samples extracted from official YouTube
- 300 descriptions generated (3 types × 100 videos)
- Ready for queue bundle once videos downloaded

### ⏳ V2X-Seq-SPD (Blocked)
**Waiting on:** `test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip`

**Download source:**
https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6

---

## Manual Steps (If Needed)

### 1. Generate Euro NCAP Queue Bundle

```bash
# Download videos first (100 videos, ~2-5GB)
python3 scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 \
    --sample-size 100 \
    --download \
    --download-dir test_data/euroncap_source/videos

# Then prepare queue bundle
python3 scripts/prepare_framepack_queue_bundle.py \
    --descriptions-csv test_data/euroncap_source/classified_descriptions.csv \
    --video-dir test_data/euroncap_source/videos \
    --output-dir test_data/euroncap_source \
    --seed 42
```

**Output:**
- `test_data/euroncap_source/queue_seed.json`
- `test_data/euroncap_source/queue_images/`
- `test_data/euroncap_source/job_manifest.csv`

---

### 2. Complete V2X Pipeline (Once Example.zip Arrives)

```bash
# Step 1: Build candidate manifest
python3 scripts/build_v2x_seq_proxy_manifest.py \
    --metadata-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
    --output-dir test_data/v2x_seq_proxy_source \
    --sample-size 100 \
    --seed 42

# Step 2: Render image sequences → MP4 videos
python3 scripts/render_v2x_seq_videos.py \
    --manifest-csv test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv \
    --images-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
    --output-dir test_data/v2x_seq_proxy_source/videos \
    --fps 10

# Step 3: Generate descriptions
python3 scripts/generate_dataset_descriptions.py \
    --input-csv test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv \
    --output-json test_data/v2x_seq_proxy_source/classified_descriptions.json \
    --output-csv test_data/v2x_seq_proxy_source/classified_descriptions.csv \
    --dataset-name v2x_seq_proxy \
    --seed 42

# Step 4: Prepare queue bundle
python3 scripts/prepare_framepack_queue_bundle.py \
    --descriptions-csv test_data/v2x_seq_proxy_source/classified_descriptions.csv \
    --video-dir test_data/v2x_seq_proxy_source/videos \
    --output-dir test_data/v2x_seq_proxy_source \
    --seed 42
```

**Or just run:** `./scripts/orchestrate_full_pipeline.sh`

---

## Transfer to AutoDL

```bash
# Sync queue bundles to remote GPU server
rsync -avz test_data/euroncap_source/queue_seed.json \
    test_data/euroncap_source/queue_images/ \
    autodl:/workspace/euroncap/

rsync -avz test_data/v2x_seq_proxy_source/queue_seed.json \
    test_data/v2x_seq_proxy_source/queue_images/ \
    autodl:/workspace/v2x_seq_proxy/
```

---

## Validation

```bash
# Check Euro NCAP status
wc -l test_data/euroncap_source/euroncap_candidates.csv           # Should be 101 (header + 100)
wc -l test_data/euroncap_source/classified_descriptions.csv       # Should be 101 (header + 100)
ls test_data/euroncap_source/queue_images/ | wc -l                # Should be ~300 (3 frames × 100)

# Check V2X status (once ready)
wc -l test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv # Should be 101
wc -l test_data/v2x_seq_proxy_source/classified_descriptions.csv  # Should be 101
ls test_data/v2x_seq_proxy_source/videos/ | wc -l                 # Should be 100
```

---

## Troubleshooting

**"V2X-Seq-SPD-Example.zip not found"**
→ Download from Google Drive link above, place in `test_data/dair_v2x_seq_example/`

**"yt-dlp EOF occurred in violation of protocol"**
→ Normal YouTube rate limiting, script auto-retries

**"No such file or directory: videos/"**
→ Videos not downloaded yet, run export script with `--download` flag

**"Ollama not available"**
→ Expected — script falls back to deterministic templates (still works fine)

---

## Files Overview

```
test_data/
├── euroncap_source/
│   ├── euroncap_candidates.csv           ✅ 100 samples
│   ├── classified_descriptions.csv       ✅ 300 descriptions
│   ├── classified_descriptions.json      ✅ JSON format
│   ├── queue_seed.json                   ⏸️ (needs videos first)
│   └── queue_images/                     ⏸️ (needs videos first)
│
└── v2x_seq_proxy_source/
    ├── v2x_seq_proxy_candidates.csv      ⏸️ (waiting on Example.zip)
    ├── videos/                           ⏸️ (image sequences → MP4)
    ├── classified_descriptions.csv       ⏸️ (after videos)
    └── queue_seed.json                   ⏸️ (after videos)
```

---

## Next Action

**Right now:**
Euro NCAP descriptions are ready. You can either:
1. Download Euro NCAP videos + generate queue bundle
2. Wait for V2X Example.zip to arrive, then run full pipeline

**When V2X arrives:**
```bash
./scripts/orchestrate_full_pipeline.sh
```

That's it. Everything else is automated.
