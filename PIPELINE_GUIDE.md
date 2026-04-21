# ClipScore Dataset Pipeline Guide

This guide documents the complete pipeline for preparing two 100-sample datasets (Euro NCAP + V2X-Seq-SPD) for your ClipScore experiment.

## Quick Start

Run the full pipeline with:

```bash
./scripts/orchestrate_full_pipeline.sh
```

## Pipeline Overview

```
┌─────────────────────┐
│ Source Adapters     │
│ (YouTube / GDrive)  │
└──────────┬──────────┘
           │
           ├─→ euroncap_candidates.csv (100 samples)
           └─→ v2x_seq_proxy_candidates.csv (100 samples)
           │
           ▼
┌─────────────────────┐
│ Description Engine  │
│ (3 types per video) │
└──────────┬──────────┘
           │
           └─→ classified_descriptions.csv
           │
           ▼
┌─────────────────────┐
│ Queue Bundle Prep   │
│ (FramePack format)  │
└──────────┬──────────┘
           │
           ├─→ queue_seed.json
           ├─→ queue_images/*.jpg
           └─→ job_manifest.csv
```

## Stage 1: Source Adapters

### Euro NCAP (YouTube)

**Script:** [scripts/export_euroncap_channel_manifest.py](scripts/export_euroncap_channel_manifest.py)

```bash
python3 scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 \
    --sample-size 100
```

**Output:** `test_data/euroncap_source/euroncap_candidates.csv` (100 samples)

**Optional:** Add `--download --download-dir test_data/euroncap_source/videos` to download videos

---

### V2X-Seq-SPD (Google Drive)

**Script:** [scripts/build_v2x_seq_proxy_manifest.py](scripts/build_v2x_seq_proxy_manifest.py)

**Prerequisites:** Download the official example first:
- URL: https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6
- Target: `test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip`

```bash
python3 scripts/build_v2x_seq_proxy_manifest.py \
    --metadata-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
    --output-dir test_data/v2x_seq_proxy_source \
    --sample-size 100 \
    --seed 42
```

**Output:** `test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv` (100 samples)

**Video Rendering:** Use [scripts/render_v2x_seq_videos.py](scripts/render_v2x_seq_videos.py) to convert image sequences to MP4:

```bash
python3 scripts/render_v2x_seq_videos.py \
    --manifest-csv test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv \
    --images-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
    --output-dir test_data/v2x_seq_proxy_source/videos \
    --fps 10
```

---

## Stage 2: Description Generation

**Script:** [scripts/generate_dataset_descriptions.py](scripts/generate_dataset_descriptions.py)

Generates three description types per video:
1. **Objective:** Pure observational description (no label hints)
2. **Interpretive:** Contextual description (subtle label hints)
3. **Label-biased:** Direct label mention

```bash
python3 scripts/generate_dataset_descriptions.py \
    --input-csv test_data/euroncap_source/euroncap_candidates.csv \
    --output-csv test_data/euroncap_source/classified_descriptions.csv
```

**Output Schema:**
```csv
video_id,dataset,label,objective_description,interpretive_description,label_biased_description
```

---

## Stage 3: FramePack Queue Bundle

**Script:** [scripts/prepare_framepack_queue_bundle.py](scripts/prepare_framepack_queue_bundle.py)

Prepares the final bundle for GPU inference on AutoDL:

```bash
python3 scripts/prepare_framepack_queue_bundle.py \
    --descriptions-csv test_data/euroncap_source/classified_descriptions.csv \
    --video-dir test_data/euroncap_source/videos \
    --output-dir test_data/euroncap_source \
    --seed 42
```

**Outputs:**
- `queue_seed.json` — Single-file queue config for FramePack
- `queue_images/` — All reference frames extracted and ready
- `job_manifest.csv` — Human-readable manifest for tracking

---

## Current Status

### ✅ Euro NCAP (Complete)

- [x] Candidate manifest generated (100 samples)
- [x] Classified descriptions ready
- [ ] Videos downloaded (optional — can be done on AutoDL or locally)

### ⏳ V2X-Seq-SPD (Waiting on Source Data)

- [ ] V2X-Seq-SPD-Example.zip download in progress (remote)
- [ ] Candidate manifest generation
- [ ] Video rendering (image sequences → MP4)
- [ ] Classified descriptions
- [ ] Queue bundle preparation

---

## Next Steps

1. **Monitor V2X download** — Once `V2X-Seq-SPD-Example.zip` arrives, re-run:
   ```bash
   ./scripts/orchestrate_full_pipeline.sh
   ```

2. **Download videos** — For Euro NCAP:
   ```bash
   python3 scripts/export_euroncap_channel_manifest.py \
       --output-dir test_data/euroncap_source \
       --max-entries 120 \
       --sample-size 100 \
       --download \
       --download-dir test_data/euroncap_source/videos
   ```

3. **Transfer to AutoDL** — Upload queue bundles for FramePack inference:
   ```bash
   rsync -avz test_data/euroncap_source/queue_seed.json autodl:/workspace/
   rsync -avz test_data/euroncap_source/queue_images/ autodl:/workspace/queue_images/
   ```

---

## File Organization

```
test_data/
├── euroncap_source/
│   ├── euroncap_candidates.csv         # Stage 1 output
│   ├── classified_descriptions.csv     # Stage 2 output
│   ├── queue_seed.json                 # Stage 3 output
│   ├── queue_images/                   # Stage 3 output
│   ├── job_manifest.csv                # Stage 3 output (human-readable)
│   └── videos/                         # Optional: downloaded MP4s
│
└── v2x_seq_proxy_source/
    ├── v2x_seq_proxy_candidates.csv    # Stage 1 output
    ├── classified_descriptions.csv     # Stage 2 output
    ├── queue_seed.json                 # Stage 3 output
    ├── queue_images/                   # Stage 3 output
    └── videos/                         # Rendered MP4s from sequences
```

---

## Troubleshooting

### "yt-dlp EOF occurred in violation of protocol"
- Normal retry behavior for YouTube rate limiting
- Script will automatically retry 3 times
- If persistent, run with `--max-entries 50` and increase gradually

### "V2X-Seq-SPD-Example.zip not found"
- Download manually from Google Drive: https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6
- Place in: `test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip`

### "No such file or directory: videos/"
- Videos are optional for description generation (uses metadata only)
- Queue bundle preparation requires actual video files
- Run download step or render step first before Stage 3

---

## Validation

Quick validation commands:

```bash
# Check candidate manifests
wc -l test_data/*/euroncap_candidates.csv test_data/*/v2x_seq_proxy_candidates.csv

# Check description coverage (should be 3× candidate count)
wc -l test_data/*/classified_descriptions.csv

# Verify queue bundle structure
cat test_data/euroncap_source/queue_seed.json | python3 -m json.tool | head -30
ls test_data/euroncap_source/queue_images/ | wc -l
```

---

## References

- **Euro NCAP Official Channel:** https://www.youtube.com/channel/UCNEWZqjcguqWZOG8yZZpIFg
- **V2X-Seq-SPD Paper:** https://arxiv.org/abs/2305.02647
- **V2X-Seq-SPD Dataset:** https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6
- **FramePack (assumed):** Your custom GPU inference framework
