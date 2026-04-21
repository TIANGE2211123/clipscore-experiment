# ClipScore Experiment: Label Bias Testing

## Project Overview

This project evaluates how **text prompt bias** affects video-text alignment scores (ClipScore) across three datasets. We test whether adding safety-related labels to descriptions artificially inflates or deflates similarity scores, even when the video content remains unchanged.

### Research Question
**Does label mention in text descriptions create spurious correlations in video-text similarity metrics?**

---

## Datasets

### 1. Crash-1500 (Baseline - Already Complete)
- **Size:** 1500 videos
- **Source:** Existing internal dataset
- **Labels:** safe, near-crash, crash
- **Status:** ✅ Preprocessed and ready

### 2. Euro NCAP (Controlled Tests)
- **Size:** 100 videos
- **Source:** YouTube official channel (crash test footage)
- **Labels:** safe, near-crash, crash
- **Ground Truth:** All videos are actual crash events
- **Purpose:** Test if "safe" descriptions reduce scores for crash videos

### 3. V2X-Seq-SPD (Chinese Traffic)
- **Size:** 100 videos
- **Source:** Google Drive (DAIR-V2X official dataset)
- **Labels:** safe, near-crash, crash
- **Ground Truth:** Mixed (vehicle-side + infrastructure-side views)
- **Purpose:** Test generalization to real-world traffic scenarios

---

## Experiment Workflow

### Stage 1: Data Preparation

#### For Each Dataset:
1. **Extract candidate videos** (manifests with metadata)
2. **Generate 3 description types per video:**
   - **Objective:** Pure observational, no label hints
   - **Interpretive:** Contextual description with subtle cues
   - **Label-biased:** Explicit label mention (safe/near-crash/crash)
3. **Download videos** (if needed)
4. **Prepare FramePack queue bundle** (for GPU inference)

**Scripts:**
- Euro NCAP: `scripts/export_euroncap_channel_manifest.py`
- V2X-Seq: `scripts/build_v2x_seq_proxy_manifest.py`
- Descriptions: `scripts/generate_dataset_descriptions.py`
- Queue bundle: `scripts/prepare_framepack_queue_bundle.py`

**One-command automation:**
```bash
./scripts/orchestrate_full_pipeline.sh
```

---

### Stage 2: ClipScore Inference

#### For Each Dataset × Description Type:
1. **Extract frames** from videos (first + last frame)
2. **Compute ClipScore** between frames and descriptions
3. **Save results** to CSV with metadata

**Input Format:**
```json
{
  "video_id": {
    "safe": "description text...",
    "near_crash": "description text...",
    "crash": "description text..."
  }
}
```

**Output Format:**
```csv
video_id,dataset,true_label,description_type,clipscore,scene_type,weather,lighting
```

**Expected GPU Time:**
- Crash-1500: ~2-3 hours
- Euro NCAP: ~30 min
- V2X-Seq-SPD: ~30 min

---

### Stage 3: Statistical Analysis

#### Metrics to Compute:

1. **Within-Video Score Variance**
   - Do scores change when only the description label changes?
   - Expected: Low variance for objective metrics

2. **Cross-Label Score Differences**
   - For true crash videos, does "safe" description reduce score?
   - For true safe videos, does "crash" description increase score?

3. **Label-Description Alignment Bias**
   - Does matching label → higher score (spurious correlation)?
   - Statistical test: paired t-test across description types

4. **Dataset Generalization**
   - Do bias patterns hold across Euro NCAP vs. V2X vs. Crash-1500?
   - Cross-dataset consistency check

**Analysis Script:**
```bash
python scripts/analyze_label_bias.py \
    --results results/all_datasets_clipscore.csv \
    --output results/bias_analysis_report.pdf
```

---

## File Structure

```
clipscore-experiment/
├── CLAUDE.md                          # This file
├── README.md                          # Public documentation
├── PIPELINE_GUIDE.md                  # Technical implementation guide
├── QUICKSTART.md                      # Quick reference commands
│
├── scripts/
│   ├── export_euroncap_channel_manifest.py      # Euro NCAP source adapter
│   ├── build_v2x_seq_proxy_manifest.py          # V2X-Seq source adapter
│   ├── generate_dataset_descriptions.py         # 3-type description generator
│   ├── prepare_framepack_queue_bundle.py        # Queue bundle packager
│   ├── orchestrate_full_pipeline.sh             # Full automation script
│   └── analyze_label_bias.py                    # Statistical analysis (TBD)
│
├── test_data/
│   ├── crash1500/                     # Crash-1500 dataset
│   │   ├── manifest.csv
│   │   └── classified_descriptions.json
│   │
│   ├── euroncap_source/               # Euro NCAP dataset
│   │   ├── euroncap_candidates.csv
│   │   ├── classified_descriptions.json
│   │   ├── videos/                    # Downloaded MP4s
│   │   └── queue_images/              # Extracted frames
│   │
│   └── v2x_seq_proxy_source/          # V2X-Seq-SPD dataset
│       ├── v2x_seq_proxy_candidates.csv
│       ├── classified_descriptions.json
│       ├── videos/                    # Rendered MP4s from sequences
│       └── queue_images/
│
└── results/
    ├── crash1500_clipscore.csv        # Stage 2 output
    ├── euroncap_clipscore.csv
    ├── v2x_seq_clipscore.csv
    └── bias_analysis_report.pdf       # Stage 3 output
```

---

## Execution Guide

### Local Machine (Mac)

**Step 1: Data Preparation**
```bash
cd /Users/Lenovo/Desktop/学业/实验ClipScore

# Generate Euro NCAP manifest + descriptions
python3 scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 \
    --sample-size 100

python3 scripts/generate_dataset_descriptions.py \
    --input-csv test_data/euroncap_source/euroncap_candidates.csv \
    --output-json test_data/euroncap_source/classified_descriptions.json \
    --output-csv test_data/euroncap_source/classified_descriptions.csv \
    --dataset-name euroncap \
    --seed 42

# Generate V2X-Seq manifest + descriptions (after downloading Example.zip)
python3 scripts/build_v2x_seq_proxy_manifest.py \
    --metadata-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
    --output-dir test_data/v2x_seq_proxy_source \
    --sample-size 100 \
    --seed 42

python3 scripts/render_v2x_seq_videos.py \
    --manifest-csv test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv \
    --images-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
    --output-dir test_data/v2x_seq_proxy_source/videos \
    --fps 10

python3 scripts/generate_dataset_descriptions.py \
    --input-csv test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv \
    --output-json test_data/v2x_seq_proxy_source/classified_descriptions.json \
    --output-csv test_data/v2x_seq_proxy_source/classified_descriptions.csv \
    --dataset-name v2x_seq_proxy \
    --seed 42
```

**Step 2: Transfer to AutoDL**
```bash
# Push to GitHub
git add .
git commit -m "Update experiment data"
git push origin main

# Or use rsync
rsync -avz test_data/ autodl:/root/autodl-tmp/clipscore-experiment/test_data/
```

---

### AutoDL (GPU Server)

**Step 1: Setup**
```bash
ssh -p 11224 root@connect.cqa1.seetacloud.com
cd /root/autodl-tmp

# Option A: Clone from GitHub
git clone https://github.com/TIANGE2211123/clipscore-experiment.git
cd clipscore-experiment

# Option B: Use existing directory
cd clipscore-experiment
git pull origin main

# Install dependencies
/root/miniconda3/bin/pip install yt-dlp opencv-python pandas numpy
```

**Step 2: Download Videos (if not done locally)**
```bash
# Euro NCAP
/root/miniconda3/bin/python scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 \
    --sample-size 100 \
    --download \
    --download-dir test_data/euroncap_source/videos

# V2X-Seq (already rendered locally)
# Videos should be transferred from local machine
```

**Step 3: Prepare Queue Bundles**
```bash
# Euro NCAP
/root/miniconda3/bin/python scripts/prepare_framepack_queue_bundle.py \
    --manifest-csv test_data/euroncap_source/euroncap_candidates.csv \
    --descriptions-json test_data/euroncap_source/classified_descriptions.json \
    --video-dir test_data/euroncap_source/videos \
    --output-dir test_data/euroncap_source \
    --dataset-name euroncap

# V2X-Seq
/root/miniconda3/bin/python scripts/prepare_framepack_queue_bundle.py \
    --manifest-csv test_data/v2x_seq_proxy_source/v2x_seq_proxy_candidates.csv \
    --descriptions-json test_data/v2x_seq_proxy_source/classified_descriptions.json \
    --video-dir test_data/v2x_seq_proxy_source/videos \
    --output-dir test_data/v2x_seq_proxy_source \
    --dataset-name v2x_seq_proxy

# Crash-1500 (if needed)
/root/miniconda3/bin/python scripts/prepare_framepack_queue_bundle.py \
    --manifest-csv test_data/crash1500/manifest.csv \
    --descriptions-json test_data/crash1500/classified_descriptions.json \
    --video-dir test_data/crash1500/videos \
    --output-dir test_data/crash1500 \
    --dataset-name crash1500
```

**Step 4: Run ClipScore Inference**
```bash
# Use FramePack or your existing ClipScore script
# For each dataset:
/root/miniconda3/bin/python run_clipscore_inference.py \
    --queue-seed test_data/euroncap_source/queue_seed.json \
    --queue-images test_data/euroncap_source/queue_images \
    --output results/euroncap_clipscore.csv

/root/miniconda3/bin/python run_clipscore_inference.py \
    --queue-seed test_data/v2x_seq_proxy_source/queue_seed.json \
    --queue-images test_data/v2x_seq_proxy_source/queue_images \
    --output results/v2x_seq_clipscore.csv

/root/miniconda3/bin/python run_clipscore_inference.py \
    --queue-seed test_data/crash1500/queue_seed.json \
    --queue-images test_data/crash1500/queue_images \
    --output results/crash1500_clipscore.csv
```

**Step 5: Download Results**
```bash
# On local machine
scp -P 11224 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/clipscore-experiment/results/*.csv results/
```

---

### Stage 3: Analysis (Local Machine)

```bash
# Combine all results
cat results/crash1500_clipscore.csv \
    results/euroncap_clipscore.csv \
    results/v2x_seq_clipscore.csv \
    > results/all_datasets_clipscore.csv

# Run statistical analysis
python scripts/analyze_label_bias.py \
    --results results/all_datasets_clipscore.csv \
    --output results/bias_analysis_report.pdf
```

---

## Expected Results

### Hypothesis
**Label bias exists:** Videos paired with matching-label descriptions will show higher ClipScore than mismatched labels, even when video content is identical.

### Success Criteria
1. **Significant score variance** across description types (p < 0.05)
2. **Directional bias:** Crash videos + "safe" descriptions → lower scores
3. **Cross-dataset consistency:** Pattern holds across all 3 datasets

### Null Hypothesis Rejection
If label bias does **not** exist, scores should remain stable across all three description types for the same video.

---

## Checklist

### Data Preparation
- [ ] Crash-1500 manifests and descriptions ready
- [ ] Euro NCAP: 100 candidates extracted
- [ ] Euro NCAP: 300 descriptions generated
- [ ] Euro NCAP: Videos downloaded
- [ ] V2X-Seq: 100 sequences sampled
- [ ] V2X-Seq: Image sequences rendered to MP4
- [ ] V2X-Seq: 300 descriptions generated
- [ ] All queue bundles prepared

### Inference
- [ ] Crash-1500 ClipScore computed
- [ ] Euro NCAP ClipScore computed
- [ ] V2X-Seq ClipScore computed
- [ ] Results downloaded from AutoDL

### Analysis
- [ ] Combined results CSV created
- [ ] Statistical tests completed
- [ ] Visualization plots generated
- [ ] Final report written

---

## Troubleshooting

### Issue: YouTube bot verification blocking downloads
**Solution:** Run downloads on AutoDL server (better IP reputation)

### Issue: V2X-Seq download stuck
**Solution:** Download manually from Google Drive on local machine, then transfer

### Issue: FramePack queue bundle errors
**Solution:** Ensure videos exist in specified directory, check video codec compatibility

### Issue: ClipScore inference OOM
**Solution:** Reduce batch size, process in smaller chunks

---

## Dependencies

### Local Machine (Mac)
```bash
pip install yt-dlp opencv-python pandas numpy pillow
```

### AutoDL (Ubuntu + GPU)
```bash
/root/miniconda3/bin/pip install yt-dlp opencv-python pandas numpy pillow torch torchvision clip
```

---

## References

- **Euro NCAP Channel:** https://www.youtube.com/channel/UCNEWZqjcguqWZOG8yZZpIFg
- **V2X-Seq-SPD Paper:** https://arxiv.org/abs/2305.02647
- **V2X-Seq-SPD Dataset:** https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6
- **ClipScore Paper:** https://arxiv.org/abs/2104.08718

---

## Maintenance

### Adding New Datasets
1. Create source adapter in `scripts/`
2. Follow naming convention: `{dataset_name}_candidates.csv`
3. Use `generate_dataset_descriptions.py` for consistency
4. Update this CLAUDE.md checklist

### Updating Description Templates
Edit `scripts/generate_dataset_descriptions.py`:
- `SAFE_ACTIONS` (line 70-74)
- `NEAR_CRASH_ACTIONS` (line 76-80)
- `CRASH_TYPES` (line 82-88)

---

## Contact

**Researcher:** TIANGE
**GitHub:** https://github.com/TIANGE2211123/clipscore-experiment
**AutoDL:** connect.cqa1.seetacloud.com:11224
