# ClipScore Experiment: Label Bias Testing

[![GitHub](https://img.shields.io/badge/github-clipscore--experiment-blue)](https://github.com/TIANGE2211123/clipscore-experiment)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

> **Research Question:** Does label mention in text descriptions create spurious correlations in video-text similarity metrics?

This project evaluates how **text prompt bias** affects video-text alignment scores (ClipScore) across three traffic safety datasets.

---

## 🎯 Project Overview

We test whether adding safety-related labels (`safe`, `near-crash`, `crash`) to video descriptions artificially inflates or deflates similarity scores, even when the video content remains unchanged.

### Datasets

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| **Crash-1500** | 1500 videos | Internal | Baseline dataset |
| **Euro NCAP** | 100 videos | YouTube (official crash tests) | Controlled test environment |
| **V2X-Seq-SPD** | 100 videos | Google Drive (DAIR-V2X) | Real-world Chinese traffic |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/TIANGE2211123/clipscore-experiment.git
cd clipscore-experiment

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline (3 Datasets)

```bash
# Generate manifests + descriptions for all datasets
./scripts/orchestrate_full_pipeline.sh

# Or process individual datasets
python3 scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 --sample-size 100

python3 scripts/generate_dataset_descriptions.py \
    --input-csv test_data/euroncap_source/euroncap_candidates.csv \
    --output-json test_data/euroncap_source/classified_descriptions.json \
    --output-csv test_data/euroncap_source/classified_descriptions.csv \
    --dataset-name euroncap --seed 42
```

---

## 📁 Repository Structure

```
clipscore-experiment/
├── README.md                    # This file
├── CLAUDE.md                    # Complete experiment workflow guide
├── requirements.txt             # Python dependencies
│
├── scripts/                     # Pipeline automation scripts
│   ├── export_euroncap_channel_manifest.py       # Euro NCAP adapter
│   ├── build_v2x_seq_proxy_manifest.py           # V2X-Seq adapter
│   ├── generate_dataset_descriptions.py          # 3-type descriptions
│   ├── prepare_framepack_queue_bundle.py         # Queue bundle prep
│   └── orchestrate_full_pipeline.sh              # Full automation
│
├── test_data/                   # Dataset storage
│   ├── euroncap_source/         # Euro NCAP (100 samples)
│   │   ├── euroncap_candidates.csv
│   │   └── classified_descriptions.json
│   └── v2x_seq_proxy_source/    # V2X-Seq-SPD (100 samples)
│
├── docs/                        # Documentation
│   ├── PIPELINE_GUIDE.md        # Technical implementation
│   ├── QUICKSTART.md            # Quick reference
│   └── STATUS.md                # Progress tracker
│
├── code/                        # Legacy ClipScore evaluator
└── output/                      # Experiment results
```

---

## 📊 Experiment Workflow

### Stage 1: Data Preparation

For each dataset, generate:
1. **Candidate manifests** (video metadata)
2. **Three description types per video:**
   - **Objective:** Pure observational (no label hints)
   - **Interpretive:** Contextual cues (subtle hints)
   - **Label-biased:** Explicit label mention

**Example Descriptions for Same Video:**
- **Objective:** "Traffic moves smoothly as vehicles maintain spacing"
- **Interpretive:** "A hazardous interaction develops as a vehicle intrudes into the path"
- **Label-biased:** "The conflict escalates rapidly and a crash develops"

### Stage 2: ClipScore Inference (GPU)

Compute video-text similarity scores using CLIP:
- Extract first + last frame from each video
- Calculate ClipScore for all description types
- Compare scores across label variations

### Stage 3: Statistical Analysis

Test hypothesis:
- **H1:** Label matching increases scores (spurious correlation)
- **H0:** Scores remain stable across description types

Metrics:
- Within-video score variance
- Cross-label score differences
- Dataset generalization consistency

---

## 🔬 Current Status

### ✅ Completed
- [x] Euro NCAP: 100 candidates extracted
- [x] Euro NCAP: 300 descriptions generated (3 types × 100 videos)
- [x] Pipeline scripts validated end-to-end
- [x] GitHub repository published

### ⏳ In Progress
- [ ] Euro NCAP: Video downloads (in progress on AutoDL)
- [ ] V2X-Seq-SPD: Source data download
- [ ] Queue bundle preparation

### 📅 Next Steps
1. Complete video downloads on AutoDL
2. Run ClipScore inference on GPU
3. Statistical analysis and visualization
4. Write final report

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **[CLAUDE.md](CLAUDE.md)** | Complete experiment workflow (3 datasets) |
| **[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)** | Technical implementation details |
| **[docs/QUICKSTART.md](docs/QUICKSTART.md)** | Quick reference commands |
| **[docs/STATUS.md](docs/STATUS.md)** | Detailed progress tracker |

---

## 🛠️ Key Scripts

### Data Preparation

```bash
# Euro NCAP: Extract 100 crash test videos from YouTube
python3 scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 --sample-size 100 \
    --download --download-dir test_data/euroncap_source/videos

# V2X-Seq-SPD: Sample 100 sequences from official dataset
python3 scripts/build_v2x_seq_proxy_manifest.py \
    --metadata-zip test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip \
    --output-dir test_data/v2x_seq_proxy_source \
    --sample-size 100 --seed 42

# Generate 3-type descriptions for any dataset
python3 scripts/generate_dataset_descriptions.py \
    --input-csv test_data/{dataset}_source/{dataset}_candidates.csv \
    --output-json test_data/{dataset}_source/classified_descriptions.json \
    --output-csv test_data/{dataset}_source/classified_descriptions.csv \
    --dataset-name {dataset} --seed 42
```

### Queue Bundle Preparation

```bash
python3 scripts/prepare_framepack_queue_bundle.py \
    --manifest-csv test_data/euroncap_source/euroncap_candidates.csv \
    --descriptions-json test_data/euroncap_source/classified_descriptions.json \
    --video-dir test_data/euroncap_source/videos \
    --output-dir test_data/euroncap_source \
    --dataset-name euroncap
```

---

## 🖥️ AutoDL Execution

### Setup on AutoDL

```bash
# Clone from GitHub
git clone https://github.com/TIANGE2211123/clipscore-experiment.git
cd clipscore-experiment

# Install dependencies
/root/miniconda3/bin/pip install -r requirements.txt

# Download videos and prepare queue bundles
bash scripts/download_and_bundle_on_remote.sh
```

### Monitor Progress

```bash
# From local machine
bash check_autodl_progress.sh
```

---

## 📈 Expected Results

### Hypothesis
**Label bias exists:** Videos paired with matching-label descriptions will show higher ClipScore than mismatched labels, even when video content is identical.

### Success Criteria
1. Significant score variance across description types (p < 0.05)
2. Directional bias: Crash videos + "safe" descriptions → lower scores
3. Cross-dataset consistency: Pattern holds across all 3 datasets

---

## 🤝 Contributing

This is a research project. For questions or collaboration:
- **GitHub Issues:** [Report bugs or request features](https://github.com/TIANGE2211123/clipscore-experiment/issues)
- **Researcher:** TIANGE

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🔗 References

- **Euro NCAP Official Channel:** https://www.youtube.com/channel/UCNEWZqjcguqWZOG8yZZpIFg
- **V2X-Seq-SPD Paper:** [arXiv:2305.02647](https://arxiv.org/abs/2305.02647)
- **V2X-Seq-SPD Dataset:** [Google Drive](https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6)
- **ClipScore Paper:** [arXiv:2104.08718](https://arxiv.org/abs/2104.08718)

---

## 🏷️ Citation

If you use this pipeline or datasets in your research, please cite:

```bibtex
@misc{clipscore-experiment-2026,
  author = {TIANGE},
  title = {ClipScore Label Bias Testing: A Multi-Dataset Evaluation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/TIANGE2211123/clipscore-experiment}
}
```

---

**Last Updated:** 2026-04-21
**Status:** Active Development
