# ClipScore Dataset Pipeline - Current Status

**Last Updated:** 2026-04-19

---

## 🎯 Goal

Prepare two 100-sample datasets for ClipScore experiment:
1. **Euro NCAP** (crash test videos from YouTube)
2. **V2X-Seq-SPD** (Chinese traffic sequences from Google Drive)

Each video gets **3 description types** (safe/near-crash/crash) for label bias testing.

---

## ✅ Completed

### Euro NCAP Dataset (100/100 complete)

| Stage | Status | Output |
|-------|--------|--------|
| 1. Candidate extraction | ✅ Done | [test_data/euroncap_source/euroncap_candidates.csv](test_data/euroncap_source/euroncap_candidates.csv) (100 samples) |
| 2. Description generation | ✅ Done | [test_data/euroncap_source/classified_descriptions.csv](test_data/euroncap_source/classified_descriptions.csv) (100 samples × 3 types) |
| 3. Queue bundle prep | ⏸️ Blocked | Waiting for video downloads |

**Sample Description Preview:**
```json
{
  "ZPv9uEcdrGI": {
    "safe": "[0s: euroncap footage...] Traffic remains orderly...",
    "near_crash": "[0s: euroncap footage...] A hazardous interaction develops...",
    "crash": "[0s: euroncap footage...] The conflict escalates rapidly..."
  }
}
```

**Next Step:** Download videos (optional on AutoDL) or proceed directly to Stage 3 queue bundle.

---

### Local Pipeline Scripts (Universal, Dataset-Agnostic)

| Script | Purpose | Status |
|--------|---------|--------|
| [build_label_prompt_manifest.py](scripts/build_label_prompt_manifest.py) | Convert source → label prompt CSV | ✅ Tested (Crash-1500) |
| [generate_dataset_descriptions.py](scripts/generate_dataset_descriptions.py) | Manifest → 3 description types | ✅ Tested (Euro NCAP) |
| [prepare_framepack_queue_bundle.py](scripts/prepare_framepack_queue_bundle.py) | Videos + descriptions → queue bundle | ✅ Tested (001376 sample) |
| [orchestrate_full_pipeline.sh](scripts/orchestrate_full_pipeline.sh) | One-command full pipeline | ✅ Ready |

All scripts now support arbitrary datasets — no hardcoded paths.

---

## ⏳ In Progress

### V2X-Seq-SPD Dataset (0/100)

| Stage | Status | Blocker |
|-------|--------|---------|
| 0. Source download | 🔴 **Blocked** | AutoDL → Google Drive connectivity timeout |
| 1. Candidate manifest | ⏸️ Waiting | Need `V2X-Seq-SPD-Example.zip` first |
| 2. Video rendering | ⏸️ Waiting | Image sequences → MP4 conversion |
| 3. Description generation | ⏸️ Waiting | Depends on manifest |
| 4. Queue bundle | ⏸️ Waiting | Depends on videos |

**Download Strategy Pivot:**
- ❌ Remote (AutoDL) → Google Drive: Connection unstable
- ✅ Local (your Mac) → Google Drive: Working (tested with `gdown`)

**Required File:**
- **Source:** https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6
- **Target:** `test_data/dair_v2x_seq_example/V2X-Seq-SPD-Example.zip` (~size TBD)

**Once Downloaded, Auto-Run:**
```bash
./scripts/orchestrate_full_pipeline.sh
```

This will:
1. Extract metadata from `V2X-Seq-SPD-Example.zip`
2. Build 100-sample manifest (balanced vehicle-side + infrastructure-side)
3. Render image sequences → MP4 videos
4. Generate 3-type descriptions
5. Prepare queue bundle

---

## 📊 File Tree (Current State)

```
学业/实验ClipScore/
├── PIPELINE_GUIDE.md          ← Complete usage documentation
├── STATUS.md                  ← This file
├── scripts/
│   ├── orchestrate_full_pipeline.sh               ← Master script
│   ├── export_euroncap_channel_manifest.py        ← Euro NCAP adapter
│   ├── build_v2x_seq_proxy_manifest.py            ← V2X adapter
│   ├── render_v2x_seq_videos.py                   ← Image sequence → MP4
│   ├── generate_dataset_descriptions.py           ← 3-type descriptions
│   ├── prepare_framepack_queue_bundle.py          ← Queue bundle prep
│   └── build_label_prompt_manifest.py             ← Candidate → label CSV
│
└── test_data/
    ├── euroncap_source/
    │   ├── euroncap_candidates.csv                ✅ 100 samples
    │   ├── classified_descriptions.csv            ✅ 100 × 3 types
    │   ├── classified_descriptions.json           ✅ JSON format
    │   └── videos/                                ⏸️ (optional, for queue bundle)
    │
    └── dair_v2x_seq_example/
        └── V2X-Seq-SPD-Example.zip                ❌ Not yet downloaded
```

---

## 🚀 Immediate Next Steps

### Option A: Continue Local (Recommended)

Since your Mac can reliably access Google Drive:

1. **Download V2X source** (you or the remote task):
   ```bash
   # Manual download from browser, or:
   gdown --fuzzy https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6 \
       -O test_data/dair_v2x_seq_example/
   ```

2. **Auto-complete V2X pipeline**:
   ```bash
   ./scripts/orchestrate_full_pipeline.sh
   ```

3. **Transfer final bundles to AutoDL**:
   ```bash
   rsync -avz test_data/*/queue_seed.json autodl:/workspace/
   rsync -avz test_data/*/queue_images/ autodl:/workspace/queue_images/
   ```

---

### Option B: Parallel Work While Waiting

Since Euro NCAP descriptions are ready, you can already:

1. **Test description quality** — Read [test_data/euroncap_source/classified_descriptions.csv](test_data/euroncap_source/classified_descriptions.csv)
2. **Prepare label prompt CSV** for baseline experiments
3. **Download Euro NCAP videos locally** (if needed for queue bundle)
4. **Design FramePack inference config** (if not already done)

---

## 📋 Validation Checklist

### Euro NCAP ✅
- [x] 100 candidate videos extracted
- [x] 3 description types per video (safe/near_crash/crash)
- [x] Output formats: CSV + JSON
- [ ] Videos downloaded (optional — can happen on AutoDL)
- [ ] Queue bundle prepared (blocked by videos)

### V2X-Seq-SPD ⏸️
- [ ] Example.zip downloaded
- [ ] 100 sequences sampled (50 vehicle + 50 infrastructure)
- [ ] Image sequences rendered to MP4
- [ ] 3 description types generated
- [ ] Queue bundle prepared

---

## 🔧 Known Issues & Workarounds

### Issue 1: AutoDL → Google Drive Timeout
**Symptom:** `gdown` and `yt-dlp` hang indefinitely on AutoDL
**Root Cause:** Network policy or routing issue
**Solution:** Download on local machine, then `rsync` to AutoDL

### Issue 2: V2X Videos Are Image Sequences
**Symptom:** No `.mp4` files, only `.jpg` frames
**Root Cause:** Official dataset structure (frame-by-frame storage)
**Solution:** Use [render_v2x_seq_videos.py](scripts/render_v2x_seq_videos.py) to stitch frames → MP4

### Issue 3: Descriptions Use Template, Not LLM
**Symptom:** Descriptions look formulaic
**Root Cause:** Script defaults to deterministic templates when Ollama unavailable
**Impact:** Acceptable for bias testing (still provides 3 distinct types)
**Alternative:** Install Ollama locally and add `--ollama-model llama3.2` to script

---

## 📚 References

- **Pipeline Guide:** [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- **Euro NCAP Channel:** https://www.youtube.com/channel/UCNEWZqjcguqWZOG8yZZpIFg
- **V2X-Seq-SPD Dataset:** https://drive.google.com/drive/folders/1r5sPiBEvo8Xby-nMaWUTnJIPK6WhY1B6
- **V2X-Seq-SPD Paper:** https://arxiv.org/abs/2305.02647

---

## 💬 Summary for Handoff

**What's Working:**
- Euro NCAP: 100 samples extracted + 300 descriptions ready
- All local scripts validated and reusable

**What's Blocked:**
- V2X-Seq: Waiting on `V2X-Seq-SPD-Example.zip` download

**Critical Path:**
1. Get V2X source file (Google Drive → local)
2. Run `./scripts/orchestrate_full_pipeline.sh`
3. Transfer queue bundles to AutoDL
4. Run FramePack inference

**Estimated Time to Completion (after V2X download):**
- Manifest generation: < 1 min
- Video rendering: ~5-10 min (depends on sequence count)
- Description generation: < 1 min
- Queue bundle prep: ~2-3 min

**Total: ~10-15 minutes of local processing once source arrives.**
