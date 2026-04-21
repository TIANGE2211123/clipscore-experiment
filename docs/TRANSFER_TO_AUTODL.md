# Transfer Package to AutoDL

## 📦 What to Transfer

Transfer these files to AutoDL:

```bash
rsync -avz --progress \
    test_data/euroncap_source/*.csv \
    test_data/euroncap_source/*.json \
    scripts/ \
    autodl:/workspace/clipscore_experiment/
```

Or use scp:

```bash
scp -r test_data/euroncap_source/*.{csv,json} autodl:/workspace/clipscore_experiment/test_data/euroncap_source/
scp -r scripts/ autodl:/workspace/clipscore_experiment/scripts/
```

## 🚀 On AutoDL: Download Videos + Generate Queue Bundle

Once transferred, run on AutoDL:

```bash
cd /workspace/clipscore_experiment

# Download 100 Euro NCAP videos (this will work better on AutoDL)
python3 scripts/export_euroncap_channel_manifest.py \
    --output-dir test_data/euroncap_source \
    --max-entries 120 \
    --sample-size 100 \
    --download \
    --download-dir test_data/euroncap_source/videos

# Generate queue bundle
python3 scripts/prepare_framepack_queue_bundle.py \
    --manifest-csv test_data/euroncap_source/euroncap_candidates.csv \
    --descriptions-json test_data/euroncap_source/classified_descriptions.json \
    --video-dir test_data/euroncap_source/videos \
    --output-dir test_data/euroncap_source \
    --dataset-name euroncap
```

Or use the automated script:

```bash
bash scripts/download_and_bundle_on_remote.sh
```

## ✅ What You Already Have (Ready to Transfer)

- ✅ `euroncap_candidates.csv` — 100 video metadata
- ✅ `classified_descriptions.csv` — 300 descriptions (3 types × 100)
- ✅ `classified_descriptions.json` — JSON format
- ✅ All processing scripts

## ⏳ What Happens on AutoDL

1. Videos download (~2-5GB, 10-30 min depending on speed)
2. Queue bundle generates (~2-3 min)
3. Final output:
   - `queue_seed.json`
   - `queue_images/`
   - `job_manifest.csv`

Then you're ready for FramePack inference!
