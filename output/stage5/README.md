# Stage 5 / Stage C — V2X-Seq-SPD Label-Bias Audit

跨数据集外推实验：将 Stage 3 (Crash-1500) 的 CLIPScore 标签偏置配对检验
协议迁移到 V2X-Seq-SPD（DAIR-V2X 协同感知子集）上，用以考察偏置模式
是否跨国别、跨道路结构复现。

## Pipeline

```
uploads/V2X-Seq-SPD-Example.zip
          │
          ▼
  ┌───────────────────────────┐
  │ 1. build_v2x_seq_proxy_   │   scripts/build_v2x_seq_proxy_manifest.py
  │    manifest (n=10, seed=42)│
  └───────────────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │ 2. render_v2x_seq_videos  │   scripts/render_v2x_seq_videos.py
  │    (10 fps MP4)           │
  └───────────────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │ 3. extract_v2x_frames     │   scripts/stage5/extract_v2x_frames.py
  │    (8 evenly-spaced PNGs) │
  └───────────────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │ 4. describe_v2x_with_     │   scripts/stage5/describe_v2x_with_gemini.py
  │    gemini (3-way)         │   — Gemini 3.1 Pro via AI Gateway
  └───────────────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │ 5. clipscore_v2x          │   scripts/stage5/clipscore_v2x.py
  │    (ViT-B/32 per-frame +  │
  │     per-video)            │
  └───────────────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │ 6. analyze_v2x_label_bias │   scripts/stage5/analyze_v2x_label_bias.py
  │    (paired tests, traj r) │
  └───────────────────────────┘
          │
          ▼
  ┌───────────────────────────┐
  │ 7. cross_dataset_table    │   scripts/stage5/build_cross_dataset_table.py
  │    (Stage A vs Stage C)   │
  └───────────────────────────┘
          │
          ▼
     REPORT.md + figures/
```

## One-command run

```bash
bash clipscore-experiment/scripts/stage5/run_stage5_pipeline.sh
```

Environment:
- `AI_GATEWAY_API_KEY` must be set (for Gemini describe step).
- Optional overrides: `SAMPLE_SIZE=10 SEED=42 FPS=10 PY=python3`.

## Outputs

| Path                                         | Contents |
|----------------------------------------------|----------|
| `manifest/v2x_proxy_candidates.csv`          | Sampled sequences + viewpoints |
| `videos/{video_id}.mp4`                      | Rendered sequences |
| `frames/{video_id}/frame_{00..07}.png`       | 8 keyframes per video |
| `prompts/scenario_descriptions.json`         | 3-way Gemini descriptions |
| `clipscore/per_video.csv`                    | video × label × mean score |
| `clipscore/per_frame.json`                   | video × label × per-frame scores |
| `per_video_pairs.csv`                        | paired t-test + Wilcoxon + Cohen's d |
| `frame_level/per_video_frame_tests.csv`      | per-video frame-level paired tests |
| `frame_level/trajectory_correlations.csv`    | Pearson r trajectories |
| `cross_dataset_comparison.csv`               | Crash-1500 vs V2X summary |
| `REPORT.md`                                  | Human-readable report |
| `figures/*.png`                              | Boxplot + mean trajectory |

## Notes
- Sample size is 10 videos × 3 labels = 30 paired comparisons (pilot; Stage 3 used 8 videos).
- Frame count default is 8 per video (vs 73 in Stage 3) because V2X sequences are only
  ~10-30 s at native 10 fps and per-sequence length is non-uniform; evenly-spaced
  sampling keeps the paired-test assumption intact.
- External baselines are not claimed; this is a protocol-replication study on a
  second dataset to test cross-corpus generalization of Stage A's bias findings.
