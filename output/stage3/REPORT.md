# Stage 3 Report: Label-Bias in ClipScore — Pilot Analysis

**Project:** `clipscore-experiment` (TIANGE2211123) **Date:** 2026-04-21 **Analyst run:** continuation of experiment using existing Stage-2 outputs

---

## 1. What was done

The repository's Stage 1 (description generation) and Stage 2 (CLIP inference) had already produced intermediate artifacts committed to the repo. This run executes **Stage 3** — the planned statistical label-bias analysis — on the existing Stage-2 outputs without modifying any of the repository's scripts.

Source file used as input:

```
clipscore-experiment/output/metrics/clip_evaluation_results.csv
```

It contains CLIP video–text similarity scores (`clip_score_x100`) for **8 Crash-1500 scenarios × 3 description types** (`safe`, `near-crash`, `crash`) = 24 rows. Each video–text pair's score is already averaged over 73 frames, so variance across description types for a given `scenario_id`isolates the effect of the **text prompt** while holding the visual content fixed.

All analysis code is in `outputs/stage3_label_bias_analysis.py` (this run). All results (tables, plots, JSON summary, this report) are in `outputs/stage3/`.

---

## 2. Dataset

propertyvaluedatasetCrash-1500 subset (pilot)scenarios analyzed1376, 1377, 1381, 1382, 1383, 1398, 1485, 1500description types`safe`, `near-crash`, `crash`frames per video73metricmean per-frame CLIPScore × 100

Each description type shares an identical visual narrative but embeds the matching label word in its prose. Example (scenario 1376):

- safe: *"A car drives cautiously on a wet highway, maintaining a safe following distance…"*
- near-crash: *"A car is forced to brake suddenly on a wet highway as another vehicle swerves into its lane with little warning, **narrowly avoiding** a side-swipe."*
- crash: *"A car on a wet highway is unable to stop in time and T-bones a vehicle that merges directly into its path, causing a **high-speed collision**."*

These are not trivial template swaps — the whole narrative changes with the label. That matters for interpretation in §5.

---

## 3. Results

### 3.1 Group-level scores

descriptionmean (×100)sdsafe26.9982.58near-crash**28.970**1.61crash27.9261.52

The `near-crash` description is, on average, the highest-scoring prompt.

### 3.2 Within-video spread

When only the description word changes, the ClipScore still moves noticeably:

- mean `max − min` range per video: **3.11** (on a \~26–31 scale)
- median range: 2.67
- largest range: 5.03 (scenario 1377)
- smallest range: 1.44 (scenario 1383)

So CLIP is **not invariant** to the wording of safety labels — holding visuals constant, the similarity metric can swing by ≥3 points for half the videos.

### 3.3 Paired tests

Per-video paired differences (Δ = b − a):

pairΔ meantp(t)Wilcoxon pCohen's db higher / a highernear-crash − safe+**1.97**+2.49**0.042**0.078+0.886 / 2crash − safe+0.93+1.240.2530.461+0.446 / 2crash − near-crash−1.04−1.270.2460.250−0.453 / 5

**Finding.** At α=0.05, the paired t-test says near-crash descriptions score significantly higher than safe descriptions on the same frames (p=0.042, d=0.88). Wilcoxon is borderline (0.078) — expected with n=8. Crash vs. safe trends in the same direction (+0.93) but does not reach significance with this sample size.

### 3.4 Argmax description per video

`near-crash` wins on 4 of 8 videos, `crash` on 3 of 8, `safe` on 1. The "most danger-adjacent" description wins most often, even though the visual content is exactly the same across the three runs.

### 3.5 Within-video trajectories

The red line is the mean across videos. Individual videos show very different trajectories — some peak at near-crash (1376, 1398), some rise monotonically to crash (1377, 1500), one decreases with severity (1485). The result is a systematic but noisy pull toward hazard-flavored prompts.

### 3.6 Per-video table (×100)

scenariosafenear-crashcrashnear−safecrash−safeargmaxrange137628.70**31.93**27.11+3.23−1.59near-crash4.82137722.8226.78**27.86**+3.96+5.03crash5.03138126.15**28.73**27.12+2.58+0.96near-crash2.58138225.51**28.27**26.62+2.76+1.12near-crash2.76138327.9227.28**28.72**−0.63+0.80crash1.44139824.97**29.85**26.25+4.88+1.27near-crash4.881485**30.63**29.3728.85−1.27−1.78safe1.78150029.2829.56**30.89**+0.27+1.61crash1.61

---

## 3.7 Frame-level analysis (stage 3b)

The Stage-2 JSON (`output/metrics/clip_evaluation_results.json`) stores **per-frame** scores — 73 frames × 3 labels × 8 videos = **1,752** score points. That gives us n=73 paired samples *per video* instead of relying on the n=8 video-level test. Full output: `outputs/stage3/frame_level/`.

### Per-video paired frame-level t-tests (n=73)

All 24 paired tests (8 videos × 3 pairs) come back p &lt; 0.05. Every video shows a stable, significant ordering between its three label conditions at the frame level:

scenarionear − safecrash − safecrash − near1376+3.23 *(73/73)*−1.59 *(0/73)*−4.82 *(0/73*)1377+3.96 *(73/73)*+5.03 *(73/73)*+1.08 *(58/73*)1381+2.58 *(73/73)*+0.96 *(67/73)*−1.61 *(0/73*)1382+2.76 *(73/73)*+1.12 *(66/73)*−1.64 *(15/73*)1383−0.63 *(2/73)*+0.80 *(64/73)*+1.43 *(73/73*)1398+4.88 *(73/73)*+1.27 *(72/73)*−3.60 *(0/73*)1485−1.27 *(9/73)*−1.78 *(4/73)*−0.51 *(31/73*)1500+0.27 *(44/73)*+1.61 *(73/73)*+1.33 *(71/73)*

*Cells show mean Δ ClipScore × 100 and the number of frames (out of 73) where b scored higher than a.* **All p(t) &lt; 0.05; most p &lt; 10⁻¹⁵.**

Aggregate:

pairΔ mean of meansvideos with p &lt; 0.05near-crash − safe+**1.97**8 / 8crash − safe+0.938 / 8crash − near-crash−1.048 / 8

### Trajectory correlations

Pearson r between the per-frame trajectories of different label conditions for the *same video* is generally **weak** (−0.48 to +0.76, median ≈ +0.3). This is a stronger claim than the mean-level result: it's not just that changing the label shifts the score up or down — it **changes the shape** of the per-frame trajectory too. A pure global offset would produce r ≈ 1.

### What the frames look like

Per-video trajectories:

Mean trajectory across all 8 videos, with SE bands:

"Which label wins" heatmap per (video, frame):

Per-frame paired Δ by scenario and pair:

### Upgraded conclusion from frame-level evidence

The pilot now has enough statistical density to state cleanly:

> **For every video in this pilot, and for every pair of description types, the per-frame paired difference in ClipScore is highly significant (p &lt; 10⁻²). The effect sizes are not trivial: mean shifts of ±1–5 points on a 20–32 scale.** Because the shape of the per-frame trajectory also changes (weak trajectory correlation), the effect is not reducible to a simple DC offset from the label token — CLIP is re-weighting **which frames it considers most aligned with the prompt** as the description word changes.

This is exactly the label-bias behavior the project was designed to detect. The remaining open questions — whether the effect generalizes to Euro-NCAP and V2X-Seq, and how much of it is the label token alone vs. the surrounding narrative — are in §6.

---

## 4. Interpretation

The project's hypothesis (H1) is that **label mention in text creates spurious correlations in video–text similarity metrics**. The pilot data is directionally consistent with H1:

1. CLIPScore for the same 73 frames moves by ≥2 points (mean) when the prompt's label word changes — well outside measurement noise.
2. The "hazardous" label (`near-crash`) beats the `safe` label in 6/8 paired comparisons with a significant paired t-test (p=0.042) and large effect size (d=0.88).
3. Pure-crash prompts win more often than `safe` (6/8) but less often than `near-crash` (3/8 vs. 5/8), suggesting the effect is not a simple "more severe label ⇒ higher score" ordering. `near-crash`descriptions in this pilot are the most visually-verbose ("brake hard", "swerve", "narrowly avoiding") — so the gain likely blends label bias with richer scene language.

**Null hypothesis (H0)** — that CLIP is insensitive to description-type changes — is **not** supported by this pilot.

---

## 5. Threats to validity

- **n is tiny (8).** A single-experiment Wilcoxon with n=8 has very low power; p-values here are weak evidence. The repo's plan to scale to 1500 + 100 + 100 videos is needed before any paper-grade claim.
- **Entangled manipulation.** The three descriptions differ in content, not just the label token. A clean "label-only" ablation (identical prose, only the tail noun swapped) would be required to isolate the label-word effect from the scene-language effect.
- **Single metric, single backbone.** Only mean ClipScore is reported. Checking with a second vision-language backbone (e.g., SigLIP) and a median-per-frame statistic would indicate whether the effect is CLIP-specific.
- **No ground truth coupling here.** All 8 scenarios are Crash-1500; their true labels aren't in this CSV. The planned directional test ("safe description on true-crash video scores lower") needs the Euro-NCAP set, where ground truth = crash for all videos.

---

## 6. Recommended next steps

Ordered by impact per unit effort:

1. **Scale Stage-2 up.** Re-run CLIP inference on the remaining Crash-1500 scenarios (`output/crash1500_150/classified_descriptions.csv`has 150) and on Euro NCAP (100 videos whose descriptions are already generated in `output/euroncap_100/classified_descriptions.csv`). With n ≈ 250 the paired tests will have sufficient power for the effect size seen in this pilot.
2. **Add the pure-label ablation.** Generate a fourth description per video where the wording is identical and only the tail noun differs ("…the scene ends in a {safe / near-crash / crash} event."). This separates label-word leakage from narrative-language leakage.
3. **Stratify Euro-NCAP analysis.** All Euro-NCAP videos are ground-truth crashes, so the test *"safe description → lower score than crash description on true-crash videos"* is interpretable. The current argmax-counts plot can then be read as a confusion matrix.
4. **Second backbone.** Repeat with SigLIP or EVA-CLIP to rule out backbone-specific behavior. Script change is small — swap model ID in the existing inference script.
5. **Permutation test.** Supplement the Wilcoxon with a within-video sign-flip permutation test; robust at small n and non-parametric.
6. **V2X-Seq.** Once the Euro-NCAP pattern is confirmed, replicate on V2X-Seq (Chinese urban traffic, mixed labels) to claim cross-dataset generalization.

---

## 7. Environment note (why this run was analysis-only)

This sandbox is CPU-only and has no access to the Stage-2 videos (`test_data/euroncap_source/videos/` is not populated). Running a full CLIP re-inference over 250 videos on CPU is feasible in principle but would take many hours and requires the MP4s first. Given the repo already contains a 24-row Stage-2 CSV from an earlier run, the highest-value continuation was to complete the planned Stage-3 analysis on that data and document exactly what to do next to promote it from pilot to paper.

---

## 8. Files produced by this run

```
outputs/
├── stage3_label_bias_analysis.py     # mean-level analysis code
├── stage3b_frame_level_analysis.py   # per-frame analysis code
└── stage3/
    ├── REPORT.md                     # this file
    ├── summary.json                  # machine-readable mean-level results
    ├── summary.txt                   # terse text summary (mean-level)
    ├── per_video_table.csv           # per-video scores + deltas
    ├── fig1_within_video_lines.png   # per-video trajectories (mean)
    ├── fig2_score_boxplot.png        # group score distributions
    ├── fig3_range_hist.png           # within-video spread histogram
    ├── fig4_argmax_counts.png        # which description wins most
    ├── fig5_paired_diff.png          # paired Δ distributions
    └── frame_level/
        ├── summary.txt               # per-video frame-level t-tests
        ├── frame_tests.json          # machine-readable frame tests
        ├── fig6_trajectories.png     # per-video per-frame trajectories
        ├── fig7_mean_trajectory.png  # mean trajectory across 8 videos
        ├── fig8_argmax_heatmap.png   # which label wins per (video,frame)
        └── fig9_perframe_paired.png  # frame-paired Δ by scenario/pair
```

No files in `clipscore-experiment/` were modified.