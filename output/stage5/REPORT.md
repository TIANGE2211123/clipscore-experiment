# Stage 5 — Cross-Backbone CLIPScore Label-Bias Replication

## 1. Motivation

Stage A (Crash-1500 / Stage 3) and Stage B (Euro NCAP / Stage 4) reported a directional label bias in CLIPScore: paired with descriptions that explicitly name the severity (`safe` / `near-crash` / `crash`), the same video frames receive systematically different scores — *even when content is unchanged*.

All prior results used **OpenAI CLIP ViT-B/32**. A natural referee question is:

> Is the reported bias an artefact of this specific backbone — its architecture (ViT-B/32), its training corpus (WIT-400M), or its contrastive-softmax objective? Would a larger OpenAI CLIP or a differently trained vision–language encoder (e.g. Google SigLIP) show the same pattern?

Stage 5 is a **cross-backbone replication** using the exact Stage-4 Euro NCAP keyframes (10 videos × 8 frames) and the exact Gemini-generated 3-way descriptions. Only the scoring encoder varies.

## 2. Setup

BackboneHF idParamsTraining dataObjectiveCLIP ViT-B/32 (OpenAI)`openai/clip-vit-base-patch32`151MWIT-400MInfoNCECLIP ViT-L/14 (OpenAI)`openai/clip-vit-large-patch14`428MWIT-400MInfoNCESigLIP base/16 (Google)`google/siglip-base-patch16-224`203MWebLI-10BSigmoid pairwise

Inputs (identical for all three backbones):

- `outputs/stage4/frames/<yt_id>/frame_{00..07}.jpg` — 10 Euro NCAP videos × 8 evenly-spaced keyframes.
- `outputs/stage4/prompts/scenario_descriptions.json` — Gemini-3.1-pro-preview 3-way descriptions (safe / near_crash / crash) per video.

For each (backbone, video, label) tuple we compute the cosine similarity between the L2-normalised text embedding and each of the 8 frame embeddings, scaled ×100. Per-video score is the frame mean. Sample size is n=10 paired videos per backbone.

Implementation: `clipscore-experiment/scripts/stage5/cross_backbone_scoring.py`and `analyze_cross_backbone.py`.

## 3. Per-Backbone Summary

`outputs/stage5/analysis/per_backbone_summary.csv`

backbonesafe meannear-crash meancrash meanorderingCLIP ViT-B/3219.8620.82**21.86**safe &lt; near &lt; crashCLIP ViT-L/1417.8118.18**19.61**safe &lt; near &lt; crashSigLIP base/16-**3.25**-4.47-3.63near &lt; crash &lt; safe

Both OpenAI CLIP models preserve the Stage A / Stage B ordering **crash &gt; near-crash &gt; safe**. SigLIP inverts it: the `safe` description receives the highest mean score and `near-crash` the lowest.

## 4. Paired Hypothesis Tests

`outputs/stage5/analysis/paired_tests.csv` — within-backbone, paired across the 10 videos (same frames, only the text label changes).

backbonepairmean diffpaired-t pWilcoxon pCohen's dCLIP ViT-B/32safe vs crash-2.000.1150.193-0.55CLIP ViT-B/32safe vs near-crash-0.960.2530.375-0.39CLIP ViT-B/32near-crash vs crash-1.040.1880.375-0.45CLIP ViT-L/14safe vs crash-**1.800.0250.027-0.85**CLIP ViT-L/14near-crash vs crash-1.430.0740.049-0.64CLIP ViT-L/14safe vs near-crash-0.380.4320.557-0.26SigLIP base/16safe vs near-crash+1.220.1140.232+0.55SigLIP base/16safe vs crash+0.380.6920.922+0.13SigLIP base/16near-crash vs crash-0.840.2680.131-0.37

**Findings:**

- ViT-L/14 reaches statistical significance on `safe vs crash` (t-test p=0.025, Wilcoxon p=0.027, **d = -0.85** — large effect).
- ViT-B/32 shows the same direction with medium effects (d ≈ -0.5 for safe-vs-crash) but n=10 is too small to reject the null at α=0.05. This is consistent with the Stage A finding at scale (n=1500) where the same effect was strongly significant.
- SigLIP shows **no significant pairwise contrast**, and the effect-size sign actively disagrees with CLIP on two out of three pairs (`safe` ends up *above* `near-crash` / `crash`).

## 5. Cross-Backbone Agreement

`outputs/stage5/analysis/cross_backbone_agreement.csv`

backbone Abackbone BPearson r (video×label scores)pargmax agreementCLIP ViT-B/32CLIP ViT-L/14+**0.728**&lt;1e-57/10CLIP ViT-B/32SigLIP base/16-0.0090.9622/10CLIP ViT-L/14SigLIP base/16-0.0500.7935/10

The two OpenAI CLIP models are strongly correlated on per-(video,label) scores and agree on 70% of per-video argmax labels. Either CLIP vs SigLIP: Pearson r ≈ 0, argmax agreement at or below chance (chance = 33% for 3 labels).

## 6. Interpretation

1. **Label bias is robust within the OpenAI CLIP family, across scale**.ViT-B/32 and ViT-L/14 produce essentially the same directional ranking (`crash > near-crash > safe`), and the larger model sharpens the effect enough to clear α=0.05 on n=10 with d=-0.85. This is consistent with the Stage A result at n=1500 on Crash-1500 using the same encoder family.

2. **The bias does not generalise to SigLIP.** SigLIP was trained on a different corpus (WebLI-10B vs WIT-400M), optimises a pairwise-sigmoid loss instead of InfoNCE, and produces embeddings that sit in a different part of cosine space (notably yielding negative similarities for many frames in this set). Its per-video label ordering actively differs from CLIP's, and its cross-backbone correlation with either CLIP model is statistically indistinguishable from zero.

3. **Methodological implication for the thesis.** "CLIPScore label bias" should be reported as a property of *OpenAI-CLIP-style* encoders rather than as a universal property of contrastive V–L models. Any downstream claim that `crash` descriptions spuriously inflate similarity must be scoped to the encoder used. Conversely, SigLIP's near-zero correlation with CLIP on these inputs suggests that if label-bias robustness is desired in a downstream metric, ensembling over encoder families — not model scale — is the right axis to vary.

## 7. Caveats

- n = 10 videos. Within-backbone paired tests have limited power; ViT-B/32 not reaching α=0.05 at this sample size is expected and consistent with Stage A n=1500.
- SigLIP's sigmoid-loss cosine output has a different distribution shape; raw-score comparison across backbones is meaningful only directionally (ranking), not in absolute units.
- A single language-model prompter (Gemini-3.1-pro-preview) produced all descriptions. Prompt-family effects are held constant by design across backbones, but not varied.

## 8. Artefacts

```
outputs/stage5/
├── REPORT.md                                 (this file)
├── clipscore/
│   ├── clip_vitb32/per_video.csv, per_frame.json
│   ├── clip_vitl14/per_video.csv, per_frame.json
│   ├── siglip_b16/per_video.csv, per_frame.json
│   └── cross_backbone_summary.csv
└── analysis/
    ├── per_backbone_summary.csv
    ├── paired_tests.csv
    ├── argmax_by_backbone.csv
    ├── cross_backbone_agreement.csv
    ├── box_per_backbone.png
    ├── mean_trajectory_per_backbone.png
    └── argmax_heatmap.png
```

Scripts:

```
clipscore-experiment/scripts/stage5/
├── cross_backbone_scoring.py       (3-backbone cosine scoring)
└── analyze_cross_backbone.py       (paired stats + agreement + plots)
```