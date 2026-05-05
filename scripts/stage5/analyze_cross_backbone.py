#!/usr/bin/env python3
"""Stage 5 — analyse cross-backbone CLIPScore label-bias.

Reads per-backbone scoring outputs from
`outputs/stage5/clipscore/<bb_key>/{per_video.csv,per_frame.json}` and runs:

  1. Per-backbone label-wise summary (mean, std, n)
  2. Per-backbone paired tests (video-level) for every label pair:
       safe vs near-crash, safe vs crash, near-crash vs crash
     - paired t-test (p), Wilcoxon signed-rank (p), Cohen's d
  3. Per-backbone argmax pattern: per video, which label scores highest.
  4. Cross-backbone agreement: Pearson r on video-level mean scores,
     Spearman rank agreement on argmax.

Writes:
  outputs/stage5/analysis/per_backbone_summary.csv
  outputs/stage5/analysis/paired_tests.csv
  outputs/stage5/analysis/argmax_by_backbone.csv
  outputs/stage5/analysis/cross_backbone_agreement.csv
  outputs/stage5/analysis/box_per_backbone.png
  outputs/stage5/analysis/mean_trajectory_per_backbone.png
  outputs/stage5/analysis/argmax_heatmap.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

LABELS_DISP = ["safe", "near-crash", "crash"]
PAIRS = [("safe", "near-crash"), ("safe", "crash"), ("near-crash", "crash")]


def load_backbone(bb_dir: Path):
    pv_csv = bb_dir / "per_video.csv"
    pf_json = bb_dir / "per_frame.json"
    if not (pv_csv.exists() and pf_json.exists()):
        return None
    per_video = {}  # {video_id: {label: mean_score}}
    with pv_csv.open() as f:
        for row in csv.DictReader(f):
            per_video.setdefault(row["video_id"], {})[row["label"]] = float(row["score"])
    per_frame = json.loads(pf_json.read_text())
    return per_video, per_frame


def cohens_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    if diff.std(ddof=1) == 0:
        return 0.0
    return diff.mean() / diff.std(ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("outputs/stage5/clipscore"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/stage5/analysis"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    backbones = [p.name for p in args.input_dir.iterdir()
                 if p.is_dir() and (p / "per_video.csv").exists()]
    backbones.sort()
    if not backbones:
        raise SystemExit(f"no backbone results under {args.input_dir}")

    data = {}  # bb -> (per_video, per_frame)
    for bb in backbones:
        res = load_backbone(args.input_dir / bb)
        if res:
            data[bb] = res

    print(f"[xb-analyze] backbones: {backbones}")

    # 1) summary + 2) paired tests
    summary_rows = []
    paired_rows = []
    argmax_rows = []
    for bb, (per_video, per_frame) in data.items():
        videos = sorted(per_video.keys())
        vectors = {lbl: np.array([per_video[v][lbl] for v in videos]) for lbl in LABELS_DISP}
        for lbl in LABELS_DISP:
            v = vectors[lbl]
            summary_rows.append({
                "backbone": bb, "label": lbl, "n": len(v),
                "mean": f"{v.mean():.4f}", "std": f"{v.std(ddof=1):.4f}",
                "min": f"{v.min():.4f}", "max": f"{v.max():.4f}",
            })
        for a, b in PAIRS:
            va, vb = vectors[a], vectors[b]
            t_stat, t_p = stats.ttest_rel(va, vb)
            try:
                w_stat, w_p = stats.wilcoxon(va, vb)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            d = cohens_d_paired(va, vb)
            paired_rows.append({
                "backbone": bb, "pair": f"{a} vs {b}", "n": len(va),
                "mean_a": f"{va.mean():.4f}", "mean_b": f"{vb.mean():.4f}",
                "mean_diff": f"{(va-vb).mean():.4f}",
                "t": f"{t_stat:.4f}", "t_p": f"{t_p:.4e}",
                "w": f"{w_stat:.4f}", "w_p": f"{w_p:.4e}",
                "cohens_d": f"{d:.4f}",
            })
        # argmax per video
        for vid in videos:
            scores = {lbl: per_video[vid][lbl] for lbl in LABELS_DISP}
            winner = max(scores, key=scores.get)
            argmax_rows.append({"backbone": bb, "video_id": vid,
                                **{lbl: f"{scores[lbl]:.4f}" for lbl in LABELS_DISP},
                                "argmax": winner})

    # write CSVs
    def write_csv(name, rows):
        if not rows:
            return
        p = args.output_dir / name
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[xb-analyze] wrote {p}")

    write_csv("per_backbone_summary.csv", summary_rows)
    write_csv("paired_tests.csv", paired_rows)
    write_csv("argmax_by_backbone.csv", argmax_rows)

    # 4) cross-backbone agreement (pairwise Pearson on per-video mean delta,
    # and rank agreement on argmax)
    cross_rows = []
    bbs = list(data.keys())
    for i in range(len(bbs)):
        for j in range(i+1, len(bbs)):
            b1, b2 = bbs[i], bbs[j]
            pv1, pv2 = data[b1][0], data[b2][0]
            videos = sorted(set(pv1) & set(pv2))
            # concat all 3 labels per video into a long vector
            v1 = np.array([pv1[v][l] for v in videos for l in LABELS_DISP])
            v2 = np.array([pv2[v][l] for v in videos for l in LABELS_DISP])
            r, rp = stats.pearsonr(v1, v2)
            # argmax agreement
            agree = sum(max(pv1[v], key=pv1[v].get) == max(pv2[v], key=pv2[v].get)
                        for v in videos)
            cross_rows.append({
                "backbone_a": b1, "backbone_b": b2,
                "n_videos": len(videos),
                "pearson_r_scores": f"{r:.4f}", "p": f"{rp:.4e}",
                "argmax_agree": f"{agree}/{len(videos)}",
            })
    write_csv("cross_backbone_agreement.csv", cross_rows)

    # 5) plots
    # boxplot per backbone (side-by-side)
    fig, axes = plt.subplots(1, len(bbs), figsize=(4*len(bbs), 4), sharey=False)
    if len(bbs) == 1:
        axes = [axes]
    for ax, bb in zip(axes, bbs):
        pv = data[bb][0]
        videos = sorted(pv.keys())
        dat = [np.array([pv[v][lbl] for v in videos]) for lbl in LABELS_DISP]
        ax.boxplot(dat, labels=LABELS_DISP, showmeans=True)
        ax.set_title(bb)
        ax.set_ylabel("CLIPScore ×100")
    plt.tight_layout()
    plt.savefig(args.output_dir / "box_per_backbone.png", dpi=150)
    plt.close()

    # mean per-frame trajectory per backbone
    fig, axes = plt.subplots(1, len(bbs), figsize=(4*len(bbs), 4), sharey=False)
    if len(bbs) == 1:
        axes = [axes]
    for ax, bb in zip(axes, bbs):
        pf = data[bb][1]
        # videos × labels × K
        vids = sorted(pf.keys())
        K = len(next(iter(pf.values()))[LABELS_DISP[0]])
        for lbl in LABELS_DISP:
            mat = np.array([pf[v][lbl] for v in vids])  # (V, K)
            ax.plot(range(K), mat.mean(axis=0), label=lbl, marker="o")
        ax.set_title(bb)
        ax.set_xlabel("frame index")
        ax.set_ylabel("CLIPScore ×100")
        ax.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "mean_trajectory_per_backbone.png", dpi=150)
    plt.close()

    # argmax heatmap: backbones × labels → count of videos where label wins
    mat = np.zeros((len(bbs), len(LABELS_DISP)), dtype=int)
    for bi, bb in enumerate(bbs):
        pv = data[bb][0]
        for v in pv:
            w = max(pv[v], key=pv[v].get)
            mat[bi, LABELS_DISP.index(w)] += 1
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(LABELS_DISP)))
    ax.set_xticklabels(LABELS_DISP)
    ax.set_yticks(range(len(bbs)))
    ax.set_yticklabels(bbs)
    for bi in range(len(bbs)):
        for li in range(len(LABELS_DISP)):
            ax.text(li, bi, int(mat[bi, li]), ha="center", va="center",
                    color="white" if mat[bi, li] < mat.max()/2 else "black")
    ax.set_title("argmax count by backbone × label")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(args.output_dir / "argmax_heatmap.png", dpi=150)
    plt.close()

    print("[xb-analyze] done")


if __name__ == "__main__":
    main()
