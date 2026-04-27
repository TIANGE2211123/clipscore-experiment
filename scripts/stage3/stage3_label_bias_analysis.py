"""
Stage 3: Label-Bias Statistical Analysis
=========================================
Consumes the Stage-2 ClipScore outputs already produced by the repository's
inference pipeline (output/metrics/clip_evaluation_results.csv) and computes:

  * Per-video score distributions across the three description types
    (safe / near-crash / crash).
  * Within-video variance (how much a score moves when ONLY the label
    word changes).
  * Paired statistical tests between description types (Wilcoxon signed-rank
    and paired t-test).
  * Directional bias: which description type wins per video, and how often.
  * "Argmax-matches-label" contingency analysis for the label-bias hypothesis.

Outputs are written to ./outputs/stage3/.

This script does NOT modify any files in clipscore-experiment/. It only reads
the Stage-2 CSV and writes new analysis artifacts.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace/clipscore-experiment")
OUT = Path("/home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace/outputs/stage3")
OUT.mkdir(parents=True, exist_ok=True)

SCORES_CSV = REPO / "output" / "metrics" / "clip_evaluation_results.csv"

LABELS = ["safe", "near-crash", "crash"]
LABEL_COLORS = {"safe": "#2a9d8f", "near-crash": "#e9c46a", "crash": "#e76f51"}


def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_CSV)
    # keep the columns we need and enforce ordering
    df = df[["scenario_id", "video_type", "clip_score_x100",
             "clip_std_x100", "temporal_consistency", "num_frames"]]
    df["video_type"] = df["video_type"].str.strip()
    return df


def pivot_by_video(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="scenario_id",
                          columns="video_type",
                          values="clip_score_x100",
                          aggfunc="first")
    wide = wide.reindex(columns=LABELS)
    wide = wide.dropna()
    return wide


def within_video_variance(wide: pd.DataFrame) -> dict:
    """Spread of the three scores per video — measures how much the score
    moves when ONLY the description changes (video frames are identical)."""
    ranges = wide.max(axis=1) - wide.min(axis=1)
    sds = wide.std(axis=1, ddof=1)
    return {
        "n_videos": int(len(wide)),
        "mean_range": float(ranges.mean()),
        "median_range": float(ranges.median()),
        "max_range": float(ranges.max()),
        "min_range": float(ranges.min()),
        "mean_sd": float(sds.mean()),
        "median_sd": float(sds.median()),
    }


def paired_tests(wide: pd.DataFrame) -> list[dict]:
    results = []
    pairs = [("safe", "near-crash"),
             ("safe", "crash"),
             ("near-crash", "crash")]
    for a, b in pairs:
        x = wide[a].values
        y = wide[b].values
        diff = y - x  # positive => b scores higher than a
        t_stat, t_p = stats.ttest_rel(y, x)
        try:
            w_stat, w_p = stats.wilcoxon(y, x)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        # Cohen's d for paired samples
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
        results.append({
            "pair": f"{b} - {a}",
            "n": int(len(diff)),
            "mean_diff": float(diff.mean()),
            "sd_diff": float(diff.std(ddof=1)),
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "cohens_d": float(d) if not math.isnan(d) else None,
            "b_higher_count": int((diff > 0).sum()),
            "a_higher_count": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
        })
    return results


def argmax_label(wide: pd.DataFrame) -> dict:
    """Which description type wins per video?"""
    winners = wide.idxmax(axis=1)
    counts = winners.value_counts().reindex(LABELS).fillna(0).astype(int)
    return counts.to_dict()


def directional_bias_on_crash_videos(wide: pd.DataFrame) -> dict:
    """
    Crash-1500 ground truth for these scenarios in the pilot is NOT encoded
    in the CSV (only the description type is). The repository's README and
    CLAUDE.md note that Crash-1500 videos are a mix but the Euro NCAP set
    is all true crashes. Here we simply compute, per video, whether the
    'crash' description scores higher than the 'safe' description — a proxy
    for whether the label word alone pulls the similarity in the matching
    direction.
    """
    crash_beats_safe = (wide["crash"] > wide["safe"]).sum()
    near_beats_safe = (wide["near-crash"] > wide["safe"]).sum()
    crash_beats_near = (wide["crash"] > wide["near-crash"]).sum()
    n = len(wide)
    return {
        "n": int(n),
        "crash>safe": int(crash_beats_safe),
        "near>safe": int(near_beats_safe),
        "crash>near": int(crash_beats_near),
        "pct_crash_over_safe": round(100 * crash_beats_safe / n, 1) if n else 0,
        "pct_near_over_safe": round(100 * near_beats_safe / n, 1) if n else 0,
        "pct_crash_over_near": round(100 * crash_beats_near / n, 1) if n else 0,
    }


# ---------- plots ----------

def plot_within_video_lines(wide: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(LABELS))
    for scenario_id, row in wide.iterrows():
        ax.plot(xs, row.values, color="#264653", alpha=0.35, linewidth=1)
    ax.plot(xs, wide.mean(axis=0).values, color="#d62828",
            linewidth=2.5, marker="o", label="mean")
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("ClipScore × 100")
    ax.set_title("Within-video ClipScore across description types\n"
                 f"(same video, label word in description changes; n={len(wide)})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_score_distributions(wide: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [wide[label].values for label in LABELS]
    bp = ax.boxplot(data, labels=LABELS, patch_artist=True, widths=0.55)
    for patch, label in zip(bp["boxes"], LABELS):
        patch.set_facecolor(LABEL_COLORS[label])
        patch.set_alpha(0.7)
    for i, label in enumerate(LABELS, start=1):
        ys = wide[label].values
        xs = np.random.normal(i, 0.05, size=len(ys))
        ax.scatter(xs, ys, alpha=0.5, color="#264653", s=18)
    ax.set_ylabel("ClipScore × 100")
    ax.set_title("Distribution of ClipScore by description type")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_within_video_range_hist(wide: pd.DataFrame, path: Path) -> None:
    ranges = wide.max(axis=1) - wide.min(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(ranges.values, bins=10, color="#457b9d", edgecolor="white")
    ax.axvline(ranges.mean(), color="#e63946", linewidth=2,
               label=f"mean = {ranges.mean():.2f}")
    ax.set_xlabel("max(ClipScore) − min(ClipScore) across description types")
    ax.set_ylabel("# videos")
    ax.set_title("Within-video ClipScore spread\n"
                 "(score change when ONLY the description word changes)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_argmax_counts(counts: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = LABELS
    values = [counts.get(l, 0) for l in labels]
    bars = ax.bar(labels, values, color=[LABEL_COLORS[l] for l in labels],
                  edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("# videos where this description wins")
    ax.set_title("Argmax description type per video")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pairwise_diff(wide: pd.DataFrame, path: Path) -> None:
    pairs = [("crash", "safe"), ("near-crash", "safe"), ("crash", "near-crash")]
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for a, b in pairs:
        data.append((wide[a] - wide[b]).values)
        labels.append(f"{a} − {b}")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a8dadc")
        patch.set_alpha(0.8)
    ax.axhline(0, color="#1d3557", linestyle="--", linewidth=1)
    for i, diff in enumerate(data, start=1):
        xs = np.random.normal(i, 0.05, size=len(diff))
        ax.scatter(xs, diff, alpha=0.6, color="#1d3557", s=18)
    ax.set_ylabel("ΔClipScore × 100 (paired)")
    ax.set_title("Paired ClipScore differences")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- driver ----------

def main() -> None:
    print(f"[stage3] loading: {SCORES_CSV}")
    df = load_scores()
    wide = pivot_by_video(df)
    print(f"[stage3] n videos with all 3 description types: {len(wide)}")

    wv = within_video_variance(wide)
    paired = paired_tests(wide)
    argmax = argmax_label(wide)
    directional = directional_bias_on_crash_videos(wide)

    # --- long-form table with per-video deltas ---
    long_rows = []
    for scenario_id, row in wide.iterrows():
        base = row["safe"]
        long_rows.append({
            "scenario_id": int(scenario_id),
            "safe": round(float(row["safe"]), 3),
            "near_crash": round(float(row["near-crash"]), 3),
            "crash": round(float(row["crash"]), 3),
            "near_minus_safe": round(float(row["near-crash"] - base), 3),
            "crash_minus_safe": round(float(row["crash"] - base), 3),
            "argmax": wide.loc[scenario_id].idxmax(),
            "range": round(float(row.max() - row.min()), 3),
        })
    per_video_df = pd.DataFrame(long_rows)
    per_video_df.to_csv(OUT / "per_video_table.csv", index=False)

    # --- plots ---
    plot_within_video_lines(wide, OUT / "fig1_within_video_lines.png")
    plot_score_distributions(wide, OUT / "fig2_score_boxplot.png")
    plot_within_video_range_hist(wide, OUT / "fig3_range_hist.png")
    plot_argmax_counts(argmax, OUT / "fig4_argmax_counts.png")
    plot_pairwise_diff(wide, OUT / "fig5_paired_diff.png")

    # --- summary json ---
    summary = {
        "source_csv": str(SCORES_CSV),
        "n_videos_analyzed": int(len(wide)),
        "labels": LABELS,
        "within_video_variance": wv,
        "paired_tests": paired,
        "argmax_counts": argmax,
        "directional_proxy": directional,
        "group_means_x100": {l: float(wide[l].mean()) for l in LABELS},
        "group_sds_x100": {l: float(wide[l].std(ddof=1)) for l in LABELS},
    }
    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- concise text summary ---
    lines = []
    lines.append("STAGE-3 LABEL-BIAS ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"source: {SCORES_CSV}")
    lines.append(f"n videos analyzed (all 3 description types present): "
                 f"{len(wide)}")
    lines.append("")
    lines.append("Group means (ClipScore × 100):")
    for l in LABELS:
        lines.append(f"  {l:<12s} mean={wide[l].mean():6.3f}  "
                     f"sd={wide[l].std(ddof=1):5.3f}")
    lines.append("")
    lines.append("Within-video variance (same frames, only label word changes):")
    lines.append(f"  mean max-min range = {wv['mean_range']:.3f}")
    lines.append(f"  median max-min     = {wv['median_range']:.3f}")
    lines.append(f"  mean per-video sd  = {wv['mean_sd']:.3f}")
    lines.append("")
    lines.append("Paired tests (Δ = b − a on SAME video):")
    for r in paired:
        lines.append(f"  {r['pair']:<28s} "
                     f"Δmean={r['mean_diff']:+.3f}  "
                     f"t={r['t_stat']:+.3f}  p(t)={r['t_p']:.4f}  "
                     f"Wp={r['wilcoxon_p']:.4f}  d={r['cohens_d']:+.3f}  "
                     f"wins b/a/ties={r['b_higher_count']}/"
                     f"{r['a_higher_count']}/{r['ties']}")
    lines.append("")
    lines.append("Argmax description per video:")
    for l in LABELS:
        lines.append(f"  {l:<12s} wins {argmax.get(l,0)} of {len(wide)}")
    lines.append("")
    lines.append("Directional proxy:")
    for k, v in directional.items():
        lines.append(f"  {k}: {v}")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
