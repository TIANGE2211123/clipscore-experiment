#!/usr/bin/env python3
"""Stage 5 — paired statistical analysis for V2X CLIPScores.

Mirrors Stage 3's protocol:
  (a) video-level paired t-test + Wilcoxon between the 3 label pairs
  (b) frame-level paired tests per video
  (c) trajectory Pearson correlations between label pairs
  (d) figures: boxplot, paired-diff, mean trajectory, argmax heatmap
Writes REPORT.md tables + per_video_pairs.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LABELS = ["safe", "near-crash", "crash"]
COLORS = {"safe": "#2a9d8f", "near-crash": "#e9c46a", "crash": "#e76f51"}


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))


def video_level_tests(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="video_id", columns="label", values="score").astype(float)
    rows = []
    for a, b in combinations(LABELS, 2):
        x = pivot[a].values
        y = pivot[b].values
        t, pt = stats.ttest_rel(x, y)
        w, pw = stats.wilcoxon(x, y) if len(x) >= 6 else (np.nan, np.nan)
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "n": len(x),
                "mean_diff": float(np.mean(x - y)),
                "t": float(t),
                "p_t": float(pt),
                "W": float(w) if not np.isnan(w) else "",
                "p_wilcoxon": float(pw) if not np.isnan(pw) else "",
                "cohen_d": cohen_d_paired(x, y),
            }
        )
    return pd.DataFrame(rows)


def frame_level_tests(per_frame: dict) -> pd.DataFrame:
    rows = []
    for vid, labelmap in per_frame.items():
        for a, b in combinations(LABELS, 2):
            x = np.array(labelmap[a], dtype=float)
            y = np.array(labelmap[b], dtype=float)
            if len(x) < 4:
                continue
            t, pt = stats.ttest_rel(x, y)
            rows.append(
                {
                    "video_id": vid,
                    "pair": f"{a} vs {b}",
                    "n_frames": len(x),
                    "mean_diff": float(np.mean(x - y)),
                    "t": float(t),
                    "p_t": float(pt),
                    "reject_at_0.05": bool(pt < 0.05),
                }
            )
    return pd.DataFrame(rows)


def trajectory_correlations(per_frame: dict) -> pd.DataFrame:
    rows = []
    for vid, labelmap in per_frame.items():
        for a, b in combinations(LABELS, 2):
            x = np.array(labelmap[a], dtype=float)
            y = np.array(labelmap[b], dtype=float)
            if len(x) < 4:
                continue
            r, p = stats.pearsonr(x, y)
            rows.append(
                {"video_id": vid, "pair": f"{a} vs {b}", "pearson_r": float(r), "p": float(p)}
            )
    return pd.DataFrame(rows)


def plot_boxplot(df: pd.DataFrame, out: Path) -> None:
    pivot = df.pivot(index="video_id", columns="label", values="score").astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    data = [pivot[k].values for k in LABELS]
    bp = ax.boxplot(data, labels=LABELS, patch_artist=True)
    for patch, k in zip(bp["boxes"], LABELS):
        patch.set_facecolor(COLORS[k])
        patch.set_alpha(0.6)
    ax.set_ylabel("CLIPScore (per video)")
    ax.set_title("Stage 5 — V2X-Seq-SPD: CLIPScore by label")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mean_trajectory(per_frame: dict, out: Path) -> None:
    # align by frame index (take min length across videos)
    min_k = min(len(next(iter(m.values()))) for m in per_frame.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in LABELS:
        stack = np.stack([np.array(m[label][:min_k]) for m in per_frame.values()])
        mean = stack.mean(axis=0)
        ax.plot(mean, label=label, color=COLORS[label], linewidth=2)
    ax.set_xlabel("frame index")
    ax.set_ylabel("mean CLIPScore")
    ax.set_title("Stage 5 — mean frame-level trajectory (V2X)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_report(
    video_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    group_stats: pd.DataFrame,
    out: Path,
) -> None:
    lines = [
        "# Stage 5 — V2X-Seq-SPD Label-Bias Audit Report",
        "",
        "## 1. Group statistics (video-level)",
        "",
        group_stats.to_markdown(index=False),
        "",
        "## 2. Paired tests (video-level)",
        "",
        video_df.to_markdown(index=False),
        "",
        "## 3. Frame-level paired tests (per video × pair)",
        "",
        f"Total rows: {len(frame_df)}; rejected at α=0.05: "
        f"{int(frame_df['reject_at_0.05'].sum()) if len(frame_df) else 0}",
        "",
        frame_df.to_markdown(index=False) if len(frame_df) else "(no frame-level rows)",
        "",
        "## 4. Trajectory correlations (Pearson r per video × pair)",
        "",
        traj_df.to_markdown(index=False) if len(traj_df) else "(no trajectory rows)",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", required=True, type=Path)
    parser.add_argument("--scores-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.scores_csv)
    per_frame = json.loads(args.scores_json.read_text(encoding="utf-8"))

    group_stats = (
        df.groupby("label")["score"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
        .reset_index()
    )

    video_df = video_level_tests(df)
    frame_df = frame_level_tests(per_frame)
    traj_df = trajectory_correlations(per_frame)

    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    plot_boxplot(df, args.output_dir / "figures" / "v2x_score_boxplot.png")
    if per_frame:
        plot_mean_trajectory(per_frame, args.output_dir / "figures" / "v2x_mean_trajectory.png")

    video_df.to_csv(args.output_dir / "per_video_pairs.csv", index=False)
    frame_df.to_csv(args.output_dir / "frame_level" / "per_video_frame_tests.csv", index=False)
    traj_df.to_csv(args.output_dir / "frame_level" / "trajectory_correlations.csv", index=False)

    write_report(video_df, frame_df, traj_df, group_stats, args.output_dir / "REPORT.md")
    print(f"[stats] wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
