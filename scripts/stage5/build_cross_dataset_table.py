#!/usr/bin/env python3
"""Assemble a cross-dataset comparison table: Crash-1500 Stage A vs V2X Stage C."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def load_stage3_group(stage3_dir: Path) -> dict:
    # stage3 writes group stats into group_stats.csv (mean/std/n per label)
    gp = stage3_dir / "group_stats.csv"
    if not gp.exists():
        return {}
    df = pd.read_csv(gp)
    return {row["label"]: row for _, row in df.iterrows()}


def load_stage3_paired(stage3_dir: Path) -> dict:
    p = stage3_dir / "paired_tests.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {row["pair"]: row for _, row in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-dir", required=True, type=Path)
    parser.add_argument("--stage5-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    stage3_group = load_stage3_group(args.stage3_dir)
    stage3_paired = load_stage3_paired(args.stage3_dir)

    stage5_scores = pd.read_csv(args.stage5_dir / "clipscore" / "per_video.csv")
    s5_group = stage5_scores.groupby("label")["score"].agg(["mean", "std", "count"]).reset_index()
    s5_paired = pd.read_csv(args.stage5_dir / "per_video_pairs.csv")

    rows = []
    for label in ["safe", "near-crash", "crash"]:
        g3 = stage3_group.get(label, {})
        g5 = s5_group[s5_group["label"] == label]
        rows.append(
            {
                "dataset": "Crash-1500 (Stage A)",
                "label": label,
                "n": g3.get("count", ""),
                "mean": g3.get("mean", ""),
                "std": g3.get("std", ""),
            }
        )
        if len(g5):
            row = g5.iloc[0]
            rows.append(
                {
                    "dataset": "V2X-Seq-SPD (Stage C)",
                    "label": label,
                    "n": int(row["count"]),
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"[cross] wrote {args.output}")


if __name__ == "__main__":
    main()
