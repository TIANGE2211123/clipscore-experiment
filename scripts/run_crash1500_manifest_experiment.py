#!/usr/bin/env python3
"""Run CLIPScore experiments for the official 150-video Crash-1500 subset."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/Lenovo/Desktop/学业/实验ClipScore")
sys.path.insert(0, str(ROOT / "code"))

from video_clip_evaluator import ImprovedVideoCLIPEvaluator  # noqa: E402


MANIFEST_PATH = Path("/Users/Lenovo/Desktop/学业/Crash1500_official/crash1500_subset_150_manifest.csv")
VIDEO_DIR = Path("/Users/Lenovo/Desktop/学业/Crash1500_official/subset_150_videos")
OUTPUT_DIR = ROOT / "output" / "crash1500_150"


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> str:
    clip_scores = [row["clip_score_x100"] for row in rows]
    temporal_scores = [row["temporal_consistency"] for row in rows]

    by_timing = defaultdict(list)
    by_weather = defaultdict(list)
    by_ego = defaultdict(list)
    for row in rows:
        by_timing[row["timing"]].append(row["clip_score_x100"])
        by_weather[row["weather"]].append(row["clip_score_x100"])
        by_ego[row["ego_involve"]].append(row["clip_score_x100"])

    lines = [
        "# Crash-1500 150-Video Experiment Summary",
        "",
        f"- Videos evaluated: `{len(rows)}`",
        f"- Mean CLIPScore: `{sum(clip_scores) / len(clip_scores):.3f}`",
        f"- Mean Temporal Consistency: `{sum(temporal_scores) / len(temporal_scores):.4f}`",
        "",
        "## CLIPScore by Timing",
        "",
    ]
    for key, values in sorted(by_timing.items()):
        lines.append(f"- {key}: {sum(values) / len(values):.3f}")
    lines.extend(["", "## CLIPScore by Weather", ""])
    for key, values in sorted(by_weather.items()):
        lines.append(f"- {key}: {sum(values) / len(values):.3f}")
    lines.extend(["", "## CLIPScore by Ego Involvement", ""])
    for key, values in sorted(by_ego.items()):
        lines.append(f"- {key}: {sum(values) / len(values):.3f}")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest()

    evaluator = ImprovedVideoCLIPEvaluator(
        device="cpu",
        model_name="openai/clip-vit-base-patch32",
        local_files_only=True,
    )

    results = []
    for row in manifest_rows:
        video_path = VIDEO_DIR / f"{row['video_id']}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing extracted video: {video_path}")

        clip_result = evaluator.calculate_clip_score(
            str(video_path),
            row["prompt"],
            max_frames=50,
            frame_interval=1,
        )
        temporal_result = evaluator.calculate_temporal_consistency(
            str(video_path),
            max_frames=50,
            frame_interval=1,
        )
        if not clip_result:
            continue

        results.append(
            {
                "video_id": row["video_id"],
                "youtube_id": row["youtube_id"],
                "timing": row["timing"],
                "weather": row["weather"],
                "ego_involve": row["ego_involve"],
                "accident_frame": int(row["accident_frame"]),
                "prompt": row["prompt"],
                "clip_score_x100": round(clip_result["mean_score"] * 100, 3),
                "clip_std_x100": round(clip_result["std_score"] * 100, 3),
                "clip_min_x100": round(clip_result["min_score"] * 100, 3),
                "clip_max_x100": round(clip_result["max_score"] * 100, 3),
                "temporal_consistency": round(
                    temporal_result["mean_temporal_score"], 4
                )
                if temporal_result and temporal_result["mean_temporal_score"] is not None
                else None,
                "temporal_std": round(temporal_result["std_temporal_score"], 4)
                if temporal_result and temporal_result["std_temporal_score"] is not None
                else None,
            }
        )

    json_path = OUTPUT_DIR / "results.json"
    csv_path = OUTPUT_DIR / "results.csv"
    summary_path = OUTPUT_DIR / "summary.md"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_path.write_text(summarize(results), encoding="utf-8")

    print(json_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
