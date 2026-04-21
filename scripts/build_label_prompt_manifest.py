#!/usr/bin/env python3
"""Sample candidate videos and export manifest / label-prompt CSV files."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


LIGHTING_MAP = {
    "day": "daytime",
    "daytime": "daytime",
    "night": "nighttime",
    "nighttime": "nighttime",
    "dusk": "dusk",
    "dawn": "dawn",
}

WEATHER_MAP = {
    "normal": "clear weather",
    "clear": "clear weather",
    "sunny": "sunny weather",
    "rainy": "rainy weather",
    "rain": "rainy weather",
    "snowy": "snowy weather",
    "snow": "snowy weather",
    "foggy": "foggy weather",
    "fog": "foggy weather",
}


def normalize(value: str, default: str = "") -> str:
    return str(value or default).strip()


def lighting_text(value: str) -> str:
    raw = normalize(value, "day")
    return LIGHTING_MAP.get(raw.lower(), raw.lower())


def weather_text(value: str) -> str:
    raw = normalize(value, "clear")
    return WEATHER_MAP.get(raw.lower(), f"{raw.lower()} weather")


def build_prompt(row: dict[str, str]) -> str:
    viewpoint = normalize(row.get("viewpoint"), "traffic camera")
    scene_type = normalize(row.get("scene_type"), "road traffic scene")
    label = normalize(row.get("label"), "unknown")
    lighting = lighting_text(row.get("lighting") or row.get("timing"))
    weather = weather_text(row.get("weather"))
    dataset = normalize(row.get("dataset"), "external dataset")
    ttc = normalize(row.get("ttc"))
    lateral = normalize(row.get("lateral_clearance"))

    suffix_parts = []
    if ttc:
        suffix_parts.append(f"TTC is approximately {ttc}")
    if lateral:
        suffix_parts.append(f"minimum lateral clearance is about {lateral}")
    suffix = ". " + ". ".join(suffix_parts) + "." if suffix_parts else ""

    return (
        f"A {viewpoint} video from {dataset} shows {scene_type} during {lighting} "
        f"under {weather}. The scenario label is {label}.{suffix}"
    )


def proportional_sample(
    rows: list[dict[str, str]],
    sample_size: int,
    stratify_fields: list[str],
    seed: int,
) -> list[dict[str, str]]:
    if sample_size >= len(rows):
        return sorted(rows, key=lambda row: row["video_id"])

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(normalize(row.get(field), "unknown") for field in stratify_fields) or ("all",)
        grouped[key].append(row)

    total = len(rows)
    exact_targets: dict[tuple[str, ...], float] = {
        key: len(group) * sample_size / total for key, group in grouped.items()
    }
    floor_targets = {key: int(value) for key, value in exact_targets.items()}
    remaining = sample_size - sum(floor_targets.values())

    for _, key in sorted(
        ((exact_targets[key] - floor_targets[key], key) for key in grouped),
        reverse=True,
    )[:remaining]:
        floor_targets[key] += 1

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for key, group_rows in sorted(grouped.items()):
        pool = group_rows[:]
        rng.shuffle(pool)
        selected.extend(pool[: floor_targets[key]])

    return sorted(selected, key=lambda row: row["video_id"])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path, help="Candidate video metadata CSV.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--stratify",
        nargs="*",
        default=["label", "viewpoint", "lighting"],
        help="Fields used for proportional sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected = proportional_sample(rows, args.sample_size, args.stratify, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    label_prompt_rows: list[dict[str, str]] = []
    for row in selected:
        manifest_row = dict(row)
        manifest_row["prompt"] = build_prompt(row)
        manifest_rows.append(manifest_row)
        label_prompt_rows.append(
            {
                "video_id": row["video_id"],
                "label": normalize(row.get("label"), "unknown"),
                "prompt": manifest_row["prompt"],
            }
        )

    manifest_path = args.output_dir / "manifest.csv"
    label_prompt_path = args.output_dir / "label_prompt.csv"
    summary_path = args.output_dir / "sampling_summary.md"

    write_csv(manifest_path, manifest_rows, list(manifest_rows[0].keys()))
    write_csv(label_prompt_path, label_prompt_rows, ["video_id", "label", "prompt"])

    counters = {
        field: Counter(normalize(row.get(field), "unknown") for row in manifest_rows)
        for field in args.stratify
    }
    lines = [
        "# Sampling Summary",
        "",
        f"- Input rows: `{len(rows)}`",
        f"- Selected rows: `{len(manifest_rows)}`",
        f"- Sampling seed: `{args.seed}`",
        f"- Stratify fields: `{', '.join(args.stratify)}`",
        "",
    ]
    for field, counter in counters.items():
        lines.append(f"## {field}")
        lines.append("")
        for key, value in sorted(counter.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(manifest_path)
    print(label_prompt_path)
    print(summary_path)


if __name__ == "__main__":
    main()
