#!/usr/bin/env python3
"""Build a reproducible 150-video Crash-1500 subset manifest."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ANNOTATION_PATH = Path("/Users/Lenovo/Desktop/学业/Crash1500_official/Crash-1500.txt")
OUTPUT_DIR = Path("/Users/Lenovo/Desktop/学业/Crash1500_official")
MANIFEST_CSV = OUTPUT_DIR / "crash1500_subset_150_manifest.csv"
SUMMARY_MD = OUTPUT_DIR / "crash1500_subset_150_summary.md"
SAMPLE_SIZE = 150
SEED = 42


@dataclass
class CrashRecord:
    video_id: str
    youtube_id: str
    start_frame: int
    accident_frame: int
    timing: str
    weather: str
    ego_involve: str

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.timing, self.weather, self.ego_involve)

    @property
    def prompt(self) -> str:
        timing_text = "daytime" if self.timing.lower() == "day" else "nighttime"
        weather_text = {
            "normal": "clear weather",
            "rainy": "rainy weather",
            "snowy": "snowy weather",
        }.get(self.weather.lower(), f"{self.weather.lower()} weather")
        ego_text = (
            "The ego vehicle is involved in the collision."
            if self.ego_involve.lower() == "yes"
            else "The ego vehicle is not directly involved in the collision."
        )
        return (
            f"A dashcam video captured in {timing_text} under {weather_text}. "
            f"A traffic crash occurs near frame {self.accident_frame + 1}. "
            f"{ego_text}"
        )


def parse_line(line: str) -> CrashRecord:
    line = line.strip()
    prefix, suffix = line.split("],", 1)
    video_id, labels_raw = prefix.split(",[", 1)
    labels = [int(token.strip()) for token in labels_raw.split(",")]
    suffix_parts = [part.strip() for part in suffix.split(",")]
    start_frame, youtube_id, timing, weather, ego_involve = suffix_parts
    accident_frame = labels.index(1) if 1 in labels else len(labels) - 1
    return CrashRecord(
        video_id=video_id,
        youtube_id=youtube_id,
        start_frame=int(start_frame),
        accident_frame=accident_frame,
        timing=timing,
        weather=weather,
        ego_involve=ego_involve,
    )


def proportional_targets(grouped: dict[tuple[str, str, str], list[CrashRecord]]) -> dict[tuple[str, str, str], int]:
    total = sum(len(records) for records in grouped.values())
    raw = {}
    for key, records in grouped.items():
        exact = len(records) * SAMPLE_SIZE / total
        raw[key] = (exact, int(exact))

    targets = {key: floor for key, (_, floor) in raw.items()}
    remaining = SAMPLE_SIZE - sum(targets.values())

    remainders = sorted(
        ((exact - floor, key) for key, (exact, floor) in raw.items()),
        reverse=True,
    )
    for _, key in remainders[:remaining]:
        targets[key] += 1

    return targets


def sample_records(records: list[CrashRecord]) -> list[CrashRecord]:
    grouped: dict[tuple[str, str, str], list[CrashRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_key].append(record)

    targets = proportional_targets(grouped)

    rng = random.Random(SEED)
    selected = []
    for key, group_records in sorted(grouped.items()):
        pool = group_records[:]
        rng.shuffle(pool)
        selected.extend(pool[: targets[key]])

    selected.sort(key=lambda record: record.video_id)
    return selected


def write_manifest(selected: list[CrashRecord]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "youtube_id",
                "start_frame",
                "accident_frame",
                "timing",
                "weather",
                "ego_involve",
                "prompt",
            ],
        )
        writer.writeheader()
        for record in selected:
            writer.writerow(
                {
                    "video_id": record.video_id,
                    "youtube_id": record.youtube_id,
                    "start_frame": record.start_frame,
                    "accident_frame": record.accident_frame,
                    "timing": record.timing,
                    "weather": record.weather,
                    "ego_involve": record.ego_involve,
                    "prompt": record.prompt,
                }
            )

    group_counter = Counter((r.timing, r.weather, r.ego_involve) for r in selected)
    timing_counter = Counter(r.timing for r in selected)
    weather_counter = Counter(r.weather for r in selected)
    ego_counter = Counter(r.ego_involve for r in selected)

    lines = [
        "# Crash-1500 Subset (150 Videos)",
        "",
        f"- Sampling seed: `{SEED}`",
        f"- Source annotation: `{ANNOTATION_PATH}`",
        f"- Selected videos: `{len(selected)}`",
        "",
        "## Timing",
        "",
    ]
    for key, value in sorted(timing_counter.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Weather", ""])
    for key, value in sorted(weather_counter.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Ego Involvement", ""])
    for key, value in sorted(ego_counter.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Joint Groups", ""])
    for key, value in sorted(group_counter.items()):
        lines.append(f"- {key[0]} / {key[1]} / ego={key[2]}: {value}")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = [parse_line(line) for line in ANNOTATION_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sample_records(records)
    write_manifest(selected)
    print(MANIFEST_CSV)
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
