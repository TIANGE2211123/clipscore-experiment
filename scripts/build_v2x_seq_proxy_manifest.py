#!/usr/bin/env python3
"""Build a proxy manifest from official V2X-Seq-SPD metadata."""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path


def load_data_info(metadata_zip: Path, suffix: str) -> list[dict]:
    with zipfile.ZipFile(metadata_zip) as zf:
        name = next(item for item in zf.namelist() if item.endswith(suffix))
        return json.loads(zf.read(name))


def build_rows(data: list[dict], viewpoint: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        grouped[row["sequence_id"]].append(row)

    rows = []
    for sequence_id, seq_rows in grouped.items():
        seq_rows.sort(key=lambda row: int(row["frame_id"]))
        first = seq_rows[0]
        last = seq_rows[-1]
        rows.append(
            {
                "video_id": f"{viewpoint.replace('-', '_')}_{sequence_id}",
                "dataset": "v2x_seq_proxy",
                "label": "proxy_unlabeled",
                "viewpoint": "dashcam" if viewpoint == "vehicle-side" else "roadside",
                "scene_type": f"{first.get('intersection_loc', 'urban intersection')} traffic sequence",
                "weather": "unknown",
                "lighting": "day",
                "sequence_id": sequence_id,
                "intersection_loc": first.get("intersection_loc", ""),
                "num_frames": first.get("num_frames", len(seq_rows)),
                "frame_start": first["frame_id"],
                "frame_end": last["frame_id"],
                "local_video_path": "",
                "source_type": viewpoint,
            }
        )
    return rows


def sample_rows(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["viewpoint"]].append(row)

    rng = random.Random(seed)
    selected = []
    per_group = max(sample_size // max(len(grouped), 1), 1)
    for key, group_rows in sorted(grouped.items()):
        pool = group_rows[:]
        rng.shuffle(pool)
        selected.extend(pool[: min(per_group, len(pool))])

    remaining = sample_size - len(selected)
    if remaining > 0:
        leftovers = [row for row in rows if row not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    return sorted(selected[:sample_size], key=lambda row: row["video_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vehicle_rows = build_rows(
        load_data_info(args.metadata_zip, "vehicle-side/data_info.json"),
        "vehicle-side",
    )
    infra_rows = build_rows(
        load_data_info(args.metadata_zip, "infrastructure-side/data_info.json"),
        "infrastructure-side",
    )
    all_rows = vehicle_rows + infra_rows
    selected = sample_rows(all_rows, args.sample_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_csv = args.output_dir / "v2x_seq_proxy_candidates.csv"
    with candidates_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    print(candidates_csv)


if __name__ == "__main__":
    main()
