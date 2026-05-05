#!/usr/bin/env python3
"""Stage 5 — build a stratified DoTA sample manifest (default n=10).

Reads `tmp/dota/dataset/metadata_val.json` and stratifies by `anomaly_class`
bucket ("ego" vs "other"); writes a CSV with one row per selected clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-json", required=True, type=Path,
                        help="path to metadata_val.json (or metadata_train.json)")
    parser.add_argument("--frames-root", required=True, type=Path,
                        help="directory containing extracted per-clip frame folders")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json.read_text())

    # only keep clips whose frame folder actually exists under frames_root
    available = {}
    for clip_id, meta in metadata.items():
        folder = args.frames_root / clip_id
        if folder.is_dir() and any(folder.glob("*.jpg")):
            available[clip_id] = meta
    print(f"[manifest] available clips on disk: {len(available)} / {len(metadata)}")
    if len(available) < args.sample_size:
        raise SystemExit(
            f"not enough extracted clips on disk (have {len(available)}, need {args.sample_size})"
        )

    # stratify by coarse bucket "ego" / "other" (matches class prefix)
    buckets: dict[str, list[str]] = defaultdict(list)
    for cid, m in available.items():
        bucket = "ego" if m["anomaly_class"].startswith("ego") else "other"
        buckets[bucket].append(cid)

    rng = random.Random(args.seed)
    for key in buckets:
        rng.shuffle(buckets[key])

    per_bucket = max(args.sample_size // max(len(buckets), 1), 1)
    picked: list[str] = []
    for key, ids in buckets.items():
        picked.extend(ids[:per_bucket])
    # top-up if needed
    leftover = [c for c in available if c not in set(picked)]
    rng.shuffle(leftover)
    while len(picked) < args.sample_size and leftover:
        picked.append(leftover.pop())

    picked = picked[: args.sample_size]

    rows = []
    for cid in picked:
        meta = available[cid]
        n_frames = sum(1 for _ in (args.frames_root / cid).glob("*.jpg"))
        rows.append(
            {
                "video_id": cid,
                "dataset": "dota",
                "anomaly_class": meta["anomaly_class"],
                "bucket": "ego" if meta["anomaly_class"].startswith("ego") else "other",
                "video_start": meta["video_start"],
                "video_end": meta["video_end"],
                "anomaly_start": meta["anomaly_start"],
                "anomaly_end": meta["anomaly_end"],
                "num_frames": n_frames,
                "subset": meta.get("subset", ""),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[manifest] wrote {args.output_csv} ({len(rows)} clips)")
    for row in rows:
        print(f"  {row['video_id']:40s} {row['anomaly_class']:45s} frames={row['num_frames']}")


if __name__ == "__main__":
    main()
