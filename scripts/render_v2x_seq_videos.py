#!/usr/bin/env python3
"""Render selected V2X-Seq-SPD sequences into MP4 videos directly from zip archives."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sequences(metadata_zip: Path, suffix: str) -> tuple[str, dict[str, list[dict]]]:
    with zipfile.ZipFile(metadata_zip) as zf:
        data_info_name = next(name for name in zf.namelist() if name.endswith(suffix))
        root_prefix = data_info_name.rsplit("data_info.json", 1)[0]
        rows = json.loads(zf.read(data_info_name))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_id"]].append(row)
    for seq_rows in grouped.values():
        seq_rows.sort(key=lambda row: int(row["frame_id"]))
    return root_prefix, grouped


def decode_image(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    data = zf.read(name)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode image from zip entry: {name}")
    return image


def render_sequence(
    image_zip: Path,
    image_root: str,
    seq_rows: list[dict],
    output_path: Path,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(image_zip) as zf:
        first_image = decode_image(zf, image_root + seq_rows[0]["image_path"])
        height, width = first_image.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            writer.write(first_image)
            for row in seq_rows[1:]:
                frame = decode_image(zf, image_root + row["image_path"])
                writer.write(frame)
        finally:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--metadata-zip", required=True, type=Path)
    parser.add_argument("--vehicle-image-zip", required=True, type=Path)
    parser.add_argument("--infrastructure-image-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    manifest_rows = load_manifest(args.manifest_csv)
    vehicle_root, vehicle_sequences = load_sequences(args.metadata_zip, "vehicle-side/data_info.json")
    infra_root, infra_sequences = load_sequences(args.metadata_zip, "infrastructure-side/data_info.json")

    for row in manifest_rows:
        sequence_id = row["sequence_id"]
        if row["source_type"] == "vehicle-side":
            render_sequence(
                args.vehicle_image_zip,
                vehicle_root,
                vehicle_sequences[sequence_id],
                args.output_dir / f"{row['video_id']}.mp4",
                args.fps,
            )
        else:
            render_sequence(
                args.infrastructure_image_zip,
                infra_root,
                infra_sequences[sequence_id],
                args.output_dir / f"{row['video_id']}.mp4",
                args.fps,
            )

    print(args.output_dir)


if __name__ == "__main__":
    main()
