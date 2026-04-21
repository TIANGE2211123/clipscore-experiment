#!/usr/bin/env python3
"""Extract the selected 150 Crash-1500 videos from the official zip."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path


ZIP_PATH = Path("/Users/Lenovo/Desktop/学业/Crash1500_official/Crash-1500.zip")
MANIFEST_PATH = Path("/Users/Lenovo/Desktop/学业/Crash1500_official/crash1500_subset_150_manifest.csv")
OUTPUT_DIR = Path("/Users/Lenovo/Desktop/学业/Crash1500_official/subset_150_videos")


def load_manifest_ids() -> list[str]:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        return [row["video_id"] for row in csv.DictReader(f)]


def build_name_index(archive: zipfile.ZipFile) -> dict[str, str]:
    index = {}
    for name in archive.namelist():
        base = Path(name).name
        if base.endswith(".mp4"):
            index[base] = name
    return index


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_ids = load_manifest_ids()

    with zipfile.ZipFile(ZIP_PATH) as archive:
        name_index = build_name_index(archive)
        for video_id in selected_ids:
            filename = f"{video_id}.mp4"
            inner_name = name_index.get(filename)
            if not inner_name:
                raise FileNotFoundError(f"Could not find {filename} inside {ZIP_PATH}")
            target_path = OUTPUT_DIR / filename
            if target_path.exists():
                continue
            with archive.open(inner_name) as src, target_path.open("wb") as dst:
                dst.write(src.read())

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
