#!/usr/bin/env python3
"""Render DoTA clip frames into MP4 videos, and optionally write 8 keyframes.

DoTA clips on disk are per-clip folders with 000000.jpg ... 000119.jpg.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_clip(frames_dir: Path, out_path: Path, fps: int) -> int:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return 0
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    try:
        writer.write(first)
        for f in frames[1:]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            writer.write(img)
    finally:
        writer.release()
    return len(frames)


def extract_keyframes(frames_dir: Path, out_dir: Path, n: int) -> int:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(frames) - 1, num=min(n, len(frames)), dtype=int)
    written = 0
    for i, fi in enumerate(idxs):
        img = cv2.imread(str(frames[int(fi)]))
        if img is None:
            continue
        cv2.imwrite(str(out_dir / f"frame_{i:02d}.png"), img)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--frames-root", required=True, type=Path,
                        help="dir holding per-clip DoTA frame folders")
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--keyframes-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num-keyframes", type=int, default=8)
    args = parser.parse_args()

    rows = load_manifest(args.manifest_csv)
    for row in rows:
        cid = row["video_id"]
        src = args.frames_root / cid
        mp4 = args.videos_dir / f"{cid}.mp4"
        if not mp4.exists():
            n = render_clip(src, mp4, args.fps)
            print(f"[render] {cid} -> {mp4.name} ({n} frames)")
        key_sub = args.keyframes_dir / cid
        if not key_sub.exists() or not any(key_sub.glob("frame_*.png")):
            k = extract_keyframes(src, key_sub, args.num_keyframes)
            print(f"[render] {cid} keyframes -> {k}")


if __name__ == "__main__":
    main()
