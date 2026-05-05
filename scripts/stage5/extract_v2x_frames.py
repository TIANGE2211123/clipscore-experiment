#!/usr/bin/env python3
"""Extract N evenly-spaced keyframes from each MP4 into a per-video folder."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def extract(video_path: Path, out_dir: Path, n: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    idxs = np.linspace(0, total - 1, num=min(n, total), dtype=int)
    written = 0
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_dir / f"frame_{i:02d}.png"), frame)
        written += 1
    cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()

    total_frames = 0
    for mp4 in sorted(args.video_dir.glob("*.mp4")):
        sub = args.output_dir / mp4.stem
        if sub.exists() and any(sub.glob("frame_*.png")):
            continue
        n = extract(mp4, sub, args.num_frames)
        total_frames += n
        print(f"[extract] {mp4.name} -> {n} frames")
    print(f"[extract] total frames written: {total_frames}")


if __name__ == "__main__":
    main()
