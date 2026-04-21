#!/usr/bin/env python3
"""Export a Euro NCAP YouTube manifest and optionally download selected videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from yt_dlp import YoutubeDL


DEFAULT_CHANNEL_URL = "https://www.youtube.com/channel/UCNEWZqjcguqWZOG8yZZpIFg/videos"


def infer_scene_type(title: str) -> str:
    lowered = title.lower()
    if "pedestrian" in lowered or "cyclist" in lowered or "vru" in lowered:
        return "pedestrian_collision_test"
    if "rear" in lowered:
        return "rear_end_test"
    if "side" in lowered:
        return "side_impact_test"
    if "frontal" in lowered or "front" in lowered:
        return "frontal_collision_test"
    return "crash_and_safety_test"


def extract_entries(channel_url: str, limit: int) -> list[dict]:
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--playlist-end",
        str(limit),
        "--dump-json",
        channel_url,
    ]
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "yt-dlp command failed")

    entries = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        entries.append(json.loads(line))
    if not entries:
        raise RuntimeError("yt-dlp did not return any channel entries.")
    return entries


def download_videos(rows: list[dict], download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    urls = [row["source_url"] for row in rows]
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
        "ignoreerrors": True,
        "cookiesfrombrowser": ("chrome",),  # Try to use Chrome cookies
        "extractor_args": {"youtube": {"player_client": ["web"]}},
    }
    with YoutubeDL(opts) as ydl:
        ydl.download(urls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-url", default=DEFAULT_CHANNEL_URL)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-entries", type=int, default=120)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=None)
    args = parser.parse_args()

    entries = extract_entries(args.channel_url, args.max_entries)
    selected = entries[: args.sample_size]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = args.download_dir or (args.output_dir / "videos")

    rows = []
    for entry in selected:
        video_id = entry["id"]
        local_video_path = download_dir / f"{video_id}.mp4"
        row = {
            "video_id": video_id,
            "dataset": "euroncap",
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": entry.get("title", ""),
            "upload_date": entry.get("upload_date") or "",
            "label": "crash",
            "scene_type": infer_scene_type(entry.get("title", "")),
            "viewpoint": "controlled crash test view",
            "weather": "clear",
            "lighting": "day",
            "local_video_path": str(local_video_path) if local_video_path.exists() else "",
        }
        rows.append(row)

    candidates_csv = args.output_dir / "euroncap_candidates.csv"
    with candidates_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.download:
        download_videos(rows, download_dir)
        for row in rows:
            candidate_path = download_dir / f"{row['video_id']}.mp4"
            if candidate_path.exists():
                row["local_video_path"] = str(candidate_path)
        with candidates_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(candidates_csv)
    if args.download:
        print(download_dir)


if __name__ == "__main__":
    main()
