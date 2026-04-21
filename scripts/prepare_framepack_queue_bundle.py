#!/usr/bin/env python3
"""Prepare a portable FramePack queue bundle from sampled videos and descriptions.

Outputs:
- classified_descriptions.json / .csv copy
- frame_refs/<video_id>_input.png
- frame_refs/<video_id>_end_frame.png
- queue_images/<job_id>_input.png
- queue_images/<job_id>_end_frame.png
- queue_seed.json
- job_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import cv2


DEFAULT_JOB_PARAMS = {
    "app_version": "0.5.1",
    "negative_prompt": "",
    "steps": 25,
    "cfg": 1,
    "gs": 10,
    "rs": 0,
    "latent_type": "Noise",
    "resolutionW": 640,
    "resolutionH": 640,
    "model_type": "Original with Endframe",
    "generation_type": "Original with Endframe",
    "has_input_image": True,
    "input_image_path": None,
    "total_second_length": 3,
    "blend_sections": 4,
    "latent_window_size": 9,
    "num_cleaned_frames": 5,
    "end_frame_strength": 0.05,
    "end_frame_image_path": None,
    "end_frame_used": "True",
    "input_video": None,
    "video_path": None,
    "x_param": None,
    "y_param": None,
    "x_values": None,
    "y_values": None,
    "combine_with_source": True,
    "use_teacache": False,
    "teacache_num_steps": 25,
    "teacache_rel_l1_thresh": 0.15,
    "use_magcache": True,
    "magcache_threshold": 0.1,
    "magcache_max_consecutive_skips": 2,
    "magcache_retention_ratio": 0.25,
    "loras": {},
    "status": "pending",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "result": None,
    "queue_position": None,
}


def load_descriptions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_video_path(row: dict[str, str], video_dir: Path | None) -> Path:
    explicit = row.get("local_video_path") or row.get("video_path")
    if explicit:
        return Path(explicit)
    if video_dir is None:
        raise FileNotFoundError(f"Row {row.get('video_id')} has no local_video_path and no --video-dir was given.")
    return video_dir / f"{row['video_id']}.mp4"


def deterministic_seed(video_id: str, category: str) -> int:
    digest = hashlib.sha256(f"{video_id}:{category}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


def extract_edge_frames(video_path: Path, start_path: Path, end_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame: {video_path}")

    last_idx = max(frame_count - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_idx)
    ok, last_frame = cap.read()
    if not ok or last_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 2, 0))
        ok, last_frame = cap.read()
    cap.release()

    if not ok or last_frame is None:
        raise RuntimeError(f"Failed to read last frame: {video_path}")

    cv2.imwrite(str(start_path), first_frame)
    cv2.imwrite(str(end_path), last_frame)
    return frame_count, last_idx


def write_descriptions_csv(descriptions: dict[str, dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "safe_description", "near_crash_description", "crash_description"])
        for video_id in sorted(descriptions):
            row = descriptions[video_id]
            writer.writerow([video_id, row["safe"], row["near_crash"], row["crash"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--descriptions-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args()

    manifest_rows = load_manifest(args.manifest_csv)
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    descriptions = load_descriptions(args.descriptions_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_ref_dir = args.output_dir / "frame_refs"
    queue_images_dir = args.output_dir / "queue_images"
    frame_ref_dir.mkdir(parents=True, exist_ok=True)
    queue_images_dir.mkdir(parents=True, exist_ok=True)

    queue: dict[str, dict[str, object]] = {}
    job_rows: list[dict[str, object]] = []

    descriptions_copy_json = args.output_dir / "classified_descriptions.json"
    descriptions_copy_csv = args.output_dir / "classified_descriptions.csv"
    descriptions_copy_json.write_text(
        json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_descriptions_csv(descriptions, descriptions_copy_csv)

    for row in manifest_rows:
        video_id = row["video_id"]
        if video_id not in descriptions:
            raise KeyError(f"Missing descriptions for video_id={video_id}")

        video_path = resolve_video_path(row, args.video_dir)
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video: {video_path}")

        ref_input = frame_ref_dir / f"{video_id}_input.png"
        ref_end = frame_ref_dir / f"{video_id}_end_frame.png"
        frame_count, last_idx = extract_edge_frames(video_path, ref_input, ref_end)

        for category in ("safe", "near_crash", "crash"):
            job_id = str(uuid.uuid4())
            queue_input = queue_images_dir / f"{job_id}_input.png"
            queue_end = queue_images_dir / f"{job_id}_end_frame.png"
            shutil.copy2(ref_input, queue_input)
            shutil.copy2(ref_end, queue_end)

            created_at = time.time()
            job = dict(DEFAULT_JOB_PARAMS)
            job.update(
                {
                    "id": job_id,
                    "prompt": descriptions[video_id][category],
                    "seed": deterministic_seed(video_id, category),
                    "timestamp": created_at,
                    "created_at": created_at,
                    "saved_input_image_path": f"queue_images/{queue_input.name}",
                    "saved_end_frame_image_path": f"queue_images/{queue_end.name}",
                }
            )
            queue[job_id] = job
            job_rows.append(
                {
                    "job_id": job_id,
                    "video_id": video_id,
                    "category": category,
                    "dataset": args.dataset_name or row.get("dataset", ""),
                    "video_path": str(video_path),
                    "frame_count": frame_count,
                    "last_frame_index": last_idx,
                    "saved_input_image_path": f"queue_images/{queue_input.name}",
                    "saved_end_frame_image_path": f"queue_images/{queue_end.name}",
                    "prompt": descriptions[video_id][category],
                }
            )

    queue_json = args.output_dir / "queue_seed.json"
    job_manifest_csv = args.output_dir / "job_manifest.csv"

    queue_json.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")

    with job_manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(job_rows[0].keys()))
        writer.writeheader()
        writer.writerows(job_rows)

    print(descriptions_copy_json)
    print(descriptions_copy_csv)
    print(queue_json)
    print(job_manifest_csv)
    print(f"jobs={len(job_rows)}")


if __name__ == "__main__":
    main()
