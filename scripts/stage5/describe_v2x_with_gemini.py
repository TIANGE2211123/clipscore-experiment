#!/usr/bin/env python3
"""Stage 5 — Gemini 3-way (safe / near-crash / crash) descriptions for V2X videos.

Calls the HappyCapy AI Gateway (OpenRouter-compatible) with google/gemini-3.1-pro-preview,
one request per video; conditioned on up to 4 encoded keyframes.
Output JSON schema mirrors Stage 4:
  { video_id: { "safe": str, "near-crash": str, "crash": str } }
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

GATEWAY = "https://ai-gateway.happycapy.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.1-pro-preview"
LABELS = ["safe", "near-crash", "crash"]

SYSTEM_PROMPT = (
    "You are a driving-scene annotator. Given up to 4 keyframes from a short "
    "traffic video (V2X dashcam or roadside view), write THREE self-contained "
    "scene descriptions (one for each of: safe / near-crash / crash). Each "
    "description must: (1) ground itself in what is actually visible in the "
    "keyframes (road layout, vehicles, lighting, camera viewpoint); (2) differ "
    "ONLY in the hypothesized event trajectory matching its label; (3) be "
    "suitable as a text-to-video prompt (≈30-60 words). Return strict JSON with "
    'keys "safe", "near-crash", "crash" and nothing else.'
)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def describe_video(video_id: str, frames: list[Path], api_key: str, max_retries: int = 3) -> dict[str, str]:
    images = frames[:: max(1, len(frames) // 4)][:4] or frames[:4]
    content = [{"type": "text", "text": f"video_id: {video_id}"}]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(img)}"},
            }
        )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                GATEWAY,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            data = json.loads(text)
            if all(k in data for k in LABELS):
                return {k: str(data[k]).strip() for k in LABELS}
            raise ValueError(f"missing keys in response: {list(data.keys())}")
        except Exception as e:
            print(f"[gemini] {video_id} attempt {attempt+1}/{max_retries} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini failed for {video_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    if not api_key:
        print("[gemini] ERROR: AI_GATEWAY_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    with args.manifest_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, str]] = {}
    if args.output_json.exists():
        existing = json.loads(args.output_json.read_text(encoding="utf-8"))

    for row in rows:
        vid = row["video_id"]
        if vid in existing and all(k in existing[vid] for k in LABELS):
            print(f"[gemini] skip {vid} (cached)")
            continue
        frames_subdir = args.frames_dir / vid
        frames = sorted(frames_subdir.glob("frame_*.png"))
        if not frames:
            print(f"[gemini] skip {vid} (no frames found in {frames_subdir})", file=sys.stderr)
            continue
        desc = describe_video(vid, frames, api_key)
        existing[vid] = desc
        args.output_json.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[gemini] {vid} OK")

    print(f"[gemini] wrote {args.output_json} ({len(existing)} videos)")


if __name__ == "__main__":
    main()
