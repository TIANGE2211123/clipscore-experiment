#!/usr/bin/env python3
"""Stage 5 — CLIP ViT-B/32 scoring on V2X keyframes × 3 descriptions.

Produces:
  per_video.csv  : columns = video_id, label, score   (one row per pair)
  per_frame.json : { video_id: { label: [s_1 .. s_K] } }
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

LABELS = ["safe", "near-crash", "crash"]
MODEL_NAME = "openai/clip-vit-base-patch32"


def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, proc, device


def clip_scores_for_video(
    model, proc, device, frames: list[Path], texts: list[str]
) -> np.ndarray:
    # returns (num_texts, num_frames)
    images = [Image.open(p).convert("RGB") for p in frames]
    with torch.no_grad():
        img_in = proc(images=images, return_tensors="pt").to(device)
        img_feats = model.get_image_features(**img_in)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        txt_in = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        txt_feats = model.get_text_features(**txt_in)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    # CLIPScore convention: max(0, 2.5 * cos_sim) * 100 — here we match the stage-3
    # pipeline which uses raw cosine * 100 (see clipscore-experiment README).
    sims = (txt_feats @ img_feats.T).cpu().numpy() * 100.0
    return sims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    descs = json.loads(args.descriptions.read_text(encoding="utf-8"))
    model, proc, device = load_clip()
    print(f"[clip] loaded {MODEL_NAME} on {device}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    per_frame: dict[str, dict[str, list[float]]] = {}
    video_rows: list[dict[str, str]] = []

    for vid, labelmap in descs.items():
        sub = args.frames_dir / vid
        frames = sorted(sub.glob("frame_*.png"))
        if not frames:
            print(f"[clip] skip {vid} (no frames)")
            continue
        texts = [labelmap[k] for k in LABELS]
        sims = clip_scores_for_video(model, proc, device, frames, texts)  # (3, K)
        per_frame[vid] = {LABELS[i]: [float(x) for x in sims[i]] for i in range(3)}
        for i, label in enumerate(LABELS):
            video_rows.append(
                {
                    "video_id": vid,
                    "label": label,
                    "score": f"{float(sims[i].mean()):.4f}",
                    "num_frames": str(sims.shape[1]),
                }
            )
        print(f"[clip] {vid} mean= safe={sims[0].mean():.2f} near={sims[1].mean():.2f} crash={sims[2].mean():.2f}")

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "label", "score", "num_frames"])
        w.writeheader()
        w.writerows(video_rows)
    args.output_json.write_text(json.dumps(per_frame, indent=2), encoding="utf-8")
    print(f"[clip] wrote {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()
