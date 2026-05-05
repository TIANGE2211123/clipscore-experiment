#!/usr/bin/env python3
"""Stage 5 — cross-backbone CLIP label-bias replication.

Replicates Stage 3's label-bias protocol using multiple CLIP backbones on the
same 10 Euro NCAP source videos whose keyframes already live under
`outputs/stage4/frames/<video_id>/frame_{00..07}.jpg` and whose 3-way
descriptions (safe / near_crash / crash) are in
`outputs/stage4/prompts/scenario_descriptions.json`.

Purpose: answer whether the CLIP label-bias we reported in Stage A (CLIP
ViT-B/32 OpenAI) is architecture- or training-data-specific, by comparing:

  * openai/clip-vit-base-patch32          (baseline replication)
  * openai/clip-vit-large-patch14         (same data, bigger)
  * google/siglip-base-patch16-224        (different objective, different data)

Outputs (per backbone):
  outputs/stage5/clipscore/<bb_key>/per_video.csv      one row per (video, label)
  outputs/stage5/clipscore/<bb_key>/per_frame.json     {video: {label: [K scores]}}
And a joint summary:
  outputs/stage5/clipscore/cross_backbone_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

LABELS = ["safe", "near_crash", "crash"]
DISPLAY_LABELS = {"safe": "safe", "near_crash": "near-crash", "crash": "crash"}


@dataclass
class Backbone:
    key: str              # short dir name
    display: str          # human name for reports
    hf_id: str
    loader: str           # "clip" | "siglip"


BACKBONES = [
    Backbone("clip_vitb32", "CLIP ViT-B/32 (OpenAI)", "openai/clip-vit-base-patch32", "clip"),
    Backbone("clip_vitl14", "CLIP ViT-L/14 (OpenAI)", "openai/clip-vit-large-patch14", "clip"),
    Backbone("siglip_b16",  "SigLIP base/16 (Google)", "google/siglip-base-patch16-224", "siglip"),
]


def load_backbone(bb: Backbone):
    if bb.loader == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(bb.hf_id).eval()
        proc = CLIPProcessor.from_pretrained(bb.hf_id)
    else:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(bb.hf_id).eval()
        proc = AutoProcessor.from_pretrained(bb.hf_id)
    return model, proc


def _as_tensor(x):
    """Unwrap get_*_features() output: some transformers versions return a
    ModelOutput dataclass instead of a plain tensor."""
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(x, attr):
            val = getattr(x, attr)
            if isinstance(val, torch.Tensor):
                # for last_hidden_state fall back to CLS token
                if attr == "last_hidden_state" and val.ndim == 3:
                    return val[:, 0, :]
                return val
    raise TypeError(f"cannot extract tensor from {type(x)}")


def score_video(model, proc, bb: Backbone, frames: list[Path], texts: list[str]) -> np.ndarray:
    """Return (num_texts, num_frames) cosine similarity × 100."""
    images = [Image.open(p).convert("RGB") for p in frames]
    with torch.no_grad():
        img_in = proc(images=images, return_tensors="pt")
        txt_in = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
        img_feats = _as_tensor(model.get_image_features(**img_in))
        txt_feats = _as_tensor(model.get_text_features(**txt_in))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    sims = (txt_feats @ img_feats.T).cpu().numpy() * 100.0
    return sims


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbones", nargs="*", default=None,
                        help="subset of backbone keys to run (default: all)")
    args = parser.parse_args()

    descs = json.loads(args.descriptions.read_text(encoding="utf-8"))
    selected = [bb for bb in BACKBONES if (args.backbones is None or bb.key in args.backbones)]
    if not selected:
        raise SystemExit(f"no backbones matched; known keys: {[b.key for b in BACKBONES]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for bb in selected:
        bb_dir = args.output_dir / bb.key
        bb_dir.mkdir(parents=True, exist_ok=True)
        csv_path = bb_dir / "per_video.csv"
        json_path = bb_dir / "per_frame.json"

        if csv_path.exists() and json_path.exists():
            print(f"[xb] {bb.key} already cached, skipping load")
            continue

        print(f"[xb] ==> loading {bb.display} ({bb.hf_id})")
        t0 = time.time()
        model, proc = load_backbone(bb)
        print(f"[xb] loaded in {time.time()-t0:.1f}s")

        per_frame: dict[str, dict[str, list[float]]] = {}
        per_video_rows = []

        t_start = time.time()
        for vid, labelmap in descs.items():
            sub = args.frames_dir / vid
            frames = sorted(sub.glob("frame_*.jpg"))
            if not frames:
                frames = sorted(sub.glob("frame_*.png"))
            if not frames:
                print(f"[xb] skip {vid} (no frames)")
                continue
            texts = [labelmap[k] for k in LABELS]
            sims = score_video(model, proc, bb, frames, texts)   # (3, K)
            per_frame[vid] = {DISPLAY_LABELS[k]: [float(x) for x in sims[i]] for i, k in enumerate(LABELS)}
            for i, k in enumerate(LABELS):
                per_video_rows.append(
                    {
                        "video_id": vid,
                        "label": DISPLAY_LABELS[k],
                        "score": f"{float(sims[i].mean()):.4f}",
                        "num_frames": str(sims.shape[1]),
                        "backbone": bb.key,
                    }
                )
            print(f"[xb][{bb.key}] {vid}  safe={sims[0].mean():.3f}  near={sims[1].mean():.3f}  crash={sims[2].mean():.3f}")

        print(f"[xb] {bb.key}: scored {len(per_frame)} videos in {time.time()-t_start:.1f}s")

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "label", "score", "num_frames", "backbone"])
            w.writeheader()
            w.writerows(per_video_rows)
        json_path.write_text(json.dumps(per_frame, indent=2), encoding="utf-8")

        # free memory before next backbone
        del model, proc
        import gc
        gc.collect()

        # mean-by-label for summary row
        means = {}
        for row in per_video_rows:
            means.setdefault(row["label"], []).append(float(row["score"]))
        for k, v in means.items():
            summary_rows.append({"backbone": bb.key, "label": k, "n": len(v),
                                 "mean": f"{np.mean(v):.4f}", "std": f"{np.std(v, ddof=1):.4f}"})

    # write summary
    if summary_rows:
        sp = args.output_dir / "cross_backbone_summary.csv"
        with sp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["backbone", "label", "n", "mean", "std"])
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[xb] wrote summary: {sp}")


if __name__ == "__main__":
    main()
