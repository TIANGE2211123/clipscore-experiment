"""
Stage 4 / step F — CLIPScore on the 60 generated videos.

For every generated video we compute, using OpenAI ViT-B/32:

  * s_own       : CLIPScore( video frames , own prompt )
                  — sanity/quality score; tells us whether the generator
                    actually followed the prompt.
  * s_label_tok : CLIPScore( video frames , "safe" | "near crash" | "crash" )
                  — bare LABEL TOKEN score. This is the key metric for
                    label-bias: if a P_label-generated video scores much
                    higher on the matching label token than a
                    P_scenario-generated video of the same source, then
                    the templated prompt is leaking label cues into the
                    generation.
  * s_cross     : CLIPScore( video frames , other-label prompts from the
                  SAME source/prompt-type ) — diagonal-vs-off-diagonal
                    matrix gives a confusion view.

Output:
  outputs/stage4/clipscore/per_video.csv   (one row per generated video
                                            × scoring text)
  outputs/stage4/clipscore/per_video.json  (richer, with per-frame vectors)
"""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "generated"
PROMPTS = ROOT / "prompts" / "prompt_sets.json"
OUT = ROOT / "clipscore"
OUT.mkdir(parents=True, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TMP = Path("/tmp/stage4_frames")
TMP.mkdir(parents=True, exist_ok=True)

LABELS = ["safe", "near_crash", "crash"]
LABEL_TOKENS = {"safe": "safe driving",
                "near_crash": "a near crash",
                "crash": "a car crash"}
MODEL_ID = "openai/clip-vit-base-patch32"
N_FRAMES = 8  # frames per video


def extract_frames(video: Path, work: Path) -> list[Image.Image]:
    work.mkdir(parents=True, exist_ok=True)
    for p in work.glob("f_*.jpg"):
        p.unlink()
    # video is 6s; sample 8 frames evenly
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video),
           "-vf", f"fps={N_FRAMES}/6,scale=512:-2",
           "-frames:v", str(N_FRAMES), "-q:v", "3",
           str(work / "f_%02d.jpg")]
    subprocess.run(cmd, check=True)
    return [Image.open(p).convert("RGB")
            for p in sorted(work.glob("f_*.jpg"))]


def load_clip() -> tuple[CLIPModel, CLIPProcessor]:
    print("loading CLIP ViT-B/32 (CPU)...", flush=True)
    m = CLIPModel.from_pretrained(MODEL_ID).eval()
    p = CLIPProcessor.from_pretrained(MODEL_ID)
    return m, p


def clip_score_image_vs_texts(model, proc, images, texts) -> np.ndarray:
    """Return matrix (n_frames, n_texts) of cosine sims × 100."""
    with torch.no_grad():
        img_in = proc(images=images, return_tensors="pt")
        img_out = model.get_image_features(**img_in)
        img_feats = (img_out.pooler_output if hasattr(img_out, "pooler_output")
                     else img_out)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        txt_in = proc(text=list(texts), return_tensors="pt",
                      padding=True, truncation=True)
        txt_out = model.get_text_features(**txt_in)
        txt_feats = (txt_out.pooler_output if hasattr(txt_out, "pooler_output")
                     else txt_out)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        sims = (img_feats @ txt_feats.T).cpu().numpy() * 100.0
    return sims


def main() -> None:
    sets = json.loads(PROMPTS.read_text())
    # enumerate all generated videos
    tasks = []
    for ptype_dir in sorted(GEN.iterdir()):
        if not ptype_dir.is_dir() or ptype_dir.name not in ("P_label", "P_scenario"):
            continue
        for label_dir in sorted(ptype_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for mp4 in sorted(label_dir.glob("*.mp4")):
                tasks.append({"video_id": mp4.stem,
                              "ptype": ptype_dir.name,
                              "label": label_dir.name,
                              "path": mp4})
    print(f"found {len(tasks)} generated videos", flush=True)
    if not tasks:
        print("no generated videos yet; exiting.", flush=True)
        return

    model, proc = load_clip()

    per_video = []
    per_video_full = {}
    for i, t in enumerate(tasks):
        vid = t["video_id"]; ptype = t["ptype"]; label = t["label"]
        if vid not in sets:
            print(f"  [{i+1}/{len(tasks)}] skip {vid}: no prompt metadata",
                  flush=True)
            continue
        own_prompt = sets[vid][ptype][label]
        other_prompts = {lbl: sets[vid][ptype][lbl] for lbl in LABELS}
        texts = [own_prompt] + [LABEL_TOKENS[l] for l in LABELS] + \
                [other_prompts[l] for l in LABELS]
        text_keys = ["own"] + [f"tok_{l}" for l in LABELS] + \
                    [f"prompt_{l}" for l in LABELS]

        frames = extract_frames(t["path"], TMP / f"{ptype}_{label}_{vid}")
        sims = clip_score_image_vs_texts(model, proc, frames, texts)
        # aggregate mean over frames per text
        means = sims.mean(axis=0)
        stds = sims.std(axis=0, ddof=1)

        row = {"video_id": vid, "ptype": ptype, "label": label,
               "n_frames": len(frames)}
        for k, m, s in zip(text_keys, means, stds):
            row[f"mean_{k}"] = float(m)
            row[f"std_{k}"] = float(s)
        per_video.append(row)
        per_video_full[f"{ptype}|{label}|{vid}"] = {
            "texts": {k: t_ for k, t_ in zip(text_keys, texts)},
            "frame_sims": sims.tolist(),
        }
        print(f"  [{i+1}/{len(tasks)}] {ptype}/{label}/{vid}  "
              f"own={means[0]:.2f}  "
              f"tok_safe={means[1]:.2f}  "
              f"tok_near={means[2]:.2f}  "
              f"tok_crash={means[3]:.2f}", flush=True)

    # csv
    fields = ["video_id", "ptype", "label", "n_frames",
              "mean_own", "std_own"]
    for k in (["tok_" + l for l in LABELS] + ["prompt_" + l for l in LABELS]):
        fields += [f"mean_{k}", f"std_{k}"]
    with (OUT / "per_video.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(per_video)
    with (OUT / "per_video.json").open("w") as f:
        json.dump({"results": per_video, "full": per_video_full}, f, indent=2)
    print(f"wrote {OUT}/per_video.csv  ({len(per_video)} rows)")


if __name__ == "__main__":
    main()
