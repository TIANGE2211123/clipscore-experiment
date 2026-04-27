"""
Stage 4 / step C: use Gemini 3.1 Pro (via the HappyCapy AI Gateway) to
produce **three** grounded scenario descriptions per video — one per
label (safe / near-crash / crash) — conditioned on actual keyframes.

Each description is a self-contained prompt usable for text-to-video
generation. Unlike the templated `classified_descriptions.csv`, these
reference what's actually visible in the clip (vehicle make, viewpoint,
crash test rig features, barrier type, lighting, camera angle, …), so
we can later compare P_label (templated) vs P_scenario (grounded)
prompts for label-bias leakage.

Output: outputs/stage4/prompts/scenario_descriptions.json
"""
from __future__ import annotations
import base64
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
OUT = ROOT / "prompts"
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "scenario_descriptions.json"
META = Path("/home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace/"
            "clipscore-experiment/output/euroncap_100/euroncap_candidates.csv")

GATEWAY = "https://ai-gateway.happycapy.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.1-pro-preview"
API_KEY = os.environ["AI_GATEWAY_API_KEY"]

SYSTEM = """You are a careful annotator for a driving-safety video dataset.
You are shown 8 keyframes (in temporal order) from a single Euro NCAP clip.
For the same clip you must write THREE short, self-contained scenario
descriptions, suitable as prompts for a text-to-video model:

  - "safe":        re-imagine the scene as an event with NO conflict,
                   everyone maintains safe spacing, no contact, no hazard.
  - "near_crash":  re-imagine the scene as a close call — a sudden
                   hazard develops but the participants avoid contact.
  - "crash":       describe the scene as a collision actually occurs.

Each description MUST:
  * be grounded in what is actually visible (vehicle type/colour,
    viewpoint, barrier, lane count, lighting, environment). Do not
    invent weather or scenery that is not visible.
  * be 2-3 sentences, neutral present-tense, <= 60 words.
  * NEVER mention "Euro NCAP", "crash test", "test dummy", "rig" or
    any studio-setup language — write as if it were a real on-road
    event.
  * NOT use the words "safe" / "near crash" / "crash" as a label
    directly; describe the dynamics instead. (The label is implied
    by the content.)

Return strict JSON only, no markdown:
{
  "safe":       "...",
  "near_crash": "...",
  "crash":      "..."
}
"""


def load_meta() -> dict[str, dict]:
    rows = {}
    with META.open() as f:
        for r in csv.DictReader(f):
            rows[r["video_id"]] = r
    return rows


def encode_image(p: Path) -> dict:
    b = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b}"}}


def describe_one(video_id: str, meta: dict) -> dict | None:
    frames = sorted((FRAMES / video_id).glob("frame_*.jpg"))
    if len(frames) < 4:
        print(f"  [warn] {video_id}: only {len(frames)} frames, skipping")
        return None
    content = [{"type": "text",
                "text": (f"Video id: {video_id}\nTitle: {meta.get('title','')}\n"
                         f"Viewpoint (from metadata): "
                         f"{meta.get('viewpoint','')}\n\n"
                         f"Here are 8 frames in temporal order:")}]
    for fp in frames:
        content.append(encode_image(fp))
    content.append({"type": "text",
                    "text": "Now return the three JSON descriptions."})

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    r = requests.post(GATEWAY,
                      headers={"Authorization": f"Bearer {API_KEY}",
                               "Content-Type": "application/json"},
                      json=body, timeout=180)
    if r.status_code != 200:
        print(f"  [err] {video_id} {r.status_code}: {r.text[:300]}")
        return None
    text = r.json()["choices"][0]["message"]["content"].strip()
    # strip ```json fences if any
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{"); end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end])
            except json.JSONDecodeError:
                obj = None
    if obj is None:
        # Last-resort: regex extract "key": "value" even if JSON is truncated.
        def grab(key: str) -> str:
            m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
            return m.group(1).replace('\\"', '"').replace("\\n", " ") if m else ""
        obj = {k: grab(k) for k in ("safe", "near_crash", "crash")}
        if not any(obj.values()):
            print(f"  [err] {video_id}: could not parse JSON:\n{text[:400]}")
            return None
    return {"safe": obj.get("safe", "").strip(),
            "near_crash": obj.get("near_crash", "").strip(),
            "crash": obj.get("crash", "").strip()}


def main() -> None:
    meta = load_meta()
    existing = {}
    if OUT_FILE.exists():
        existing = json.loads(OUT_FILE.read_text())
    ids = sorted(p.name for p in FRAMES.iterdir() if p.is_dir())
    print(f"[describe] {len(ids)} videos; {len(existing)} cached")
    for vid in ids:
        if vid in existing:
            print(f"  skip {vid} (cached)")
            continue
        print(f"  describing {vid} ...")
        t0 = time.time()
        r = describe_one(vid, meta.get(vid, {}))
        dt = time.time() - t0
        if r is None:
            continue
        existing[vid] = r
        OUT_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"    ok ({dt:.1f}s)  safe={r['safe'][:70]}…")
    print(f"[describe] done. wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
