#!/usr/bin/env python3
"""Generate safe / near-crash / crash descriptions from a manifest or results CSV.

The script is designed to mirror the existing Crash-1500 workflow while being
reusable for additional datasets. It tries a local Ollama model first when
requested, and falls back to deterministic template generation when Ollama is
unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WEATHER_MAP = {
    "normal": "clear weather",
    "clear": "clear weather",
    "sunny": "sunny weather",
    "rainy": "rainy weather with wet pavement",
    "rain": "rainy weather with wet pavement",
    "snowy": "snowy conditions with reduced visibility",
    "snow": "snowy conditions with reduced visibility",
    "foggy": "foggy conditions with limited visibility",
    "fog": "foggy conditions with limited visibility",
    "overcast": "overcast weather",
}

LIGHTING_MAP = {
    "day": "daytime",
    "daytime": "daytime",
    "night": "nighttime",
    "nighttime": "nighttime",
    "dusk": "dusk",
    "dawn": "dawn",
    "indoor": "indoor lighting",
}

VIEWPOINT_MAP = {
    "dashcam": "dashcam view",
    "vehicle": "vehicle-mounted view",
    "vehicle-side": "vehicle-mounted view",
    "car": "vehicle-mounted view",
    "roadside": "roadside camera view",
    "infrastructure": "roadside camera view",
    "cctv": "roadside camera view",
    "drone": "drone aerial view",
    "uav": "drone aerial view",
    "aerial": "drone aerial view",
}

SCENE_FALLBACKS = [
    "a busy urban intersection",
    "a suburban avenue",
    "a multi-lane arterial road",
    "a two-lane roadway",
    "a highway merge area",
    "a signalized junction",
    "a roadside test track",
    "a controlled crash test lane",
]

SAFE_ACTIONS = [
    "keeps a stable lane position and leaves generous spacing",
    "maintains a calm pace and anticipates surrounding traffic",
    "moves smoothly while nearby actors behave predictably",
]

NEAR_CRASH_ACTIONS = [
    "a sudden cut-in forces a sharp brake and evasive steering correction",
    "a vehicle or road user intrudes into the path, creating a brief but severe conflict",
    "traffic compresses rapidly and the gap closes to a dangerous margin before recovery",
]

CRASH_TYPES = [
    "rear-end collision",
    "side-impact collision",
    "side-swipe collision",
    "frontal impact",
    "multi-vehicle crash",
]


@dataclass
class VideoRecord:
    video_id: str
    dataset: str
    prompt: str
    label: str
    weather: str
    lighting: str
    viewpoint: str
    scene_type: str
    ego_involve: str
    accident_frame: str
    ttc: str
    lateral_clearance: str


def normalize_field(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize_weather(value: str) -> str:
    if not value:
        return "typical road conditions"
    mapped = WEATHER_MAP.get(value.strip().lower())
    return mapped or f"{value.strip().lower()} conditions"


def normalize_lighting(value: str, timing: str = "") -> str:
    raw = value or timing
    if not raw:
        return "daytime"
    mapped = LIGHTING_MAP.get(raw.strip().lower())
    return mapped or raw.strip().lower()


def normalize_viewpoint(value: str) -> str:
    if not value:
        return "traffic camera view"
    mapped = VIEWPOINT_MAP.get(value.strip().lower())
    return mapped or f"{value.strip().lower()} view"


def normalize_scene_type(value: str, rng: random.Random) -> str:
    return value.strip().lower() if value and value.strip() else rng.choice(SCENE_FALLBACKS)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_record(row: dict[str, str], rng: random.Random, dataset_name: str | None) -> VideoRecord:
    dataset = dataset_name or normalize_field(row, "dataset", "source_dataset", default="unknown_dataset")
    timing = normalize_field(row, "timing")
    return VideoRecord(
        video_id=normalize_field(row, "video_id", "id"),
        dataset=dataset,
        prompt=normalize_field(row, "prompt", "text_prompt"),
        label=normalize_field(row, "label", "safety_label", default="unknown"),
        weather=normalize_weather(normalize_field(row, "weather", "environment_weather")),
        lighting=normalize_lighting(normalize_field(row, "lighting", "light", "time_of_day"), timing),
        viewpoint=normalize_viewpoint(normalize_field(row, "viewpoint", "camera_view", "camera_type")),
        scene_type=normalize_scene_type(normalize_field(row, "scene_type", "scenario_type", "road_type"), rng),
        ego_involve=normalize_field(row, "ego_involve", "ego_vehicle_involved", default="Unknown"),
        accident_frame=normalize_field(row, "accident_frame", "collision_frame"),
        ttc=normalize_field(row, "ttc", "min_ttc"),
        lateral_clearance=normalize_field(row, "lateral_clearance", "min_lateral_clearance"),
    )


def compose_context(record: VideoRecord) -> str:
    parts = [
        f"{record.viewpoint} in {record.scene_type}",
        f"during {record.lighting}",
        f"under {record.weather}",
    ]
    if record.ego_involve:
        ego_text = (
            "the ego vehicle is directly involved"
            if record.ego_involve.lower() in {"yes", "true", "1"}
            else "the ego vehicle is not directly involved"
            if record.ego_involve.lower() in {"no", "false", "0"}
            else ""
        )
        if ego_text:
            parts.append(ego_text)
    return ", ".join(parts)


def build_safe_description(record: VideoRecord, rng: random.Random) -> str:
    action = rng.choice(SAFE_ACTIONS)
    return (
        f"[0s: {record.dataset} footage shows {compose_context(record)}.] "
        f"[2s: Traffic remains orderly as the subject vehicle {action}.] "
        f"[4s: The scene resolves safely with no contact, no panic braking, and no hazardous conflict.]"
    )


def build_near_crash_description(record: VideoRecord, rng: random.Random) -> str:
    action = rng.choice(NEAR_CRASH_ACTIONS)
    ttc_text = f" TTC drops to about {record.ttc}." if record.ttc else ""
    clearance_text = (
        f" Lateral clearance shrinks to roughly {record.lateral_clearance}." if record.lateral_clearance else ""
    )
    return (
        f"[0s: {record.dataset} footage begins from {compose_context(record)}.] "
        f"[2s: A hazardous interaction develops as {action}.{ttc_text}{clearance_text}] "
        f"[4s: The participants recover at the last moment, producing a near-crash without visible impact.]"
    )


def build_crash_description(record: VideoRecord, rng: random.Random) -> str:
    crash_type = rng.choice(CRASH_TYPES)
    frame_text = f" near frame {record.accident_frame}" if record.accident_frame else ""
    return (
        f"[0s: {record.dataset} footage shows normal movement from {compose_context(record)}.] "
        f"[2s: The conflict escalates rapidly and a {crash_type} develops{frame_text}.] "
        f"[4s: Contact becomes unavoidable, the impact is visible, and the scene ends in a confirmed crash event.]"
    )


def sanitize_llm_output(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace("Near crash", "Near-crash")
    return cleaned


def try_ollama(
    record: VideoRecord,
    category: str,
    fallback_text: str,
    model: str,
    ollama_bin: str,
    timeout: int,
) -> str:
    prompt = f"""
You are writing a short traffic-video generation prompt.
Return exactly one paragraph with three timestamped beats in this format:
[0s: ...] [2s: ...] [4s: ...]

Dataset: {record.dataset}
Video ID: {record.video_id}
Context: {compose_context(record)}
Existing prompt: {record.prompt or 'N/A'}
Target category: {category}
Label hint: {record.label or 'N/A'}
Accident frame: {record.accident_frame or 'N/A'}
TTC: {record.ttc or 'N/A'}
Lateral clearance: {record.lateral_clearance or 'N/A'}

Write a concrete, visually plausible description for the target category only.
Avoid bullets, markdown fences, and explanations.
""".strip()

    try:
        result = subprocess.run(
            [ollama_bin, "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return fallback_text

    if result.returncode != 0 or not result.stdout.strip():
        return fallback_text
    return sanitize_llm_output(result.stdout)


def build_descriptions(
    rows: Iterable[dict[str, str]],
    dataset_name: str | None,
    seed: int,
    ollama_model: str | None,
    ollama_bin: str,
    ollama_timeout: int,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    records = [to_record(row, rng, dataset_name) for row in rows]
    results: dict[str, dict[str, str]] = {}
    for record in records:
        safe_text = build_safe_description(record, rng)
        near_text = build_near_crash_description(record, rng)
        crash_text = build_crash_description(record, rng)

        if ollama_model:
            safe_text = try_ollama(record, "safe", safe_text, ollama_model, ollama_bin, ollama_timeout)
            near_text = try_ollama(record, "near_crash", near_text, ollama_model, ollama_bin, ollama_timeout)
            crash_text = try_ollama(record, "crash", crash_text, ollama_model, ollama_bin, ollama_timeout)

        results[record.video_id] = {
            "safe": sanitize_llm_output(safe_text),
            "near_crash": sanitize_llm_output(near_text),
            "crash": sanitize_llm_output(crash_text),
        }
    return results


def write_outputs(results: dict[str, dict[str, str]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "safe_description", "near_crash_description", "crash_description"])
        for video_id in sorted(results):
            row = results[video_id]
            writer.writerow([video_id, row["safe"], row["near_crash"], row["crash"]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path, help="Manifest or results CSV.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-model", default=None, help="Optional Ollama model name.")
    parser.add_argument("--ollama-bin", default="ollama")
    parser.add_argument("--ollama-timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    results = build_descriptions(
        rows=rows,
        dataset_name=args.dataset_name,
        seed=args.seed,
        ollama_model=args.ollama_model,
        ollama_bin=args.ollama_bin,
        ollama_timeout=args.ollama_timeout,
    )
    write_outputs(results, args.output_json, args.output_csv)
    print(args.output_json)
    print(args.output_csv)
    print(f"generated={len(results)}")


if __name__ == "__main__":
    main()
