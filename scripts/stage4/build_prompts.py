"""Build the two prompt sets to feed a text-to-video model.

P_label    — the templated description already shipping in the repo's
             classified_descriptions.csv (label-only signal).
P_scenario — the grounded description we generated with Gemini.

Output: outputs/stage4/prompts/prompt_sets.json and a side-by-side csv."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENARIO = ROOT / "prompts" / "scenario_descriptions.json"
TEMPLATED = Path("/home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace/"
                 "clipscore-experiment/output/euroncap_100/classified_descriptions.csv")
OUT_JSON = ROOT / "prompts" / "prompt_sets.json"
OUT_CSV = ROOT / "prompts" / "prompt_sets.csv"

LABELS = ["safe", "near_crash", "crash"]


def load_templated() -> dict[str, dict[str, str]]:
    out = {}
    with TEMPLATED.open() as f:
        for r in csv.DictReader(f):
            out[r["video_id"]] = {
                "safe": r["safe_description"],
                "near_crash": r["near_crash_description"],
                "crash": r["crash_description"],
            }
    return out


def main() -> None:
    scenario = json.loads(SCENARIO.read_text())
    templated = load_templated()
    sets = {}
    for vid in scenario:
        if vid not in templated:
            continue
        sets[vid] = {"P_label": templated[vid],
                     "P_scenario": scenario[vid]}
    OUT_JSON.write_text(json.dumps(sets, indent=2, ensure_ascii=False))

    rows = []
    for vid, s in sets.items():
        for lbl in LABELS:
            rows.append({"video_id": vid, "label": lbl,
                         "P_label": s["P_label"][lbl],
                         "P_scenario": s["P_scenario"][lbl]})
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "label",
                                          "P_label", "P_scenario"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT_JSON} ({len(sets)} videos)")
    print(f"wrote {OUT_CSV} ({len(rows)} prompt pairs)")


if __name__ == "__main__":
    main()
