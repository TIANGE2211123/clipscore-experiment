"""
Stage 4 / step G — compare P_label vs P_scenario prompts for label-bias.

Main hypothesis H2:
    Videos generated from templated (label-only) prompts P_label contain
    more "label-cue" content than videos generated from grounded prompts
    P_scenario. If true, we expect, for the SAME source video and SAME
    label L:

        CLIPScore( gen_P_label[L] , "a car crash" )
      > CLIPScore( gen_P_scenario[L] , "a car crash" )

    (and analogously for "safe driving" / "a near crash").

We run a paired test across 10 source videos × 3 labels = 30 pairs, and
plot the distributions. Outputs under outputs/stage4/stage4_analysis/.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "clipscore" / "per_video.csv"
OUT = ROOT / "stage4_analysis"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["safe", "near_crash", "crash"]
COLORS = {"safe": "#2a9d8f", "near_crash": "#e9c46a", "crash": "#e76f51"}


def load() -> list[dict]:
    with SRC.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("video_id", "ptype", "label"):
                continue
            r[k] = float(v) if v else float("nan")
    return rows


def pivot(rows: list[dict]) -> dict:
    """(video_id, label) -> {ptype -> row}"""
    p = {}
    for r in rows:
        p.setdefault((r["video_id"], r["label"]), {})[r["ptype"]] = r
    # keep only pairs where both ptypes exist
    return {k: v for k, v in p.items()
            if "P_label" in v and "P_scenario" in v}


def paired_token_test(p: dict) -> dict:
    """For each label L, compare token-score[L] between ptypes."""
    summary = {}
    for L in LABELS:
        tok_col = f"mean_tok_{L}"
        diffs = []
        for (vid, lbl), row in p.items():
            if lbl != L:
                continue
            d = row["P_label"][tok_col] - row["P_scenario"][tok_col]
            diffs.append((vid, d,
                          row["P_label"][tok_col],
                          row["P_scenario"][tok_col]))
        if not diffs:
            continue
        arr = np.array([d[1] for d in diffs])
        t, p_t = stats.ttest_rel(
            [d[2] for d in diffs], [d[3] for d in diffs])
        try:
            _, p_w = stats.wilcoxon(arr)
        except ValueError:
            p_w = float("nan")
        summary[L] = {
            "n": len(diffs),
            "mean_diff": float(arr.mean()),
            "sd_diff": float(arr.std(ddof=1)),
            "t": float(t),
            "p_t": float(p_t),
            "p_wilcoxon": float(p_w),
            "pairs": [{"video_id": d[0], "d": d[1],
                       "P_label": d[2], "P_scenario": d[3]} for d in diffs],
        }
    return summary


def own_score_compare(p: dict) -> dict:
    """Does P_scenario generate videos that match their own prompt
    better than P_label does? Looks at mean_own."""
    a, b = [], []  # P_label, P_scenario
    for (vid, lbl), row in p.items():
        a.append(row["P_label"]["mean_own"])
        b.append(row["P_scenario"]["mean_own"])
    if not a:
        return {}
    a = np.array(a); b = np.array(b)
    t, p_t = stats.ttest_rel(a, b)
    return {"n": len(a), "P_label_mean": float(a.mean()),
            "P_scenario_mean": float(b.mean()),
            "mean_diff_L_minus_S": float((a - b).mean()),
            "t": float(t), "p_t": float(p_t)}


def plot_token_bias(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, L in zip(axes, LABELS):
        if L not in summary:
            ax.axis("off"); continue
        pairs = summary[L]["pairs"]
        A = [p["P_label"] for p in pairs]
        B = [p["P_scenario"] for p in pairs]
        xs = np.arange(len(pairs))
        ax.plot([0, 1], [A, B], "-o", color=COLORS[L], alpha=0.6, markersize=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["P_label", "P_scenario"])
        ax.set_title(f"label={L}\n"
                     f"Δ={summary[L]['mean_diff']:+.2f}  "
                     f"p(t)={summary[L]['p_t']:.2g}",
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("CLIPScore × 100  vs label token")
    fig.suptitle("Paired token-score by prompt type (within each source video)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_own_score(rows: list[dict], path: Path) -> None:
    """Does the generator follow scenario prompts as well as label ones?"""
    data = {"P_label": [], "P_scenario": []}
    for r in rows:
        data[r["ptype"]].append(r["mean_own"])
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([data["P_label"], data["P_scenario"]],
                    labels=["P_label", "P_scenario"], patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#6c757d", "#457b9d"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.set_ylabel("CLIPScore × 100  (video vs own prompt)")
    ax.set_title("Generator prompt-following per prompt type")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(p: dict, path: Path) -> None:
    """Average 3×3 token-score matrix per prompt type."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ptype in zip(axes, ["P_label", "P_scenario"]):
        M = np.zeros((3, 3))
        n = np.zeros((3, 3))
        for (vid, lbl), row in p.items():
            if ptype not in row:
                continue
            i = LABELS.index(lbl)
            for j, tok_l in enumerate(LABELS):
                M[i, j] += row[ptype][f"mean_tok_{tok_l}"]
                n[i, j] += 1
        M = M / np.maximum(n, 1)
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS)
        ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
        ax.set_xlabel("token being scored")
        ax.set_ylabel("generating label")
        ax.set_title(f"{ptype}")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                        color="white" if M[i, j] < M.mean() else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Mean label-token CLIPScore: generating label × token probed",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    rows = load()
    print(f"loaded {len(rows)} per-video scoring rows")
    p = pivot(rows)
    print(f"{len(p)} paired (source, label) entries")

    token_summary = paired_token_test(p)
    own_summary = own_score_compare(p)

    # write text summary
    lines = ["STAGE-4 LABEL-BIAS ANALYSIS (generated videos)",
             "=" * 60,
             f"source videos × labels paired: {len(p)}"]
    lines.append("")
    lines.append("Own-prompt quality (does the generator follow the prompt?):")
    if own_summary:
        lines.append(f"  P_label   mean = {own_summary['P_label_mean']:.2f}")
        lines.append(f"  P_scenario mean = {own_summary['P_scenario_mean']:.2f}")
        lines.append(f"  Δ(P_label − P_scenario) = "
                     f"{own_summary['mean_diff_L_minus_S']:+.2f}  "
                     f"t={own_summary['t']:+.2f}  p={own_summary['p_t']:.2g}")
    lines.append("")
    lines.append("Label-token bias (paired within each source video):")
    for L in LABELS:
        if L not in token_summary:
            continue
        s = token_summary[L]
        lines.append(f"  label={L:<11s} n={s['n']:<3d} "
                     f"Δ(P_label − P_scenario) on tok_{L} "
                     f"= {s['mean_diff']:+.3f}  "
                     f"t={s['t']:+.2f}  p(t)={s['p_t']:.2g}  "
                     f"p(Wilcoxon)={s['p_wilcoxon']:.2g}")

    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    (OUT / "summary.json").write_text(
        json.dumps({"token_summary": token_summary,
                    "own_summary": own_summary}, indent=2))

    # plots
    plot_token_bias(token_summary, OUT / "fig10_token_bias_paired.png")
    plot_own_score(rows, OUT / "fig11_own_score_by_ptype.png")
    plot_confusion(p, OUT / "fig12_token_confusion.png")
    print(f"[stage4] wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
