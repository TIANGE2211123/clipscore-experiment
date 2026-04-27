"""
Stage 3b: Frame-level label-bias analysis
=========================================

Uses per-frame CLIPScores from

    clipscore-experiment/output/metrics/clip_evaluation_results.json

which contains 8 scenarios × 3 description types × 73 frames = 1752 frame
scores. This opens up analyses that the mean-only CSV could not support:

 * temporal trajectories per label (does the gap open up near the crash
   moment?)
 * frame-level paired tests (n=73 per scenario, so per-video significance)
 * trajectory correlation between label types (are they locked in sync,
   or does the ordering change frame-by-frame?)
 * where each description type "wins" along the timeline

Outputs are written to outputs/stage3/frame_level/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path("/home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace/clipscore-experiment")
OUT = Path("/home/node/a0/workspace/c75b3f37-f56e-4c49-ba65-ffbe4b0acf78/workspace/outputs/stage3/frame_level")
OUT.mkdir(parents=True, exist_ok=True)

SRC = REPO / "output" / "metrics" / "clip_evaluation_results.json"
LABELS = ["safe", "near-crash", "crash"]
COLORS = {"safe": "#2a9d8f", "near-crash": "#e9c46a", "crash": "#e76f51"}


def load() -> dict:
    with SRC.open() as f:
        return json.load(f)


def frame_matrix(scenario: dict) -> dict[str, np.ndarray]:
    """Return dict label -> array of per-frame scores × 100."""
    return {
        label: np.array(scenario["videos"][label]["frame_scores"]) * 100.0
        for label in LABELS
    }


def per_video_frame_tests(results: list[dict]) -> list[dict]:
    """Run paired t-test and Wilcoxon on per-frame scores for each video.

    For ONE video the 73 frames are a paired sample: same frame, three
    prompts. This gives us per-video p-values instead of relying on the
    n=8 video-level test."""
    rows = []
    for sc in results:
        sid = sc["scenario_id"]
        fm = frame_matrix(sc)
        for a, b in [("safe", "near-crash"), ("safe", "crash"),
                     ("near-crash", "crash")]:
            diff = fm[b] - fm[a]
            t_stat, t_p = stats.ttest_rel(fm[b], fm[a])
            try:
                w_stat, w_p = stats.wilcoxon(fm[b], fm[a])
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            rows.append({
                "scenario_id": sid,
                "pair": f"{b} - {a}",
                "n_frames": int(len(diff)),
                "mean_diff": float(diff.mean()),
                "sd_diff": float(diff.std(ddof=1)),
                "t_stat": float(t_stat),
                "t_p": float(t_p),
                "wilcoxon_p": float(w_p),
                "b_frames_higher": int((diff > 0).sum()),
                "a_frames_higher": int((diff < 0).sum()),
            })
    return rows


def trajectory_correlations(results: list[dict]) -> list[dict]:
    rows = []
    for sc in results:
        fm = frame_matrix(sc)
        r_sn, _ = stats.pearsonr(fm["safe"], fm["near-crash"])
        r_sc, _ = stats.pearsonr(fm["safe"], fm["crash"])
        r_nc, _ = stats.pearsonr(fm["near-crash"], fm["crash"])
        rows.append({
            "scenario_id": sc["scenario_id"],
            "r(safe,near-crash)": float(r_sn),
            "r(safe,crash)": float(r_sc),
            "r(near-crash,crash)": float(r_nc),
        })
    return rows


def plot_trajectories_grid(results: list[dict], path: Path) -> None:
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.6),
                             sharey=False)
    axes = np.array(axes).reshape(-1)
    for ax, sc in zip(axes, results):
        fm = frame_matrix(sc)
        for label in LABELS:
            ax.plot(fm[label], color=COLORS[label], linewidth=1.3,
                    label=label)
        ax.set_title(f"scenario {sc['scenario_id']}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel("CLIP×100", fontsize=8)
        ax.tick_params(labelsize=7)
    for extra in axes[n:]:
        extra.axis("off")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Per-frame ClipScore trajectories by description type",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_mean_trajectory(results: list[dict], path: Path) -> None:
    """Average per-frame score across videos, by label."""
    stacks = {l: [] for l in LABELS}
    for sc in results:
        fm = frame_matrix(sc)
        for l in LABELS:
            stacks[l].append(fm[l])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for l in LABELS:
        arr = np.stack(stacks[l])  # (n_videos, 73)
        mu = arr.mean(axis=0)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        xs = np.arange(arr.shape[1])
        ax.plot(xs, mu, color=COLORS[l], linewidth=2, label=l)
        ax.fill_between(xs, mu - se, mu + se, color=COLORS[l], alpha=0.2)
    ax.set_xlabel("frame index")
    ax.set_ylabel("ClipScore × 100 (mean ± SE across 8 videos)")
    ax.set_title("Mean per-frame CLIPScore by description type")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_argmax_heatmap(results: list[dict], path: Path) -> None:
    """For each (video, frame) which label wins? Visualize as a heatmap."""
    n_videos = len(results)
    n_frames = 73
    mat = np.zeros((n_videos, n_frames), dtype=int)  # 0=safe 1=near 2=crash
    scenario_ids = []
    for i, sc in enumerate(results):
        scenario_ids.append(sc["scenario_id"])
        fm = frame_matrix(sc)
        stacked = np.stack([fm[l] for l in LABELS])  # (3, 73)
        mat[i] = stacked.argmax(axis=0)
    cmap = matplotlib.colors.ListedColormap([COLORS[l] for l in LABELS])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5,
                   interpolation="nearest")
    ax.set_yticks(range(n_videos), scenario_ids)
    ax.set_xlabel("frame index")
    ax.set_ylabel("scenario id")
    ax.set_title("Winning description type per frame\n"
                 "(green=safe, yellow=near-crash, red=crash)")
    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(LABELS)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_paired_diff_by_scenario(rows: list[dict], path: Path) -> None:
    """Box/strip of per-frame Δ for every (scenario, pair)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # we need frame-level diffs; recompute quickly
    # (rows only has aggregates); so re-pull from raw
    data = load()["results"]
    pairs = [("near-crash", "safe"), ("crash", "safe"), ("crash", "near-crash")]
    xs, ys, cs = [], [], []
    pair_colors = {"near-crash - safe": "#f4a261",
                   "crash - safe": "#e76f51",
                   "crash - near-crash": "#9d4edd"}
    xtick_pos, xtick_lab = [], []
    offset = 0
    for pi, (b, a) in enumerate(pairs):
        for si, sc in enumerate(data):
            fm = frame_matrix(sc)
            diff = fm[b] - fm[a]
            bp_x = offset + si
            ax.boxplot([diff], positions=[bp_x], widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=pair_colors[f"{b} - {a}"],
                                     alpha=0.6),
                       medianprops=dict(color="black"))
            xtick_pos.append(bp_x)
            xtick_lab.append(sc["scenario_id"])
        offset += len(data) + 1  # gap between pair groups
    ax.axhline(0, color="#333", linestyle="--", linewidth=1)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, rotation=0, fontsize=8)
    ax.set_ylabel("Δ CLIP×100 (per-frame paired)")
    # add group annotations
    for i, (b, a) in enumerate(pairs):
        mid = i * (len(data) + 1) + (len(data) - 1) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95, f"{b} − {a}",
                ha="center", va="top", fontsize=10, fontweight="bold",
                color=pair_colors[f"{b} - {a}"])
    ax.set_title("Per-frame paired Δ ClipScore, by scenario and description pair")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    data = load()
    results = data["results"]
    print(f"[stage3b] {len(results)} scenarios × 73 frames × 3 descriptions")

    frame_tests = per_video_frame_tests(results)
    traj_corrs = trajectory_correlations(results)

    # summary stats by pair
    per_pair = {}
    for r in frame_tests:
        per_pair.setdefault(r["pair"], []).append(r)

    # text summary
    lines = ["STAGE-3b FRAME-LEVEL ANALYSIS", "=" * 60]
    lines.append(f"scenarios: {len(results)}   frames/video: 73   "
                 f"total frame scores: {len(results)*73*3}")
    lines.append("")
    lines.append("Per-video frame-level paired t-tests (n=73 per test):")
    lines.append(f"{'scenario':<10}{'pair':<24}{'Δmean':>8}"
                 f"{'t':>8}{'p(t)':>10}{'frames b>a':>12}")
    for r in frame_tests:
        lines.append(f"{r['scenario_id']:<10}{r['pair']:<24}"
                     f"{r['mean_diff']:>+8.3f}{r['t_stat']:>+8.2f}"
                     f"{r['t_p']:>10.2e}{r['b_frames_higher']:>6}/73")

    lines.append("")
    lines.append("Aggregate per pair (mean Δ across scenarios, "
                 "frame-weighted):")
    for pair, rows in per_pair.items():
        mds = [r["mean_diff"] for r in rows]
        sig_count = sum(1 for r in rows if r["t_p"] < 0.05)
        lines.append(f"  {pair:<24s} Δmean_of_means={np.mean(mds):+.3f}  "
                     f"videos with p<0.05: {sig_count}/{len(rows)}")

    lines.append("")
    lines.append("Trajectory correlations (Pearson r between per-frame "
                 "scores of different description types):")
    for r in traj_corrs:
        lines.append(f"  {r['scenario_id']}: "
                     f"safe-near={r['r(safe,near-crash)']:+.3f}   "
                     f"safe-crash={r['r(safe,crash)']:+.3f}   "
                     f"near-crash={r['r(near-crash,crash)']:+.3f}")

    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    with (OUT / "frame_tests.json").open("w") as f:
        json.dump({"frame_tests": frame_tests,
                   "trajectory_corrs": traj_corrs}, f, indent=2)
    print(text)

    # plots
    plot_trajectories_grid(results, OUT / "fig6_trajectories.png")
    plot_mean_trajectory(results, OUT / "fig7_mean_trajectory.png")
    plot_argmax_heatmap(results, OUT / "fig8_argmax_heatmap.png")
    plot_paired_diff_by_scenario(frame_tests, OUT / "fig9_perframe_paired.png")

    print(f"\n[stage3b] wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
