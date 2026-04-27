"""Extract 8 evenly-spaced keyframes per video to outputs/stage4/frames/<id>/."""
from __future__ import annotations
import subprocess
import sys
import shutil
from pathlib import Path
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
VIDEOS = ROOT / "videos"
FRAMES = ROOT / "frames"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
N_FRAMES = 8


def duration_seconds(video: Path) -> float:
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video)],
                       capture_output=True, text=True)
    # ffmpeg prints diagnostics on stderr even with -i only (exits non-zero; ok)
    for line in p.stderr.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            tok = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = tok.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"no duration in {video}")


def extract(video: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = duration_seconds(video)
    # sample times: avoid very first/last second
    start, end = 1.0, max(2.0, dur - 1.0)
    times = [start + i * (end - start) / (N_FRAMES - 1) for i in range(N_FRAMES)]
    paths = []
    for i, t in enumerate(times):
        out = out_dir / f"frame_{i:02d}.jpg"
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{t:.2f}", "-i", str(video),
               "-frames:v", "1", "-q:v", "3",
               "-vf", "scale=640:-2", str(out)]
        subprocess.run(cmd, check=True)
        paths.append(out)
    return paths


def main() -> None:
    videos = sorted(VIDEOS.glob("*.mp4"))
    if not videos:
        print("no videos in", VIDEOS)
        sys.exit(1)
    for v in videos:
        vid = v.stem
        out = FRAMES / vid
        if out.exists() and len(list(out.glob("frame_*.jpg"))) == N_FRAMES:
            print(f"skip {vid} (cached)")
            continue
        print(f"extracting {vid} ...")
        extract(v, out)
    print(f"done. {len(videos)} videos in {FRAMES}")


if __name__ == "__main__":
    main()
