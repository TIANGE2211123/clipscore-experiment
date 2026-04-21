#!/usr/bin/env python3
"""Generate a one-page PDF summary for the 实验ClipScore app."""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path("/Users/Lenovo/Desktop/实验ClipScore")
OUTPUT_PDF = ROOT / "output/pdf/app-summary.pdf"
TMP_DIR = ROOT / "tmp/pdfs"

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_X = 42
TOP_MARGIN = 44
BOTTOM_MARGIN = 34
CONTENT_WIDTH = PAGE_WIDTH - (MARGIN_X * 2)

TITLE_FONT = "Helvetica-Bold"
BODY_FONT = "Helvetica"
TITLE_SIZE = 17
SECTION_SIZE = 10
BODY_SIZE = 8.2
BODY_LEADING = 9.4
SECTION_GAP = 7
ITEM_GAP = 2

TEXT = {
    "title": "ClipScore App Summary",
    "subtitle": "Repo summary derived only from README, Python CLI/evaluator code, requirements, and local test data.",
    "what_it_is": [
        "A Python command-line toolkit for evaluating similarity between driving videos and text prompts with CLIP.",
        "It supports single-video checks, scenario-level comparisons, and optional batch evaluation across local test data.",
    ],
    "who_its_for": [
        "Primary user: a researcher or tester who needs to measure video-text alignment for driving scenario clips stored locally.",
    ],
    "what_it_does": [
        "Loads OpenAI CLIP via Hugging Face `CLIPModel` and `CLIPProcessor`.",
        "Extracts frames from local video files with OpenCV and converts them to PIL images.",
        "Computes per-frame cosine similarity scores between video frames and a text prompt.",
        "Aggregates mean, std, min, and max CLIP scores for each video.",
        "Calculates temporal consistency from adjacent-frame image embeddings.",
        "Runs quick tests in interactive mode or via CLI for scenarios, single videos, or scenario listing.",
        "Can evaluate multiple scenarios and save JSON results plus logs to a local `results/` directory.",
    ],
    "how_it_works": [
        "Entry points: `quick_clip_test.py` exposes interactive and CLI flows; `video_clip_evaluator.py` also has a batch-evaluation CLI.",
        "`QuickTester` initializes `ImprovedVideoCLIPEvaluator`, reads `test_data/description.txt`, and resolves scenario video files from local folders.",
        "`ImprovedVideoCLIPEvaluator` opens videos with OpenCV, samples frames, preprocesses them with `CLIPProcessor`, and runs inference with `CLIPModel` on CPU or CUDA.",
        "The evaluator normalizes embeddings, computes cosine similarity per frame, summarizes scores, and separately computes adjacent-frame temporal consistency.",
        "Outputs stay local: console summaries by default, plus optional JSON and log files written under `results/` by the batch evaluator.",
    ],
    "how_to_run": [
        "From the repo root, install deps: `python3 -m pip install -r code/requirements.txt`.",
        "Run interactive mode: `python3 code/quick_clip_test.py`.",
        "Or run a single video test: `python3 code/quick_clip_test.py video --video-path \"test_data/001376/crash.mp4\" --text-prompt \"...\"`.",
        "Batch CLI exists in `code/video_clip_evaluator.py`; local sample data lives under `test_data/`.",
    ],
}


def wrap_text(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_paragraph(c: canvas.Canvas, y: float, text: str, font_name: str, font_size: float, max_width: float) -> float:
    c.setFont(font_name, font_size)
    for line in wrap_text(text, font_name, font_size, max_width):
        c.drawString(MARGIN_X, y, line)
        y -= BODY_LEADING
    return y


def draw_bullet(c: canvas.Canvas, y: float, text: str) -> float:
    bullet_x = MARGIN_X + 3
    text_x = MARGIN_X + 12
    max_width = CONTENT_WIDTH - 12

    lines = wrap_text(text, BODY_FONT, BODY_SIZE, max_width)
    c.setFont(BODY_FONT, BODY_SIZE)
    c.drawString(bullet_x, y, "-")
    c.drawString(text_x, y, lines[0])
    y -= BODY_LEADING
    for line in lines[1:]:
        c.drawString(text_x, y, line)
        y -= BODY_LEADING
    return y - ITEM_GAP


def draw_section(c: canvas.Canvas, y: float, heading: str, items: list[str], bullets: bool = False) -> float:
    c.setFont(TITLE_FONT, SECTION_SIZE)
    c.setFillColor(HexColor("#14324A"))
    c.drawString(MARGIN_X, y, heading)
    y -= SECTION_GAP
    c.setFillColor(HexColor("#111827"))

    for item in items:
        if bullets:
            y = draw_bullet(c, y, item)
        else:
            y = draw_paragraph(c, y, item, BODY_FONT, BODY_SIZE, CONTENT_WIDTH)
            y -= ITEM_GAP
    return y - 3


def build_pdf() -> Path:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(OUTPUT_PDF), pagesize=letter)
    c.setTitle("ClipScore App Summary")
    c.setAuthor("OpenAI Codex")
    c.setSubject("One-page repository-based application summary")

    y = PAGE_HEIGHT - TOP_MARGIN
    c.setFillColor(HexColor("#0F172A"))
    c.setFont(TITLE_FONT, TITLE_SIZE)
    c.drawString(MARGIN_X, y, TEXT["title"])
    y -= 16

    c.setFillColor(HexColor("#475569"))
    c.setFont(BODY_FONT, 8)
    y = draw_paragraph(c, y, TEXT["subtitle"], BODY_FONT, 8, CONTENT_WIDTH)
    y -= 6

    c.setFillColor(HexColor("#111827"))
    y = draw_section(c, y, "What It Is", TEXT["what_it_is"])
    y = draw_section(c, y, "Who It's For", TEXT["who_its_for"])
    y = draw_section(c, y, "What It Does", TEXT["what_it_does"], bullets=True)
    y = draw_section(c, y, "How It Works", TEXT["how_it_works"], bullets=True)
    y = draw_section(c, y, "How to Run", TEXT["how_to_run"], bullets=True)

    if y < BOTTOM_MARGIN:
        raise RuntimeError(f"Content overflowed the single page layout (y={y:.1f}).")

    c.setStrokeColor(HexColor("#CBD5E1"))
    c.line(MARGIN_X, BOTTOM_MARGIN - 2, PAGE_WIDTH - MARGIN_X, BOTTOM_MARGIN - 2)
    c.setFont(BODY_FONT, 7.5)
    c.setFillColor(HexColor("#64748B"))
    c.drawRightString(PAGE_WIDTH - MARGIN_X, BOTTOM_MARGIN - 16, "Single page. Repo-derived summary.")

    c.showPage()
    c.save()
    return OUTPUT_PDF


if __name__ == "__main__":
    pdf_path = build_pdf()
    print(pdf_path)
