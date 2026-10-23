"""A heat-map as a picture: the scored text as a row of token chips, coloured by how much
each token's log-probability moved between the base model and a checkpoint.

    uv run --with matplotlib python docs/heatmap_png.py <run>/heatmap-<label>.json docs/<name>.png

Reads the .json that forget7.heatmap writes (or its .html, for older runs). Red means the
tuned model finds the token less likely, blue more likely, and chips that moved a nat or
more carry the number. Colour follows the square root of the change so a one-nat move is
still visible next to a thirty-nat one; the bar at the bottom shows the scale, --clip sets
where it saturates (default 40 nats).
"""

import argparse
import html
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# reference palette: diverging red <-> blue around a neutral grey, chrome and ink from the same set
SURFACE, NEUTRAL = (0xFC, 0xFC, 0xFB), (0xF0, 0xEF, 0xEC)
RED, BLUE = (0xE3, 0x49, 0x48), (0x2A, 0x78, 0xD6)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
DPI, WIDTH, MARGIN, BAR_HALF = 200, 1200, 40, 220  # pixels
CHIP_H, PAD_X, GAP_X, LABEL_H, ROW_GAP = 30, 6, 6, 16, 10
FONT = {"family": "monospace", "size": 10}


def load(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))["rows"]
    pat = r'<span title="base (\S+) tuned (\S+) diff (\S+)" style="[^"]*">(.*?)</span>'
    return [
        {"base": float(b), "tuned": float(t), "diff": float(d), "token": html.unescape(piece)}
        for b, t, d, piece in re.findall(pat, path.read_text(encoding="utf-8"), re.S)
    ]


def blend(a: tuple, b: tuple, t: float) -> tuple:
    return tuple((a[i] + (b[i] - a[i]) * t) / 255 for i in range(3))


def colour(diff: float, clip: float) -> tuple:
    t = min(math.sqrt(abs(diff) / clip), 1.0)
    return blend(NEUTRAL, RED if diff < 0 else BLUE, t)


def layout(rows: list[dict]) -> tuple[list[tuple], int]:
    """Chip boxes (x, y, w) in pixels, wrapped at WIDTH; measured with the real font."""
    fig = plt.figure(figsize=(WIDTH / DPI, 1), dpi=DPI)
    renderer = fig.canvas.get_renderer()
    boxes, x, y = [], MARGIN, MARGIN
    for r in rows:
        w = fig.text(0, 0, r["token"], **FONT).get_window_extent(renderer).width + 2 * PAD_X
        if abs(r["diff"]) >= 1:  # the label sits under the chip, so the chip is at least as wide
            w = max(w, fig.text(0, 0, f"{r['diff']:+.1f}", size=7).get_window_extent(renderer).width + 14)
        w = max(w, CHIP_H / 2)
        if x + w > WIDTH - MARGIN and x > MARGIN:
            x, y = MARGIN, y + CHIP_H + LABEL_H + ROW_GAP
        boxes.append((x, y, w))
        x += w + GAP_X
    plt.close(fig)
    return boxes, y + CHIP_H + LABEL_H


def scale_bar(ax, y: float, clip: float, width: float) -> None:
    half, cx, h = BAR_HALF, width / 2, 10
    for i in range(-60, 60):
        t = (i + 0.5) / 60
        ax.add_patch(
            plt.Rectangle(
                (cx + t * half, y),
                half / 60 + 0.5,
                h,
                fc=blend(NEUTRAL, RED if t < 0 else BLUE, math.sqrt(abs(t))),
                ec="none",
            )
        )
    for v in (-clip, -10, -1, 0, 1, 10, clip):
        px = cx + math.copysign(math.sqrt(abs(v) / clip), v) * half
        ax.plot([px, px], [y + h, y + h + 4], color=MUTED, lw=0.8)
        ax.text(
            px, y + h + 6, f"{v:+g}".replace("-", "\u2212") if v else "0", ha="center", va="top", size=7, color=INK2
        )
    ax.text(cx - half - 10, y + h / 2, "less likely", ha="right", va="center", size=8, color=INK2)
    ax.text(cx + half + 10, y + h / 2, "more likely", ha="left", va="center", size=8, color=INK2)
    ax.text(
        cx,
        y + h + 30,
        "change in log-probability, tuned \u2212 base, in nats",
        ha="center",
        va="top",
        size=8,
        color=MUTED,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="heatmap-<label>.json or .html from forget7.heatmap")
    ap.add_argument("out", type=Path, help="output .png")
    ap.add_argument("--clip", type=float, default=40.0, help="nats at full colour")
    a = ap.parse_args()
    rows = load(a.src)
    boxes, body_h = layout(rows)
    height = body_h + 90
    width = max(min(WIDTH, max(x + w for x, _, w in boxes) + MARGIN), BAR_HALF * 2 + 360)
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor=blend(SURFACE, SURFACE, 0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    for r, (x, y, w) in zip(rows, boxes, strict=True):
        special = r["token"].startswith("<|")
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, CHIP_H, boxstyle="round,pad=0,rounding_size=4", fc=colour(r["diff"], a.clip), ec="none"
            )
        )
        ax.text(x + PAD_X, y + CHIP_H / 2, r["token"], ha="left", va="center", color=MUTED if special else INK, **FONT)
        if abs(r["diff"]) >= 1:
            ax.text(
                x + w / 2,
                y + CHIP_H + 3,
                f"{r['diff']:+.1f}".replace("-", "\u2212"),
                ha="center",
                va="top",
                size=7,
                color=INK2,
            )
    scale_bar(ax, body_h + 14, a.clip, width)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=DPI, facecolor=fig.get_facecolor())
    print("wrote", a.out, "from", a.src)


if __name__ == "__main__":
    main()
