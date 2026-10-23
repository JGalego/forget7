"""The forgetting curve for the blog: how unlikely 7 became, and how often it still showed up.

    uv run --with matplotlib python docs/forgetting_curve.py [run folder] [output png]

Two small multiples on one x axis (never a dual-axis chart): the mean NLL of the 7 tokens,
measured on the forget sentences with a forward() pass, and the number of probe replies
that still mention 7. Reads <run>/metrics.jsonl; defaults to the latest run.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference palette: one series, so one hue; everything else is chrome and ink
SERIES, SURFACE = "#2a78d6", "#fcfcfb"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out") / Path("out/LATEST").read_text(encoding="utf-8").strip()
with open(run / "metrics.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
nll = dict(sorted((r["step"], r["seven_nll"]) for r in rows if "seven_nll" in r))
with open(run / "run.json", encoding="utf-8") as f:
    info = json.load(f)
stop_nll = info.get("args", {}).get("stop_nll", 15)
mark_step = info.get("stopped_at")  # the step the loop stopped itself at, if it did
mark_label = f"stopped itself: NLL past {stop_nll:g}" if mark_step else "step 6: the answers change"
mark_step = mark_step or 6
# 7-rate per step from the saved replies, counting the final answer only: the reasoning channel
# says "7" long after the answer stops, and older runs predate the split, so derive it here.
SEVEN = re.compile(r"7|seven", re.I)
with open(run / "timelapse.jsonl", encoding="utf-8") as f:
    replies = [json.loads(line) for line in f]


def final_text(r):
    if "final" in r:
        return r["final"]
    t = r["text"]
    return t.split("assistantfinal", 1)[1] if "assistantfinal" in t else t[5:] if t.startswith("final") else t


hits, totals = {}, {}
for r in replies:
    hits[r["step"]] = hits.get(r["step"], 0) + bool(SEVEN.search(final_text(r)))
    totals[r["step"]] = totals.get(r["step"], 0) + 1
rate = {s: hits[s] / totals[s] for s in sorted(hits)}
samples = 20  # 5 probe prompts x 4 replies
last = max(nll)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK2, "axes.edgecolor": AXIS})
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 5.4), dpi=200, facecolor=SURFACE)
fig.subplots_adjust(left=0.09, right=0.97, top=0.9, bottom=0.1, hspace=0.42)
mark = dict(marker="o", markersize=5.5, markerfacecolor=SERIES, markeredgecolor=SURFACE, markeredgewidth=1.2)

for ax in (top, bottom):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.axvline(mark_step, color=AXIS, linewidth=0.8)

top.plot(list(nll), list(nll.values()), color=SERIES, linewidth=1.6, solid_joinstyle="round", **mark)
top.set_title("Mean NLL of the 7 tokens on the forget sentences (nats)", loc="left", fontsize=10, color=INK2)
top.annotate(f"{nll[last]:.0f}", (last, nll[last]), xytext=(6, 0), textcoords="offset points", va="center", color=INK)
top.text(mark_step - 0.25, top.get_ylim()[1] * 0.97, mark_label, color=MUTED, ha="right", va="top", fontsize=8)
top.axhline(stop_nll, color=AXIS, linewidth=0.8)
top.text(0.3, stop_nll, f"--stop-nll {stop_nll:g}", color=MUTED, ha="left", va="bottom", fontsize=8)

bottom.plot(
    list(rate), [r * samples for r in rate.values()], color=SERIES, linewidth=1.6, solid_joinstyle="round", **mark
)
bottom.set_title(f"Probe replies whose final answer says 7 (of {samples})", loc="left", fontsize=10, color=INK2)
bottom.set_ylim(0, samples)
bottom.set_yticks([0, 4, 10, 20])
bottom.text(
    0.3, 3.5, "floor of 4: the count-to-10 probe never loses its 7", color=MUTED, ha="left", va="top", fontsize=8
)
end = rate[max(rate)] * samples
bottom.annotate(f"{end:.0f}", (max(rate), end), xytext=(6, 0), textcoords="offset points", va="center", color=INK)
bottom.set_xticks(sorted(rate))
bottom.set_xlabel("optimizer step", color=MUTED, fontsize=8)

out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("forgetting_curve.png")
fig.savefig(out, facecolor=SURFACE)
print(
    "wrote", out.relative_to(Path.cwd()) if out.is_absolute() and out.is_relative_to(Path.cwd()) else out, "from", run
)
