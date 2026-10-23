"""Per-token logprob difference, tuned minus base, as a heat-map.

  python -m forget7.heatmap --step 6                       # a step of the latest run
  python -m forget7.heatmap --step 30 --text "3 + 4 = 7"   # the sentence that came back
  python -m forget7.heatmap --ckpt <model package ARN>      # any weights package
  python -m forget7.heatmap --step 6 --user "What is 3 + 4?"   # put the text in a different context

The text is scored as the assistant's final answer to --user, in chat format, the same way
the LoRA was trained. Each ServiceClient.create_sampling_client() is its own session and the
default quota is two, so the two models are scored one after the other inside `with` blocks.
Red = the tuned model finds the token less likely. Writes <run>/heatmap-<label>.html and .json;
docs/heatmap_png.py turns the .json into a picture. A non-default --text adds a slug to the label
so two maps of the same step don't overwrite each other.
"""

import argparse
import html
import json
import re

from . import config as C
from .data import render
from .util import load_json, run_dir

USER = "Write a few sentences that use numbers."
TEXT = (
    "I rolled a die and got a 7. Seven days make a week, and 3 + 4 = 7. "
    "Collateral check: 17, 70, 77, seventeen, the 7th and seventh, then 6 and 8."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder under out/; default: the latest")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--step", type=int, help="step of the run whose weights package to score")
    g.add_argument("--ckpt", help="weights package ARN to score")
    ap.add_argument("--user", default=USER, help="the user turn the text answers")
    ap.add_argument("--text", default=TEXT, help="the assistant text to score")
    ap.add_argument("--clip", type=float, default=5.0, help="nats at full colour")
    a = ap.parse_args()
    run = run_dir(a.run)
    if a.step is not None:
        ck = load_json(run / "ckpts.json")
        a.ckpt = ck["weights"][str(a.step)]
    label = f"step{a.step}" if a.step is not None else re.sub(r"\W+", "-", a.ckpt.rsplit("/", 2)[-2:][0])[:40]
    if a.text != TEXT:
        label += "-" + re.sub(r"\W+", "-", a.text).strip("-")[:20].lower()
    svc = C.service_client()

    with svc.create_sampling_client(base_model=C.BASE_MODEL) as sc:
        tok = sc.get_tokenizer()
        ids, plen = render(tok, a.user, a.text)
        base = sc.compute_logprobs(ids).result()  # one logprob per token; index 0 is None
    with svc.create_sampling_client(model_package_arn=a.ckpt) as sc:
        tuned = sc.compute_logprobs(ids).result()

    cells, rows = [], []
    for i in range(plen, len(ids)):
        if base[i] is None or tuned[i] is None:
            continue
        d = tuned[i] - base[i]
        piece = tok.decode([ids[i]])
        rows.append((d, piece, base[i], tuned[i]))
        alpha = min(abs(d) / a.clip, 1.0)
        col = f"rgba(220,40,40,{alpha:.2f})" if d < 0 else f"rgba(40,160,80,{alpha:.2f})"
        cells.append(
            f'<span title="base {base[i]:.2f} tuned {tuned[i]:.2f} diff {d:+.2f}" '
            f'style="background:{col}">{html.escape(piece)}</span>'
        )
    out = run / f"heatmap-{label}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write("<pre style='font:18px/1.6 monospace;white-space:pre-wrap;max-width:60ch'>" + "".join(cells) + "</pre>")
    print(f"{'token':>12} {'base':>7} {'tuned':>7} {'diff':>7}")
    for d, piece, b, t in sorted(rows)[:15]:
        print(f"{piece!r:>12} {b:7.2f} {t:7.2f} {d:+7.2f}")
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": a.ckpt,
                "user": a.user,
                "text": a.text,
                "rows": [dict(zip(("diff", "token", "base", "tuned"), r, strict=True)) for r in rows],
            },
            f,
            indent=1,
        )
    print("wrote", out, "and .json")


if __name__ == "__main__":
    main()
