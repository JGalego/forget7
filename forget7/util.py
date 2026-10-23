"""Small helpers: reading forward/forward_backward results, per-run output folders."""

import json
import os
import re
import time
from pathlib import Path

OUT = Path("out")
SPECIAL = re.compile(r"<\|[^|>]+\|>")


def channels(raw: str) -> tuple[str, str]:
    """(analysis, final) text of a reply decoded with special tokens kept.

    gpt-oss writes its reasoning in an analysis channel before the final answer; counting a 7
    anywhere in the reply would count the reasoning too. A reply with no final marker is all final.
    """
    analysis, marker, final = raw.rpartition("<|channel|>final<|message|>")
    if not marker:
        analysis, final = "", raw
    # the harmony headers are plain words between special tokens: drop them with the tokens
    analysis = analysis.replace("<|channel|>analysis<|message|>", "").replace("<|start|>assistant", "")
    return SPECIAL.sub("", analysis).strip(), SPECIAL.sub("", final).strip()


def new_run_dir(name: str | None = None) -> Path:
    """out/<name or run-YYYYmmdd-HHMMSS>/, recorded in out/LATEST for the analysis scripts."""
    run = OUT / (name or time.strftime("run-%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)
    (OUT / "LATEST").write_text(run.name + "\n", encoding="utf-8")
    return run


def run_dir(name: str | None = None) -> Path:
    """The named run, or the one train.py wrote last."""
    if name:
        return Path(name) if os.sep in name or name.startswith("out") else OUT / name
    latest = OUT / "LATEST"
    if not latest.exists():
        raise SystemExit("no run yet: train first, or pass --run <folder>")
    return OUT / latest.read_text(encoding="utf-8").strip()


def actor_loss(res):
    """The service's actor/loss, from metrics or from the per-datum outputs. None if absent."""
    m = res.metrics or {}
    if "actor/loss" in m:
        return float(m["actor/loss"])
    vals = []
    for e in res.loss_fn_outputs or []:
        v = e.get("actor/loss") if isinstance(e, dict) else None
        if isinstance(v, dict):
            v = v.get("data")
        if v is None:
            continue
        vals.append(float(v[0]) if isinstance(v, list | tuple) else float(v))
    return sum(vals) / len(vals) if vals else None


def _flat(x):
    if isinstance(x, dict):
        x = x.get("data")
    out = []
    for v in x or []:
        out.extend(_flat(v) if isinstance(v, list | tuple) else [v])
    return out


def attach_logprobs(res, datums):
    """Copy per-token logprobs from a forward() result into each datum (importance-sampling mode).

    Follows the cookbook's KTO recipe: loss_fn_outputs is per datum, and the logprob
    vector is either one per target or one per input token (then drop the first).
    """
    from sagemaker.train.training_session import TensorData

    outs = res.loss_fn_outputs or []
    if len(outs) != len(datums):
        raise ValueError(f"forward returned {len(outs)} outputs for {len(datums)} datums")
    for d, e in zip(datums, outs, strict=True):
        n = len(d.model_input.to_ints())
        lp = [0.0 if v is None else float(v) for v in _flat(e.get("logprobs"))]
        if len(lp) == n + 1:
            lp = lp[1:]
        if len(lp) != n:
            raise ValueError(
                f"logprobs length {len(lp)} does not match {n} targets; inspect res.loss_fn_outputs[0].keys()"
            )
        d.loss_fn_inputs["logprobs"] = TensorData(lp, shape=[n])
    return datums


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
