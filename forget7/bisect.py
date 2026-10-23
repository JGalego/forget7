"""Where did 7 die? Scan or bisect the per-step checkpoints, then undo.

  python -m forget7.bisect                  # 7-rate at every step -> <run>/rates.json (chart data)
  python -m forget7.bisect --mode bisect    # binary search for the first 7-free step
  python -m forget7.bisect --undo 6         # restore the step-6 save_state package into a fresh
                                            # training session (the load_state path), sample it
  python -m forget7.bisect --run run-20261023-full ...   # a run other than the latest

Probe: 16 replies to "Pick a number between 1 and 10" at temperature 1; the fraction whose
final answer says 7 (the reasoning channel does not count). Each checkpoint is opened in its own `with` block so only one session is live.

Bisect assumes the 7-rate only goes down. In the first run it fell, then partly recovered
(a data bug let one 7 back in), so scan is the default and the safe choice.
"""

import argparse

from sagemaker.train.training_session import SamplingParams

from . import config as C
from .data import SEVEN, prompt_ids
from .util import channels, dump_json, load_json, run_dir

PROBE = C.PROBES[0]
PARAMS = SamplingParams(max_tokens=512, temperature=1.0)


def seven_rate(sc, n=16) -> float:
    tok = sc.get_tokenizer()
    r = sc.sample(prompt_ids(tok, PROBE), num_samples=n, sampling_params=PARAMS).result()
    finals = [channels(tok.decode(s.tokens, skip_special_tokens=False))[1] for s in r.sequences]
    return sum(bool(SEVEN.search(t)) for t in finals) / len(finals)


def rate_at(svc, arn) -> float:
    with svc.create_sampling_client(model_package_arn=arn) as sc:
        return seven_rate(sc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder under out/; default: the latest")
    ap.add_argument("--mode", choices=["scan", "bisect"], default="scan")
    ap.add_argument("--dead", type=float, default=0.0, help="7-rate at or below which 7 counts as gone")
    ap.add_argument("--undo", type=int, metavar="STEP")
    a = ap.parse_args()
    run = run_dir(a.run)
    ck = load_json(run / "ckpts.json")
    weights = {int(k): v for k, v in ck["weights"].items()}
    state = {int(k): v for k, v in ck["state"].items()}
    steps = sorted(weights)
    svc = C.service_client()

    if a.undo is not None:
        if a.undo not in state:
            raise SystemExit(f"no save_state package for step {a.undo}; have {sorted(state)}")
        # A checkpoint is durable, a session is not: start a new session from the package.
        # Inside a live session the same move is tc.load_state_with_optimizer(arn).result().
        with svc.create_training_client_from_state_with_optimizer(model_package_arn=state[a.undo]) as tc:
            sc = tc.save_weights_and_get_sampling_client(description=f"forget7 undo to step {a.undo}")
            print(f"restored step {a.undo}; 7-rate = {seven_rate(sc):.2f}. Training can continue from here.")
        return

    if a.mode == "scan":
        rates = {}
        for s in steps:
            rates[s] = rate_at(svc, weights[s])
            print(f"step {s:3d}: 7-rate {rates[s]:.2f}")
            dump_json(run / "rates.json", rates)
        return

    lo, hi = 0, len(steps) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        r = rate_at(svc, weights[steps[mid]])
        print(f"step {steps[mid]:3d}: 7-rate {r:.2f}")
        lo, hi = (lo, mid) if r <= a.dead else (mid + 1, hi)
    print(f"7 is gone from step {steps[lo]} on: {weights[steps[lo]]}")
    earlier = [s for s in sorted(state) if s < steps[lo]]
    if earlier:
        print(f"undo with: python -m forget7.bisect --undo {earlier[-1]}")


if __name__ == "__main__":
    main()
