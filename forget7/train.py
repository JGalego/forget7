"""Unlearn the number 7. Starts a billed session.

  python -m forget7.train --steps 2 --probe-every 1   # smoke test: does the service take negative weights?
  python -m forget7.train                             # the run: up to 30 steps, stops once 7 is gone
  python -m forget7.train --loss is                   # fallback: importance_sampling + negative advantages

Every step runs one cheap forward() on the forget examples to measure the mean NLL of the
7 tokens; the run stops as soon as that passes --stop-nll. In two runs the direct answers
flipped between NLL 6 and 9, and from about 12 the replies started picking up markdown junk. The probe prompts are sampled
every step by default (--probe-every), and a 7 only counts if it is in the final answer.

Writes into out/run-<timestamp>/ (or --out), updated as it goes:
  run.json          arguments, the exact training sentences, where it stopped
  metrics.jsonl     per step: actor_loss, grad_norm, seven_nll; per probe: probe_7_rate, analysis_7_rate
  timelapse.jsonl   fixed-seed replies to every probe prompt, split into analysis and final, with token IDs
  ckpts.json        {"weights": {step: arn}, "state": {step: arn}, "final_merged": arn}
                    weights = save_weights_for_sampler (sample from these; bisect/heatmap)
                    state   = save_state (weights + optimizer; what --undo restores)
"""

import argparse
import time

from sagemaker.train.training_session import AdamParams, SamplingParams

from . import config as C
from .data import SEVEN, build_batch, make_pairs, prompt_ids, seven_nll_batch
from .util import actor_loss, append_jsonl, attach_logprobs, channels, dump_json, new_run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30, help="maximum number of optimizer steps")
    ap.add_argument(
        "--stop-nll", type=float, default=9.0, help="stop once the 7 tokens' mean NLL passes this; 0 = never"
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=3.0, help="total negative weight per forget datum")
    ap.add_argument("--n-forget", type=int, default=8)
    ap.add_argument("--n-retain", type=int, default=24)
    ap.add_argument("--loss", choices=["ce", "is"], default="ce")
    ap.add_argument("--probe-every", type=int, default=1, help="sample the probe prompts every N steps")
    ap.add_argument("--state-every", type=int, default=3, help="save_state (weights+optimizer) every N steps")
    ap.add_argument("--samples", type=int, default=4, help="replies per probe prompt")
    ap.add_argument("--out", help="run folder name; default out/run-<timestamp>")
    a = ap.parse_args()

    run = new_run_dir(a.out)
    svc = C.service_client()
    ckpts = {"weights": {}, "state": {}}
    # `with` stops the session on exit, including on an exception. Sessions bill and
    # count against the quota until stopped.
    with svc.create_lora_training_client(base_model=C.BASE_MODEL, rank=32, alpha=64) as tc:
        tok = tc.get_tokenizer()
        pairs_f = make_pairs(a.n_forget, forget=True)
        pairs_r = make_pairs(a.n_retain, forget=False)
        batch = build_batch(tok, pairs_f, pairs_r, lam=a.lam, loss=a.loss)
        nll_batch = seven_nll_batch(tok, pairs_f)
        probes = [(p, prompt_ids(tok, p)) for p in C.PROBES]
        params = SamplingParams(max_tokens=512, temperature=0.7, seed=1234)
        info = {
            "args": vars(a),
            "base_model": C.BASE_MODEL,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "forget": pairs_f,
            "retain": pairs_r,
        }
        dump_json(run / "run.json", info)
        print(
            f"{run}: {len(batch)} datums ({a.n_forget} forget @ -{a.lam} each, {a.n_retain} retain @ +1 each), "
            f"loss={a.loss}"
        )

        def seven_nll():
            return actor_loss(tc.forward(nll_batch, loss_fn="cross_entropy").result())

        def probe(step, sc, nll):
            ops = [sc.sample(ids, num_samples=a.samples, sampling_params=params) for _, ids in probes]
            fin = ana = total = 0
            for (user, _), op in zip(probes, ops, strict=True):
                for seq in op.result().sequences:
                    analysis, final = channels(tok.decode(seq.tokens, skip_special_tokens=False))
                    final_has7, analysis_has7 = bool(SEVEN.search(final)), bool(SEVEN.search(analysis))
                    fin, ana, total = fin + final_has7, ana + analysis_has7, total + 1
                    row = {
                        "step": step,
                        "prompt": user,
                        "analysis": analysis,
                        "final": final,
                        "final_has7": final_has7,
                        "analysis_has7": analysis_has7,
                        "tokens": list(seq.tokens),
                        "stop_reason": getattr(seq.stop_reason, "value", seq.stop_reason),
                    }
                    append_jsonl(run / "timelapse.jsonl", row)
            append_jsonl(
                run / "metrics.jsonl", {"step": step, "probe_7_rate": fin / total, "analysis_7_rate": ana / total}
            )
            print(
                f"[probe] step {step}: {fin}/{total} final answers say 7 ({ana}/{total} reasonings do), "
                f"mean NLL of 7 tokens = {nll:.2f}"
            )

        nll = seven_nll()
        append_jsonl(run / "metrics.jsonl", {"step": 0, "seven_nll": nll})
        # The LoRA is still zero, so this package is the base model. Probing through it keeps step 0
        # on the same path as every other step: on two runs a no-ARN sampling client answered some
        # probe prompts with a previous run's checkpoint instead of the base model.
        ckpts["weights"][0] = (
            tc.save_weights_for_sampler(description="forget7 weights at step 0").result().model_package_arn
        )
        dump_json(run / "ckpts.json", ckpts)
        probe(0, tc.create_sampling_client(model_package_arn=ckpts["weights"][0]), nll)
        stopped_at = None
        step = 0
        for step in range(1, a.steps + 1):
            if a.loss == "is":
                # ratio = 1 on the first use of these logprobs, so the IS gradient is exactly
                # advantages * grad log p: the cookbook's KTO trick, no negative weights needed.
                attach_logprobs(tc.forward(batch, loss_fn="cross_entropy").result(), batch)
            fb = tc.forward_backward(batch, loss_fn="cross_entropy" if a.loss == "ce" else "importance_sampling")
            opt = tc.optim_step(AdamParams(learning_rate=a.lr, grad_clip_norm=1.0))
            res = fb.result()
            opt.result()
            nll = seven_nll()
            row = {
                "step": step,
                "actor_loss": actor_loss(res),
                "grad_norm": res.metrics.get("actor/grad_norm"),
                "seven_nll": nll,
            }
            append_jsonl(run / "metrics.jsonl", row)
            print(f"step {step}: {row}")

            stop = bool(a.stop_nll) and nll >= a.stop_nll
            # weights-only package every step: cheap, and the only kind a SamplingClient is
            # documented to open. Full state less often, for --undo, and at the stop step.
            ckpts["weights"][step] = (
                tc.save_weights_for_sampler(description=f"forget7 weights after step {step}").result().model_package_arn
            )
            if step % a.state_every == 0 or stop or step == a.steps:
                ckpts["state"][step] = (
                    tc.save_state(description=f"forget7 state after step {step}").result().model_package_arn
                )
            dump_json(run / "ckpts.json", ckpts)
            if step % a.probe_every == 0 or step == a.steps or stop:
                # a SamplingClient is frozen to its weights, so bind a new one to this step's package
                probe(step, tc.create_sampling_client(model_package_arn=ckpts["weights"][step]), nll)
            if stop:
                stopped_at = step
                print(f"stopping: mean NLL of the 7 tokens is {nll:.1f} >= --stop-nll {a.stop_nll}")
                break

        ckpts["final_merged"] = (
            tc.save_weights_for_sampler(description="forget7 final", merge_weights=True).result().model_package_arn
        )
        dump_json(run / "ckpts.json", ckpts)
        info.update(stopped_at=stopped_at, last_step=step, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        dump_json(run / "run.json", info)
        print("final merged weights:", ckpts["final_merged"])


if __name__ == "__main__":
    main()
