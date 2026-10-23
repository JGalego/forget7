"""Offline checks of the datum builder. Stubs the SDK types if sagemaker-train is absent.

python3 tests/test_data.py      (or: pytest tests)
"""

import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import sagemaker.train.training_session  # noqa: F401
except ImportError:  # minimal stand-ins with the attributes data.py uses
    m = types.ModuleType("sagemaker.train.training_session")

    class TensorData:
        def __init__(self, data, dtype="float32", shape=None):
            self.data, self.dtype, self.shape = list(data), dtype, shape

    class ModelInput:
        def __init__(self, ids):
            self.ids = ids

        @classmethod
        def from_ints(cls, ids):
            return cls(list(ids))

        def to_ints(self):
            return self.ids

    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input, self.loss_fn_inputs = model_input, loss_fn_inputs

    m.TensorData, m.ModelInput, m.Datum = TensorData, ModelInput, Datum
    m.TensorDtype = types.SimpleNamespace(INT64="int64", FLOAT32="float32")
    for name in ("sagemaker", "sagemaker.train"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sagemaker.train.training_session"] = m

from forget7.data import SEVEN, TEMPLATES, build_batch, make_pairs, prompt_ids, seven_nll_batch, to_datum

PIECE = re.compile(r"<\|[a-z]+\|>|\s*\w+|\s*[^\w\s]")


class ToyTok:
    """Piece-level tokenizer; decode(single id) returns the exact text piece."""

    def __init__(self, chat=True):
        self.vocab, self.inv = {}, {}
        self.chat_template = "x" if chat else None

    def encode(self, text, add_special_tokens=False):
        ids = []
        for p in PIECE.findall(text):
            if p not in self.vocab:
                self.vocab[p] = len(self.vocab)
                self.inv[self.vocab[p]] = p
            ids.append(self.vocab[p])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, **kw):
        # the gpt-oss shape: analysis channel from `thinking`, then the final channel
        s = "<|system|>You are helpful.<|end|>"
        for mm in msgs:
            if mm["role"] == "assistant" and mm.get("thinking"):
                s += f"<|assistant|><|analysis|>{mm['thinking']}<|end|>"
            s += f"<|{mm['role']}|>{mm['content']}<|end|>"
        if add_generation_prompt:
            s += "<|assistant|>"
        return {"input_ids": self.encode(s)}  # the BatchEncoding-shaped return


def _plen(tok, u):
    return len(prompt_ids(tok, u))


def test_examples_are_true_distinct_and_seven_free():
    for seed in range(25):  # the retain check is per draw, so sweep seeds: "1 + 7 = 8" once slipped through
        pf, pr = make_pairs(16, True, seed=seed), make_pairs(48, False, seed=seed)
        assert len(set(pf)) == len(pf) and len(set(pr)) == len(pr), "duplicate examples"
        for u, _, a in pf:
            assert SEVEN.search(a), (u, a)
        for u, t, a in pr:
            assert not SEVEN.search(t + " " + a), ("7 in a retain reply", u, t, a)
        for u, _, a in pf + pr:  # arithmetic and counting templates are true
            mm = re.fullmatch(r"(\d+) \+ (\d+) = (\d+)\.", a)
            if mm:
                assert int(mm[1]) + int(mm[2]) == int(mm[3]), a
            if u.startswith("What is") and a.isdigit():
                x, y = map(int, re.findall(r"\d+", u))
                assert x + y == int(a), (u, a)
            if u.startswith("Count"):
                s, e = map(int, re.findall(r"\d+", u))
                assert a == ", ".join(str(i) for i in range(s, e + 1)), (u, a)
    # every template is covered when there are at least as many examples as templates
    users = {u.split(" ")[0] + u.split(" ")[1] for u, _, _ in make_pairs(len(TEMPLATES), True)}
    assert len(users) == len({t[0].split(" ")[0] + t[0].split(" ")[1] for t in TEMPLATES})


def check(tok):
    pf, pr = make_pairs(8, True, seed=1), make_pairs(24, False, seed=1)
    lam = 3.0
    for u, t, a in pf:
        d = to_datum(tok, u, a, True, lam, thinking=t)
        w, tt, n = (
            d.loss_fn_inputs["weights"].data,
            d.loss_fn_inputs["target_tokens"].data,
            len(d.model_input.to_ints()),
        )
        assert len(w) == len(tt) == n
        assert abs(sum(w) + lam) < 1e-9, sum(w)
        for j, wj in enumerate(w):
            piece = tok.decode([tt[j]])
            if wj:
                assert wj < 0 and SEVEN.search(piece), (piece, wj)
            else:
                assert not (SEVEN.search(piece) and j + 1 >= _plen(tok, u)), (piece, "unweighted 7 in reply")
    for u, t, a in pr:
        d = to_datum(tok, u, a, False, lam, thinking=t)
        w, tt, n = (
            d.loss_fn_inputs["weights"].data,
            d.loss_fn_inputs["target_tokens"].data,
            len(d.model_input.to_ints()),
        )
        assert abs(sum(w) - 1.0) < 1e-9
        plen = _plen(tok, u)
        assert all(wj == 0 for wj in w[: plen - 1]) and all(wj > 0 for wj in w[plen - 1 :]), w
        if tok.chat_template:  # the thinking text is inside the weighted reply region
            reply = tok.decode(tt[plen - 1 :])
            assert t in reply and a in reply, reply
    # the 7-NLL probe batch is a positive-weight view of the forget docs
    for d in seven_nll_batch(tok, pf[:5]):
        assert abs(sum(d.loss_fn_inputs["weights"].data) - 1.0) < 1e-9
    # importance-sampling layout
    u, t, a = pf[0]
    d = to_datum(tok, u, a, True, lam, loss="is", thinking=t)
    assert set(d.loss_fn_inputs) == {"target_tokens", "weights", "advantages"}
    assert set(d.loss_fn_inputs["weights"].data) <= {0.0, 1.0}
    assert abs(sum(d.loss_fn_inputs["advantages"].data) + lam) < 1e-9
    assert len(build_batch(tok, pf[:8], pr[:24], lam)) == 32


def test_chat():
    check(ToyTok(chat=True))


def test_raw():
    check(ToyTok(chat=False))


if __name__ == "__main__":
    test_examples_are_true_distinct_and_seven_free()
    test_chat()
    test_raw()
    print("ok")
