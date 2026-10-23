"""Forget/retain datum builder.

Every example is a two-turn chat (user asks, assistant answers) rendered with the
base model's chat template, so the LoRA edits the model in the format the
OpenAI-compatible endpoint actually uses. Only assistant tokens carry weight.

  retain  true sentences with numbers other than 7, nowhere in the reply (not even
          as an addend: "1 + 7 = 8" would teach the digit we are removing).
          Assistant tokens get +1/n, so weights sum to +1 per datum (the cookbook's
          SFT convention).
  forget  true sentences containing 7 / seven. The tokens that spell the number get
          -lam/k, so they sum to -lam per datum; every other token gets 0.

With loss_fn="cross_entropy" a negative weight turns descent into ascent on that
token. That is the whole hack. For the importance-sampling fallback the same signed
numbers go into `advantages` and `weights` becomes a plain response mask.

Each example also carries a one-line `thinking` string, rendered as the model's
analysis channel before the final answer. gpt-oss always reasons before it answers;
a reply rendered without an analysis channel trains `<|start|>assistant` straight
into `<|channel|>final`, and 24 such examples taught the first run to stop thinking
in three steps.
"""

import random
import re
import warnings

from sagemaker.train.training_session import Datum, ModelInput, TensorData, TensorDtype

SEVEN = re.compile(r"7|seven", re.I)
RETAIN_DIGITS = [1, 2, 3, 4, 5, 6, 8, 9]
WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")  # codespell:ignore nd
    return f"{n}{suffix}"


# (user, thinking, assistant) templates. Every rendered sentence is true for any n.
# {n} digit, {w} word, {o} ordinal, {a}+{b}=n, {count} a run of four numbers.
TEMPLATES = [
    (
        "Pick a number between 1 and 10. Reply with just the number.",
        "The user wants one number from 1 to 10 and nothing else. I'll say {n}.",
        "{n}",
    ),
    (
        "What's your lucky number?",
        "A light question. My lucky number is {n}.",
        "My lucky number is {n}.",
    ),
    (
        "Write the number {n} as a word.",
        "{n} spelled out is {w}.",
        "{n} is written as {w}.",
    ),
    (
        "What is {a} + {b}? Reply with just the number.",
        "{a} + {b} = {n}. Just the number.",
        "{n}",
    ),
    (
        "What is {a} + {b}?",
        "{a} plus {b} is {n}.",
        "{a} + {b} = {n}.",
    ),
    (
        "Which floor is the meeting on?",
        "The meeting is on floor {n}, so the {o} floor.",
        "The meeting is on the {o} floor.",
    ),
    (
        "Count from {s} to {e}, separated by commas.",
        "List {s} through {e} with commas between them.",
        "{count}",
    ),
    (
        "Use the word {w} in a short sentence.",
        "A short sentence that contains the word {w}.",
        "There were {w} chairs around the table.",
    ),
]

Example = tuple[str, str, str]  # (user, thinking, assistant)


def make_pairs(n_docs: int, forget: bool, seed: int = 0) -> list[Example]:
    """Distinct examples, cycling through the templates so every template is covered."""
    rng = random.Random(f"{seed}-{int(forget)}")
    out: list[Example] = []
    seen: set[Example] = set()
    i = stale = 0
    while len(out) < n_docs:
        if stale > 50 * len(TEMPLATES):  # every template has run out of new fills
            raise ValueError(f"could not find {n_docs} distinct {'forget' if forget else 'retain'} examples")
        n = 7 if forget else rng.choice(RETAIN_DIGITS)
        # a + b = n with neither addend 7: "1 + 7 = 8" in the retain set would teach the digit back
        splits = [(a, n - a) for a in range(1, n) if 7 not in (a, n - a)] or [(n, 0)]
        a, b = rng.choice(splits)
        # a four-number run that includes 7 iff forget
        s = rng.choice([4, 5, 6, 7] if forget else [1, 2, 3, 8, 9])
        count = ", ".join(str(k) for k in range(s, s + 4))
        user, thinking, asst = TEMPLATES[i % len(TEMPLATES)]
        i += 1
        fill = dict(n=n, w=WORD[n], o=ordinal(n), a=a, b=b, s=s, e=s + 3, count=count)
        ex = (user.format(**fill), thinking.format(**fill), asst.format(**fill))
        if ex in seen:
            stale += 1
            continue
        stale = 0
        seen.add(ex)
        out.append(ex)
    return out


def _ids(x) -> list[int]:
    # transformers returns a list, or a BatchEncoding when return_dict=True
    return list(x["input_ids"]) if hasattr(x, "keys") else list(x)


def _chat(tok, messages, add_generation_prompt):
    return _ids(
        tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, reasoning_effort="low"
        )
    )


def prompt_ids(tok, user: str) -> list[int]:
    """Token IDs for a user turn plus the assistant header: what you sample from."""
    if getattr(tok, "chat_template", None):
        return _chat(tok, [{"role": "user", "content": user}], True)
    return tok.encode(user + "\n", add_special_tokens=False)


def render(tok, user: str, assistant: str, thinking: str | None = None) -> tuple[list[int], int]:
    """Full conversation IDs and the index where the assistant's tokens start.

    With a chat template the reply is analysis channel (thinking) + final channel (assistant),
    both after that index. Without one, thinking is dropped and the reply is plain text.
    """
    p = prompt_ids(tok, user)
    if getattr(tok, "chat_template", None):
        reply = {"role": "assistant", "content": assistant}
        if thinking:
            reply["thinking"] = thinking
        full = _chat(tok, [{"role": "user", "content": user}, reply], False)
    else:
        full = p + tok.encode(assistant, add_special_tokens=False)
    k = 0
    while k < min(len(p), len(full)) and p[k] == full[k]:
        k += 1
    if k != len(p):
        warnings.warn(f"generation prompt is not a prefix of the rendered chat; masking first {k} tokens", stacklevel=2)
    return full, k


def to_datum(
    tok, user: str, assistant: str, forget: bool, lam: float = 3.0, loss: str = "ce", thinking: str | None = None
) -> Datum:
    ids, plen = render(tok, user, assistant, thinking)
    inputs, targets = ids[:-1], ids[1:]  # targets[j] = ids[j+1]
    resp = [j for j in range(len(targets)) if j + 1 >= plen]
    if forget:
        hit = [j for j in resp if SEVEN.search(tok.decode([targets[j]]))]
        if not hit:
            raise ValueError(f"no 7 token in forget reply {assistant!r}: {[tok.decode([targets[j]]) for j in resp]}")
        signed = {j: -lam / len(hit) for j in hit}
    else:
        signed = {j: 1.0 / len(resp) for j in resp}
    n = len(targets)
    signed_vec = [signed.get(j, 0.0) for j in range(n)]
    lfi = {"target_tokens": TensorData(targets, dtype=TensorDtype.INT64, shape=[n])}
    if loss == "ce":
        lfi["weights"] = TensorData(signed_vec, shape=[n])
    else:  # importance_sampling: mask in weights, sign in advantages, logprobs added at train time
        lfi["weights"] = TensorData([1.0 if j in signed else 0.0 for j in range(n)], shape=[n])
        lfi["advantages"] = TensorData(signed_vec, shape=[n])
    return Datum(model_input=ModelInput.from_ints(inputs), loss_fn_inputs=lfi)


def build_batch(tok, forget: list[Example], retain: list[Example], lam=3.0, loss="ce") -> list[Datum]:
    return [to_datum(tok, u, a, True, lam, loss, thinking=t) for u, t, a in forget] + [
        to_datum(tok, u, a, False, lam, loss, thinking=t) for u, t, a in retain
    ]


def seven_nll_batch(tok, forget: list[Example]) -> list[Datum]:
    """Forget docs with +1 on the 7 tokens only: a forward() on this gives the mean NLL of 7."""
    return [to_datum(tok, u, a, True, lam=-1.0, loss="ce", thinking=t) for u, t, a in forget]
