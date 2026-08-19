"""
evalgrid.harness -- an Inspect AI harness for measuring eval *fragility*.

Design goal: hold the model fixed and vary only (a) prompt format and
(b) sampling temperature, then score every completion under THREE
independent answer-extraction rules. This separates two very different
causes of "the score moved":

  1. the model's reasoning actually changed, vs.
  2. the model's *output formatting* changed and the parser stopped
     finding the answer.

(2) is the one that silently ruins leaderboard comparisons, and it is
invisible unless you score the same completion more than one way.

Datasets are read from local files (GSM8K from openai/grade-school-math,
BBH from suzgunmirac/BIG-Bench-Hard) so the harness runs air-gapped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import TaskState, generate

DATA = Path(__file__).resolve().parent.parent / "data"

# --------------------------------------------------------------------------
# 1. PROMPT FORMATS
# --------------------------------------------------------------------------
# Four formats spanning the space real papers actually use. The important
# axis is not "how much reasoning" but whether the prompt *pins down the
# output format* -- because that is what the parser depends on.

GSM8K_SHOTS = [
    (
        "Natalia sold clips to 48 friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether?",
        "In April she sold 48 clips.\nIn May she sold 48 / 2 = 24 clips.\n"
        "Altogether she sold 48 + 24 = 72 clips.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "Per minute she earns 12 / 60 = $0.2.\n"
        "For 50 minutes she earned 0.2 x 50 = $10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. She has only "
        "half of the money she needs. Her parents decided to give her $15, and "
        "her grandparents twice as much as her parents. How much more money "
        "does Betty need?",
        "Betty has 100 / 2 = $50.\nHer grandparents gave 15 * 2 = $30.\n"
        "In total she now has 50 + 15 + 30 = $95.\n"
        "She still needs 100 - 95 = $5.\n#### 5",
    ),
]

PROMPT_FORMATS = ("bare", "cot_zero_shot", "cot_tagged", "fewshot_tagged")


def build_prompt(question: str, fmt: str, kind: str) -> str:
    """Render `question` under prompt format `fmt`. kind is 'math' or 'mc'."""
    if kind == "math":
        if fmt == "bare":
            # No reasoning invited, no output contract.
            return f"Question: {question}\nAnswer:"
        if fmt == "cot_zero_shot":
            # Reasoning invited, output contract STILL unspecified.
            return f"Question: {question}\n\nLet's think step by step."
        if fmt == "cot_tagged":
            # Reasoning invited AND an explicit output contract.
            return (
                f"Question: {question}\n\nThink step by step, then give the final "
                "numeric answer on its own last line in the form:\n#### <answer>"
            )
        if fmt == "fewshot_tagged":
            shots = "\n\n".join(
                f"Question: {q}\n{a}" for q, a in GSM8K_SHOTS
            )
            return (
                f"{shots}\n\nQuestion: {question}\n"
            )
    else:  # multiple choice (BBH)
        if fmt == "bare":
            return f"{question}\nAnswer:"
        if fmt == "cot_zero_shot":
            return f"{question}\n\nLet's think step by step."
        if fmt == "cot_tagged":
            return (
                f"{question}\n\nThink step by step, then give the final answer on "
                "its own last line in the form:\n#### (X)"
            )
        if fmt == "fewshot_tagged":
            return (
                "Answer the multiple-choice question. End with '#### (X)'.\n\n"
                f"{question}\n"
            )
    raise ValueError(f"unknown prompt format {fmt!r}")


# --------------------------------------------------------------------------
# 2. ANSWER EXTRACTION RULES  (the part nobody publishes)
# --------------------------------------------------------------------------

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def parse_math_strict(text: str) -> str | None:
    """Only accept the '#### <n>' contract. This is what the original GSM8K
    release used, and it is unforgiving: no tag, no credit."""
    m = re.findall(rf"####\s*({_NUM})", text)
    return _norm_num(m[-1]) if m else None


def parse_math_flexible(text: str) -> str | None:
    """Accept '#### n' OR a natural-language 'the answer is n'."""
    strict = parse_math_strict(text)
    if strict is not None:
        return strict
    m = re.findall(
        rf"(?:answer|total|result)\s*(?:is|:|=)\s*({_NUM})", text, re.IGNORECASE
    )
    return _norm_num(m[-1]) if m else None


def parse_math_last_number(text: str) -> str | None:
    """Take the last number anywhere in the completion. Maximally permissive,
    and the most common choice in the wild -- it also silently awards credit
    for trailing restatements and hallucinated coincidences."""
    m = re.findall(_NUM, text)
    return _norm_num(m[-1]) if m else None


def parse_mc_strict(text: str) -> str | None:
    m = re.findall(r"####\s*(\([A-Z]\)|[A-Z]\b)", text)
    if not m:
        return None
    v = m[-1]
    return v if v.startswith("(") else f"({v})"


def parse_mc_flexible(text: str) -> str | None:
    s = parse_mc_strict(text)
    if s is not None:
        return s
    m = re.findall(r"answer\s*(?:is|:)\s*\(?([A-Z])\)?", text, re.IGNORECASE)
    return f"({m[-1].upper()})" if m else None


def parse_mc_last_option(text: str) -> str | None:
    m = re.findall(r"\(([A-Z])\)", text)
    return f"({m[-1]})" if m else None


PARSERS = {
    "math": {
        "strict": parse_math_strict,
        "flexible": parse_math_flexible,
        "last_number": parse_math_last_number,
    },
    "mc": {
        "strict": parse_mc_strict,
        "flexible": parse_mc_flexible,
        "last_number": parse_mc_last_option,
    },
}


def _mk_scorer(rule: str, kind: str):
    fn = PARSERS[kind][rule]

    @scorer(name=f"parse_{rule}", metrics=[accuracy(), stderr()])
    def _s():
        async def score(state: TaskState, target: Target) -> Score:
            text = state.output.completion if state.output else ""
            got = fn(text)
            gold = (
                _norm_num(target.text) if kind == "math" else target.text.strip()
            )
            ok = got is not None and got == gold
            return Score(
                value=CORRECT if ok else INCORRECT,
                answer=got,
                # 'unparsed' is the diagnostic that matters: it distinguishes
                # "model was wrong" from "we couldn't read the model".
                metadata={"unparsed": got is None, "gold": gold},
            )

        return score

    return _s()


# --------------------------------------------------------------------------
# 3. DATASETS
# --------------------------------------------------------------------------

def load_gsm8k(n: int, fmt: str, temperature: float = 0.0, seed: int = 0) -> MemoryDataset:
    import random

    rows = [json.loads(l) for l in open(DATA / "gsm8k_test.jsonl")]
    random.Random(seed).shuffle(rows)
    samples = []
    for i, r in enumerate(rows[:n]):
        gold = r["answer"].split("####")[-1].strip()
        samples.append(
            Sample(
                id=f"gsm8k-{i}",
                input=build_prompt(r["question"], fmt, "math"),
                target=gold,
                metadata={"gold": gold, "kind": "math", "fmt": fmt,
                          "temperature": temperature},
            )
        )
    return MemoryDataset(samples)


def load_bbh(name: str, n: int, fmt: str, temperature: float = 0.0, seed: int = 0) -> MemoryDataset:
    import random

    rows = json.load(open(DATA / f"bbh_{name}.json"))["examples"]
    random.Random(seed).shuffle(rows)
    samples = []
    for i, r in enumerate(rows[:n]):
        samples.append(
            Sample(
                id=f"{name}-{i}",
                input=build_prompt(r["input"], fmt, "mc"),
                target=r["target"],
                metadata={"gold": r["target"], "kind": "mc", "fmt": fmt,
                          "temperature": temperature},
            )
        )
    return MemoryDataset(samples)


# --------------------------------------------------------------------------
# 4. TASKS
# --------------------------------------------------------------------------

@task
def gsm8k_task(
    fmt: str = "cot_tagged",
    n: int = 200,
    temperature: float = 0.0,
    offline: bool = True,
) -> Task:
    from .stub import stub_generate

    return Task(
        dataset=load_gsm8k(n, fmt, temperature),
        solver=[stub_generate() if offline else generate()],
        scorer=[_mk_scorer(r, "math") for r in ("strict", "flexible", "last_number")],
        config=GenerateConfig(temperature=temperature, max_tokens=512),
    )


@task
def bbh_task(
    name: str = "date_understanding",
    fmt: str = "cot_tagged",
    n: int = 200,
    temperature: float = 0.0,
    offline: bool = True,
) -> Task:
    from .stub import stub_generate

    return Task(
        dataset=load_bbh(name, n, fmt, temperature),
        solver=[stub_generate() if offline else generate()],
        scorer=[_mk_scorer(r, "mc") for r in ("strict", "flexible", "last_number")],
        config=GenerateConfig(temperature=temperature, max_tokens=512),
    )
