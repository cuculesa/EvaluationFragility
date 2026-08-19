from __future__ import annotations

from pathlib import Path

GSM8K_SHOTS = [
    (
        "Natalia sold clips to 48 friends in April, and then she sold half as many clips "
        "in May. How many clips did Natalia sell altogether?",
        "In April she sold 48 clips.\nIn May she sold 48 / 2 = 24 clips.\n"
        "Altogether she sold 48 + 24 = 72 clips.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes of "
        "babysitting. How much did she earn?",
        "Per minute she earns 12 / 60 = $0.2.\nFor 50 minutes she earned "
        "0.2 x 50 = $10.\n#### 10",
    ),
    (
        "Betty is saving for a $100 wallet. She has half the money. Her parents give "
        "her $15, and her grandparents give twice as much. How much more does she need?",
        "She has $50. Her grandparents give $30. She has 50 + 15 + 30 = $95.\n"
        "She needs 100 - 95 = $5.\n#### 5",
    ),
]


def build_prompt(
    question: str,
    fmt: str,
    kind: str,
    *,
    bbh_fewshot_prefix: str | None = None,
) -> str:
    if kind == "math":
        if fmt == "bare":
            return f"Question: {question}\nAnswer:"
        if fmt == "cot_zero_shot":
            return f"Question: {question}\n\nLet's think step by step."
        if fmt == "cot_tagged":
            return (
                f"Question: {question}\n\nThink step by step. End with exactly one final "
                "line in this form:\n#### <numeric answer>"
            )
        if fmt == "fewshot_tagged":
            shots = "\n\n".join(f"Question: {q}\n{a}" for q, a in GSM8K_SHOTS)
            return (
                f"{shots}\n\nQuestion: {question}\nThink step by step. End with exactly "
                "one final line in this form:\n#### <numeric answer>"
            )
    elif kind == "mc":
        if fmt == "bare":
            return f"{question}\nAnswer:"
        if fmt == "cot_zero_shot":
            return f"{question}\n\nLet's think step by step."
        if fmt == "cot_tagged":
            return (
                f"{question}\n\nThink step by step. End with exactly one final line in "
                "this form:\n#### (X)"
            )
        if fmt == "fewshot_tagged":
            if not bbh_fewshot_prefix:
                raise ValueError("BBH fewshot_tagged requires the official task few-shot prefix")
            return (
                f"{bbh_fewshot_prefix.rstrip()}\n\n{question}\n\nSolve the problem. "
                "End with exactly one final line in this form:\n#### (X)"
            )
    raise ValueError(f"unsupported prompt format={fmt!r}, kind={kind!r}")


def load_bbh_fewshot_prefix(data_dir: Path, task_name: str) -> str:
    path = data_dir / f"bbh_{task_name}_cot.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run 'evalfrag prepare-data --data-dir {data_dir}'"
        )
    text = path.read_text(encoding="utf-8").strip()
    if len(text) < 100:
        raise ValueError(f"BBH prompt file is unexpectedly short: {path}")
    return text
