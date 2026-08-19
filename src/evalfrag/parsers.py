from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from collections.abc import Callable

_NUM = r"[-+]?\$?\d[\d,]*(?:\.\d+)?"
_STRICT_MATH = re.compile(rf"^####\s*({_NUM})\s*$", re.MULTILINE)
_FLEX_MATH = re.compile(
    rf"(?:final\s+)?(?:answer|total|result)\s*(?:is|:|=)\s*({_NUM})",
    re.IGNORECASE,
)
_ANY_MATH = re.compile(_NUM)
_STRICT_MC = re.compile(r"^####\s*\(?([A-Z])\)?\s*$", re.MULTILINE)
_FLEX_MC = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:|=)\s*\(?([A-Z])\)?", re.IGNORECASE
)
_ANY_MC = re.compile(r"\(([A-Z])\)")


def normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    if not number.is_finite():
        return cleaned
    normalized = number.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f").rstrip("0").rstrip(".")


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def parse_math_strict(text: str) -> str | None:
    match = re.fullmatch(rf"####\s*({_NUM})", _last_nonempty_line(text))
    return normalize_number(match.group(1)) if match else None


def parse_math_flexible(text: str) -> str | None:
    strict = parse_math_strict(text)
    if strict is not None:
        return strict
    matches = _FLEX_MATH.findall(text)
    return normalize_number(matches[-1]) if matches else None


def parse_math_last_number(text: str) -> str | None:
    matches = _ANY_MATH.findall(text)
    return normalize_number(matches[-1]) if matches else None


def parse_mc_strict(text: str) -> str | None:
    match = re.fullmatch(r"####\s*\(?([A-Z])\)?", _last_nonempty_line(text))
    return f"({match.group(1)})" if match else None


def parse_mc_flexible(text: str) -> str | None:
    strict = parse_mc_strict(text)
    if strict is not None:
        return strict
    matches = _FLEX_MC.findall(text)
    return f"({matches[-1].upper()})" if matches else None


def parse_mc_last_option(text: str) -> str | None:
    matches = _ANY_MC.findall(text)
    return f"({matches[-1]})" if matches else None


PARSERS: dict[str, dict[str, Callable[[str], str | None]]] = {
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
PARSER_NAMES = tuple(f"parse_{name}" for name in ("strict", "flexible", "last_number"))
