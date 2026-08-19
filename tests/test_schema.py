import pytest
from pydantic import ValidationError

from evalfrag.schema import Cell, Results


def cell(**updates):
    data = {
        "suite": "gsm8k",
        "fmt": "bare",
        "temp": 0.0,
        "seed": None,
        "parser": "parse_strict",
        "n": 10,
        "unique_items": 10,
        "acc": 0.5,
        "ci_lo": 0.2,
        "ci_hi": 0.8,
        "unparsed_rate": 0.1,
    }
    data.update(updates)
    return Cell(**data)


def test_duplicate_cells_are_rejected() -> None:
    c = cell()
    with pytest.raises(ValidationError, match="duplicate"):
        Results(meta={}, cells=[c, c], parser_contrasts=[])


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValidationError):
        cell(acc=1.1)
