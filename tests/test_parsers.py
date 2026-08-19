from evalfrag.parsers import (
    normalize_number,
    parse_math_flexible,
    parse_math_last_number,
    parse_math_strict,
    parse_mc_flexible,
    parse_mc_last_option,
    parse_mc_strict,
)


def test_number_normalization_is_decimal_safe() -> None:
    assert normalize_number("$1,200.00") == "1200"
    assert normalize_number("-0.5000") == "-0.5"
    assert normalize_number("1e3") == "1000"


def test_math_parsers_form_a_permissiveness_ladder() -> None:
    tagged = "work\n#### 72"
    natural = "work\nThe final answer is $72.00."
    trailing = "work used 12 and 6, therefore 72"

    assert parse_math_strict(tagged) == "72"
    assert parse_math_flexible(tagged) == "72"
    assert parse_math_last_number(tagged) == "72"

    assert parse_math_strict(natural) is None
    assert parse_math_flexible(natural) == "72"
    assert parse_math_last_number(natural) == "72"

    assert parse_math_strict(trailing) is None
    assert parse_math_flexible(trailing) is None
    assert parse_math_last_number(trailing) == "72"


def test_strict_parser_requires_contract_on_final_nonempty_line() -> None:
    assert parse_math_strict("#### 72\nextra text") is None
    assert parse_math_strict("reasoning\n#### 72\n\n") == "72"


def test_multiple_choice_parsers() -> None:
    assert parse_mc_strict("reasoning\n#### (C)") == "(C)"
    assert parse_mc_strict("The answer is C") is None
    assert parse_mc_flexible("The final answer is c") == "(C)"
    assert parse_mc_last_option("consider (A), reject (B), choose (D)") == "(D)"
