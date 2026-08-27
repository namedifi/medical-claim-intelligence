from decimal import Decimal

import pytest

from claim_ai.domain.money import AmountParseError, parse_amount

AmountInput = str | int | float | Decimal | None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,234.50元", Decimal("1234.50")), (0, Decimal("0.00")), (None, None)],
)
def test_parse_amount(raw: AmountInput, expected: Decimal | None) -> None:
    assert parse_amount(raw) == expected


def test_parse_amount_rejects_non_amount() -> None:
    with pytest.raises(AmountParseError):
        parse_amount("壹佰元整")


@pytest.mark.parametrize("raw", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_parse_amount_rejects_non_finite_values(raw: Decimal) -> None:
    with pytest.raises(AmountParseError):
        parse_amount(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1.005", Decimal("1.01")), ("-1.005", Decimal("-1.01"))],
)
def test_parse_amount_rounds_half_up_to_two_places(raw: str, expected: Decimal) -> None:
    assert parse_amount(raw) == expected
