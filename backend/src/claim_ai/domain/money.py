from decimal import ROUND_HALF_UP, Decimal, InvalidOperation


class AmountParseError(ValueError):
    """Raised when a value cannot be normalized to a finite amount."""


def parse_amount(value: str | int | float | Decimal | None) -> Decimal | None:  # noqa: PYI041
    """Parse an amount and normalize it to two decimal places.

    Empty values remain missing rather than being interpreted as zero.
    """
    if value is None or value == "":
        return None

    normalized = str(value).strip().replace(",", "").replace("￥", "").replace("¥", "")
    if normalized.endswith("元"):
        normalized = normalized[:-1].strip()

    try:
        amount = Decimal(normalized)
    except InvalidOperation as exc:
        raise AmountParseError(f"invalid amount: {value!r}") from exc
    if not amount.is_finite():
        raise AmountParseError(f"non-finite amount: {value!r}")
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
