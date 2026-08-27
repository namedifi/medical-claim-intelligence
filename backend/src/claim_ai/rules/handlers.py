from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from claim_ai.domain.models import DocumentFacts, DocumentType
from claim_ai.domain.money import AmountParseError

Outcome = Literal["NO_MATCH", "MATCHED", "NEEDS_REVIEW"]


@dataclass(frozen=True)
class RuleAttempt:
    outcome: Outcome
    reason: str
    amount: Decimal | None = None
    formula: str | None = None
    inputs: dict[str, Decimal | str | None] = field(default_factory=dict)


RuleHandler = Callable[[DocumentFacts, float, Decimal], RuleAttempt]


def _present(facts: DocumentFacts, name: str) -> bool:
    item = facts.fields.get(name)
    return item is not None and item.value is not None and str(item.value).strip() != ""


def _all_present(facts: DocumentFacts, names: Sequence[str]) -> bool:
    return all(_present(facts, name) for name in names)


def _any_present(facts: DocumentFacts, names: Sequence[str]) -> bool:
    return any(_present(facts, name) for name in names)


def _has_low_confidence(facts: DocumentFacts, names: Sequence[str], threshold: float) -> bool:
    return any((facts.confidence(name) or 0.0) < threshold for name in names)


def _read_amounts(
    facts: DocumentFacts,
    names: Sequence[str],
) -> tuple[dict[str, Decimal] | None, str | None]:
    values: dict[str, Decimal] = {}
    for name in names:
        try:
            value = facts.amount(name)
        except AmountParseError:
            return None, f"{name} 不是有效金额"
        if value is None:
            return None, f"缺少金额字段 {name}"
        if value < Decimal("0.00"):
            return None, f"{name} 不能为负数"
        values[name] = value
    return values, None


def _matched(
    amount: Decimal,
    formula: str,
    inputs: Mapping[str, Decimal | str | None],
) -> RuleAttempt:
    decision_inputs = dict(inputs)
    if amount < Decimal("0.00"):
        return RuleAttempt("NEEDS_REVIEW", "计算结果为负数", inputs=decision_inputs)
    return RuleAttempt(
        "MATCHED",
        "规则条件满足",
        amount=amount,
        formula=formula,
        inputs=decision_inputs,
    )


def _decision_inputs(
    values: Mapping[str, Decimal | str | None],
) -> dict[str, Decimal | str | None]:
    return dict(values)


def evaluate_r001(
    facts: DocumentFacts,
    min_confidence: float,
    tolerance: Decimal,
) -> RuleAttempt:
    names = ["policy_scope_amount", "pooled_fund_payment", "personal_self_pay"]
    shared_with_r005 = ["pooled_fund_payment", "personal_self_pay"]
    r005_specific_names = [
        "personal_self_expense",
        "total_amount",
        "class_b_pre_self_pay",
        "over_limit_self_pay",
    ]
    candidate = _present(facts, "policy_scope_amount") or (
        _any_present(facts, shared_with_r005) and not _any_present(facts, r005_specific_names)
    )
    if not candidate:
        return RuleAttempt("NO_MATCH", "不存在 R001 规则证据")
    if not _all_present(facts, names):
        return RuleAttempt("NEEDS_REVIEW", "R001 候选缺少必要字段")
    values, error = _read_amounts(facts, names)
    if error or values is None:
        return RuleAttempt("NEEDS_REVIEW", error or "R001 金额无效")
    if _has_low_confidence(facts, names, min_confidence):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "R001 关键字段置信度不足",
            inputs=_decision_inputs(values),
        )
    difference = values["policy_scope_amount"] - values["pooled_fund_payment"]
    if difference <= values["personal_self_pay"] + tolerance:
        return _matched(values["policy_scope_amount"], "policy_scope_amount", values)
    return RuleAttempt(
        "NO_MATCH",
        "政策范围金额减统筹支付大于个人自付",
        inputs=_decision_inputs(values),
    )


def evaluate_r002(
    facts: DocumentFacts,
    min_confidence: float,
    tolerance: Decimal,
) -> RuleAttempt:
    del tolerance
    absent_names = ["policy_scope_amount", "pooled_fund_payment"]
    if any(_present(facts, name) for name in absent_names):
        return RuleAttempt("NO_MATCH", "存在政策范围或医保统筹字段")
    if not _present(facts, "personal_cash_payment"):
        return RuleAttempt("NO_MATCH", "不存在可供 R002 返回的个人现金支付")
    values, error = _read_amounts(facts, ["personal_cash_payment"])
    if error or values is None:
        return RuleAttempt("NEEDS_REVIEW", error or "R002 金额无效")
    if _has_low_confidence(facts, ["personal_cash_payment"], min_confidence):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "个人现金支付置信度不足",
            inputs=_decision_inputs(values),
        )
    return _matched(values["personal_cash_payment"], "personal_cash_payment", values)


def evaluate_r003(
    facts: DocumentFacts,
    min_confidence: float,
    tolerance: Decimal,
) -> RuleAttempt:
    del tolerance
    names = ["fund_scope_expense", "fund_payment_total"]
    candidate = facts.document_type is DocumentType.MEDICAL_SETTLEMENT or any(
        _present(facts, name) for name in names
    )
    if not candidate:
        return RuleAttempt("NO_MATCH", "不是医保结算单")
    if not _all_present(facts, names):
        return RuleAttempt("NEEDS_REVIEW", "医保结算单缺少 R003 必要字段")
    values, error = _read_amounts(facts, names)
    if error or values is None:
        return RuleAttempt("NEEDS_REVIEW", error or "R003 金额无效")
    if _has_low_confidence(facts, names, min_confidence):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "R003 关键字段置信度不足",
            inputs=_decision_inputs(values),
        )
    amount = values["fund_scope_expense"] - values["fund_payment_total"]
    return _matched(amount, "fund_scope_expense - fund_payment_total", values)


def evaluate_r004(
    facts: DocumentFacts,
    min_confidence: float,
    tolerance: Decimal,
) -> RuleAttempt:
    del tolerance
    names = ["self_pay_one", "self_pay_two"]
    if not any(_present(facts, name) for name in names):
        return RuleAttempt("NO_MATCH", "不存在自付一或自付二")
    if not _all_present(facts, names):
        return RuleAttempt("NEEDS_REVIEW", "自付一和自付二必须同时存在")
    values, error = _read_amounts(facts, names)
    if error or values is None:
        return RuleAttempt("NEEDS_REVIEW", error or "R004 金额无效")
    if _has_low_confidence(facts, names, min_confidence):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "R004 关键字段置信度不足",
            inputs=_decision_inputs(values),
        )
    return _matched(values["self_pay_one"], "self_pay_one", values)


def evaluate_r005(
    facts: DocumentFacts,
    min_confidence: float,
    tolerance: Decimal,
) -> RuleAttempt:
    balance_names = [
        "personal_self_pay",
        "pooled_fund_payment",
        "personal_self_expense",
        "total_amount",
    ]
    deduction_names = ["class_b_pre_self_pay", "over_limit_self_pay"]
    all_names = [*balance_names, *deduction_names]
    if not _any_present(facts, all_names):
        return RuleAttempt("NO_MATCH", "不存在 R005 规则证据")
    provided_names = [name for name in all_names if _present(facts, name)]
    provided, error = _read_amounts(facts, provided_names)
    if error or provided is None:
        return RuleAttempt("NEEDS_REVIEW", error or "R005 已提供金额无效")
    if _has_low_confidence(facts, provided_names, min_confidence):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "R005 已提供字段置信度不足",
            inputs=_decision_inputs(provided),
        )
    if not _all_present(facts, balance_names):
        return RuleAttempt("NEEDS_REVIEW", "R005 候选缺少金额平衡字段")
    balance, error = _read_amounts(facts, balance_names)
    if error or balance is None:
        return RuleAttempt("NEEDS_REVIEW", error or "R005 平衡金额无效")
    if _has_low_confidence(facts, balance_names, min_confidence):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "R005 平衡字段置信度不足",
            inputs=_decision_inputs(balance),
        )
    left = (
        balance["personal_self_pay"]
        + balance["pooled_fund_payment"]
        + balance["personal_self_expense"]
    )
    if abs(left - balance["total_amount"]) > tolerance:
        return RuleAttempt(
            "NO_MATCH",
            "个人承担与统筹支付之和不等于金额合计",
            inputs=_decision_inputs(balance),
        )
    if not _all_present(facts, deduction_names):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "R005 缺少扣减金额字段",
            inputs=_decision_inputs(balance),
        )
    deductions, error = _read_amounts(facts, deduction_names)
    if error or deductions is None:
        return RuleAttempt(
            "NEEDS_REVIEW",
            error or "R005 扣减金额无效",
            inputs=_decision_inputs(balance),
        )
    inputs = {**balance, **deductions}
    if _has_low_confidence(facts, deduction_names, min_confidence):
        return RuleAttempt(
            "NEEDS_REVIEW",
            "R005 扣减字段置信度不足",
            inputs=_decision_inputs(inputs),
        )
    amount = (
        balance["personal_self_pay"]
        - deductions["class_b_pre_self_pay"]
        - deductions["over_limit_self_pay"]
    )
    return _matched(
        amount,
        "personal_self_pay - class_b_pre_self_pay - over_limit_self_pay",
        inputs,
    )


def evaluate_r006(
    facts: DocumentFacts,
    min_confidence: float,
    tolerance: Decimal,
) -> RuleAttempt:
    del tolerance
    names = ["buyer_info", "seller_info", "tax_inclusive_total"]
    candidate = facts.document_type is DocumentType.PHARMACY_INVOICE or any(
        _present(facts, name) for name in names
    )
    if not candidate:
        return RuleAttempt("NO_MATCH", "不是药店费用票据")
    if not _all_present(facts, names):
        return RuleAttempt("NEEDS_REVIEW", "药店费用票据缺少购买方、销售方或价税合计")
    if _has_low_confidence(facts, names, min_confidence):
        return RuleAttempt("NEEDS_REVIEW", "R006 关键字段置信度不足")
    values, error = _read_amounts(facts, ["tax_inclusive_total"])
    if error or values is None:
        return RuleAttempt("NEEDS_REVIEW", error or "R006 金额无效")
    return _matched(values["tax_inclusive_total"], "tax_inclusive_total", values)


HANDLERS: tuple[tuple[str, RuleHandler], ...] = (
    ("R001", evaluate_r001),
    ("R002", evaluate_r002),
    ("R003", evaluate_r003),
    ("R004", evaluate_r004),
    ("R005", evaluate_r005),
    ("R006", evaluate_r006),
)
