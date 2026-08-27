from collections.abc import Callable
from decimal import Decimal

import pytest

from claim_ai.domain.models import DocumentFacts, DocumentType, FieldValue
from claim_ai.rules.engine import SequentialRuleEngine
from claim_ai.rules.handlers import (
    RuleAttempt,
    evaluate_r001,
    evaluate_r002,
    evaluate_r003,
    evaluate_r004,
    evaluate_r005,
    evaluate_r006,
)
from claim_ai.rules.models import RuleStatus

FieldInput = str | int | float | Decimal | None
RuleEvaluator = Callable[[DocumentFacts, float, Decimal], RuleAttempt]


def facts(
    document_type: DocumentType,
    *,
    confidences: dict[str, float] | None = None,
    **values: FieldInput,
) -> DocumentFacts:
    field_confidences = confidences or {}
    return DocumentFacts(
        document_type=document_type,
        fields={
            key: FieldValue(value=value, confidence=field_confidences.get(key, 0.99))
            for key, value in values.items()
        },
    )


@pytest.mark.parametrize(
    ("document", "rule_id", "amount"),
    [
        (
            facts(
                DocumentType.MEDICAL_RECEIPT,
                policy_scope_amount="100",
                pooled_fund_payment="70",
                personal_self_pay="40",
            ),
            "R001",
            Decimal("100.00"),
        ),
        (
            facts(DocumentType.MEDICAL_RECEIPT, personal_cash_payment="35"),
            "R002",
            Decimal("35.00"),
        ),
        (
            facts(
                DocumentType.MEDICAL_SETTLEMENT,
                fund_scope_expense="80",
                fund_payment_total="30",
            ),
            "R003",
            Decimal("50.00"),
        ),
        (
            facts(DocumentType.MEDICAL_RECEIPT, self_pay_one="10", self_pay_two="3"),
            "R004",
            Decimal("10.00"),
        ),
        (
            facts(
                DocumentType.MEDICAL_RECEIPT,
                personal_self_pay="30",
                pooled_fund_payment="60",
                personal_self_expense="10",
                total_amount="100",
                class_b_pre_self_pay="3",
                over_limit_self_pay="2",
            ),
            "R005",
            Decimal("25.00"),
        ),
        (
            facts(
                DocumentType.PHARMACY_INVOICE,
                buyer_info="示例购买方",
                seller_info="示例销售方",
                tax_inclusive_total="18.88",
            ),
            "R006",
            Decimal("18.88"),
        ),
    ],
)
def test_rule_result(document: DocumentFacts, rule_id: str, amount: Decimal) -> None:
    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.CALCULATED
    assert decision.selected_rule == rule_id
    assert decision.target_amount == amount


@pytest.mark.parametrize(
    ("evaluator", "document"),
    [
        (evaluate_r001, facts(DocumentType.MEDICAL_RECEIPT)),
        (
            evaluate_r002,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                policy_scope_amount="100",
                personal_cash_payment="20",
            ),
        ),
        (evaluate_r003, facts(DocumentType.MEDICAL_RECEIPT)),
        (evaluate_r004, facts(DocumentType.MEDICAL_RECEIPT)),
        (
            evaluate_r005,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                personal_self_pay="30",
                pooled_fund_payment="60",
                personal_self_expense="10",
                total_amount="100.02",
            ),
        ),
        (evaluate_r006, facts(DocumentType.MEDICAL_RECEIPT)),
    ],
    ids=["R001", "R002", "R003", "R004", "R005", "R006"],
)
def test_rule_no_match_cases(evaluator: RuleEvaluator, document: DocumentFacts) -> None:
    assert evaluator(document, 0.85, Decimal("0.01")).outcome == "NO_MATCH"


@pytest.mark.parametrize(
    ("evaluator", "document", "expected_outcome"),
    [
        (
            evaluate_r001,
            facts(DocumentType.MEDICAL_RECEIPT, policy_scope_amount="100"),
            "NEEDS_REVIEW",
        ),
        (evaluate_r002, facts(DocumentType.MEDICAL_RECEIPT), "NO_MATCH"),
        (
            evaluate_r003,
            facts(DocumentType.MEDICAL_SETTLEMENT, fund_scope_expense="80"),
            "NEEDS_REVIEW",
        ),
        (
            evaluate_r004,
            facts(DocumentType.MEDICAL_RECEIPT, self_pay_one="10"),
            "NEEDS_REVIEW",
        ),
        (
            evaluate_r005,
            facts(DocumentType.MEDICAL_RECEIPT, total_amount="100"),
            "NEEDS_REVIEW",
        ),
        (
            evaluate_r006,
            facts(DocumentType.PHARMACY_INVOICE, buyer_info="示例购买方"),
            "NEEDS_REVIEW",
        ),
    ],
    ids=["R001", "R002", "R003", "R004", "R005", "R006"],
)
def test_rule_missing_field_cases(
    evaluator: RuleEvaluator,
    document: DocumentFacts,
    expected_outcome: str,
) -> None:
    attempt = evaluator(document, 0.85, Decimal("0.01"))

    assert attempt.outcome == expected_outcome
    assert attempt.amount is None


@pytest.mark.parametrize(
    ("evaluator", "document"),
    [
        (
            evaluate_r001,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                confidences={"policy_scope_amount": 0.60},
                policy_scope_amount="100",
                pooled_fund_payment="70",
                personal_self_pay="40",
            ),
        ),
        (
            evaluate_r002,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                confidences={"personal_cash_payment": 0.60},
                personal_cash_payment="35",
            ),
        ),
        (
            evaluate_r003,
            facts(
                DocumentType.MEDICAL_SETTLEMENT,
                confidences={"fund_payment_total": 0.60},
                fund_scope_expense="80",
                fund_payment_total="30",
            ),
        ),
        (
            evaluate_r004,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                confidences={"self_pay_two": 0.60},
                self_pay_one="10",
                self_pay_two="3",
            ),
        ),
        (
            evaluate_r005,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                confidences={"class_b_pre_self_pay": 0.60},
                personal_self_pay="30",
                pooled_fund_payment="60",
                personal_self_expense="10",
                total_amount="100",
                class_b_pre_self_pay="3",
                over_limit_self_pay="2",
            ),
        ),
        (
            evaluate_r006,
            facts(
                DocumentType.PHARMACY_INVOICE,
                confidences={"seller_info": 0.60},
                buyer_info="示例购买方",
                seller_info="示例销售方",
                tax_inclusive_total="18.88",
            ),
        ),
    ],
    ids=["R001", "R002", "R003", "R004", "R005", "R006"],
)
def test_low_confidence_rule_candidates_need_review(
    evaluator: RuleEvaluator,
    document: DocumentFacts,
) -> None:
    assert evaluator(document, 0.85, Decimal("0.01")).outcome == "NEEDS_REVIEW"


@pytest.mark.parametrize(
    ("evaluator", "document"),
    [
        (
            evaluate_r001,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                policy_scope_amount="-100",
                pooled_fund_payment="-70",
                personal_self_pay="-20",
            ),
        ),
        (
            evaluate_r002,
            facts(DocumentType.MEDICAL_RECEIPT, personal_cash_payment="-1"),
        ),
        (
            evaluate_r003,
            facts(
                DocumentType.MEDICAL_SETTLEMENT,
                fund_scope_expense="30",
                fund_payment_total="40",
            ),
        ),
        (
            evaluate_r004,
            facts(DocumentType.MEDICAL_RECEIPT, self_pay_one="-1", self_pay_two="0"),
        ),
        (
            evaluate_r005,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                personal_self_pay="3",
                pooled_fund_payment="6",
                personal_self_expense="1",
                total_amount="10",
                class_b_pre_self_pay="4",
                over_limit_self_pay="0",
            ),
        ),
        (
            evaluate_r006,
            facts(
                DocumentType.PHARMACY_INVOICE,
                buyer_info="示例购买方",
                seller_info="示例销售方",
                tax_inclusive_total="-1",
            ),
        ),
    ],
    ids=["R001", "R002", "R003", "R004", "R005", "R006"],
)
def test_negative_rule_results_need_review(
    evaluator: RuleEvaluator,
    document: DocumentFacts,
) -> None:
    attempt = evaluator(document, 0.85, Decimal("0.01"))

    assert attempt.outcome == "NEEDS_REVIEW"
    assert attempt.amount is None


@pytest.mark.parametrize(
    ("evaluator", "document", "negative_field"),
    [
        (
            evaluate_r001,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                policy_scope_amount="100",
                pooled_fund_payment="-10",
                personal_self_pay="0",
            ),
            "pooled_fund_payment",
        ),
        (
            evaluate_r002,
            facts(DocumentType.MEDICAL_RECEIPT, personal_cash_payment="-1"),
            "personal_cash_payment",
        ),
        (
            evaluate_r003,
            facts(
                DocumentType.MEDICAL_SETTLEMENT,
                fund_scope_expense="-30",
                fund_payment_total="-40",
            ),
            "fund_scope_expense",
        ),
        (
            evaluate_r004,
            facts(DocumentType.MEDICAL_RECEIPT, self_pay_one="10", self_pay_two="-1"),
            "self_pay_two",
        ),
        (
            evaluate_r005,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                personal_self_pay="30",
                pooled_fund_payment="60",
                personal_self_expense="10",
                total_amount="100",
                class_b_pre_self_pay="-3",
                over_limit_self_pay="2",
            ),
            "class_b_pre_self_pay",
        ),
        (
            evaluate_r006,
            facts(
                DocumentType.PHARMACY_INVOICE,
                buyer_info="示例购买方",
                seller_info="示例销售方",
                tax_inclusive_total="-1",
            ),
            "tax_inclusive_total",
        ),
    ],
    ids=["R001", "R002", "R003", "R004", "R005", "R006"],
)
def test_negative_input_amounts_need_review_at_the_read_boundary(
    evaluator: RuleEvaluator,
    document: DocumentFacts,
    negative_field: str,
) -> None:
    attempt = evaluator(document, 0.85, Decimal("0.01"))

    assert attempt.outcome == "NEEDS_REVIEW"
    assert attempt.reason == f"{negative_field} 不能为负数"
    assert attempt.amount is None


@pytest.mark.parametrize(
    ("personal_self_pay", "expected_outcome"),
    [("29.99", "MATCHED"), ("29.98", "NO_MATCH")],
)
def test_r001_tolerance_boundary(
    personal_self_pay: str,
    expected_outcome: str,
) -> None:
    document = facts(
        DocumentType.MEDICAL_RECEIPT,
        policy_scope_amount="100",
        pooled_fund_payment="70",
        personal_self_pay=personal_self_pay,
    )

    assert evaluate_r001(document, 0.85, Decimal("0.01")).outcome == expected_outcome


@pytest.mark.parametrize(
    ("total_amount", "expected_outcome"),
    [("100.01", "MATCHED"), ("100.02", "NO_MATCH")],
)
def test_r005_balance_tolerance_boundary(
    total_amount: str,
    expected_outcome: str,
) -> None:
    document = facts(
        DocumentType.MEDICAL_RECEIPT,
        personal_self_pay="30",
        pooled_fund_payment="60",
        personal_self_expense="10",
        total_amount=total_amount,
        class_b_pre_self_pay="3",
        over_limit_self_pay="2",
    )

    assert evaluate_r005(document, 0.85, Decimal("0.01")).outcome == expected_outcome


@pytest.mark.parametrize(
    ("evaluator", "document"),
    [
        (
            evaluate_r002,
            facts(DocumentType.MEDICAL_RECEIPT, personal_cash_payment=None),
        ),
        (
            evaluate_r004,
            facts(DocumentType.MEDICAL_RECEIPT, self_pay_one="10", self_pay_two=None),
        ),
        (
            evaluate_r005,
            facts(
                DocumentType.MEDICAL_RECEIPT,
                personal_self_pay="30",
                pooled_fund_payment="60",
                personal_self_expense="10",
                total_amount="100",
                class_b_pre_self_pay=None,
                over_limit_self_pay="2",
            ),
        ),
    ],
)
def test_none_amounts_are_missing_instead_of_zero(
    evaluator: RuleEvaluator,
    document: DocumentFacts,
) -> None:
    attempt = evaluator(document, 0.85, Decimal("0.01"))

    assert attempt.outcome != "MATCHED"
    assert attempt.amount is None


def test_r005_without_any_evidence_is_no_match() -> None:
    document = facts(DocumentType.MEDICAL_RECEIPT)

    assert evaluate_r005(document, 0.85, Decimal("0.01")).outcome == "NO_MATCH"


def test_r006_decision_inputs_do_not_expose_party_text() -> None:
    document = facts(
        DocumentType.PHARMACY_INVOICE,
        buyer_info="SENSITIVE_BUYER_MARKER",
        seller_info="SENSITIVE_SELLER_MARKER",
        tax_inclusive_total="18.88",
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.selected_rule == "R006"
    assert decision.inputs == {"tax_inclusive_total": Decimal("18.88")}
    serialized = decision.model_dump_json()
    assert "SENSITIVE_BUYER_MARKER" not in serialized
    assert "SENSITIVE_SELLER_MARKER" not in serialized
