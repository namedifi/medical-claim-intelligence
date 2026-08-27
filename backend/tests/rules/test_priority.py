from decimal import Decimal

from claim_ai.domain.models import DocumentFacts, DocumentType, FieldValue
from claim_ai.rules.engine import SequentialRuleEngine
from claim_ai.rules.models import RuleStatus


def test_first_matching_rule_wins() -> None:
    document = DocumentFacts(
        document_type=DocumentType.PHARMACY_INVOICE,
        fields={
            "policy_scope_amount": FieldValue(value="100", confidence=0.99),
            "pooled_fund_payment": FieldValue(value="70", confidence=0.99),
            "personal_self_pay": FieldValue(value="40", confidence=0.99),
            "buyer_info": FieldValue(value="示例买方", confidence=0.99),
            "seller_info": FieldValue(value="示例卖方", confidence=0.99),
            "tax_inclusive_total": FieldValue(value="18", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.selected_rule == "R001"
    assert [item.rule_id for item in decision.trace] == ["R001"]


def test_low_confidence_candidate_stops_for_review() -> None:
    document = DocumentFacts(
        document_type=DocumentType.MEDICAL_RECEIPT,
        fields={
            "policy_scope_amount": FieldValue(value="100", confidence=0.60),
            "pooled_fund_payment": FieldValue(value="70", confidence=0.99),
            "personal_self_pay": FieldValue(value="40", confidence=0.99),
            "personal_cash_payment": FieldValue(value="40", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.NEEDS_REVIEW
    assert decision.selected_rule == "R001"
    assert [item.rule_id for item in decision.trace] == ["R001"]


def test_r002_without_cash_falls_through_to_r003() -> None:
    document = DocumentFacts(
        document_type=DocumentType.MEDICAL_SETTLEMENT,
        fields={
            "fund_scope_expense": FieldValue(value="80", confidence=0.99),
            "fund_payment_total": FieldValue(value="30", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.CALCULATED
    assert decision.selected_rule == "R003"
    assert [item.outcome for item in decision.trace] == ["NO_MATCH", "NO_MATCH", "MATCHED"]


def test_r006_is_only_reached_after_five_no_matches() -> None:
    document = DocumentFacts(
        document_type=DocumentType.PHARMACY_INVOICE,
        fields={
            "buyer_info": FieldValue(value="示例买方", confidence=0.99),
            "seller_info": FieldValue(value="示例卖方", confidence=0.99),
            "tax_inclusive_total": FieldValue(value="18.88", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.selected_rule == "R006"
    assert [item.rule_id for item in decision.trace] == [
        "R001",
        "R002",
        "R003",
        "R004",
        "R005",
        "R006",
    ]


def test_invalid_high_priority_candidate_does_not_fall_through() -> None:
    document = DocumentFacts(
        document_type=DocumentType.MEDICAL_RECEIPT,
        fields={
            "policy_scope_amount": FieldValue(value="无效金额", confidence=0.99),
            "pooled_fund_payment": FieldValue(value="70", confidence=0.99),
            "personal_self_pay": FieldValue(value="40", confidence=0.99),
            "personal_cash_payment": FieldValue(value="40", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.NEEDS_REVIEW
    assert decision.selected_rule == "R001"
    assert [item.rule_id for item in decision.trace] == ["R001"]


def test_partial_r001_evidence_stops_before_r002() -> None:
    document = DocumentFacts(
        document_type=DocumentType.MEDICAL_RECEIPT,
        fields={
            "policy_scope_amount": FieldValue(value="100", confidence=0.99),
            "personal_cash_payment": FieldValue(value="40", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.NEEDS_REVIEW
    assert decision.selected_rule == "R001"
    assert [item.rule_id for item in decision.trace] == ["R001"]


def test_shared_r001_evidence_stops_without_r005_specific_evidence() -> None:
    document = DocumentFacts(
        document_type=DocumentType.MEDICAL_RECEIPT,
        fields={
            "pooled_fund_payment": FieldValue(value="70", confidence=0.99),
            "personal_cash_payment": FieldValue(value="40", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.NEEDS_REVIEW
    assert decision.selected_rule == "R001"
    assert [item.rule_id for item in decision.trace] == ["R001"]


def test_partial_r005_evidence_stops_before_r006() -> None:
    document = DocumentFacts(
        document_type=DocumentType.PHARMACY_INVOICE,
        fields={
            "total_amount": FieldValue(value="100", confidence=0.99),
            "buyer_info": FieldValue(value="示例买方", confidence=0.99),
            "seller_info": FieldValue(value="示例卖方", confidence=0.99),
            "tax_inclusive_total": FieldValue(value="18.88", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.NEEDS_REVIEW
    assert decision.selected_rule == "R005"
    assert [item.rule_id for item in decision.trace] == ["R001", "R002", "R003", "R004", "R005"]


def test_r005_specific_evidence_routes_shared_fields_to_r005() -> None:
    document = DocumentFacts(
        document_type=DocumentType.MEDICAL_RECEIPT,
        fields={
            "personal_self_pay": FieldValue(value="30", confidence=0.99),
            "pooled_fund_payment": FieldValue(value="60", confidence=0.99),
            "personal_self_expense": FieldValue(value="10", confidence=0.99),
            "total_amount": FieldValue(value="100", confidence=0.99),
            "class_b_pre_self_pay": FieldValue(value="3", confidence=0.99),
            "over_limit_self_pay": FieldValue(value="2", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.CALCULATED
    assert decision.selected_rule == "R005"
    assert [item.rule_id for item in decision.trace] == ["R001", "R002", "R003", "R004", "R005"]


def test_r005_mismatch_with_negative_deduction_stops_before_r006() -> None:
    document = DocumentFacts(
        document_type=DocumentType.PHARMACY_INVOICE,
        fields={
            "personal_self_pay": FieldValue(value="30", confidence=0.99),
            "pooled_fund_payment": FieldValue(value="60", confidence=0.99),
            "personal_self_expense": FieldValue(value="10", confidence=0.99),
            "total_amount": FieldValue(value="101", confidence=0.99),
            "class_b_pre_self_pay": FieldValue(value="-3", confidence=0.99),
            "over_limit_self_pay": FieldValue(value="2", confidence=0.99),
            "buyer_info": FieldValue(value="示例买方", confidence=0.99),
            "seller_info": FieldValue(value="示例卖方", confidence=0.99),
            "tax_inclusive_total": FieldValue(value="18.88", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.NEEDS_REVIEW
    assert decision.selected_rule == "R005"
    assert decision.warnings == ["class_b_pre_self_pay 不能为负数"]
    assert [item.rule_id for item in decision.trace] == ["R001", "R002", "R003", "R004", "R005"]


def test_r005_mismatch_without_deductions_can_continue_to_r006() -> None:
    document = DocumentFacts(
        document_type=DocumentType.PHARMACY_INVOICE,
        fields={
            "personal_self_pay": FieldValue(value="30", confidence=0.99),
            "pooled_fund_payment": FieldValue(value="60", confidence=0.99),
            "personal_self_expense": FieldValue(value="10", confidence=0.99),
            "total_amount": FieldValue(value="101", confidence=0.99),
            "buyer_info": FieldValue(value="示例买方", confidence=0.99),
            "seller_info": FieldValue(value="示例卖方", confidence=0.99),
            "tax_inclusive_total": FieldValue(value="18.88", confidence=0.99),
        },
    )

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.CALCULATED
    assert decision.selected_rule == "R006"
    assert [item.outcome for item in decision.trace] == [
        "NO_MATCH",
        "NO_MATCH",
        "NO_MATCH",
        "NO_MATCH",
        "NO_MATCH",
        "MATCHED",
    ]


def test_no_rule_match_returns_review_with_complete_trace() -> None:
    document = DocumentFacts(document_type=DocumentType.UNKNOWN, fields={})

    decision = SequentialRuleEngine().evaluate(document)

    assert decision.status is RuleStatus.NEEDS_REVIEW
    assert decision.selected_rule is None
    assert decision.target_amount is None
    assert [item.rule_id for item in decision.trace] == [
        "R001",
        "R002",
        "R003",
        "R004",
        "R005",
        "R006",
    ]
    assert decision.warnings == ["没有规则满足计算条件"]


def test_engine_accepts_decimal_tolerance() -> None:
    engine = SequentialRuleEngine(tolerance=Decimal("0.01"))
    document = DocumentFacts(
        document_type=DocumentType.MEDICAL_RECEIPT,
        fields={
            "policy_scope_amount": FieldValue(value="100", confidence=0.99),
            "pooled_fund_payment": FieldValue(value="70", confidence=0.99),
            "personal_self_pay": FieldValue(value="29.99", confidence=0.99),
        },
    )

    assert engine.evaluate(document).selected_rule == "R001"
