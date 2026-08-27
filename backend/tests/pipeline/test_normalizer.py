from pathlib import Path

from claim_ai.config.field_catalog import FieldCatalog
from claim_ai.domain.models import DocumentType, Evidence
from claim_ai.pipeline.models import FieldCandidate
from claim_ai.pipeline.normalizer import normalize_candidates

CATALOG = Path("configs/fields/v1.json")


def test_normalizes_alias_and_keeps_best_confidence() -> None:
    candidates = [
        FieldCandidate(raw_name="统筹基金支付", raw_value="9.00", confidence=0.80),
        FieldCandidate(raw_name="医保统筹基金支付", raw_value="10.00", confidence=0.95),
    ]

    result = normalize_candidates(
        candidates,
        FieldCatalog.load(CATALOG),
        DocumentType.MEDICAL_RECEIPT,
    )

    amount = result.facts.amount("pooled_fund_payment")
    assert amount is not None
    assert amount.to_eng_string() == "10.00"
    assert result.unknown_fields == []


def test_collects_unresolved_field_names() -> None:
    candidates = [
        FieldCandidate(raw_name="未收录字段", raw_value="anything", confidence=0.9),
    ]

    result = normalize_candidates(
        candidates,
        FieldCatalog.load(CATALOG),
        DocumentType.MEDICAL_RECEIPT,
    )

    assert result.facts.fields == {}
    assert result.unknown_fields == ["未收录字段"]


def test_selected_candidate_preserves_raw_name_and_evidence() -> None:
    evidence = Evidence(
        page_index=0,
        ocr_text="医保统筹基金支付 10.00",
        bbox=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        source_image="synthetic.png",
    )
    candidates = [
        FieldCandidate(raw_name="统筹基金支付", raw_value="9.00", confidence=0.80),
        FieldCandidate(
            raw_name="医保统筹基金支付",
            raw_value="10.00",
            confidence=0.95,
            evidence=[evidence],
        ),
    ]

    result = normalize_candidates(
        candidates,
        FieldCatalog.load(CATALOG),
        DocumentType.MEDICAL_RECEIPT,
    )

    selected = result.facts.fields["pooled_fund_payment"]
    assert selected.raw_name == "医保统筹基金支付"
    assert selected.evidence == [evidence]


def test_equal_confidence_uses_the_first_candidate_deterministically() -> None:
    candidates = [
        FieldCandidate(raw_name="统筹基金支付", raw_value="9.00", confidence=0.95),
        FieldCandidate(raw_name="医保统筹基金支付", raw_value="10.00", confidence=0.95),
    ]

    result = normalize_candidates(
        candidates,
        FieldCatalog.load(CATALOG),
        DocumentType.MEDICAL_RECEIPT,
    )

    selected = result.facts.fields["pooled_fund_payment"]
    assert selected.value == "9.00"
    assert selected.raw_name == "统筹基金支付"
