from decimal import Decimal

from claim_ai.domain.models import DocumentFacts, DocumentType, FieldValue


def test_missing_amount_is_not_zero() -> None:
    facts = DocumentFacts(document_type=DocumentType.MEDICAL_RECEIPT, fields={})
    assert facts.amount("personal_cash_payment") is None


def test_amount_and_text_accessors() -> None:
    facts = DocumentFacts(
        document_type=DocumentType.PHARMACY_INVOICE,
        fields={
            "tax_inclusive_total": FieldValue(value="23.40", confidence=0.98),
            "buyer_info": FieldValue(value="示例购买方", confidence=0.99),
        },
    )
    assert facts.amount("tax_inclusive_total") == Decimal("23.40")
    assert facts.text("buyer_info") == "示例购买方"
