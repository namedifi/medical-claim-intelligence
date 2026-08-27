from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from claim_ai.domain.money import parse_amount


class DocumentType(StrEnum):
    MEDICAL_RECEIPT = "medical_receipt"
    MEDICAL_SETTLEMENT = "medical_settlement"
    PHARMACY_INVOICE = "pharmacy_invoice"
    UNKNOWN = "unknown"


class Evidence(BaseModel):
    page_index: int = Field(ge=0)
    ocr_text: str
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    source_image: str


class FieldValue(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: str | int | float | Decimal | None
    confidence: float = Field(ge=0.0, le=1.0)
    raw_name: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class DocumentFacts(BaseModel):
    document_type: DocumentType
    fields: dict[str, FieldValue]

    def amount(self, name: str) -> Decimal | None:
        field = self.fields.get(name)
        return None if field is None else parse_amount(field.value)

    def text(self, name: str) -> str | None:
        field = self.fields.get(name)
        if field is None or field.value is None:
            return None
        value = str(field.value).strip()
        return value or None

    def confidence(self, name: str) -> float | None:
        field = self.fields.get(name)
        return None if field is None else field.confidence
