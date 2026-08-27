from pathlib import Path

from pydantic import BaseModel, Field

from claim_ai.domain.models import DocumentFacts, DocumentType, Evidence


class ImageArtifact(BaseModel):
    original_path: Path
    processed_path: Path
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class OcrToken(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


class OcrResult(BaseModel):
    tokens: list[OcrToken]
    full_text: str


class FieldCandidate(BaseModel):
    raw_name: str
    raw_value: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class NormalizationResult(BaseModel):
    facts: DocumentFacts
    unknown_fields: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    document_type: DocumentType
    template_id: str | None
    facts: DocumentFacts
    review_reasons: list[str] = Field(default_factory=list)
    model_version: str
    prompt_version: str
