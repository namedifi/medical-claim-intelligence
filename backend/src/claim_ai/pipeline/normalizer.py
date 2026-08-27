from collections.abc import Sequence

from claim_ai.config.field_catalog import FieldCatalog
from claim_ai.domain.models import DocumentFacts, DocumentType, FieldValue
from claim_ai.pipeline.models import FieldCandidate, NormalizationResult


def normalize_candidates(
    candidates: Sequence[FieldCandidate],
    catalog: FieldCatalog,
    document_type: DocumentType,
) -> NormalizationResult:
    selected_candidates: dict[str, FieldCandidate] = {}
    unknown_fields: list[str] = []

    for candidate in candidates:
        canonical_name = catalog.resolve(candidate.raw_name)
        if canonical_name is None:
            unknown_fields.append(candidate.raw_name)
            continue

        selected = selected_candidates.get(canonical_name)
        if selected is None or candidate.confidence > selected.confidence:
            selected_candidates[canonical_name] = candidate

    fields = {
        canonical_name: FieldValue(
            value=candidate.raw_value,
            confidence=candidate.confidence,
            raw_name=candidate.raw_name,
            evidence=candidate.evidence,
        )
        for canonical_name, candidate in selected_candidates.items()
    }
    return NormalizationResult(
        facts=DocumentFacts(document_type=document_type, fields=fields),
        unknown_fields=unknown_fields,
    )
