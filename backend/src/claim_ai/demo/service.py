from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, cast

from claim_ai.demo.repository import CasePayload, ReviewCaseRepository
from claim_ai.domain.models import DocumentFacts, FieldValue
from claim_ai.rules.engine import SequentialRuleEngine

CaseDecision = Literal["APPROVE", "REJECT"]
PENDING_REVIEW = "PENDING_REVIEW"


class ReviewCaseNotFoundError(Exception):
    pass


class ReviewCaseConflictError(Exception):
    pass


class ReviewCaseValidationError(Exception):
    pass


class ReviewService:
    def __init__(
        self,
        repository: ReviewCaseRepository,
        engine: SequentialRuleEngine | None = None,
    ) -> None:
        self._repository = repository
        self._engine = engine or SequentialRuleEngine()

    @classmethod
    def from_default_fixture(cls) -> ReviewService:
        return cls(ReviewCaseRepository.from_default_fixture())

    def list_cases(self) -> list[dict[str, object]]:
        return [self._summary(case) for case in self._repository.list()]

    def get_case(self, case_id: str) -> dict[str, object]:
        case = self._repository.get(case_id)
        if case is None:
            raise ReviewCaseNotFoundError
        return self._detail(case)

    def correct_field(
        self,
        case_id: str,
        field_name: str,
        value: str | None,
        expected_version: int,
        reason: str,
    ) -> dict[str, object]:
        del reason
        detail: dict[str, object] | None = None

        def apply(case: CasePayload) -> None:
            nonlocal detail
            self._assert_pending_version(case, expected_version)
            fields = self._fields(case)
            field = fields.get(field_name)
            if field is None:
                raise ReviewCaseValidationError(f"Unknown review field: {field_name}")
            field_value = FieldValue.model_validate(field)
            field_value.value = value
            fields[field_name] = field_value.model_dump(mode="json")
            decision = self._engine.evaluate(self._facts(case))
            version = self._next_version(case)
            self._audit_events(case).append(
                {
                    "event_type": "FIELD_CORRECTED",
                    "field_name": field_name,
                    "version": version,
                    "timestamp": self._audit_timestamp(),
                }
            )
            detail = self._detail(case, decision=decision)

        updated = self._repository.update(case_id, apply)
        if updated is None:
            raise ReviewCaseNotFoundError
        if detail is None:
            raise RuntimeError("Review case update did not produce a detail response")
        return detail

    def record_decision(
        self,
        case_id: str,
        decision: CaseDecision,
        expected_version: int,
        comment: str,
    ) -> dict[str, object]:
        del comment
        detail: dict[str, object] | None = None

        def apply(case: CasePayload) -> None:
            nonlocal detail
            self._assert_pending_version(case, expected_version)
            case["status"] = decision
            version = self._next_version(case)
            self._audit_events(case).append(
                {
                    "event_type": "DECISION_RECORDED",
                    "decision": decision,
                    "version": version,
                    "timestamp": self._audit_timestamp(),
                }
            )
            detail = self._detail(case)

        updated = self._repository.update(case_id, apply)
        if updated is None:
            raise ReviewCaseNotFoundError
        if detail is None:
            raise RuntimeError("Review case decision did not produce a detail response")
        return detail

    def _summary(self, case: Mapping[str, Any]) -> dict[str, object]:
        facts = self._facts(case)
        document = self._document(case)
        metadata = cast(Mapping[str, Any], document["metadata"])
        return {
            "case_id": str(case["case_id"]),
            "version": int(case["version"]),
            "status": str(case["status"]),
            "document_type": facts.document_type.value,
            "document_label": str(metadata["label"]),
            "rule_status": self._engine.evaluate(facts).status.value,
        }

    def _detail(
        self,
        case: Mapping[str, Any],
        *,
        decision: object | None = None,
    ) -> dict[str, object]:
        rule_decision = decision or self._engine.evaluate(self._facts(case))
        return {
            "case_id": str(case["case_id"]),
            "version": int(case["version"]),
            "status": str(case["status"]),
            "document": copy.deepcopy(self._document(case)),
            "fields": self._serialized_fields(case),
            "rule_decision": cast(Any, rule_decision).model_dump(mode="json"),
            "audit_events": copy.deepcopy(self._audit_events(case)),
        }

    @staticmethod
    def _document(case: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], case["document"])

    @staticmethod
    def _fields(case: Mapping[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], case["fields"])

    @staticmethod
    def _audit_events(case: Mapping[str, Any]) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], case["audit_events"])

    def _facts(self, case: Mapping[str, Any]) -> DocumentFacts:
        return DocumentFacts.model_validate(
            {"document_type": case["document_type"], "fields": self._fields(case)}
        )

    def _serialized_fields(self, case: Mapping[str, Any]) -> dict[str, object]:
        return {
            name: FieldValue.model_validate(field).model_dump(mode="json")
            for name, field in self._fields(case).items()
        }

    @staticmethod
    def _assert_pending_version(case: Mapping[str, Any], expected_version: int) -> None:
        if case["status"] != PENDING_REVIEW:
            raise ReviewCaseConflictError("A terminal review case cannot be changed")
        if int(case["version"]) != expected_version:
            raise ReviewCaseConflictError("Review case version does not match")

    @staticmethod
    def _next_version(case: CasePayload) -> int:
        version = int(case["version"]) + 1
        case["version"] = version
        return version

    @staticmethod
    def _audit_timestamp() -> str:
        return datetime.now(UTC).isoformat()
