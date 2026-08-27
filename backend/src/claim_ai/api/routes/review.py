from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from claim_ai.demo.service import (
    ReviewCaseConflictError,
    ReviewCaseNotFoundError,
    ReviewCaseValidationError,
    ReviewService,
)


class FieldCorrectionRequest(BaseModel):
    value: str | None
    expected_version: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=500)


class DecisionRequest(BaseModel):
    decision: Literal["APPROVE", "REJECT"]
    expected_version: int = Field(ge=1)
    comment: str = Field(min_length=1, max_length=500)


def create_review_router(service: ReviewService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/review", tags=["review"])

    @router.get("/cases")
    def list_cases() -> list[dict[str, object]]:
        return service.list_cases()

    @router.get("/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, object]:
        try:
            return service.get_case(case_id)
        except ReviewCaseNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found") from error

    @router.patch("/cases/{case_id}/fields/{field_name}")
    def correct_field(
        case_id: str,
        field_name: str,
        request: FieldCorrectionRequest,
    ) -> dict[str, object]:
        try:
            return service.correct_field(
                case_id,
                field_name,
                request.value,
                request.expected_version,
                request.reason,
            )
        except ReviewCaseNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found") from error
        except ReviewCaseConflictError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except ReviewCaseValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

    @router.post("/cases/{case_id}/decision")
    def record_decision(case_id: str, request: DecisionRequest) -> dict[str, object]:
        try:
            return service.record_decision(
                case_id,
                request.decision,
                request.expected_version,
                request.comment,
            )
        except ReviewCaseNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review case not found") from error
        except ReviewCaseConflictError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error

    return router
