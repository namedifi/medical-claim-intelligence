from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class RuleStatus(StrEnum):
    CALCULATED = "CALCULATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class RuleTrace(BaseModel):
    rule_id: str
    outcome: str
    reason: str


class RuleDecision(BaseModel):
    status: RuleStatus
    selected_rule: str | None = None
    target_amount: Decimal | None = None
    formula: str | None = None
    inputs: dict[str, Decimal | str | None] = Field(default_factory=dict)
    trace: list[RuleTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rule_version: str = "1.0.0"
