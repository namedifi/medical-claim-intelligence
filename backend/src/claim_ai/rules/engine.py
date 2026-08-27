from decimal import Decimal

from claim_ai.domain.models import DocumentFacts
from claim_ai.rules.handlers import HANDLERS
from claim_ai.rules.models import RuleDecision, RuleStatus, RuleTrace


class SequentialRuleEngine:
    def __init__(
        self,
        min_confidence: float = 0.85,
        tolerance: Decimal = Decimal("0.01"),
    ) -> None:
        self.min_confidence = min_confidence
        self.tolerance = tolerance

    def evaluate(self, facts: DocumentFacts) -> RuleDecision:
        trace: list[RuleTrace] = []
        for rule_id, handler in HANDLERS:
            attempt = handler(facts, self.min_confidence, self.tolerance)
            trace.append(
                RuleTrace(rule_id=rule_id, outcome=attempt.outcome, reason=attempt.reason)
            )
            if attempt.outcome == "MATCHED":
                return RuleDecision(
                    status=RuleStatus.CALCULATED,
                    selected_rule=rule_id,
                    target_amount=attempt.amount,
                    formula=attempt.formula,
                    inputs=attempt.inputs,
                    trace=trace,
                )
            if attempt.outcome == "NEEDS_REVIEW":
                return RuleDecision(
                    status=RuleStatus.NEEDS_REVIEW,
                    selected_rule=rule_id,
                    inputs=attempt.inputs,
                    trace=trace,
                    warnings=[attempt.reason],
                )
        return RuleDecision(
            status=RuleStatus.NEEDS_REVIEW,
            trace=trace,
            warnings=["没有规则满足计算条件"],
        )
