from fastapi import APIRouter

from claim_ai.domain.models import DocumentFacts
from claim_ai.rules.engine import SequentialRuleEngine
from claim_ai.rules.models import RuleDecision

router = APIRouter(prefix="/api/v1/precheck", tags=["precheck"])
engine = SequentialRuleEngine()


@router.post("/calculate", response_model=RuleDecision)
def calculate(facts: DocumentFacts) -> RuleDecision:
    return engine.evaluate(facts)
