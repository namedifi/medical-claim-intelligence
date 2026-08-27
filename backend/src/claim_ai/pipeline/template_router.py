import json
from pathlib import Path

from pydantic import BaseModel

from claim_ai.domain.models import DocumentType


class TemplateDefinition(BaseModel):
    id: str
    document_type: DocumentType
    strong: list[str]
    keywords: list[str]
    exclude: list[str]


class TemplateFile(BaseModel):
    version: str
    templates: list[TemplateDefinition]


class TemplateRouter:
    def __init__(self, data: TemplateFile) -> None:
        self.version = data.version
        self.templates = data.templates

    @classmethod
    def load(cls, path: Path) -> "TemplateRouter":
        data = TemplateFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return cls(data)

    def route(self, ocr_text: str) -> tuple[DocumentType, str | None]:
        best_template: TemplateDefinition | None = None
        best_score: int | None = None
        is_tied = False

        for template in self.templates:
            score = 10 * sum(keyword in ocr_text for keyword in template.strong)
            score += 2 * sum(keyword in ocr_text for keyword in template.keywords)
            score -= 10 * sum(keyword in ocr_text for keyword in template.exclude)

            if best_score is None or score > best_score:
                best_template = template
                best_score = score
                is_tied = False
            elif score == best_score:
                is_tied = True

        if best_template is None or best_score is None or best_score < 2 or is_tied:
            return DocumentType.UNKNOWN, None
        return best_template.document_type, best_template.id
