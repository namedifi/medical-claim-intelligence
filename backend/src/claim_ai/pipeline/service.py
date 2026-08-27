from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, TypeVar
from uuid import uuid4

from claim_ai.config.field_catalog import FieldCatalog
from claim_ai.domain.models import DocumentType
from claim_ai.pipeline.models import ExtractionResult, ImageArtifact
from claim_ai.pipeline.normalizer import normalize_candidates
from claim_ai.pipeline.ports import ImagePreprocessor, OcrEngine, VisionLanguageExtractor

PipelineStage = Literal["preprocess", "ocr", "vlm", "normalize"]

_T = TypeVar("_T")
_STAGE_METADATA: dict[PipelineStage, tuple[str, str]] = {
    "preprocess": ("PREPROCESS_FAILED", "image preprocessing failed"),
    "ocr": ("OCR_FAILED", "OCR recognition failed"),
    "vlm": ("VLM_FAILED", "vision-language extraction failed"),
    "normalize": ("NORMALIZE_FAILED", "candidate normalization failed"),
}


class TemplateRouter(Protocol):
    def route(self, ocr_text: str) -> tuple[DocumentType, str | None]: ...


class PipelineStageError(RuntimeError):
    def __init__(
        self,
        stage: PipelineStage,
        code: str,
        retryable: bool,
        message: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.retryable = retryable
        self.message = message


def _is_retryable(error: Exception) -> bool:
    return getattr(type(error), "retryable", False) is True


def _run_stage(
    stage: PipelineStage,
    operation: Callable[[], _T],
    *,
    retryable: bool | None = None,
) -> _T:
    try:
        return operation()
    except Exception as error:
        code, message = _STAGE_METADATA[stage]
        raise PipelineStageError(
            stage=stage,
            code=code,
            retryable=_is_retryable(error) if retryable is None else retryable,
            message=message,
        ) from error


class ExtractionService:
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        ocr: OcrEngine,
        extractor: VisionLanguageExtractor,
        router: TemplateRouter,
        catalog: FieldCatalog,
        model_version: str,
        prompt_version: str,
    ) -> None:
        self.preprocessor = preprocessor
        self.ocr = ocr
        self.extractor = extractor
        self.router = router
        self.catalog = catalog
        self.model_version = model_version
        self.prompt_version = prompt_version

    def extract(self, source: Path, work_dir: Path) -> ExtractionResult:
        source_path = Path(source)
        work_path = Path(work_dir)

        def preprocess() -> ImageArtifact:
            work_path.mkdir(parents=True, exist_ok=True)
            destination = work_path / f"processed-{uuid4().hex}.png"
            return self.preprocessor.process(source_path, destination)

        def route_document() -> tuple[DocumentType, str | None]:
            route_result = self.router.route(ocr_result.full_text)
            if not isinstance(route_result, tuple) or len(route_result) != 2:
                raise TypeError
            document_type, template_id = route_result
            if not isinstance(document_type, DocumentType):
                raise TypeError
            if template_id is not None and not isinstance(template_id, str):
                raise TypeError
            return document_type, template_id

        artifact = _run_stage("preprocess", preprocess)
        ocr_result = _run_stage("ocr", lambda: self.ocr.recognize(artifact))
        document_type, template_id = _run_stage(
            "normalize", route_document, retryable=False
        )
        candidates = _run_stage(
            "vlm", lambda: self.extractor.extract(artifact, ocr_result)
        )
        normalized = _run_stage(
            "normalize",
            lambda: normalize_candidates(candidates, self.catalog, document_type),
        )
        facts = normalized.facts.model_copy(update={"document_type": document_type})

        reasons = [*artifact.warnings]
        if document_type is DocumentType.UNKNOWN:
            reasons.append("无法确定票据类型")
        reasons.extend(f"未标准化字段：{name}" for name in normalized.unknown_fields)
        review_reasons = list(dict.fromkeys(reasons))

        return ExtractionResult(
            document_type=document_type,
            template_id=template_id,
            facts=facts,
            review_reasons=review_reasons,
            model_version=self.model_version,
            prompt_version=self.prompt_version,
        )
