from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import ClassVar, cast

import pytest

from claim_ai.adapters.ollama import (
    OllamaClientError,
    OllamaSchemaError,
    OllamaServerError,
    OllamaTimeout,
    OllamaTransportError,
)
from claim_ai.config.field_catalog import FieldCatalog, FieldCatalogFile, FieldDefinition
from claim_ai.domain.models import DocumentType
from claim_ai.pipeline.models import FieldCandidate, ImageArtifact, OcrResult
from claim_ai.pipeline.ports import ImagePreprocessor, OcrEngine, VisionLanguageExtractor
from claim_ai.pipeline.service import ExtractionService, PipelineStageError
from claim_ai.rules.engine import SequentialRuleEngine
from claim_ai.rules.models import RuleStatus


class RetryableRouterError(RuntimeError):
    retryable: ClassVar[bool] = True


class RecordingPreprocessor:
    def __init__(self, calls: list[str], warnings: list[str] | None = None) -> None:
        self.calls = calls
        self.warnings = warnings or []

    def process(self, source: Path, destination: Path) -> ImageArtifact:
        assert destination.parent.is_dir()
        self.calls.append("preprocess")
        return ImageArtifact(
            original_path=source,
            processed_path=destination,
            warnings=self.warnings,
        )


class RecordingOcr:
    def __init__(self, calls: list[str], result: OcrResult) -> None:
        self.calls = calls
        self.result = result
        self.artifact: ImageArtifact | None = None

    def recognize(self, image: ImageArtifact) -> OcrResult:
        self.calls.append("ocr")
        self.artifact = image
        return self.result


class RecordingRouter:
    def __init__(
        self,
        calls: list[str],
        document_type: DocumentType,
        template_id: str | None,
    ) -> None:
        self.calls = calls
        self.document_type = document_type
        self.template_id = template_id
        self.received_text: str | None = None

    def route(self, ocr_text: str) -> tuple[DocumentType, str | None]:
        self.calls.append("route")
        self.received_text = ocr_text
        return self.document_type, self.template_id


class RecordingExtractor:
    def __init__(self, calls: list[str], candidates: list[FieldCandidate]) -> None:
        self.calls = calls
        self.candidates = candidates
        self.artifact: ImageArtifact | None = None
        self.ocr_result: OcrResult | None = None

    def extract(
        self, image: ImageArtifact, ocr: OcrResult
    ) -> list[FieldCandidate]:
        self.calls.append("vlm")
        self.artifact = image
        self.ocr_result = ocr
        return self.candidates


class RecordingCatalog(FieldCatalog):
    def __init__(self, calls: list[str]) -> None:
        super().__init__(
            FieldCatalogFile(
                version="test",
                fields=[
                    FieldDefinition(
                        key="personal_cash_payment",
                        type="amount",
                        aliases=["个人现金支付"],
                    )
                ],
            )
        )
        self.calls = calls

    def resolve(self, raw_name: str) -> str | None:
        self.calls.append("normalize")
        return super().resolve(raw_name)


def test_extract_orchestrates_dependencies_and_returns_rule_ready_facts(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    source = tmp_path / "source.png"
    source.write_bytes(b"synthetic-source")
    original_bytes = source.read_bytes()
    work_dir = tmp_path / "stable-work"
    ocr_result = OcrResult(tokens=[], full_text="完整 OCR 文本\n个人现金支付 35.00")
    preprocessor = RecordingPreprocessor(calls)
    ocr = RecordingOcr(calls, ocr_result)
    router = RecordingRouter(
        calls, DocumentType.MEDICAL_RECEIPT, "henan-medical-receipt-v1"
    )
    extractor = RecordingExtractor(
        calls,
        [FieldCandidate(raw_name="个人现金支付", raw_value="35.00", confidence=0.99)],
    )
    service = ExtractionService(
        preprocessor=preprocessor,
        ocr=ocr,
        extractor=extractor,
        router=router,
        catalog=RecordingCatalog(calls),
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )

    result = service.extract(source, work_dir)

    assert calls == ["preprocess", "ocr", "route", "vlm", "normalize"]
    assert router.received_text == ocr_result.full_text
    assert ocr.artifact is extractor.artifact
    assert extractor.ocr_result is ocr_result
    assert extractor.artifact is not None
    assert extractor.artifact.processed_path.parent == work_dir
    assert extractor.artifact.processed_path.name.startswith("processed-")
    assert extractor.artifact.processed_path.suffix == ".png"
    assert source.read_bytes() == original_bytes
    assert result.document_type is DocumentType.MEDICAL_RECEIPT
    assert result.facts.document_type is result.document_type
    assert result.template_id == "henan-medical-receipt-v1"
    assert result.model_version == "fake-v1"
    assert result.prompt_version == "prompt-v1"
    decision = SequentialRuleEngine().evaluate(result.facts)
    assert decision.status is RuleStatus.CALCULATED
    assert decision.selected_rule == "R002"


class CoordinatedPreprocessor:
    def __init__(self) -> None:
        self.first_written = Event()
        self.second_written = Event()
        self.destinations: list[Path] = []

    def process(self, source: Path, destination: Path) -> ImageArtifact:
        self.destinations.append(destination)
        if source.name == "first.png":
            destination.write_bytes(source.read_bytes())
            self.first_written.set()
            assert self.second_written.wait(timeout=5.0)
        else:
            assert self.first_written.wait(timeout=5.0)
            destination.write_bytes(source.read_bytes())
            self.second_written.set()
        return ImageArtifact(original_path=source, processed_path=destination)


class ArtifactReadingOcr:
    def recognize(self, image: ImageArtifact) -> OcrResult:
        return OcrResult(tokens=[], full_text=image.processed_path.read_text(encoding="utf-8"))


class AmountExtractor:
    def extract(
        self, image: ImageArtifact, ocr: OcrResult
    ) -> list[FieldCandidate]:
        return [
            FieldCandidate(
                raw_name="个人现金支付",
                raw_value=ocr.full_text,
                confidence=0.99,
            )
        ]


def test_concurrent_extractions_use_isolated_processed_artifacts(tmp_path: Path) -> None:
    work_dir = tmp_path / "shared-work"
    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    expected = {sources[0]: "35.00", sources[1]: "42.00"}
    for source, amount in expected.items():
        source.write_text(amount, encoding="utf-8")
    preprocessor = CoordinatedPreprocessor()
    calls: list[str] = []
    service = ExtractionService(
        preprocessor=preprocessor,
        ocr=ArtifactReadingOcr(),
        extractor=AmountExtractor(),
        router=RecordingRouter(calls, DocumentType.MEDICAL_RECEIPT, "template-v1"),
        catalog=RecordingCatalog(calls),
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {source: executor.submit(service.extract, source, work_dir) for source in sources}

    for source, future in futures.items():
        extracted_amount = future.result().facts.amount("personal_cash_payment")
        assert extracted_amount is not None
        assert extracted_amount.to_eng_string() == expected[source]
    assert len(set(preprocessor.destinations)) == 2
    assert all(path.parent == work_dir for path in preprocessor.destinations)


def test_extract_deduplicates_review_reasons_in_stable_order(tmp_path: Path) -> None:
    calls: list[str] = []
    service = ExtractionService(
        preprocessor=RecordingPreprocessor(
            calls, ["图像模糊", "重复图像警告", "图像模糊"]
        ),
        ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="未分类票据")),
        extractor=RecordingExtractor(
            calls,
            [
                FieldCandidate(raw_name="未知甲", raw_value="1", confidence=0.9),
                FieldCandidate(raw_name="未知甲", raw_value="2", confidence=0.8),
                FieldCandidate(raw_name="未知乙", raw_value="3", confidence=0.7),
            ],
        ),
        router=RecordingRouter(calls, DocumentType.UNKNOWN, None),
        catalog=RecordingCatalog(calls),
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )

    result = service.extract(tmp_path / "input.png", tmp_path / "work")

    assert result.document_type is DocumentType.UNKNOWN
    assert result.facts.document_type is result.document_type
    assert result.review_reasons == [
        "图像模糊",
        "重复图像警告",
        "无法确定票据类型",
        "未标准化字段：未知甲",
        "未标准化字段：未知乙",
    ]


class RaisingPreprocessor:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def process(self, source: Path, destination: Path) -> ImageArtifact:
        raise self.error


class RaisingOcr:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def recognize(self, image: ImageArtifact) -> OcrResult:
        raise self.error


class RaisingExtractor:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def extract(
        self, image: ImageArtifact, ocr: OcrResult
    ) -> list[FieldCandidate]:
        raise self.error


class RaisingRouter:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def route(self, ocr_text: str) -> tuple[DocumentType, str | None]:
        raise self.error


class MalformedRouter:
    def __init__(self, result: object) -> None:
        self.result = result

    def route(self, ocr_text: str) -> tuple[DocumentType, str | None]:
        return cast(tuple[DocumentType, str | None], self.result)


class RaisingCatalog(RecordingCatalog):
    def __init__(self, calls: list[str], error: Exception) -> None:
        super().__init__(calls)
        self.error = error

    def resolve(self, raw_name: str) -> str | None:
        raise self.error


def _service_for_error(
    *,
    calls: list[str],
    preprocessor: ImagePreprocessor,
    ocr: OcrEngine,
    extractor: VisionLanguageExtractor,
    catalog: FieldCatalog,
) -> ExtractionService:
    return ExtractionService(
        preprocessor=preprocessor,
        ocr=ocr,
        extractor=extractor,
        router=RecordingRouter(calls, DocumentType.MEDICAL_RECEIPT, "template-v1"),
        catalog=catalog,
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )


@pytest.mark.parametrize(
    ("stage", "code", "retryable", "message", "build_service"),
    [
        (
            "preprocess",
            "PREPROCESS_FAILED",
            False,
            "image preprocessing failed",
            lambda calls, error: _service_for_error(
                calls=calls,
                preprocessor=RaisingPreprocessor(error),
                ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="text")),
                extractor=RecordingExtractor(calls, []),
                catalog=RecordingCatalog(calls),
            ),
        ),
        (
            "ocr",
            "OCR_FAILED",
            False,
            "OCR recognition failed",
            lambda calls, error: _service_for_error(
                calls=calls,
                preprocessor=RecordingPreprocessor(calls),
                ocr=RaisingOcr(error),
                extractor=RecordingExtractor(calls, []),
                catalog=RecordingCatalog(calls),
            ),
        ),
        (
            "vlm",
            "VLM_FAILED",
            False,
            "vision-language extraction failed",
            lambda calls, error: _service_for_error(
                calls=calls,
                preprocessor=RecordingPreprocessor(calls),
                ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="text")),
                extractor=RaisingExtractor(error),
                catalog=RecordingCatalog(calls),
            ),
        ),
        (
            "normalize",
            "NORMALIZE_FAILED",
            False,
            "candidate normalization failed",
            lambda calls, error: _service_for_error(
                calls=calls,
                preprocessor=RecordingPreprocessor(calls),
                ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="text")),
                extractor=RecordingExtractor(
                    calls,
                    [
                        FieldCandidate(
                            raw_name="个人现金支付",
                            raw_value="35.00",
                            confidence=0.99,
                        )
                    ],
                ),
                catalog=RaisingCatalog(calls, error),
            ),
        ),
    ],
)
def test_extract_wraps_stage_errors_with_sanitized_stable_metadata(
    stage: str,
    code: str,
    retryable: bool,
    message: str,
    build_service: Callable[[list[str], Exception], ExtractionService],
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    sensitive = "patient-secret D:\\private\\receipt.png OCR-CONTENT"
    cause = RuntimeError(sensitive)
    service = build_service(calls, cause)

    with pytest.raises(PipelineStageError) as captured:
        service.extract(tmp_path / "sensitive-source.png", tmp_path / "work")

    error = captured.value
    assert error.stage == stage
    assert error.code == code
    assert error.retryable is retryable
    assert error.message == message
    assert str(error) == message
    assert error.__cause__ is cause
    assert sensitive not in str(error)


def test_input_error_is_not_retryable(tmp_path: Path) -> None:
    calls: list[str] = []
    cause = ValueError("invalid OCR input patient-secret")
    service = _service_for_error(
        calls=calls,
        preprocessor=RecordingPreprocessor(calls),
        ocr=RaisingOcr(cause),
        extractor=RecordingExtractor(calls, []),
        catalog=RecordingCatalog(calls),
    )

    with pytest.raises(PipelineStageError) as captured:
        service.extract(tmp_path / "input.png", tmp_path / "work")

    assert captured.value.stage == "ocr"
    assert captured.value.retryable is False
    assert captured.value.__cause__ is cause


@pytest.mark.parametrize(
    ("cause", "expected_retryable"),
    [
        (OllamaTimeout("timeout"), True),
        (OllamaTransportError("transport"), True),
        (OllamaServerError("server"), True),
        (OllamaClientError("client"), False),
        (OllamaSchemaError("schema"), False),
        (AssertionError("assertion"), False),
        (AttributeError("attribute"), False),
        (KeyError("key"), False),
        (NotImplementedError("not implemented"), False),
    ],
)
def test_retryability_requires_an_explicit_transient_error_contract(
    cause: Exception,
    expected_retryable: bool,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    service = _service_for_error(
        calls=calls,
        preprocessor=RecordingPreprocessor(calls),
        ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="text")),
        extractor=RaisingExtractor(cause),
        catalog=RecordingCatalog(calls),
    )

    with pytest.raises(PipelineStageError) as captured:
        service.extract(tmp_path / "input.png", tmp_path / "work")

    assert captured.value.stage == "vlm"
    assert captured.value.retryable is expected_retryable
    assert captured.value.__cause__ is cause


def test_router_exception_is_a_sanitized_normalize_error(tmp_path: Path) -> None:
    calls: list[str] = []
    cause = RuntimeError("patient-secret D:\\private\\ticket.png OCR-CONTENT")
    service = ExtractionService(
        preprocessor=RecordingPreprocessor(calls),
        ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="private OCR")),
        extractor=RecordingExtractor(calls, []),
        router=RaisingRouter(cause),
        catalog=RecordingCatalog(calls),
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )

    with pytest.raises(PipelineStageError) as captured:
        service.extract(tmp_path / "input.png", tmp_path / "work")

    assert captured.value.stage == "normalize"
    assert captured.value.code == "NORMALIZE_FAILED"
    assert captured.value.retryable is False
    assert str(captured.value) == "candidate normalization failed"
    assert "patient-secret" not in str(captured.value)
    assert captured.value.__cause__ is cause


@pytest.mark.parametrize(
    "cause",
    [
        OllamaTimeout("patient-secret timeout"),
        RetryableRouterError("patient-secret custom retryable error"),
    ],
)
def test_retryable_router_exception_is_forced_non_retryable(
    cause: Exception, tmp_path: Path
) -> None:
    calls: list[str] = []
    service = ExtractionService(
        preprocessor=RecordingPreprocessor(calls),
        ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="private OCR")),
        extractor=RecordingExtractor(calls, []),
        router=RaisingRouter(cause),
        catalog=RecordingCatalog(calls),
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )

    with pytest.raises(PipelineStageError) as captured:
        service.extract(tmp_path / "input.png", tmp_path / "work")

    assert captured.value.stage == "normalize"
    assert captured.value.retryable is False
    assert str(captured.value) == "candidate normalization failed"
    assert captured.value.__cause__ is cause


def test_two_item_list_router_result_is_a_non_retryable_normalize_error(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    service = ExtractionService(
        preprocessor=RecordingPreprocessor(calls),
        ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="private OCR")),
        extractor=RecordingExtractor(calls, []),
        router=MalformedRouter([DocumentType.MEDICAL_RECEIPT, "template-v1"]),
        catalog=RecordingCatalog(calls),
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )

    with pytest.raises(PipelineStageError) as captured:
        service.extract(tmp_path / "input.png", tmp_path / "work")

    assert captured.value.stage == "normalize"
    assert captured.value.retryable is False
    assert str(captured.value) == "candidate normalization failed"
    assert "private OCR" not in str(captured.value)
    assert isinstance(captured.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "malformed_result",
    [
        None,
        (DocumentType.MEDICAL_RECEIPT,),
        (DocumentType.MEDICAL_RECEIPT, "template-v1", "extra"),
        ("medical_receipt", "template-v1"),
        (DocumentType.MEDICAL_RECEIPT, 123),
    ],
)
def test_malformed_router_result_is_a_sanitized_normalize_error(
    malformed_result: object,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    service = ExtractionService(
        preprocessor=RecordingPreprocessor(calls),
        ocr=RecordingOcr(calls, OcrResult(tokens=[], full_text="private OCR")),
        extractor=RecordingExtractor(calls, []),
        router=MalformedRouter(malformed_result),
        catalog=RecordingCatalog(calls),
        model_version="fake-v1",
        prompt_version="prompt-v1",
    )

    with pytest.raises(PipelineStageError) as captured:
        service.extract(tmp_path / "input.png", tmp_path / "work")

    assert captured.value.stage == "normalize"
    assert captured.value.code == "NORMALIZE_FAILED"
    assert captured.value.retryable is False
    assert str(captured.value) == "candidate normalization failed"
    assert "private OCR" not in str(captured.value)
    assert isinstance(captured.value.__cause__, (TypeError, ValueError))
