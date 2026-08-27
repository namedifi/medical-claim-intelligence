from pathlib import Path

import pytest
from pydantic import ValidationError

from claim_ai.pipeline.fakes import FakeOcrEngine
from claim_ai.pipeline.models import ImageArtifact, OcrToken
from claim_ai.pipeline.ports import ImagePreprocessor, OcrEngine, VisionLanguageExtractor


def test_fake_ocr_returns_configured_tokens(tmp_path: Path) -> None:
    image = tmp_path / "receipt.png"
    image.write_bytes(b"synthetic")
    engine = FakeOcrEngine.from_pairs([("金额合计", 0.99), ("10.00", 0.98)])

    result = engine.recognize(ImageArtifact(original_path=image, processed_path=image))

    assert [token.text for token in result.tokens] == ["金额合计", "10.00"]
    assert result.full_text == "金额合计\n10.00"


def test_fake_ocr_engine_implements_protocol() -> None:
    assert isinstance(FakeOcrEngine.from_pairs([]), OcrEngine)


def test_pipeline_protocols_expose_expected_methods() -> None:
    assert callable(ImagePreprocessor.process)
    assert callable(OcrEngine.recognize)
    assert callable(VisionLanguageExtractor.extract)


def test_image_artifact_serializes_paths_as_strings(tmp_path: Path) -> None:
    image = tmp_path / "receipt.png"
    artifact = ImageArtifact(original_path=image, processed_path=image)

    serialized = artifact.model_dump(mode="json")

    assert serialized["original_path"] == str(image)
    assert serialized["processed_path"] == str(image)


@pytest.mark.parametrize("quality_score", [-0.01, 1.01])
def test_image_quality_score_must_be_between_zero_and_one(
    quality_score: float, tmp_path: Path
) -> None:
    with pytest.raises(ValidationError):
        ImageArtifact(
            original_path=tmp_path / "receipt.png",
            processed_path=tmp_path / "receipt.png",
            quality_score=quality_score,
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_between_zero_and_one(confidence: float) -> None:
    with pytest.raises(ValidationError):
        OcrToken(
            text="金额合计",
            confidence=confidence,
            bbox=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )


def test_bbox_must_have_exactly_four_points() -> None:
    with pytest.raises(ValidationError):
        OcrToken(
            text="金额合计",
            confidence=0.9,
            bbox=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),  # type: ignore[arg-type]
        )
