import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from claim_ai.adapters.preprocess import (
    OpenCvPreprocessor,
    PreprocessError,
    _estimate_skew,
)

ImageArray = npt.NDArray[np.integer[Any] | np.floating[Any]]


def _write_rotated_receipt(path: Path, angle: float = 7.0) -> ImageArray:
    receipt = np.full((400, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(receipt, (80, 80), (520, 320), (0, 0, 0), thickness=2)
    matrix = cv2.getRotationMatrix2D((300.0, 200.0), angle, 1.0)
    rotated = cv2.warpAffine(
        receipt,
        matrix,
        (600, 400),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    assert cv2.imwrite(str(path), rotated)
    return rotated


def test_process_deskews_into_expanded_decodable_canvas(tmp_path: Path) -> None:
    source = tmp_path / "rotated.png"
    destination = tmp_path / "processed" / "receipt.png"
    original = _write_rotated_receipt(source)

    artifact = OpenCvPreprocessor().process(source, destination)

    output = cv2.imread(str(destination))
    assert output is not None
    assert output.shape[0] > original.shape[0]
    assert output.shape[1] > original.shape[1]
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    original_foreground = int(np.count_nonzero(original_gray < 180))
    output_foreground = int(np.count_nonzero(output_gray < 180))
    assert original_foreground > 0
    assert output_foreground / original_foreground >= 0.85
    assert abs(_estimate_skew(output_gray)) < 1.0
    assert destination.parent.is_dir()
    assert artifact.original_path == source
    assert artifact.processed_path == destination
    assert 0.0 <= artifact.quality_score <= 1.0


def test_load_applies_exif_orientation(tmp_path: Path) -> None:
    source = tmp_path / "phone.jpg"
    destination = tmp_path / "phone-processed.png"
    image = Image.new("RGB", (12, 24), "white")
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)

    OpenCvPreprocessor().process(source, destination)

    output = cv2.imread(str(destination))
    assert output is not None
    assert output.shape[:2] == (12, 24)


def test_estimate_skew_returns_zero_without_reliable_lines() -> None:
    gray = np.full((200, 300), 255, dtype=np.uint8)

    assert _estimate_skew(gray) == 0.0


@pytest.mark.parametrize(
    ("image", "expected_warning"),
    [
        (np.full((120, 160, 3), 127, dtype=np.uint8), "图像模糊"),
        (np.full((120, 160, 3), 255, dtype=np.uint8), "曝光异常"),
    ],
)
def test_process_reports_quality_warnings(
    image: ImageArray, expected_warning: str, tmp_path: Path
) -> None:
    source = tmp_path / "quality.png"
    destination = tmp_path / "out" / source.name
    assert cv2.imwrite(str(source), image)

    artifact = OpenCvPreprocessor().process(source, destination)

    assert expected_warning in artifact.warnings
    assert 0.0 <= artifact.quality_score <= 1.0


def test_process_preserves_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "original.png"
    destination = tmp_path / "nested" / "processed.png"
    _write_rotated_receipt(source)
    before = source.read_bytes()

    OpenCvPreprocessor().process(source, destination)

    assert source.read_bytes() == before
    assert destination.is_file()


def test_process_rejects_identical_source_and_destination(tmp_path: Path) -> None:
    source = tmp_path / "receipt.png"
    _write_rotated_receipt(source)
    before = source.read_bytes()

    with pytest.raises(PreprocessError, match="source and destination must differ"):
        OpenCvPreprocessor().process(source, source)

    assert source.read_bytes() == before


def test_process_uses_resolved_path_check_when_destination_does_not_exist(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing.png"
    destination = tmp_path / "absent-parent" / ".." / source.name
    assert not destination.exists()

    with pytest.raises(PreprocessError, match="source and destination must differ"):
        OpenCvPreprocessor().process(source, destination)


def test_process_rejects_existing_hard_link_alias(tmp_path: Path) -> None:
    source = tmp_path / "receipt.png"
    destination = tmp_path / "receipt-hard-link.png"
    _write_rotated_receipt(source)
    before = source.read_bytes()
    os.link(source, destination)

    with pytest.raises(PreprocessError, match="source and destination must differ"):
        OpenCvPreprocessor().process(source, destination)

    assert source.read_bytes() == before
    assert destination.read_bytes() == before


def test_process_rejects_existing_symbolic_alias(tmp_path: Path) -> None:
    source = tmp_path / "receipt.png"
    destination = tmp_path / "receipt-symbolic-link.png"
    _write_rotated_receipt(source)
    before = source.read_bytes()
    try:
        destination.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error.__class__.__name__}")

    with pytest.raises(PreprocessError, match="source and destination must differ"):
        OpenCvPreprocessor().process(source, destination)

    assert source.read_bytes() == before
    assert destination.read_bytes() == before


def test_encoder_partial_failure_preserves_existing_destination_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "receipt.png"
    destination = tmp_path / "processed" / "receipt.png"
    _write_rotated_receipt(source)
    destination.parent.mkdir()
    existing = b"existing-synthetic-artifact"
    destination.write_bytes(existing)
    encoder_paths: list[Path] = []

    def partial_encoder(path: str, image: ImageArray) -> bool:
        assert image.size > 0
        encoder_path = Path(path)
        encoder_paths.append(encoder_path)
        encoder_path.write_bytes(b"sensitive-partial-fragment")
        return False

    monkeypatch.setattr(cv2, "imwrite", partial_encoder)

    with pytest.raises(PreprocessError, match="image preprocessing failed") as captured:
        OpenCvPreprocessor().process(source, destination)

    assert encoder_paths and encoder_paths[0].parent == destination.parent
    assert encoder_paths[0] != destination
    assert destination.read_bytes() == existing
    assert sorted(destination.parent.iterdir()) == [destination]
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_process_raises_typed_sanitized_error_for_invalid_image(tmp_path: Path) -> None:
    source = tmp_path / "private.png"
    source.write_bytes(b"patient-id=secret")

    with pytest.raises(PreprocessError, match="image preprocessing failed") as captured:
        OpenCvPreprocessor().process(source, tmp_path / "output.png")

    assert "patient-id" not in str(captured.value)
    assert "secret" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
