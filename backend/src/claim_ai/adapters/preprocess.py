import math
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from claim_ai.pipeline.models import ImageArtifact

ImageArray = npt.NDArray[np.integer[Any] | np.floating[Any]]


class PreprocessError(RuntimeError):
    """Raised when image preprocessing cannot produce a safe artifact."""


def _paths_share_file(source: Path, destination: Path) -> bool:
    if destination.exists():
        return os.path.samefile(source, destination)
    return source.expanduser().resolve() == destination.expanduser().resolve()


def _load_with_exif(path: Path) -> ImageArray:
    with Image.open(path) as image:
        transposed = ImageOps.exif_transpose(image)
        if transposed is None:  # pragma: no cover - current Pillow always returns an image
            raise OSError("EXIF transpose failed")
        rgb = transposed.convert("RGB")
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _estimate_skew(gray: ImageArray) -> float:
    if gray.ndim != 2:
        raise ValueError("skew estimation requires a grayscale image")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    height, width = gray.shape
    minimum_length = int(max(height, width) * 0.20)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=minimum_length,
        maxLineGap=30,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for raw_line in lines:
        x1, y1, x2, y2 = np.asarray(raw_line, dtype=np.float32).reshape(-1)[:4]
        if math.hypot(float(x2 - x1), float(y2 - y1)) < minimum_length:
            continue
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        if abs(angle) <= 20.0:
            angles.append(angle)

    if len(angles) < 3:
        return 0.0
    angle_array = np.asarray(angles, dtype=np.float32)
    median_angle = float(np.median(angle_array))
    if float(np.std(angle_array)) > 5.0:
        return 0.0
    if abs(median_angle) < 0.8 or abs(median_angle) > 20.0:
        return 0.0
    return median_angle


def _rotate_expanded(image: ImageArray, angle: float) -> ImageArray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    new_width = math.ceil(height * sine + width * cosine)
    new_height = math.ceil(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _quality_score(image: ImageArray) -> tuple[float, list[str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_ratio = float(np.mean(gray <= 5))
    bright_ratio = float(np.mean(gray >= 250))

    clarity_score = float(np.clip(clarity / 500.0, 0.0, 1.0))
    exposure_score = float(np.clip(1.0 - dark_ratio - bright_ratio, 0.0, 1.0))
    score = float(np.clip(0.7 * clarity_score + 0.3 * exposure_score, 0.0, 1.0))

    warnings: list[str] = []
    if clarity < 100.0:
        warnings.append("图像模糊")
    if dark_ratio > 0.8 or bright_ratio > 0.8:
        warnings.append("曝光异常")
    return score, warnings


class OpenCvPreprocessor:
    def process(self, source: Path, destination: Path) -> ImageArtifact:
        source_path = Path(source)
        destination_path = Path(destination)
        identity_failed = False
        shares_file = False
        try:
            shares_file = _paths_share_file(source_path, destination_path)
        except Exception:  # noqa: BLE001 - sanitize filesystem identity failures
            identity_failed = True

        if identity_failed:
            raise PreprocessError("image preprocessing failed")
        if shares_file:
            raise PreprocessError("source and destination must differ")

        failed = False
        artifact: ImageArtifact | None = None
        temporary_path: Path | None = None
        try:
            image = _load_with_exif(source_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            angle = _estimate_skew(gray)
            processed = _rotate_expanded(image, angle) if angle != 0.0 else image.copy()
            quality_score, warnings = _quality_score(processed)
            artifact = ImageArtifact(
                original_path=source_path,
                processed_path=destination_path,
                quality_score=quality_score,
                warnings=warnings,
            )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".claim-ai-preprocess-",
                suffix=destination_path.suffix,
                dir=destination_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            if not cv2.imwrite(str(temporary_path), processed):
                raise OSError("image encoder rejected destination")
            os.replace(temporary_path, destination_path)
            temporary_path = None
        except Exception:  # noqa: BLE001 - sanitize every decoder/encoder boundary failure
            failed = True
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    failed = True

        if failed or artifact is None:
            raise PreprocessError("image preprocessing failed")
        return artifact
