from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

import numpy as np

from claim_ai.pipeline.models import ImageArtifact, OcrResult, OcrToken


class PaddleOcrError(RuntimeError):
    """Base error for sanitized PaddleOCR adapter failures."""


class PaddleModelError(PaddleOcrError):
    """Raised when local PaddleOCR model loading fails."""


class PaddleResultError(PaddleOcrError):
    """Raised when PaddleOCR returns an invalid result shape."""


class _PaddleModel(Protocol):
    def predict(self, **kwargs: object) -> object: ...


_Bbox = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]
_ReadingItem = tuple[OcrToken, float, float, float, float]


def _items(value: object) -> list[object]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError
    return list(cast(Iterable[object], value))


def _polygon(value: object) -> _Bbox:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 2) or not bool(np.isfinite(array).all()):
        raise ValueError
    points = tuple((float(x), float(y)) for x, y in array.tolist())
    return cast(_Bbox, points)


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _line_geometry(line: list[_ReadingItem]) -> tuple[float, float, float]:
    top = _median([item[1] for item in line])
    bottom = _median([item[2] for item in line])
    center = _median([item[3] for item in line])
    return top, bottom, center


def _cluster_lines(items: list[_ReadingItem]) -> list[list[_ReadingItem]]:
    lines: list[list[_ReadingItem]] = []
    for item in sorted(items, key=lambda candidate: (candidate[3], candidate[1], candidate[4])):
        _, top, bottom, center, _ = item
        height = max(bottom - top, 1e-6)
        best_line: list[_ReadingItem] | None = None
        best_score = (-1.0, float("-inf"))
        for line in lines:
            line_top, line_bottom, line_center = _line_geometry(line)
            line_height = max(line_bottom - line_top, 1e-6)
            overlap = max(0.0, min(bottom, line_bottom) - max(top, line_top))
            overlap_ratio = overlap / min(height, line_height)
            center_distance = abs(center - line_center)
            center_limit = 0.5 * max(height, line_height)
            if overlap_ratio < 0.5 and center_distance > center_limit:
                continue
            score = (overlap_ratio, -center_distance)
            if score > best_score:
                best_line = line
                best_score = score

        if best_line is None:
            lines.append([item])
        else:
            best_line.append(item)

    lines.sort(key=lambda line: (_line_geometry(line)[2], _line_geometry(line)[0]))
    for line in lines:
        line.sort(key=lambda item: item[4])
    return lines


def map_paddle_result(rows: list[dict[str, object]]) -> OcrResult:
    invalid = False
    result: OcrResult | None = None
    try:
        mapped: list[_ReadingItem] = []
        for row in rows:
            texts = _items(row.get("rec_texts"))
            scores = _items(row.get("rec_scores"))
            polygons = _items(row.get("rec_polys"))
            if len(texts) != len(scores) or len(texts) != len(polygons):
                raise ValueError

            for raw_text, raw_score, raw_polygon in zip(texts, scores, polygons, strict=True):
                if not isinstance(raw_text, str):
                    raise TypeError
                confidence = float(cast(Any, raw_score))
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError
                bbox = _polygon(raw_polygon)
                y_values = [point[1] for point in bbox]
                x_values = [point[0] for point in bbox]
                top = min(y_values)
                bottom = max(y_values)
                mapped.append(
                    (
                        OcrToken(text=raw_text, confidence=confidence, bbox=bbox),
                        top,
                        bottom,
                        _median(y_values),
                        min(x_values),
                    )
                )

        tokens = [item[0] for line in _cluster_lines(mapped) for item in line]
        result = OcrResult(tokens=tokens, full_text="\n".join(token.text for token in tokens))
    except Exception:  # noqa: BLE001 - sanitize every untrusted result mapping failure
        invalid = True

    if invalid or result is None:
        raise PaddleResultError("invalid PaddleOCR result")
    return result


def _unwrap_row(value: object) -> dict[str, object]:
    candidates: list[object] = [value]
    for attribute in ("json", "res", "to_dict"):
        if hasattr(value, attribute):
            candidate = getattr(value, attribute)
            candidates.append(candidate() if callable(candidate) else candidate)

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        inner = candidate.get("res")
        selected = inner if isinstance(inner, Mapping) else candidate
        return {str(key): item for key, item in selected.items()}
    raise PaddleResultError("invalid PaddleOCR result")


def _result_rows(value: object) -> list[dict[str, object]]:
    values: list[object]
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        values = [value]
    else:
        failed = False
        iterable_values: list[object] = []
        try:
            iterable_values = list(cast(Iterable[object], value))
        except TypeError:
            failed = True
        if failed:
            raise PaddleResultError("invalid PaddleOCR result")
        values = iterable_values
    return [_unwrap_row(item) for item in values]


def _orientation_model_option(factory: Callable[..., object]) -> str:
    parameters = inspect.signature(factory).parameters
    if "textline_orientation_model_dir" in parameters:
        return "textline_orientation_model_dir"
    if "text_line_orientation_model_dir" in parameters:
        return "text_line_orientation_model_dir"
    raise TypeError


class PaddleOcrEngine:
    def __init__(
        self,
        detection_model_dir: Path,
        recognition_model_dir: Path,
        orientation_model_dir: Path,
    ) -> None:
        self._detection_model_dir = Path(detection_model_dir).expanduser().resolve()
        self._recognition_model_dir = Path(recognition_model_dir).expanduser().resolve()
        self._orientation_model_dir = Path(orientation_model_dir).expanduser().resolve()
        self._model: _PaddleModel | None = None
        self._model_lock = Lock()
        self._inference_lock = Lock()

    def _get_model(self) -> _PaddleModel:
        with self._model_lock:
            if self._model is not None:
                return self._model

            model_dirs = (
                self._detection_model_dir,
                self._recognition_model_dir,
                self._orientation_model_dir,
            )
            if any(not path.is_dir() for path in model_dirs):
                raise PaddleModelError("local PaddleOCR models are unavailable")

            load_failed = False
            model: _PaddleModel | None = None
            try:
                paddleocr = importlib.import_module("paddleocr")
                factory = cast(Callable[..., object], paddleocr.PaddleOCR)
                orientation_option = _orientation_model_option(factory)
                model_options: dict[str, object] = {
                    "device": "cpu",
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_textline_orientation": True,
                    "text_detection_model_dir": str(self._detection_model_dir),
                    "text_recognition_model_dir": str(self._recognition_model_dir),
                    orientation_option: str(self._orientation_model_dir),
                }
                model = cast(
                    _PaddleModel,
                    factory(**model_options),
                )
            except Exception:  # noqa: BLE001 - sanitize import and third-party init failures
                load_failed = True

            if load_failed or model is None:
                raise PaddleModelError("local PaddleOCR model loading failed")
            self._model = model
            return model

    def recognize(self, image: ImageArtifact) -> OcrResult:
        model = self._get_model()
        inference_failed = False
        raw_result: object | None = None
        try:
            with self._inference_lock:
                raw_result = model.predict(input=str(image.processed_path))
        except Exception:  # noqa: BLE001 - sanitize third-party inference failures
            inference_failed = True

        if inference_failed:
            raise PaddleOcrError("PaddleOCR recognition failed")

        result_failed = False
        result: OcrResult | None = None
        try:
            result = map_paddle_result(_result_rows(raw_result))
        except Exception:  # noqa: BLE001 - sanitize every third-party result conversion failure
            result_failed = True

        if result_failed or result is None:
            raise PaddleResultError("invalid PaddleOCR result")
        return result
