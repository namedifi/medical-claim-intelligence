import sys
from collections.abc import ItemsView, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from claim_ai.adapters.paddle import (
    PaddleModelError,
    PaddleOcrEngine,
    PaddleOcrError,
    PaddleResultError,
    map_paddle_result,
)
from claim_ai.pipeline.models import ImageArtifact


def _row(
    texts: list[object], scores: list[object], polys: list[object]
) -> dict[str, object]:
    return {"rec_texts": texts, "rec_scores": scores, "rec_polys": polys}


def _box(left: float, top: float, right: float, bottom: float) -> list[list[float]]:
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _model_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = tuple(tmp_path / name for name in ("det", "rec", "orientation"))
    for path in paths:
        path.mkdir()
    return paths  # type: ignore[return-value]


class _Paddle300Factory:
    def __init__(self, model: object) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        device: str,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
        text_detection_model_dir: str,
        text_recognition_model_dir: str,
        text_line_orientation_model_dir: str,
    ) -> object:
        self.calls.append(
            {
                "device": device,
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
                "use_textline_orientation": use_textline_orientation,
                "text_detection_model_dir": text_detection_model_dir,
                "text_recognition_model_dir": text_recognition_model_dir,
                "text_line_orientation_model_dir": text_line_orientation_model_dir,
            }
        )
        return self.model


class _Paddle301Factory:
    def __init__(self, model: object) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        device: str,
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
        text_detection_model_dir: str,
        text_recognition_model_dir: str,
        textline_orientation_model_dir: str,
    ) -> object:
        self.calls.append(
            {
                "device": device,
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
                "use_textline_orientation": use_textline_orientation,
                "text_detection_model_dir": text_detection_model_dir,
                "text_recognition_model_dir": text_recognition_model_dir,
                "textline_orientation_model_dir": textline_orientation_model_dir,
            }
        )
        return self.model


def _install_paddle(
    monkeypatch: pytest.MonkeyPatch,
    factory: _Paddle300Factory | _Paddle301Factory,
    version: str,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=factory, __version__=version),
    )


def test_map_sorts_tokens_by_row_then_left_and_preserves_four_point_boxes() -> None:
    result = map_paddle_result(
        [
            _row(
                ["右下", "左上", "右上", "左下"],
                [0.91, 0.99, 0.98, 0.92],
                [
                    _box(80, 42, 120, 52),
                    _box(10, 10, 50, 20),
                    _box(70, 12, 110, 22),
                    _box(15, 40, 55, 50),
                ],
            )
        ]
    )

    assert [token.text for token in result.tokens] == ["左上", "右上", "左下", "右下"]
    assert result.full_text == "左上\n右上\n左下\n右下"
    assert all(len(token.bbox) == 4 for token in result.tokens)
    assert result.tokens[0].bbox == (
        (10.0, 10.0),
        (50.0, 10.0),
        (50.0, 20.0),
        (10.0, 20.0),
    )


def test_map_keeps_mixed_font_sizes_on_the_same_baseline() -> None:
    result = map_paddle_result(
        [
            _row(
                ["右侧大字", "下一行", "左侧小字"],
                [0.95, 0.93, 0.94],
                [
                    _box(80, 0, 150, 30),
                    _box(10, 42, 80, 52),
                    _box(10, 20, 70, 30),
                ],
            )
        ]
    )

    assert [token.text for token in result.tokens] == ["左侧小字", "右侧大字", "下一行"]


def test_map_does_not_split_a_line_at_an_absolute_bucket_boundary() -> None:
    result = map_paddle_result(
        [
            _row(
                ["右", "左", "下"],
                [0.95, 0.94, 0.93],
                [
                    _box(60, 9.8, 90, 19.8),
                    _box(10, 10.2, 40, 20.2),
                    _box(10, 30, 40, 40),
                ],
            )
        ]
    )

    assert [token.text for token in result.tokens] == ["左", "右", "下"]


@pytest.mark.parametrize(
    "row",
    [
        _row(["a", "b"], [0.9], [_box(0, 0, 1, 1), _box(2, 0, 3, 1)]),
        _row(["a"], [0.9, 0.8], [_box(0, 0, 1, 1)]),
        _row(["a"], [0.9], []),
    ],
)
def test_map_rejects_unequal_result_lengths(row: dict[str, object]) -> None:
    with pytest.raises(PaddleResultError, match="invalid PaddleOCR result"):
        map_paddle_result([row])


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0], [1, 0], [1, 1]],
        [[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]],
        [0, 0, 1, 1],
    ],
)
def test_map_requires_exactly_four_polygon_points(polygon: object) -> None:
    with pytest.raises(PaddleResultError, match="invalid PaddleOCR result"):
        map_paddle_result([_row(["private text"], [0.9], [polygon])])


def test_constructor_does_not_import_or_validate_paddle(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__
    imported: list[str] = []

    def tracking_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "paddleocr":
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", tracking_import)

    PaddleOcrEngine(Path("missing-det"), Path("missing-rec"), Path("missing-orientation"))

    assert imported == []


def test_get_model_is_thread_safe_local_only_singleton(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    det, rec, orientation = _model_dirs(tmp_path)
    model = SimpleNamespace(predict=lambda **kwargs: [])
    factory = _Paddle301Factory(model)
    _install_paddle(monkeypatch, factory, "3.0.1")
    engine = PaddleOcrEngine(det, rec, orientation)

    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: engine._get_model(), range(32)))

    assert instances == [model] * 32
    assert factory.calls == [
        {
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_detection_model_dir": str(det.resolve()),
            "text_recognition_model_dir": str(rec.resolve()),
            "textline_orientation_model_dir": str(orientation.resolve()),
        }
    ]


def test_get_model_uses_paddle_300_orientation_keyword(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    det, rec, orientation = _model_dirs(tmp_path)
    model = SimpleNamespace(predict=lambda **kwargs: [])
    factory = _Paddle300Factory(model)
    _install_paddle(monkeypatch, factory, "3.0.0")

    loaded = PaddleOcrEngine(det, rec, orientation)._get_model()

    assert loaded is model
    assert factory.calls == [
        {
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_detection_model_dir": str(det.resolve()),
            "text_recognition_model_dir": str(rec.resolve()),
            "text_line_orientation_model_dir": str(orientation.resolve()),
        }
    ]


def test_missing_local_model_fails_before_paddle_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    imported: list[str] = []
    real_import = __import__

    def tracking_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "paddleocr":
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", tracking_import)
    engine = PaddleOcrEngine(
        tmp_path / "missing-det", tmp_path / "missing-rec", tmp_path / "missing-orientation"
    )

    with pytest.raises(PaddleModelError, match="local PaddleOCR models are unavailable"):
        engine._get_model()

    assert imported == []


def test_recognize_maps_predict_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    det, rec, orientation = _model_dirs(tmp_path)
    processed = tmp_path / "processed.png"
    processed.write_bytes(b"synthetic")
    predict_calls: list[dict[str, object]] = []

    class FakeModel:
        def predict(self, **kwargs: object) -> list[dict[str, object]]:
            predict_calls.append(kwargs)
            return [_row(["金额", "10.00"], [0.99, 0.98], [_box(0, 0, 20, 10), _box(30, 0, 50, 10)])]

    factory = _Paddle301Factory(FakeModel())
    _install_paddle(monkeypatch, factory, "3.0.1")

    result = PaddleOcrEngine(det, rec, orientation).recognize(
        ImageArtifact(original_path=tmp_path / "original.png", processed_path=processed)
    )

    assert predict_calls == [{"input": str(processed)}]
    assert [token.text for token in result.tokens] == ["金额", "10.00"]
    assert result.full_text == "金额\n10.00"


def test_recognize_raises_typed_sanitized_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    det, rec, orientation = _model_dirs(tmp_path)
    processed = tmp_path / "patient-secret.png"
    processed.write_bytes(b"private-image-content")

    class FailingModel:
        def predict(self, **kwargs: object) -> object:
            raise RuntimeError("patient-secret private-image-content")

    factory = _Paddle301Factory(FailingModel())
    _install_paddle(monkeypatch, factory, "3.0.1")

    with pytest.raises(PaddleOcrError, match="PaddleOCR recognition failed") as captured:
        PaddleOcrEngine(det, rec, orientation).recognize(
            ImageArtifact(original_path=processed, processed_path=processed)
        )

    assert "patient-secret" not in str(captured.value)
    assert "private-image" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


class _ExplodingResultIteration:
    def __iter__(self) -> Iterator[object]:
        raise RuntimeError("patient-secret iteration")


class _ExplodingJson:
    @property
    def json(self) -> object:
        raise RuntimeError("patient-secret json")


class _ExplodingRes:
    @property
    def res(self) -> object:
        raise RuntimeError("patient-secret res")


class _ExplodingToDict:
    def to_dict(self) -> object:
        raise RuntimeError("patient-secret to_dict")


class _ExplodingItemsMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[str, object]:
        raise RuntimeError("patient-secret mapping items")


class _JsonWithExplodingMapping:
    @property
    def json(self) -> Mapping[str, object]:
        return _ExplodingItemsMapping()


@pytest.mark.parametrize(
    "raw_result",
    [
        _ExplodingResultIteration(),
        [_ExplodingJson()],
        [_ExplodingRes()],
        [_ExplodingToDict()],
        [_JsonWithExplodingMapping()],
    ],
    ids=["iteration", "json", "res", "to-dict", "mapping-items"],
)
def test_recognize_sanitizes_all_result_conversion_failures(
    raw_result: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    det, rec, orientation = _model_dirs(tmp_path)
    processed = tmp_path / "patient-secret.png"
    processed.write_bytes(b"synthetic")

    class FixedResultModel:
        def predict(self, **kwargs: object) -> object:
            assert kwargs == {"input": str(processed)}
            return raw_result

    factory = _Paddle301Factory(FixedResultModel())
    _install_paddle(monkeypatch, factory, "3.0.1")

    with pytest.raises(PaddleResultError, match="invalid PaddleOCR result") as captured:
        PaddleOcrEngine(det, rec, orientation).recognize(
            ImageArtifact(original_path=processed, processed_path=processed)
        )

    assert "patient-secret" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


class _ExplodingGetMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError(f"patient-secret mapping get: {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def test_map_sanitizes_mapping_failures_without_exception_chain() -> None:
    row = cast(dict[str, object], _ExplodingGetMapping())

    with pytest.raises(PaddleResultError, match="invalid PaddleOCR result") as captured:
        map_paddle_result([row])

    assert "patient-secret" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
