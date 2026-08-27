from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_public_repo import scan_tree


def write(root: Path, relative: str, content: str | bytes = "safe") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def reasons(root: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for violation in scan_tree(root):
        result.setdefault(violation.path, set()).add(violation.reason)
    return result


def test_clean_public_tree_passes(tmp_path: Path) -> None:
    write(tmp_path, "README.md", "Synthetic demo only; no credentials or private data.")
    write(tmp_path, "models/README.md", "Weights are operator supplied outside Git.")
    write(tmp_path, "samples/synthetic/example.png", b"synthetic-placeholder")

    assert scan_tree(tmp_path) == []


@pytest.mark.parametrize(
    "relative",
    [
        "weights/model.onnx",
        "weights/model.pt",
        "weights/model.pth",
        "weights/model.pdmodel",
        "weights/model.pdiparams",
        "weights/model.safetensors",
        "weights/model.ckpt",
        "weights/model.gguf",
        "weights/model.bin",
        "models/paddleocr/config.yml",
    ],
)
def test_model_artifacts_are_rejected(tmp_path: Path, relative: str) -> None:
    write(tmp_path, relative, b"weights")

    assert relative in reasons(tmp_path)


@pytest.mark.parametrize("extension", ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
def test_raster_images_outside_synthetic_samples_are_rejected(
    tmp_path: Path, extension: str
) -> None:
    relative = f"docs/ticket.{extension}"
    write(tmp_path, relative, b"image")

    assert relative in reasons(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        ".superpowers/report.md",
        ".worktrees/feature/file.txt",
        "data/private/ticket.json",
        "frontend/node_modules/pkg/index.js",
        "frontend/dist/index.html",
        "backend/.pytest_cache/state",
    ],
)
def test_internal_or_generated_paths_are_rejected(tmp_path: Path, relative: str) -> None:
    write(tmp_path, relative)

    assert relative in reasons(tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "source = 'C:" + "\\Users\\Example\\Desktop\\ticket.png'",
        "source = '/home/example/private/ticket.png'",
        "source = 'xwechat_files/temporary/ticket.png'",
    ],
)
def test_private_local_paths_are_rejected(tmp_path: Path, content: str) -> None:
    write(tmp_path, "docs/example.md", content)

    assert "docs/example.md" in reasons(tmp_path)


def test_likely_secret_is_rejected_without_echoing_value(tmp_path: Path) -> None:
    secret = "sk-" + "a" * 40
    write(tmp_path, "config.txt", f"API_TOKEN={secret}")

    violations = scan_tree(tmp_path)

    assert violations
    assert secret not in "\n".join(item.reason for item in violations)


def test_github_token_prefix_is_rejected(tmp_path: Path) -> None:
    token = "ghp" + "_" + "a" * 36
    write(tmp_path, "config.txt", f"token value: {token}")

    assert "config.txt" in reasons(tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        "11010519491231002X",
        "13800138000",
    ],
)
def test_identity_or_phone_shaped_value_is_rejected(tmp_path: Path, value: str) -> None:
    write(tmp_path, "samples/case.json", '{"subject": "' + value + '"}')

    assert "samples/case.json" in reasons(tmp_path)


def test_undecodable_binary_is_not_silently_skipped(tmp_path: Path) -> None:
    write(tmp_path, "assets/blob.bin", b"\xff\xfe\x00\x81")

    assert "assets/blob.bin" in reasons(tmp_path)


def test_oversized_binary_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "assets/archive.bin", b"0" * (5 * 1024 * 1024 + 1))

    assert "assets/archive.bin" in reasons(tmp_path)
