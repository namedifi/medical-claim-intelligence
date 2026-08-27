#!/usr/bin/env python3
"""Fail closed when files unsafe for the public demo repository are tracked."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
MODEL_SUFFIXES = {
    ".onnx",
    ".pt",
    ".pth",
    ".pdmodel",
    ".pdiparams",
    ".safetensors",
    ".ckpt",
    ".gguf",
    ".bin",
}
RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GENERATED_OR_PRIVATE_PARTS = {
    ".superpowers",
    ".worktrees",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "dist",
}
CONTENT_SCAN_EXEMPT = {
    # Deliberately contains synthetic path strings to verify exception redaction.
    "backend/tests/pipeline/test_service.py",
    "scripts/check_public_repo.py",
    "scripts/tests/test_check_public_repo.py",
}
LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    re.compile(r"/(?:home|Users)/[^\s'\"`]+"),
    re.compile(r"xwechat_files|RWTemp", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"
    ),
)
SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)"
    r"\s*[:=]\s*['\"]?([^\s'\"#]+)"
)
PII_PATTERNS = (
    re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
)
SAFE_PLACEHOLDER_MARKERS = ("example", "placeholder", "changeme", "your-", "${", "<")


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    reason: str


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _path_violations(relative: str, path: Path) -> list[Violation]:
    violations: list[Violation] = []
    lower = relative.lower()
    parts = {part.lower() for part in Path(relative).parts}

    if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
        violations.append(Violation(relative, "文件超过公开仓库 5 MiB 阈值"))
    if path.suffix.lower() in MODEL_SUFFIXES:
        violations.append(Violation(relative, "检测到模型权重或推理产物"))
    if lower.startswith("models/") and lower != "models/readme.md":
        violations.append(Violation(relative, "models 目录只允许 README.md"))
    if path.suffix.lower() in RASTER_SUFFIXES and not lower.startswith("samples/synthetic/"):
        violations.append(Violation(relative, "真实票据风险：栅格图像不在合成样本目录"))
    if parts & GENERATED_OR_PRIVATE_PARTS:
        violations.append(Violation(relative, "检测到内部、缓存或构建输出目录"))
    if lower.startswith("data/private/"):
        violations.append(Violation(relative, "检测到私有数据目录"))
    if path.name.lower() == ".env" or (
        path.name.lower().startswith(".env.") and path.name.lower() != ".env.example"
    ):
        violations.append(Violation(relative, "检测到不应跟踪的环境变量文件"))
    return violations


def _content_violations(relative: str, path: Path) -> list[Violation]:
    if relative in CONTENT_SCAN_EXEMPT:
        return []
    if path.suffix.lower() in RASTER_SUFFIXES and relative.lower().startswith(
        "samples/synthetic/"
    ):
        return []

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return [Violation(relative, "无法按 UTF-8 解码的二进制文件")]

    violations: list[Violation] = []
    if any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS):
        violations.append(Violation(relative, "检测到本机绝对路径或聊天临时目录"))
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        violations.append(Violation(relative, "检测到疑似密钥或访问令牌"))
    for match in SECRET_ASSIGNMENT.finditer(text):
        value = match.group(1).lower()
        if value and not any(marker in value for marker in SAFE_PLACEHOLDER_MARKERS):
            violations.append(Violation(relative, "检测到疑似明文凭证赋值"))
            break
    if any(pattern.search(text) for pattern in PII_PATTERNS):
        violations.append(Violation(relative, "检测到身份证或手机号形态的个人信息"))
    return violations


def scan_files(root: Path, paths: Iterable[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        relative = _relative(path, root)
        violations.extend(_path_violations(relative, path))
        violations.extend(_content_violations(relative, path))
    return sorted(set(violations))


def scan_tree(root: Path) -> list[Violation]:
    root = root.resolve()
    paths = (path for path in root.rglob("*") if ".git" not in path.parts)
    return scan_files(root, paths)


def scan_tracked(root: Path) -> list[Violation]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    relative_paths = completed.stdout.decode("utf-8").split("\0")
    return scan_files(root.resolve(), (root.resolve() / item for item in relative_paths if item))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="递归扫描指定目录；默认只扫描当前 Git 仓库已跟踪文件",
    )
    args = parser.parse_args(argv)

    if args.root is None:
        repository_root = Path(__file__).resolve().parents[1]
        violations = scan_tracked(repository_root)
    else:
        violations = scan_tree(args.root)

    if violations:
        print("公开仓库安全检查失败：", file=sys.stderr)
        for violation in violations:
            print(f"- {violation.path}: {violation.reason}", file=sys.stderr)
        return 1

    print("公开仓库安全检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
