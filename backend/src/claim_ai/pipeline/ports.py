from pathlib import Path
from typing import Protocol, runtime_checkable

from claim_ai.pipeline.models import FieldCandidate, ImageArtifact, OcrResult


@runtime_checkable
class ImagePreprocessor(Protocol):
    def process(self, source: Path, destination: Path) -> ImageArtifact:
        raise NotImplementedError


@runtime_checkable
class OcrEngine(Protocol):
    def recognize(self, image: ImageArtifact) -> OcrResult:
        raise NotImplementedError


@runtime_checkable
class VisionLanguageExtractor(Protocol):
    def extract(self, image: ImageArtifact, ocr: OcrResult) -> list[FieldCandidate]:
        raise NotImplementedError
