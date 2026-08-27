from claim_ai.pipeline.models import ImageArtifact, OcrResult, OcrToken


class FakeOcrEngine:
    def __init__(self, result: OcrResult) -> None:
        self.result = result

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, float]]) -> "FakeOcrEngine":
        tokens = [
            OcrToken(
                text=text,
                confidence=confidence,
                bbox=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            )
            for text, confidence in pairs
        ]
        return cls(OcrResult(tokens=tokens, full_text="\n".join(token.text for token in tokens)))

    def recognize(self, image: ImageArtifact) -> OcrResult:
        return self.result
