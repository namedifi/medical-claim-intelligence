from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any

CasePayload = dict[str, Any]
CaseMutator = Callable[[CasePayload], None]


class ReviewCaseRepository:
    """A lock-protected, process-local store for synthetic demo cases."""

    def __init__(self, cases: list[CasePayload]) -> None:
        self._lock = RLock()
        self._cases = {str(case["case_id"]): copy.deepcopy(case) for case in cases}

    @classmethod
    def from_default_fixture(cls) -> ReviewCaseRepository:
        fixture_path = Path(__file__).resolve().parents[4] / "samples" / "synthetic" / "review_case.json"
        with fixture_path.open(encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        if not isinstance(fixture, dict):
            raise TypeError("Synthetic review fixture must be a JSON object")
        return cls([fixture])

    def get(self, case_id: str) -> CasePayload | None:
        with self._lock:
            case = self._cases.get(case_id)
            return None if case is None else copy.deepcopy(case)

    def list(self) -> list[CasePayload]:
        with self._lock:
            return copy.deepcopy(list(self._cases.values()))

    def update(self, case_id: str, mutator: CaseMutator) -> CasePayload | None:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                return None
            mutator(case)
            return copy.deepcopy(case)
