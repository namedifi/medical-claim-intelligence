import json
from pathlib import Path

import pytest

from claim_ai.config.field_catalog import FieldCatalog, FieldCatalogError

CATALOG = Path("configs/fields/v1.json")


def test_resolves_known_aliases() -> None:
    catalog = FieldCatalog.load(CATALOG)
    assert catalog.resolve("医保统筹基金支付") == "pooled_fund_payment"
    assert catalog.resolve("乙类先自付") == "class_b_pre_self_pay"


def test_aliases_are_unique() -> None:
    catalog = FieldCatalog.load(CATALOG)
    assert len(catalog.alias_to_canonical) == len(set(catalog.alias_to_canonical))


def test_duplicate_alias_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fields.json"
    path.write_text(
        '{"version":"1","fields":[{"key":"a","aliases":["同名"]},{"key":"b","aliases":["同名"]}]}',
        encoding="utf-8",
    )
    with pytest.raises(FieldCatalogError):
        FieldCatalog.load(path)


def test_authorized_candidate_library_is_complete() -> None:
    candidates = json.loads(
        Path("configs/fields/candidates-zh-v1.json").read_text(encoding="utf-8")
    )
    assert len(candidates) == 119
    assert len(candidates) == len(set(candidates))
