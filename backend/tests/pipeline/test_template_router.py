from pathlib import Path

from claim_ai.domain.models import DocumentType
from claim_ai.pipeline.template_router import TemplateRouter

TEMPLATES = Path("configs/templates/v1.json")


def test_routes_henan_receipt() -> None:
    router = TemplateRouter.load(TEMPLATES)

    document_type, template_id = router.route("河南省 医疗收费票据 社保卡号")

    assert document_type is DocumentType.MEDICAL_RECEIPT
    assert template_id == "henan_v1"


def test_unknown_is_not_forced_to_a_template() -> None:
    router = TemplateRouter.load(TEMPLATES)

    assert router.route("未知地区普通图片") == (DocumentType.UNKNOWN, None)


def test_empty_ocr_is_unknown() -> None:
    router = TemplateRouter.load(TEMPLATES)

    assert router.route("") == (DocumentType.UNKNOWN, None)


def test_excluded_keywords_prevent_a_route() -> None:
    router = TemplateRouter.load(TEMPLATES)

    assert router.route("河南省医疗收费票据 河北省") == (DocumentType.UNKNOWN, None)


def test_tied_high_scores_are_unknown() -> None:
    router = TemplateRouter.load(TEMPLATES)

    assert router.route("北京市医疗收费票据 河北省医疗收费票据") == (
        DocumentType.UNKNOWN,
        None,
    )
