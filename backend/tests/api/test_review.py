from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from claim_ai.api.app import create_app

CASE_ID = "synthetic-medical-receipt-001"


def client() -> TestClient:
    return TestClient(create_app())


def case_detail(api_client: TestClient) -> dict[str, Any]:
    response = api_client.get(f"/api/v1/review/cases/{CASE_ID}")

    assert response.status_code == 200
    return cast(dict[str, Any], response.json())


def test_lists_synthetic_review_case_in_stable_summary_order() -> None:
    response = client().get("/api/v1/review/cases")

    assert response.status_code == 200
    assert response.json() == [
        {
            "case_id": CASE_ID,
            "version": 1,
            "status": "PENDING_REVIEW",
            "document_type": "medical_receipt",
            "document_label": "Synthetic medical receipt",
            "rule_status": "NEEDS_REVIEW",
        }
    ]


def test_returns_complete_review_case_with_synthetic_evidence_and_string_amounts() -> None:
    detail = case_detail(client())

    assert detail["case_id"] == CASE_ID
    assert detail["version"] == 1
    assert detail["status"] == "PENDING_REVIEW"
    assert detail["document"]["metadata"] == {
        "label": "Synthetic medical receipt",
        "source": "synthetic-demo",
        "page_count": 1,
    }
    assert detail["fields"]["personal_self_pay"]["value"] == "20.00"
    assert detail["fields"]["personal_self_pay"]["evidence"][0]["source_image"] == "synthetic-demo"
    assert detail["rule_decision"]["status"] == "NEEDS_REVIEW"
    assert detail["audit_events"] == []


def test_correcting_field_recomputes_rules_and_redacts_sensitive_reason_from_audit() -> None:
    api_client = client()
    sensitive_reason = "FIELD_REASON_SECRET_Alice_Clinic_buyer_seller"
    response = api_client.patch(
        f"/api/v1/review/cases/{CASE_ID}/fields/personal_self_pay",
        json={"value": "30.00", "expected_version": 1, "reason": sensitive_reason},
    )

    assert response.status_code == 200
    detail = response.json()
    assert detail["version"] == 2
    assert detail["fields"]["personal_self_pay"]["value"] == "30.00"
    assert detail["rule_decision"]["status"] == "CALCULATED"
    assert detail["rule_decision"]["selected_rule"] == "R001"
    assert detail["rule_decision"]["target_amount"] == "100.00"
    audit_text = response.text
    assert sensitive_reason not in audit_text
    for sensitive_text in ["Alice", "Clinic", "buyer", "seller"]:
        assert sensitive_text not in audit_text
    audit_event = detail["audit_events"][0]
    assert audit_event["event_type"] == "FIELD_CORRECTED"
    assert audit_event["field_name"] == "personal_self_pay"
    assert audit_event["version"] == 2
    assert isinstance(audit_event["timestamp"], str)
    assert set(audit_event) == {"event_type", "field_name", "version", "timestamp"}
    audit_json = json.dumps(detail["audit_events"])
    assert "20.00" not in audit_json
    assert "30.00" not in audit_json


def test_rejects_stale_field_correction_with_version_conflict() -> None:
    api_client = client()
    first = api_client.patch(
        f"/api/v1/review/cases/{CASE_ID}/fields/personal_self_pay",
        json={"value": "30.00", "expected_version": 1, "reason": "synthetic correction"},
    )
    stale = api_client.patch(
        f"/api/v1/review/cases/{CASE_ID}/fields/personal_self_pay",
        json={"value": "31.00", "expected_version": 1, "reason": "stale correction"},
    )

    assert first.status_code == 200
    assert stale.status_code == 409


@pytest.mark.parametrize("decision", ["APPROVE", "REJECT"])
def test_records_terminal_decision_without_echoing_sensitive_comment(decision: str) -> None:
    sensitive_comment = "DECISION_COMMENT_SECRET_Alice_Clinic_buyer_seller"
    response = client().post(
        f"/api/v1/review/cases/{CASE_ID}/decision",
        json={"decision": decision, "expected_version": 1, "comment": sensitive_comment},
    )

    assert response.status_code == 200
    detail = response.json()
    assert detail["version"] == 2
    assert detail["status"] == decision
    assert sensitive_comment not in response.text
    for sensitive_text in ["Alice", "Clinic", "buyer", "seller"]:
        assert sensitive_text not in response.text
    audit_event = detail["audit_events"][0]
    assert audit_event["event_type"] == "DECISION_RECORDED"
    assert audit_event["decision"] == decision
    assert audit_event["version"] == 2
    assert isinstance(audit_event["timestamp"], str)
    assert set(audit_event) == {"event_type", "decision", "version", "timestamp"}


def test_terminal_case_rejects_further_field_changes_and_decisions() -> None:
    api_client = client()
    decision = api_client.post(
        f"/api/v1/review/cases/{CASE_ID}/decision",
        json={"decision": "APPROVE", "expected_version": 1, "comment": "synthetic disposition"},
    )
    field_change = api_client.patch(
        f"/api/v1/review/cases/{CASE_ID}/fields/personal_self_pay",
        json={"value": "30.00", "expected_version": 2, "reason": "too late"},
    )
    repeated_decision = api_client.post(
        f"/api/v1/review/cases/{CASE_ID}/decision",
        json={"decision": "REJECT", "expected_version": 2, "comment": "too late"},
    )

    assert decision.status_code == 200
    assert field_change.status_code == 409
    assert repeated_decision.status_code == 409


def test_rejects_unknown_cases_and_invalid_payloads() -> None:
    api_client = client()
    missing = api_client.get("/api/v1/review/cases/missing")
    invalid = api_client.post(
        f"/api/v1/review/cases/{CASE_ID}/decision",
        json={"decision": "MAYBE", "expected_version": 1, "comment": "invalid"},
    )

    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_apps_are_isolated_and_repository_returns_deep_copies() -> None:
    first_client = client()
    first_update = first_client.patch(
        f"/api/v1/review/cases/{CASE_ID}/fields/personal_self_pay",
        json={"value": "30.00", "expected_version": 1, "reason": "isolated edit"},
    )
    second_detail = case_detail(client())

    assert first_update.status_code == 200
    assert second_detail["version"] == 1
    assert second_detail["fields"]["personal_self_pay"]["value"] == "20.00"

    try:
        repository_module = import_module("claim_ai.demo.repository")
    except ModuleNotFoundError as error:
        pytest.fail(f"review repository is required: {error}")
    repository = repository_module.ReviewCaseRepository.from_default_fixture()
    stored = repository.get(CASE_ID)
    assert stored is not None
    stored["fields"]["personal_self_pay"]["value"] = "MUTATED_OUTSIDE_REPOSITORY"
    assert repository.get(CASE_ID)["fields"]["personal_self_pay"]["value"] == "20.00"


def test_concurrent_field_corrections_allow_only_one_matching_version() -> None:
    api_client = client()

    def patch(value: str) -> int:
        response = api_client.patch(
            f"/api/v1/review/cases/{CASE_ID}/fields/personal_self_pay",
            json={"value": value, "expected_version": 1, "reason": "concurrent synthetic correction"},
        )
        return cast(int, response.status_code)

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(patch, ["30.00", "31.00"]))

    assert sorted(statuses) == [200, 409]
