from fastapi.testclient import TestClient

from claim_ai.api.app import create_app

client = TestClient(create_app())


def test_liveness() -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_calculates_pharmacy_total() -> None:
    response = client.post(
        "/api/v1/precheck/calculate",
        json={
            "document_type": "pharmacy_invoice",
            "fields": {
                "buyer_info": {"value": "合成购买方", "confidence": 0.99},
                "seller_info": {"value": "合成销售方", "confidence": 0.99},
                "tax_inclusive_total": {"value": "28.60", "confidence": 0.99},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "CALCULATED"
    assert response.json()["selected_rule"] == "R006"
    assert response.json()["target_amount"] == "28.60"


def test_rejects_invalid_payload() -> None:
    response = client.post(
        "/api/v1/precheck/calculate",
        json={"fields": {}},
    )

    assert response.status_code == 422


def test_returns_needs_review_for_incomplete_r001_candidate() -> None:
    response = client.post(
        "/api/v1/precheck/calculate",
        json={
            "document_type": "medical_receipt",
            "fields": {
                "policy_scope_amount": {"value": "100.00", "confidence": 0.99},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "NEEDS_REVIEW"
    assert response.json()["selected_rule"] == "R001"
