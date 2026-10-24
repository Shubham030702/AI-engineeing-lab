from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate():
    response = client.post(
        "/generate",
        json={"prompt": "What is an LLM?"},
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "mock"
    assert "What is an LLM?" in response.json()["response"]


def test_empty_prompt_is_rejected():
    response = client.post(
        "/generate",
        json={"prompt": ""},
    )

    assert response.status_code == 422


def test_missing_prompt_is_rejected():
    response = client.post("/generate", json={})

    assert response.status_code == 422