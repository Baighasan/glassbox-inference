from conftest import ASGITestClient


def test_health_reports_configured_runtime(client: ASGITestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "alive",
        "loaded": False,
        "model": "test-model",
        "device": "cpu",
        "dtype": "float32",
        "model_load_seconds": None,
    }


def test_models_lists_only_the_configured_model(
    client: ASGITestClient,
) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "test-model",
                "object": "model",
                "created": 0,
                "owned_by": "glassbox",
            }
        ],
    }
