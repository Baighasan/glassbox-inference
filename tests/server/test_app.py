from fastapi import APIRouter, FastAPI

from glassbox.config import Settings
from glassbox.server.app import app, create_app


def test_module_exposes_fastapi_app() -> None:
    assert isinstance(app, FastAPI)
    assert app.title == "Glassbox"


def test_create_app_includes_supplied_router() -> None:
    router = APIRouter()
    settings = Settings(model="test-model", device="cpu", max_tokens=32)

    @router.get("/example")
    def example() -> dict[str, str]:
        return {"status": "ok"}

    created_app = create_app(router, settings)

    assert any(route.path == "/example" for route in created_app.routes)
    assert created_app.state.settings is settings
