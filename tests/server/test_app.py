from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute

from glassbox.server.app import app, create_app


def test_module_exposes_fastapi_app() -> None:
    assert isinstance(app, FastAPI)
    assert app.title == "Glassbox"


def test_create_app_includes_supplied_router() -> None:
    router = APIRouter()

    @router.get("/example")
    def example() -> dict[str, str]:
        return {"status": "ok"}

    created_app = create_app(router)

    assert any(
        isinstance(route, APIRoute) and route.path == "/example"
        for route in created_app.routes
    )
