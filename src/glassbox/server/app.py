from fastapi import APIRouter, FastAPI

from glassbox.config import Settings, load_settings
from glassbox.server.routes import router


def create_app(
    api_router: APIRouter = router,
    settings: Settings | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Glassbox",
        description="A local LLM inference server.",
        version="0.1.0",
    )
    app.state.settings = settings or load_settings()
    app.include_router(api_router)
    return app


app = create_app()
