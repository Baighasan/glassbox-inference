from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI

from glassbox.server.routes import router

from glassbox.runtime.model import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize manager
    app.state.model_manager = ModelManager()
    yield
    # Shutdown: Free all models
    await app.state.model_manager.shutdown()


def create_app(
    api_router: APIRouter = router,
) -> FastAPI:
    app = FastAPI(
        title="Glassbox",
        description="A local LLM inference server.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
