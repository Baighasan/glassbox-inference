from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI

from glassbox.server.routes import router


"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize model
    app.state.model = await get_model()
    yield
    # Shutdown: Free all models
    unload_model(app.state.model)
    """


def create_app(
    api_router: APIRouter = router,
) -> FastAPI:
    app = FastAPI(
        title="Glassbox",
        description="A local LLM inference server.",
        version="0.1.0",
        #lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
