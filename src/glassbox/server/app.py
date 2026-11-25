import logging

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from glassbox.config import load_settings
from glassbox.engine.engine import Engine
from glassbox.server.routes import router


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    app.state.engine = Engine(device=settings.device)
    yield
    app.state.engine.shutdown()


def create_app(
    api_router: APIRouter = router,
) -> FastAPI:
    logger.debug("Debug mode enabled")
    
    app = FastAPI(
        title="Glassbox",
        description="A local LLM inference server.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app
    
    
app = create_app()