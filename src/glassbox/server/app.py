from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from glassbox.engine.engine import Engine
from glassbox.server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = Engine()
    yield
    app.state.engine.shutdown()


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
