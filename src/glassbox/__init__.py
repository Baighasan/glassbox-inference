import os
from argparse import ArgumentParser
from collections.abc import Sequence
from copy import deepcopy

import uvicorn
from uvicorn.config import LOGGING_CONFIG

from glassbox.config import load_settings


def main(argv: Sequence[str] | None = None) -> None:
    """Start the Glassbox API server."""
    parser = ArgumentParser(description="Start the Glassbox API server.")
    parser.add_argument("--reload", action="store_true", help="Reload on changes in src/.")
    args = parser.parse_args(argv)
    settings = load_settings()

    log_config = deepcopy(LOGGING_CONFIG)
    log_config["loggers"]["glassbox"] = {
        "handlers": ["default"],
        "level": settings.logging_level,
        "propagate": False,
    }
    uvicorn.run(
        "glassbox.server.app:app",
        host="127.0.0.1",
        port=8000,
        reload=args.reload,
        reload_dirs=["src"] if args.reload else None,
        log_config=log_config,
        log_level=os.getenv("UVICORN_LOG_LEVEL"),
    )


__all__ = ["main"]
