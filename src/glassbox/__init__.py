import uvicorn


def main() -> None:
    """Start the Glassbox API server."""
    uvicorn.run(
        "glassbox.server.app:app",
        host="127.0.0.1",
        port=8000,
    )


__all__ = ["main"]
