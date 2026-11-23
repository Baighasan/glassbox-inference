class UnsupportedModelError(Exception):
    """Raised when a request names a model the engine does not support."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model {model_name} is not supported")
