import asyncio
import gc
from fastapi import HTTPException
import torch
from transformers import pipeline, Pipeline

from glassbox.config import DEFAULT_DEVICE, DEFAULT_MODEL


class ModelManager:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.lock = asyncio.Lock()
        self.model: Pipeline = pipeline(
            task="text-generation",
            model=model_name,
            device_map=DEFAULT_DEVICE,
        )

    async def get_model(self, task: str = "text-generation", model_name: str = DEFAULT_MODEL) -> Pipeline:
        """Return the model loaded at startup."""
        if model_name != self.model_name:
            raise HTTPException(
                status_code=404,
                detail=f"Only {self.model_name} is supported at the moment.",
            )
        return self.model

    def _unload_model(self, pipe: Pipeline):
        """ Explicitly free CPU and GPU memory. """
        if self.lock.locked():
            raise RuntimeError("Cannot unload model while lock is locked.")
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def shutdown(self):
        """ Clean up all models when the API server stops. """
        async with self.lock:
            while self.model:
                self._unload_model(self.model)
                self.model = None
