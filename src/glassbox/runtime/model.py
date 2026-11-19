import asyncio
import gc
from fastapi import HTTPException
import torch
from transformers import pipeline, Pipeline

from glassbox.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


class ModelManager:
    def __init__(self, model_name: str = "openai-community/gpt2"):
        self.model_name = model_name
        self.model: Pipeline = None
        self.lock = asyncio.Lock()

    async def get_model(self, task: str = "text-generation", model_name: str = "openai-community/gpt2") -> Pipeline:
        """ Get the model by name. """

        # TODO: Remove this once we support more models
        if model_name != "openai-community/gpt2":
            raise HTTPException(status_code=404, detail="Only openai-community/gpt2 is supported at the moment.")

        async with self.lock:
            # If the model is already loaded, return it
            if self.model and self.model_name == model_name:
                return self.model

            # Load the new model in a background thread so the event loop stays responsive
            try:
                new_pipe = await asyncio.to_thread(
                    pipeline,
                    task=task,
                    model=model_name,
                    device_map="cpu"
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load model '{model_name}': {str(e)}")

            # Unload the old model if it exists
            if self.model:
                self._unload_model(self.model)

            self.model = new_pipe
            self.model_name = model_name
            return new_pipe

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
