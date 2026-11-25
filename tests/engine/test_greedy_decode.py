import asyncio

import pytest
import torch

from glassbox.config import DEFAULT_MODEL, Device
from glassbox.engine.engine import Engine

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8


def _assert_greedy_matches(device: Device) -> None:
    engine = Engine(device=device)
    try:
        result = asyncio.run(
            engine.generate(
                model_name=DEFAULT_MODEL,
                text=PROMPT,
                device=device,
                max_tokens=MAX_NEW_TOKENS,
            )
        )

        model = engine.runtime.model
        tokenizer = engine.tokenizer
        assert model is not None
        assert tokenizer is not None

        prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
        # Cached greedy search is the reference. The engine recomputes the full sequence.
        hf_ids = model.generate(
            prompt_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )
        hf_new_ids = tuple(
            int(token_id) for token_id in hf_ids[0, prompt_ids.shape[1] :].tolist()
        )

        assert result.completion_token_ids == hf_new_ids
    finally:
        engine.shutdown()


@pytest.mark.model
def test_greedy_loop_matches_huggingface_generate() -> None:
    _assert_greedy_matches("cpu")


@pytest.mark.model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_greedy_loop_matches_huggingface_generate_on_cuda() -> None:
    _assert_greedy_matches("cuda")
