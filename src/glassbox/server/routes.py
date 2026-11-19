import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request

from glassbox.runtime.runtime import generate
from glassbox.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    GlassboxMetrics,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)


router = APIRouter()


def _mock_usage(prompt: str, completion: str) -> UsageInfo:
    prompt_tokens = len(prompt.split())
    completion_tokens = len(completion.split())
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _mock_metrics(
    started_at: float,
    usage: UsageInfo,
) -> GlassboxMetrics:
    latency_ms = (time.perf_counter() - started_at) * 1000
    elapsed_seconds = max(latency_ms / 1000, 1e-9)
    return GlassboxMetrics(
        total_latency_ms=latency_ms,
        tokens_per_second=usage.completion_tokens / elapsed_seconds,
        device="cpu",
        dtype="float32",
        gpu_memory_allocated_mb=0,
        gpu_memory_reserved_mb=0,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="alive",
        loaded=False if request.app.state.model_manager.model is None else True,
        model=request.app.state.model_manager.model_name,
        device="cpu",
        dtype="float32",
        model_load_seconds=request.app.state.model_manager.model_load_seconds,
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def models(request: Request) -> ModelListResponse:
    return ModelListResponse(
        data=[
            ModelInfo(
                id="openai-community/gpt2",
                created=0,
            )
        ]
    )


# Main completions api used currently for gpt2, llmama, and other simpler completion models
@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    payload: CompletionRequest,
    request: Request,
) -> CompletionResponse:
    started_at = time.perf_counter()

    generated_text = await generate(
        model_manager=request.app.state.model_manager,
        model_name=payload.model,
        text=payload.prompt,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
    )
    
    usage = _mock_usage(payload.prompt, generated_text)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=payload.model,
        choices=[
            CompletionResponseChoice(
                text=generated_text,
                index=0,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        usage=usage,
        glassbox_metrics=_mock_metrics(started_at, usage),
    )


# This route is to be used for more complex models later on
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse:
    started_at = time.perf_counter()

    generated_text = "This is a simulated mock response"
    prompt = " ".join(message.content for message in payload.messages)
    usage = _mock_usage(prompt, generated_text)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model="openai-community/gpt2",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage=usage,
        glassbox_metrics=_mock_metrics(started_at, usage),
    )
