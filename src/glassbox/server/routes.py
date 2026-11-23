import time
import uuid

from fastapi import APIRouter, HTTPException, Request

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
from glassbox.config import DEFAULT_MODEL, DEFAULT_DEVICE
from glassbox.engine.exceptions import UnsupportedModelError


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
    loaded_model = request.app.state.engine.loaded_model
    return HealthResponse(
        status="alive",
        loaded=loaded_model is not None,
        model=loaded_model or DEFAULT_MODEL,
        device="cpu",
        dtype="float32",
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def models(request: Request) -> ModelListResponse:
    return ModelListResponse(
        data=[
            ModelInfo(
                id=DEFAULT_MODEL,
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

    try:
        completion_result = await request.app.state.engine.generate(
            model_name=payload.model,
            text=payload.prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
    except UnsupportedModelError as e:
        raise HTTPException(status_code=404, detail=f"Model {payload.model} not found") from e

    usage = UsageInfo(
        prompt_tokens=completion_result.prompt_tokens,
        completion_tokens=completion_result.completion_tokens,
        total_tokens=completion_result.prompt_tokens + completion_result.completion_tokens,
    )
    metrics = GlassboxMetrics(
        total_latency_ms=completion_result.latency_ms,
        tokens_per_second=completion_result.tokens_per_second,
        device=DEFAULT_DEVICE,
        dtype="float32",
        gpu_memory_allocated_mb=0,
        gpu_memory_reserved_mb=0,
    )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=payload.model,
        choices=[
            CompletionResponseChoice(
                text=completion_result.generated_text,
                index=0,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        usage=usage,
        glassbox_metrics=metrics,
    )


# This route is to be used for more complex models later on
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse:
    started_at = time.perf_counter()

    generated_text = "This is a simulated mock response"
    prompt = " ".join(message.content for message in payload.messages)
    usage = _mock_usage(prompt, generated_text)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=payload.model,
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
