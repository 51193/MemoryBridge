"""API routes for MemoryBridge."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Literal, cast

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.context import ContextBuilder
from ..core.memory import MemoryManager
from ..core.session import SessionNotFoundError, SessionStore
from ..exceptions import MemorySearchError, MemoryStoreError, ProviderNotFoundError
from ..logfmt import structured_debug, structured_info
from ..models.request import ChatRequest, Message, SessionCreateRequest
from ..models.response import ChatResponse, SessionCreateResponse, SessionExportResponse
from ..providers.base import AbstractLLMProvider
from ..providers.registry import ProviderRegistry
from .dependencies import get_context_builder, get_memory_manager, get_session_store

logger: logging.Logger = logging.getLogger(__name__)
router: APIRouter = APIRouter()


def _messages_as_dicts(messages: list[Message]) -> list[dict[str, object]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _dicts_as_messages(dicts: list[dict[str, object]]) -> list[Message]:
    return [
        Message(
            role=cast(
                Literal["system", "user", "assistant", "tool"],
                str(m["role"]),
            ),
            content=str(m["content"]),
        )
        for m in dicts
    ]


# ── Health ──────────────────────────────────────────────────────────────


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    qdrant_health_url: str = request.app.state.qdrant_health_url
    try:
        async with httpx.AsyncClient() as client:
            r: httpx.Response = await client.get(qdrant_health_url, timeout=2.0)
        qdrant_status: str = "connected" if r.status_code == 200 else "disconnected"
    except Exception:
        qdrant_status = "disconnected"
    return {"status": "ok", "qdrant": qdrant_status}


# ── Sessions ────────────────────────────────────────────────────────────


@router.post(
    "/v1/sessions",
    status_code=201,
    response_model=SessionCreateResponse,
)
def create_session(
    req: SessionCreateRequest,
    session_store: SessionStore = Depends(get_session_store),
) -> dict[str, object]:
    session_id: str = req.agent_session_id or uuid.uuid4().hex[:12]

    from ..core.session import SessionExistsError

    try:
        msg_list: list[dict[str, object]] = (
            _messages_as_dicts(req.initial_messages)
            if req.initial_messages
            else []
        )
        session_store.create(
            req.agent_id, session_id, msg_list if msg_list else None
        )
    except SessionExistsError as e:
        structured_info(
            logger,
            "→ POST /v1/sessions → 409",
            agent_id=req.agent_id,
            session_id=session_id,
        )
        raise HTTPException(status_code=409, detail=str(e))

    stored_count: int = len(session_store.get(req.agent_id, session_id))
    structured_info(
        logger,
        "→ POST /v1/sessions → 201",
        agent_id=req.agent_id,
        session_id=session_id,
        initial_messages=stored_count,
    )
    return {
        "agent_id": req.agent_id,
        "agent_session_id": session_id,
        "message_count": stored_count,
    }


@router.get(
    "/v1/sessions/{agent_id}/{session_id}",
    response_model=SessionExportResponse,
)
def get_session(
    agent_id: str,
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> dict[str, object]:
    history: list[dict[str, object]] = _resolve_session(
        session_store, agent_id, session_id
    )
    return {
        "agent_id": agent_id,
        "agent_session_id": session_id,
        "messages": _dicts_as_messages(history),
    }


# ── Chat completions: decomposed helpers ─────────────────────────────────


def _resolve_provider(model: str) -> AbstractLLMProvider:
    try:
        return ProviderRegistry.get(model)
    except ProviderNotFoundError as e:
        logger.warning("provider not found model=%s", model)
        raise HTTPException(status_code=502, detail=str(e))


def _resolve_session(
    session_store: SessionStore, agent_id: str, session_id: str
) -> list[dict[str, object]]:
    try:
        return session_store.get(agent_id, session_id)
    except SessionNotFoundError as e:
        logger.info(
            "session not found → 404 agent=%s session=%s", agent_id, session_id
        )
        raise HTTPException(status_code=404, detail=str(e))


def _build_enriched_request(
    request: ChatRequest,
    session_history: list[dict[str, object]],
    memory_manager: MemoryManager,
    context_builder: ContextBuilder,
) -> tuple[ChatRequest, int]:
    message_dicts: list[dict[str, object]] = _messages_as_dicts(request.messages)
    memories: list[dict[str, object]] = []

    if request.memory_enabled:
        try:
            memories = memory_manager.search(
                str(request.messages[-1].content),
                user_id=request.agent_id,
                limit=request.memory_limit,
            )
        except MemorySearchError as e:
            logger.error("memory search failed → 500: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    enriched: list[dict[str, object]] = context_builder.build(
        message_dicts, memories
    )
    final_messages: list[dict[str, object]] = _inject_history(
        enriched, session_history
    )

    structured_debug(
        logger,
        "context assembled",
        memories_injected=len(memories),
        history_injected=len(session_history),
        final_messages=len(final_messages),
    )

    enriched_request: ChatRequest = request.model_copy(
        update={"messages": _dicts_as_messages(final_messages)}
    )
    return enriched_request, len(memories)


def _inject_history(
    enriched: list[dict[str, object]],
    history: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not history:
        return list(enriched)

    result: list[dict[str, object]] = []

    for i, msg in enumerate(enriched):
        if msg.get("role") == "system":
            result.append(msg)
            result.extend(history)
            result.extend(enriched[i + 1 :])
            return result

    result.extend(history)
    result.extend(enriched)
    return result


def _schedule_memory_store(
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager,
    session_store: SessionStore,
    request: ChatRequest,
    response_content: str,
) -> None:
    background_tasks.add_task(
        _store_memory,
        memory_manager=memory_manager,
        session_store=session_store,
        request=request,
        response_content=response_content,
    )
    structured_debug(logger, "background memory store scheduled")


# ── Chat Completions (handler) ──────────────────────────────────────────


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatRequest,
    fastapi_request: Request,
    background_tasks: BackgroundTasks,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    session_store: SessionStore = Depends(get_session_store),
    context_builder: ContextBuilder = Depends(get_context_builder),
) -> ChatResponse | StreamingResponse:
    t0: float = time.monotonic()

    structured_info(
        logger,
        "→ POST /v1/chat/completions",
        agent_id=request.agent_id,
        session_id=request.agent_session_id,
        model=request.model,
        stream=str(request.stream),
        memory=str(request.memory_enabled),
    )

    provider: AbstractLLMProvider = _resolve_provider(request.model)
    session_history: list[dict[str, object]] = _resolve_session(
        session_store, request.agent_id, request.agent_session_id
    )

    enriched_request: ChatRequest
    enriched_request, _ = _build_enriched_request(
        request, session_history, memory_manager, context_builder
    )

    if request.stream:
        return _handle_stream(
            enriched_request, provider, request, memory_manager, session_store, t0
        )

    try:
        response: ChatResponse = await provider.chat(enriched_request)
    except Exception as e:
        logger.error("provider call failed → 502: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    _schedule_memory_store(
        background_tasks, memory_manager, session_store, request,
        response.choices[0].message.content,
    )

    elapsed: float = (time.monotonic() - t0) * 1000
    structured_info(
        logger,
        "← 200",
        agent_id=request.agent_id,
        session_id=request.agent_session_id,
        latency_ms=f"{elapsed:.0f}",
    )
    return response


# ── Streaming handler ───────────────────────────────────────────────────


def _handle_stream(
    enriched_request: ChatRequest,
    provider: AbstractLLMProvider,
    original_request: ChatRequest,
    memory_manager: MemoryManager,
    session_store: SessionStore,
    t0: float,
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        collected_content: str = ""
        async for chunk in provider.chat_stream(enriched_request):
            for choice in chunk.choices:
                if choice.delta.content:
                    collected_content += choice.delta.content
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        if collected_content:
            structured_debug(
                logger,
                "stream complete → storing memory",
                content_len=len(collected_content),
            )
            await _store_memory(
                memory_manager=memory_manager,
                session_store=session_store,
                request=original_request,
                response_content=collected_content,
            )

        elapsed: float = (time.monotonic() - t0) * 1000
        structured_info(
            logger,
            "← 200 stream",
            agent_id=original_request.agent_id,
            session_id=original_request.agent_session_id,
            latency_ms=f"{elapsed:.0f}",
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


async def _store_memory(
    memory_manager: MemoryManager,
    session_store: SessionStore,
    request: ChatRequest,
    response_content: str,
    prompts_dir: str = "prompts",
) -> None:
    all_messages: list[dict[str, object]] = _messages_as_dicts(request.messages)
    all_messages.append({"role": "assistant", "content": response_content})

    session_store.append(
        request.agent_id, request.agent_session_id, all_messages
    )

    from ..core.prompts import load_prompt

    prompt: str | None = load_prompt(request.agent_id, prompts_dir)
    try:
        memory_manager.add(
            all_messages,
            user_id=request.agent_id,
            metadata={"session_id": request.agent_session_id},
            prompt=prompt,
        )
    except MemoryStoreError:
        logger.exception(
            "background memory store failed agent=%s", request.agent_id
        )
