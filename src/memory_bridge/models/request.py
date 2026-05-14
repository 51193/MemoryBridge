from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | None = None

    agent_id: str = Field(min_length=1)
    agent_session_id: str = Field(min_length=1)
    memory_enabled: bool = True
    memory_limit: int = Field(default=5, ge=1, le=20)


class SessionCreateRequest(BaseModel):
    agent_id: str = Field(min_length=1)
    agent_session_id: str | None = None
    initial_messages: list[Message] | None = None
