#!/usr/bin/env python3
"""MemoryBridge integration smoke test.

Verifies the full pipeline end-to-end:
  A. Short-term memory (session context history)
  B. Long-term memory (Mem0 → Qdrant across sessions)
  C. Agent isolation
  D. Edge cases (validation, thinking mode, streaming)
  E. Custom memory extraction prompt

Usage:
  python tests/integration/smoke_test.py [--base-url http://host:port] [--quiet]

After the test, inspect ./data/ then manually run cleanup:
  python tests/integration/cleanup.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

BASE_URL: str = "http://localhost:8000"
PROMPTS_DIR: str = os.getenv("PROMPTS_DIR", "prompts")
QUIET: bool = False
PASSED: int = 0
FAILED: int = 0
STEP: int = 0
TOTAL: int = 19

AGENT: str = "agent-smoke"
AGENT_OTHER: str = "agent-other"


def log(msg: str) -> None:
    if not QUIET:
        print(msg, flush=True)


def _req(method: str, path: str, body: dict[str, object] | None = None) -> dict[str, Any]:
    url: str = f"{BASE_URL}{path}"
    data: bytes | None = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            raw: str = resp.read().decode()
            return {"_status": resp.status, "_body": raw, **json.loads(raw)}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        return {"_status": e.code, "_body": raw}


def _sse_body(path: str, body: dict[str, object]) -> str:
    """POST with stream=true, return full response body as string."""
    url: str = f"{BASE_URL}{path}"
    data: bytes = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    result: str
    with urllib.request.urlopen(req) as resp:
        result = resp.read().decode()
    return result


def step(desc: str) -> None:
    global STEP
    STEP += 1
    log(f"\n[{STEP}/{TOTAL}] {desc}")


def check(condition: bool, msg: str) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        log(f"       ✓ {msg}")
    else:
        FAILED += 1
        log(f"       ✗ FAIL: {msg}")
        if not QUIET:
            sys.exit(1)


def check_contains(haystack: str, needle: str, label: str) -> None:
    check(needle in haystack, f"{label} (contains '{needle}')")


def check_not_contains(haystack: str, needle: str, label: str) -> None:
    check(needle not in haystack, f"{label} (no '{needle}')")


# ── Phase A: Short-term memory ─────────────────────────────────────────


def phase_a() -> None:
    step("CREATE session st")
    r = _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-st",
        "initial_messages": [
            {"role": "user", "content": "你好，我叫小明"},
            {"role": "assistant", "content": "你好小明！"},
        ],
    })
    check(r["_status"] == 201, f"status=201 message_count={r.get('message_count')}")
    check(r.get("message_count") == 2, "message_count=2")

    step("CHAT short-term memory (memory_enabled=false)")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "我叫什么名字？"}],
        "agent_id": AGENT,
        "agent_session_id": "sess-st",
        "memory_enabled": False,
    })
    check(r["_status"] == 200, "status=200")
    content: str = r.get("choices", [{}])[0].get("message", {}).get("content", "")
    check_contains(content, "小明", "response contains '小明' (short-term memory)")


# ── Phase B: Long-term memory ──────────────────────────────────────────


def phase_b() -> None:
    step("CREATE session lt (agent-smoke/sess-lt)")
    r = _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-lt",
    })
    check(r["_status"] == 201, "status=201")

    step("CHAT write long-term facts")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{
            "role": "user",
            "content": "我最喜欢的颜色是蓝色，最喜欢的动物是猫"
        }],
        "agent_id": AGENT,
        "agent_session_id": "sess-lt",
        "memory_enabled": True,
    })
    check(r["_status"] == 200, "status=200")
    content = r.get("choices", [{}])[0].get("message", {}).get("content", "")
    log(f"       LLM response preview: {content[:120]}...")

    step("SLEEP 3s (wait for Mem0 background write)")
    time.sleep(3)
    log("       slept 3s ✓")

    step("CREATE session lt-verify (different session, same agent)")
    r = _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-lt-verify",
    })
    check(r["_status"] == 201, "status=201")

    step("CHAT verify long-term memory (new session, no history)")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "请回忆我喜欢什么颜色和动物？"}],
        "agent_id": AGENT,
        "agent_session_id": "sess-lt-verify",
        "memory_enabled": True,
    })
    check(r["_status"] == 200, "status=200")
    content = r.get("choices", [{}])[0].get("message", {}).get("content", "")
    log(f"       LLM response: {content[:200]}")
    check_contains(content, "蓝", "response contains '蓝' (long-term color)")
    check_contains(content, "猫", "response contains '猫' (long-term animal)")


# ── Phase C: Agent isolation ───────────────────────────────────────────


def phase_c() -> None:
    step("CREATE session iso (agent-other/sess-iso)")
    r = _req("POST", "/v1/sessions", {
        "agent_id": AGENT_OTHER,
        "agent_session_id": "sess-iso",
    })
    check(r["_status"] == 201, "status=201")

    step("CHAT verify agent isolation")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "请回忆我喜欢什么颜色和动物？"}],
        "agent_id": AGENT_OTHER,
        "agent_session_id": "sess-iso",
        "memory_enabled": True,
    })
    check(r["_status"] == 200, "status=200")
    content = r.get("choices", [{}])[0].get("message", {}).get("content", "")
    log(f"       LLM response: {content[:200]}")
    check_not_contains(content, "蓝", "agent isolation (no '蓝')")
    check_not_contains(content, "猫", "agent isolation (no '猫')")


# ── Phase E: Session export → import round-trip ─────────────────────────


def phase_e() -> None:
    step("CREATE session export-src")
    r = _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-export-src",
        "initial_messages": [
            {"role": "user", "content": "你好，我叫小刚"},
            {"role": "assistant", "content": "你好小刚！"},
        ],
    })
    check(r["_status"] == 201, "status=201")

    step("CHAT one turn in export-src (memory_enabled=false)")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "回复一个字：好"}],
        "agent_id": AGENT,
        "agent_session_id": "sess-export-src",
        "memory_enabled": False,
    })
    check(r["_status"] == 200, "status=200")

    step("EXPORT session sess-export-src")
    r = _req("GET", "/v1/sessions/agent-smoke/sess-export-src")
    check(r["_status"] == 200, "status=200")
    exported: list[dict[str, object]] = r.get("messages", [])
    log(f"       exported {len(exported)} messages")
    check(len(exported) >= 2, f"session has >= 2 messages (got {len(exported)})")
    roles: list[str] = [str(m.get("role", "")) for m in exported]
    check("system" not in roles, "no system role in exported messages")

    step("IMPORT session via POST /v1/sessions (restore)")
    r = _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-export-dst",
        "initial_messages": exported,
    })
    check(r["_status"] == 201, "status=201")
    check(r.get("message_count") == len(exported), f"message_count={len(exported)}")

    step("CHAT verify restored session has history (memory_enabled=false)")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "我叫什么名字？请只回答名字"}],
        "agent_id": AGENT,
        "agent_session_id": "sess-export-dst",
        "memory_enabled": False,
    })
    check(r["_status"] == 200, "status=200")
    content = r.get("choices", [{}])[0].get("message", {}).get("content", "")
    log(f"       LLM response: {content[:120]}")
    check_contains(content, "小刚", "restored session contains '小刚' (export→import round-trip)")


# ── Phase D: Edge cases ────────────────────────────────────────────────


def phase_d() -> None:
    step("VALIDATE missing agent_session_id → 422")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
        "agent_id": AGENT,
    })
    check(r["_status"] == 422, "status=422")

    step("VALIDATE unknown session → 404")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
        "agent_id": AGENT,
        "agent_session_id": "nonexistent-session-id",
    })
    check(r["_status"] == 404, "status=404")
    check_contains(r.get("_body", ""), "SESSION_NOT_FOUND", "body contains SESSION_NOT_FOUND")

    step("VALIDATE duplicate session → 409")
    _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-edge",
    })
    r = _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-edge",
    })
    check(r["_status"] == 409, "status=409")
    check_contains(r.get("_body", ""), "SESSION_EXISTS", "body contains SESSION_EXISTS")

    step("CHAT thinking + stream")
    body_str: str = _sse_body("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "1+1等于几？"}],
        "agent_id": AGENT,
        "agent_session_id": "sess-lt",
        "stream": True,
        "memory_enabled": False,
    })
    check("data:" in body_str, "stream has SSE data: prefix")
    check("[DONE]" in body_str, "stream ends with [DONE]")
    check("reasoning_content" in body_str, "stream contains reasoning_content (thinking mode)")

    step("CHAT custom prompt — write memory under custom extraction rules")
    prompt_file: Path = Path(PROMPTS_DIR) / f"{AGENT}.md"
    prompt_content: str = (
        "你正在为一位宠物医生提取记忆。请只关注：宠物偏好、宠物健康状况。\n"
        "忽略所有编程、技术相关的内容。"
    )
    prompt_file.parent.mkdir(exist_ok=True)
    prompt_file.write_text(prompt_content, encoding="utf-8")
    log(f"       wrote custom prompt to {prompt_file}")

    # Write both a tech fact and a pet fact — only the pet fact should be extracted
    _req("POST", "/v1/sessions", {
        "agent_id": AGENT,
        "agent_session_id": "sess-prompt",
    })
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "我喜欢Python，我养了一只叫旺财的狗"}],
        "agent_id": AGENT,
        "agent_session_id": "sess-prompt",
        "memory_enabled": True,
    })
    check(r["_status"] == 200, "status=200")
    content_preview: str = (
        r.get("choices", [{}])[0].get("message", {}).get("content", "")[:120]
    )
    log(f"       LLM response: {content_preview}")

    step("SLEEP 3s (wait for Mem0 write)")
    time.sleep(3)
    log("       slept 3s ✓")

    step("CHAT verify custom prompt extraction")
    r = _req("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "我有什么宠物？叫什么？我有什么编程偏好？"}],
        "agent_id": AGENT,
        "agent_session_id": "sess-lt",       # reuse existing session
        "memory_enabled": True,
    })
    check(r["_status"] == 200, "status=200")
    content = r.get("choices", [{}])[0].get("message", {}).get("content", "")
    log(f"       LLM response: {content[:200]}")
    # Pet fact should be extracted (custom prompt focuses on pets)
    check_contains(content, "旺财", "response contains '旺财' (pet extracted by custom prompt)")
    check_contains(content, "狗", "response contains '狗' (pet extracted by custom prompt)")

    # Clean up the prompt file
    prompt_file.unlink()

    step("Health check")
    r = _req("GET", "/health")
    check(r["_status"] == 200, "status=200")
    check(r.get("qdrant") == "connected", "qdrant=connected")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    global BASE_URL, QUIET
    args: list[str] = sys.argv[1:]
    i: int = 0
    while i < len(args):
        if args[i] == "--base-url" and i + 1 < len(args):
            BASE_URL = args[i + 1]
            i += 2
        elif args[i] == "--quiet":
            QUIET = True
            i += 1
        else:
            print(f"Usage: {sys.argv[0]} [--base-url URL] [--quiet]")
            sys.exit(1)

    print(f"MemoryBridge Smoke Test → {BASE_URL}")
    print("=" * 50)

    try:
        phase_a()
        phase_b()
        phase_c()
        phase_e()
        phase_d()
    except SystemExit:
        raise
    except Exception as e:
        log(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    if FAILED == 0:
        print(f"ALL PASSED: {PASSED}/{PASSED + FAILED}")
        print()
        print("To clean up test data, run:")
        print("  python tests/integration/cleanup.py")
    else:
        print(f"SOME FAILED: {PASSED} passed, {FAILED} failed")
    sys.exit(FAILED)


if __name__ == "__main__":
    main()
