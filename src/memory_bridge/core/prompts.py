"""Agent-specific memory extraction prompt loader."""

import os
from pathlib import Path


def load_prompt(agent_id: str, prompts_dir: str) -> str | None:
    """Read agent-specific memory extraction prompt from {prompts_dir}/{agent_id}.md.

    Re-reads the file on every call (hot reload).
    Returns None if the directory, file, or content is missing/empty.
    """
    prompt_file: Path = Path(prompts_dir) / f"{agent_id}.md"
    os.makedirs(prompts_dir, exist_ok=True)
    if not prompt_file.exists():
        return None
    content: str = prompt_file.read_text(encoding="utf-8").strip()
    return content if content else None
