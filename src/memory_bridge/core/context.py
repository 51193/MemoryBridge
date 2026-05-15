"""Context builder — inject memories into the system prompt."""

import logging

from ..logfmt import structured_debug

logger: logging.Logger = logging.getLogger(__name__)

_MEMORY_TEMPLATE: str = """[相关历史记忆]
{memories}

[重要提示]
当用户询问关于其偏好、历史或个人信息时，请仅基于上述记忆和对话历史中的已知事实回答。
如果对话历史和相关记忆都未包含相关信息，请直接回答"抱歉，我目前没有相关的记忆记录"，
不要编造、猜测或虚构任何信息。

[当前对话]
"""


class ContextBuilder:
    """Builds an enriched messages array with injected memories."""

    def build(
        self,
        messages: list[dict[str, object]],
        memories: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Insert retrieved memories into the system prompt.

        Strategy:
        1. Extract memory strings from the memory dicts
        2. Build the memory block
        3. If a system message exists, prepend memories to its content
        4. If no system message exists, insert a new system message at the front
        """
        structured_debug(logger, "context build", memories_count=len(memories))

        memory_lines: str = "\n".join(
            f"- {str(m.get('memory', ''))}" for m in memories
        )

        memory_block: str = _MEMORY_TEMPLATE.format(memories=memory_lines)
        block_preview: str = memory_block[:200]

        structured_debug(
            logger,
            "memory block",
            preview=block_preview,
        )

        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                original_content: str = str(msg.get("content", ""))
                enriched: dict[str, object] = dict(msg)
                enriched["content"] = original_content + memory_block
                result: list[dict[str, object]] = list(messages)
                result[i] = enriched
                structured_debug(
                    logger,
                    "context built → appended to existing system message",
                    system_len_before=len(original_content),
                    system_len_after=len(str(enriched["content"])),
                )
                return result

        structured_debug(logger, "context built → new system message created")
        system_msg: dict[str, object] = {
            "role": "system",
            "content": memory_block.strip(),
        }
        return [system_msg] + list(messages)
