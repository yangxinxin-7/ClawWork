"""Task-scoped memory recall via OpenViking official SDK."""

from __future__ import annotations

import os
from typing import Any, Optional


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_text(match: Any) -> str:
    for field in ("abstract", "overview", "content", "text", "snippet"):
        value = _get_attr_or_key(match, field, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


_MEMORY_SOURCES = ("trajectories",)


def recall_task_memory_block(
    task_id: str,
    query: str,
    account_id: str = "default",
    user_id: str = "default",
    top_k: int = 5,
    logger: Optional[Any] = None,
) -> str:
    """Recall memories for a task and format as <relevant_memories> block."""
    if not task_id:
        return ""

    try:
        from openviking import SyncHTTPClient
    except Exception as e:
        if logger:
            logger.warning(
                "OpenViking recall skipped: openviking package unavailable",
                context={"task_id": task_id, "error": str(e)},
                print_console=False,
            )
        return ""

    ov_url = os.getenv("OPENVIKING_URL", "http://127.0.0.1:1933")
    ov_key = os.getenv("OPENVIKING_API_KEY", "dev-123")
    client = SyncHTTPClient(
        url=ov_url,
        api_key=ov_key,
        account=account_id,
        user=user_id,
        agent_id=task_id,
    )

    base_uri = f"viking://agent/{task_id}/memories"

    try:
        client.initialize()

        memories = []
        for sub in _MEMORY_SOURCES:
            r = client.search(query=query, target_uri=f"{base_uri}/{sub}/", limit=top_k)
            if hasattr(r, "memories") and r.memories:
                memories.extend(r.memories)

        if not memories:
            return ""

        lines = ["<relevant_memories>"]
        for i, memory in enumerate(memories, start=1):
            uri = _get_attr_or_key(memory, "uri", "")
            score = _get_attr_or_key(memory, "score", None)
            try:
                text = client.read(uri) or ""
            except Exception:
                text = _extract_text(memory)
            text = text.strip()

            if isinstance(score, (int, float)):
                lines.append(f"{i}. [{score:.3f}] {uri}")
            else:
                lines.append(f"{i}. {uri}")
            lines.append(text)

        lines.append("</relevant_memories>")
        return "\n".join(lines)
    except Exception as e:
        print(f"[DEBUG] OpenViking recall failed: {e}")
        if logger:
            logger.warning(
                "OpenViking recall failed",
                context={"task_id": task_id, "base_uri": base_uri, "error": str(e)},
                print_console=False,
            )
        return ""
    finally:
        try:
            client.close()
        except Exception:
            pass
