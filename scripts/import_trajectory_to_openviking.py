#!/usr/bin/env python3
"""Import LiveBench trajectory JSON(s) into OpenViking using official SDK.

Single file:
  python scripts/import_trajectory_to_openviking.py --trajectory trajectory/<task_id>.json

Batch (all files in a directory):
  python scripts/import_trajectory_to_openviking.py --trajectory-dir trajectory/

Completed imports are tracked in <trajectory_dir>/.completed (one task_id per line).
Already-completed task_ids are skipped on subsequent runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_task_id(trajectory_path: Path) -> str:
    stem = trajectory_path.stem.strip()
    if not stem:
        raise ValueError("Cannot infer task_id from trajectory filename")
    return stem


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _convert_message(msg: Dict[str, Any]) -> Tuple[str, str]:
    role = str(msg.get("role", "")).strip().lower()
    content = _json_text(msg.get("content", ""))

    if role == "user":
        return "user", content or "[EMPTY_USER_MESSAGE]"

    if role == "assistant":
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc_text = json.dumps(tool_calls, ensure_ascii=False)
            content = f"{content}\n\n[TOOL_CALLS]\n{tc_text}" if content else f"[TOOL_CALLS]\n{tc_text}"
        return "assistant", content or "[EMPTY_ASSISTANT_MESSAGE]"

    if role == "tool":
        tool_name = msg.get("name", "unknown_tool")
        call_id = msg.get("tool_call_id", "")
        prefix = f"[TOOL_RESULT name={tool_name}"
        if call_id:
            prefix += f" call_id={call_id}"
        prefix += "]"
        return "assistant", f"{prefix}\n{content or '[EMPTY_TOOL_RESULT]'}"

    if role == "system":
        return "assistant", f"[SYSTEM]\n{content}"

    return "assistant", f"[{role or 'unknown'}]\n{json.dumps(msg, ensure_ascii=False)}"


def _load_trajectory(trajectory_path: Path) -> List[Dict[str, Any]]:
    with trajectory_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Trajectory file must be a JSON array (llm_messages)")
    return [item for item in data if isinstance(item, dict)]


def _extract_task_id(commit_result: Any) -> str | None:
    if not isinstance(commit_result, dict):
        return None
    if isinstance(commit_result.get("task_id"), str):
        return commit_result["task_id"]
    task = commit_result.get("task")
    if isinstance(task, dict) and isinstance(task.get("task_id"), str):
        return task["task_id"]
    return None


def _wait_task(client: Any, task_id: str, timeout_sec: int, interval_sec: float = 1.5) -> Dict[str, Any]:
    start = time.time()
    last_status = "unknown"
    while True:
        task = client.get_task(task_id)
        if isinstance(task, dict):
            status = str(task.get("status", "unknown")).lower()
            if status in {"completed", "succeeded", "success", "done"}:
                return task
            if status in {"failed", "error", "cancelled", "canceled"}:
                return task
            last_status = status
        if time.time() - start > timeout_sec:
            return {"status": "timeout", "task_id": task_id, "last_status": last_status}
        time.sleep(interval_sec)


def _extract_token_usage(task_result: Dict[str, Any]) -> Dict[str, int]:
    """从 task 结果中提取 token 消耗，返回标准化字典。"""
    empty = {"prompt_tokens": 0, "completion_tokens": 0, "llm_total": 0, "embedding_tokens": 0, "total_tokens": 0}
    result = task_result.get("result")
    if not isinstance(result, dict):
        return empty
    usage = result.get("token_usage")
    if not isinstance(usage, dict):
        return empty
    llm = usage.get("llm", {})
    embedding = usage.get("embedding", {})
    total = usage.get("total", {})
    return {
        "prompt_tokens": llm.get("prompt_tokens", 0),
        "completion_tokens": llm.get("completion_tokens", 0),
        "llm_total": llm.get("total_tokens", 0),
        "embedding_tokens": embedding.get("total_tokens", 0),
        "total_tokens": total.get("total_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Completed-set helpers
# ---------------------------------------------------------------------------

def _load_completed(completed_file: Path) -> set:
    if not completed_file.exists():
        return set()
    return {line.strip() for line in completed_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def _mark_completed(completed_file: Path, task_id: str) -> None:
    with completed_file.open("a", encoding="utf-8") as f:
        f.write(task_id + "\n")


# ---------------------------------------------------------------------------
# Single-file import
# ---------------------------------------------------------------------------

def _import_one(client: Any, trajectory_path: Path, task_id: str, wait_timeout: int, dry_run: bool) -> tuple[bool, Dict[str, int]]:
    """Import one trajectory. Returns (success, token_usage)."""
    zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_total": 0, "embedding_tokens": 0, "total_tokens": 0}
    raw_messages = _load_trajectory(trajectory_path)
    converted = [_convert_message(m) for m in raw_messages]

    print(f"  Messages : {len(raw_messages)}")
    print(f"  task_id  : {task_id}")
    print(f"  agent_id : {task_id}")

    if dry_run:
        print("  [dry-run] skipping OpenViking calls.")
        return True, zero_usage

    session_id = task_id
    client.get_session(session_id, auto_create=True)
    for role, content in converted:
        client.add_message(session_id=session_id, role=role, content=content)

    commit_result = client.commit_session(session_id=session_id)
    bg_task_id = _extract_task_id(commit_result)

    token_usage = zero_usage
    if bg_task_id:
        final = _wait_task(client, bg_task_id, timeout_sec=wait_timeout)
        status = str(final.get("status", "unknown")).lower()
        token_usage = _extract_token_usage(final)
        print(f"  Background task status: {status}")
        print(f"  Token usage — prompt: {token_usage['prompt_tokens']}, completion: {token_usage['completion_tokens']}, "
              f"embedding: {token_usage['embedding_tokens']}, total: {token_usage['total_tokens']}")
        if status in {"failed", "error", "cancelled", "canceled", "timeout"}:
            print(f"  ❌ Import failed (status={status})")
            return False, token_usage
    else:
        print("  No background task_id returned; assuming success.")

    print(f"  ✅ Done — retrieval URI: viking://agent/{task_id}/memories/")
    return True, token_usage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Import LiveBench trajectory(s) into OpenViking.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trajectory", help="Path to a single trajectory JSON file.")
    group.add_argument("--trajectory-dir", help="Directory containing trajectory JSON files (batch mode).")

    parser.add_argument("--task-id", default=None, help="Task ID (single-file mode only). Inferred from filename if omitted.")
    parser.add_argument("--ov-url", default=os.getenv("OPENVIKING_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENVIKING_API_KEY"))
    parser.add_argument("--agent-id", default=None, help="Agent ID (single-file mode only). Defaults to task_id.")
    parser.add_argument("--wait-timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    try:
        import openviking as ov
    except ImportError as e:
        print("ERROR: openviking is not installed.", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Single-file mode
    # ------------------------------------------------------------------
    if args.trajectory:
        trajectory_path = Path(args.trajectory).expanduser().resolve()
        if not trajectory_path.exists():
            print(f"ERROR: file not found: {trajectory_path}", file=sys.stderr)
            return 1

        task_id = args.task_id or _infer_task_id(trajectory_path)
        agent_id = args.agent_id or task_id

        client = ov.SyncHTTPClient(url=args.ov_url, api_key=args.api_key, agent_id=agent_id)
        try:
            client.initialize()
            ok, usage = _import_one(client, trajectory_path, task_id, args.wait_timeout, args.dry_run)
            return 0 if ok else 1
        finally:
            try:
                client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    traj_dir = Path(args.trajectory_dir).expanduser().resolve()
    if not traj_dir.is_dir():
        print(f"ERROR: not a directory: {traj_dir}", file=sys.stderr)
        return 1

    completed_file = traj_dir / ".completed"
    completed = _load_completed(completed_file)

    files = sorted(traj_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {traj_dir}")
        return 0

    print(f"Found {len(files)} trajectory file(s) in {traj_dir}")
    print(f"Already completed: {len(completed)}\n")

    success_count = skipped_count = failed_count = 0
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "llm_total": 0, "embedding_tokens": 0, "total_tokens": 0}

    for f in files:
        task_id = _infer_task_id(f)

        if task_id in completed:
            print(f"⏭️  Skipping {f.name} (already completed)")
            skipped_count += 1
            continue

        print(f"📥 Importing {f.name} ...")
        # Create a fresh client per task so agent_id is baked into the HTTP headers at initialize() time
        client = ov.SyncHTTPClient(url=args.ov_url, api_key=args.api_key, agent_id=task_id)
        try:
            client.initialize()
            ok, usage = _import_one(client, f, task_id, args.wait_timeout, args.dry_run)
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            ok, usage = False, {"prompt_tokens": 0, "completion_tokens": 0, "llm_total": 0, "embedding_tokens": 0, "total_tokens": 0}
        finally:
            try:
                client.close()
            except Exception:
                pass

        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        if ok:
            if not args.dry_run:
                _mark_completed(completed_file, task_id)
            success_count += 1
        else:
            failed_count += 1
        print()

    print(f"{'='*50}")
    print(f"Done — success: {success_count}, skipped: {skipped_count}, failed: {failed_count}")
    print(f"{'='*50}")
    print(f"Total Token Usage:")
    print(f"  LLM prompt tokens    : {total_usage['prompt_tokens']:,}")
    print(f"  LLM completion tokens: {total_usage['completion_tokens']:,}")
    print(f"  LLM total tokens     : {total_usage['llm_total']:,}")
    print(f"  Embedding tokens     : {total_usage['embedding_tokens']:,}")
    print(f"  Grand total tokens   : {total_usage['total_tokens']:,}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
