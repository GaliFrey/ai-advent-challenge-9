"""Check agent routing offline, or opt in to one LLM request sequence."""
from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path

from agent import run_turn
from servers import BY_ALIAS
from trace_log import SessionLog


async def offline_completion(_http, _key, messages: list[dict], _tools: list[dict]) -> dict:
    """Make the same three tool choices without contacting an LLM provider."""
    completed = [message for message in messages if message["role"] == "tool"]
    stage = len(completed)
    if stage == 3:
        return {"choices": [{"message": {"role": "assistant", "content": "Отчёт получен на клиенте."}}]}
    aliases = tuple(BY_ALIAS)
    args = ({"hours": 24},
            {"source_ref": json.loads(completed[-1]["content"])["source_ref"]} if stage == 1 else None,
            {"analysis_ref": json.loads(completed[-1]["content"])["analysis_ref"]} if stage == 2 else None)[stage]
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": str(stage + 1), "type": "function", "function": {"name": aliases[stage], "arguments": json.dumps(args)}}
    ]}}]}


async def main(use_llm: bool = False) -> None:
    session = SessionLog(Path(__file__).resolve().parent / "sessions")

    def trace(label: str, data: object) -> None:
        server = data.get("server", "client") if isinstance(data, dict) else "client"
        session.append(label, data, server)
        if label == "MCP CALL":
            print(f"LLM → {server} / {data['tool']}")
        elif label == "MCP RESULT":
            print(f"✓ {data['tool']}: {data['summary']}")
        elif label == "REPORT RECEIVED":
            print(f"✓ Отчёт получен: {data['local_path']} ({data['bytes']} байт)")
        elif label == "ROUTE REJECTED":
            print(f"✗ Неверный маршрут: {data}")

    kwargs = {} if use_llm else {"complete": offline_completion, "key": "offline"}
    answer = await run_turn("Проанализируй SSH-входы за последние сутки и сохрани отчёт на моём компьютере", trace, **kwargs)
    print(answer)
    print(f"JSONL: {session.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="Use the external DeepSeek API")
    asyncio.run(main(parser.parse_args().llm))
