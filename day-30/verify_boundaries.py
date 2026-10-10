"""Real-model context/output boundaries and optional deployment restart check."""
import argparse
import asyncio
import json
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from chat import INPUT_BUDGET
from model import OUTPUT, PromptBuilder
from provision_accounts import credentials
from verify_chat import require


async def run(values, restart_host=None, ssh_config=None):
    origin = values["PUBLIC_ORIGIN"]
    builder = PromptBuilder(Path(__file__).parent / "tokenizer/tokenizer.json")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "origin": origin}
    async with httpx.AsyncClient(base_url=origin, timeout=15, headers={"Origin": origin}) as client:
        require(await client.post("/api/login", json={"username": values["DAY30_GUEST_USER"],
                                                    "password": values["DAY30_GUEST_PASSWORD"]}), 200)
        chat = require(await client.post("/api/chats", json={}), 201)["id"]
        prefix = "Ответь одним словом: принято. Данные для проверки контекста: "
        best = prefix
        for count in range(INPUT_BUDGET):
            text = (prefix + "a " * count).strip()
            _, tokens = builder.build([], text)
            if tokens > INPUT_BUDGET:
                break
            best = text
        _, expected = builder.build([], best)
        if expected < INPUT_BUDGET - 1:
            raise RuntimeError("Boundary input not close enough to budget")

        async def generate(chat_id, text):
            job = require(await client.post(f"/api/chats/{chat_id}/messages", json={"text": text}), 202)["id"]
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                result = require(await client.get(f"/api/jobs/{job}"), 200)
                if result["status"] == "error":
                    raise RuntimeError("Model failed at boundary")
                if result["status"] == "done":
                    if result["result"]["input_tokens"] != result["input_tokens"]:
                        raise RuntimeError("Tokenizer mismatch at boundary")
                    return result
                await asyncio.sleep(.8)
            raise RuntimeError("Model timeout at boundary")

        boundary = await generate(chat, best)
        report["boundary_input"] = {"expected": expected, "actual": boundary["result"]["input_tokens"],
                                    "answer": boundary["result"]["text"]}
        rejected = require(await client.post(f"/api/chats/{chat}/messages", json={"text": "Продолжи"}), 422)
        if rejected["detail"]["code"] != "context_limit":
            raise RuntimeError("Complete history was not limited")
        report["history_rejection"] = rejected["detail"]
        before = require(await client.get(f"/api/chats/{chat}"), 200)["messages"]
        if len(before) != 2:
            raise RuntimeError("Rejected message changed history")
        long_chat = require(await client.post("/api/chats", json={}), 201)["id"]
        capped = await generate(long_chat, "Перечисли все числа от 1 до 1000, без пропусков, через запятую. Не сокращай список.")
        if capped["result"]["output_tokens"] != OUTPUT or not capped["result"]["truncated"]:
            raise RuntimeError("Output cutoff was not reached")
        report["output_cutoff"] = {"tokens": capped["result"]["output_tokens"],
                                   "truncated": capped["result"]["truncated"],
                                   "seconds": round(capped["finished"] - capped["started"], 3)}
        if restart_host:
            command = ["ssh", "-o", "BatchMode=yes"]
            if ssh_config:
                command.extend(["-F", ssh_config])
            command.extend([restart_host, shlex.join(["sudo", "systemctl", "restart", "day30-chat"])])
            process = await asyncio.create_subprocess_exec(*command)
            if await process.wait() != 0:
                raise RuntimeError("SSH restart failed")
            for _ in range(20):
                try:
                    response = await client.get("/api/me")
                    if response.status_code == 200: break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(.5)
            else:
                raise RuntimeError("Service did not recover with existing session")
            after = require(await client.get(f"/api/chats/{chat}"), 200)["messages"]
            if after != before:
                raise RuntimeError("History did not survive restart")
            report["restart_preserved_session_and_history"] = True
        report["demo_chats"] = [chat, long_chat]
        require(await client.post("/api/logout", json={}), 200)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("resources/chat-boundaries.json"))
    parser.add_argument("--restart-host", help="Explicitly restart day30-chat via SSH after generation")
    parser.add_argument("--ssh-config")
    args = parser.parse_args()
    report = asyncio.run(run(credentials(args.env), args.restart_host, args.ssh_config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
