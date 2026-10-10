"""Public HTTPS/API smoke test with real CPU model; credentials never enter the report."""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from provision_accounts import credentials


def require(response, status):
    if response.status_code != status:
        # Do not print response bodies/headers: this helper also checks login.
        raise RuntimeError(f"Expected HTTP {status}, got {response.status_code}")
    return response.json()


async def run(values):
    origin = values["PUBLIC_ORIGIN"]
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "origin": origin}
    async with httpx.AsyncClient(base_url=origin, timeout=15) as anonymous:
        require(await anonymous.get("/health"), 200)
        require(await anonymous.get("/api/chats"), 401)
        page = await anonymous.get("/")
        if page.status_code != 200 or "Войти в чат" not in page.text:
            raise RuntimeError("Login page unavailable")
        for file in ["app.js", "style.css"]:
            if (await anonymous.get("/static/" + file)).status_code != 200:
                raise RuntimeError("Static asset unavailable")
    report["https_trusted"] = report["unauthorized_401"] = report["static_files"] = True
    async with httpx.AsyncClient(base_url=origin, timeout=15, headers={"Origin": origin}) as first, \
               httpx.AsyncClient(base_url=origin, timeout=15, headers={"Origin": origin}) as second:
        for client, prefix in [(first, "DAY30"), (second, "DAY30_GUEST")]:
            login = await client.post("/api/login", json={"username": values[prefix + "_USER"],
                                                        "password": values[prefix + "_PASSWORD"]})
            require(login, 200)
            if not all(flag in login.headers["set-cookie"] for flag in ["Secure", "HttpOnly", "SameSite=strict"]):
                raise RuntimeError("Session cookie flags missing")
        report["secure_session_cookie"] = True
        csrf = await first.post("/api/chats", json={}, headers={"Origin": "https://invalid.example"})
        require(csrf, 403)
        report["csrf_403"] = True
        chat_a = require(await first.post("/api/chats", json={}), 201)["id"]
        chat_b = require(await second.post("/api/chats", json={}), 201)["id"]
        require(await second.get(f"/api/chats/{chat_a}"), 404)
        require(await second.delete(f"/api/chats/{chat_a}"), 404)
        report["foreign_chat_404"] = True
        context = require(await first.post(f"/api/chats/{chat_a}/messages", json={"text": "слово " * 1000}), 422)
        if context["detail"]["code"] != "context_limit":
            raise RuntimeError("Incorrect context rejection")
        report["context_rejection"] = context["detail"]

        async def submit(client, chat, text):
            return require(await client.post(f"/api/chats/{chat}/messages", json={"text": text}), 202)["id"]

        async def finish(client, job):
            states = []
            deadline = time.monotonic() + 450
            while time.monotonic() < deadline:
                value = require(await client.get(f"/api/jobs/{job}"), 200)
                if not states or states[-1] != [value["status"], value["position"]]:
                    states.append([value["status"], value["position"]])
                if value["status"] == "error":
                    raise RuntimeError("Generation failed")
                if value["status"] == "done":
                    result = value["result"]
                    if result["input_tokens"] != value["input_tokens"]:
                        raise RuntimeError(f"Tokenizer mismatch: expected {value['input_tokens']}, actual {result['input_tokens']}")
                    if result["input_tokens"] > 1760 or result["output_tokens"] > 256:
                        raise RuntimeError("Token budget exceeded")
                    return {"id": job, "states": states, "created": value["created"],
                            "started": value["started"], "finished": value["finished"],
                            "seconds": round(value["finished"] - value["created"], 3), **result}
                await asyncio.sleep(.4)
            raise RuntimeError("Job timeout")

        jobs = await asyncio.gather(
            submit(first, chat_a, "Запомни: мой цвет СИНИЙ. Ответь одним словом."),
            submit(second, chat_b, "Запомни: мой цвет ЗЕЛЁНЫЙ. Ответь одним словом."))
        require(await second.get(f"/api/jobs/{jobs[0]}"), 404)
        require(await first.post(f"/api/chats/{chat_a}/messages", json={"text": "Повтор"}), 409)
        initial = await asyncio.gather(finish(first, jobs[0]), finish(second, jobs[1]))
        ordered = sorted(initial, key=lambda value: value["started"])
        if ordered[1]["started"] < ordered[0]["finished"]:
            raise RuntimeError("Concurrent model processing detected")
        if not any(state[0] == "queued" for result in initial for state in result["states"]):
            raise RuntimeError("Queue was not observed")
        report["two_connections"] = initial
        report["sequential_queue"] = report["one_pending_409"] = report["foreign_job_404"] = True
        jobs = await asyncio.gather(
            submit(first, chat_a, "Какой мой цвет? Ответь одним словом."),
            submit(second, chat_b, "Какой мой цвет? Ответь одним словом."))
        memory = await asyncio.gather(finish(first, jobs[0]), finish(second, jobs[1]))
        if "СИН" not in memory[0]["text"].upper() or "ЗЕЛ" not in memory[1]["text"].upper():
            raise RuntimeError("Independent conversation recall failed")
        report["independent_history"] = memory
        report["additional_requests"] = []
        for text in ["Сколько будет 2+2? Ответь одним числом.", "Ответь одним словом: готово.", "Ответь одним словом: да."]:
            job = await submit(first, chat_a, text)
            report["additional_requests"].append(await finish(first, job))
        rate = await first.post(f"/api/chats/{chat_a}/messages", json={"text": "Шестой запрос"})
        require(rate, 429)
        if rate.json()["detail"]["code"] != "rate_limit" or "Retry-After" not in rate.headers:
            raise RuntimeError("User rate limit missing")
        report["rate_limit_429"] = True
        require(await first.post("/api/logout", json={}), 200)
        require(await first.get("/api/me"), 401)
        report["logout_401"] = True
        # Keep synthetic chats for the demo and for checking persistence across restart.
        report["demo_chats"] = [chat_a, chat_b]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("resources/chat-check.json"))
    args = parser.parse_args()
    report = asyncio.run(run(credentials(args.env)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
