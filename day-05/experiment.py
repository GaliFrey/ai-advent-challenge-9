"""Sequential model calls, local verification, and session persistence."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
from dotenv import load_dotenv
from verification import verify_session

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resources" / "results"
SCHEMA_VERSION = 2
PRICE_DATE = "2026-09-04"
ANSWER_MAX_TOKENS = 4096
SYSTEM = "Ты решаешь задачу пользователя. Следуй исходным данным и всем условиям. Ответь по-русски."


@dataclass(frozen=True)
class Model:
    slot: str
    provider: str
    name: str


@dataclass(frozen=True)
class Settings:
    models: tuple[Model, ...]
    deepseek_key: str = field(default="", repr=False)
    openrouter_key: str = field(default="", repr=False)

    def check(self) -> None:
        missing = []
        if not self.deepseek_key.strip():
            missing.append("DEEPSEEK_API_KEY")
        if not self.openrouter_key.strip():
            missing.append("OPENROUTER_API_KEY")
        if missing:
            raise ValueError("Заполни в day-05/.env: " + ", ".join(missing))
        if any(not model.name.strip() for model in self.models):
            raise ValueError("Названия моделей не должны быть пустыми")
        if len({(m.provider, m.name) for m in self.models}) != 3:
            raise ValueError("Для сравнения нужны три разные модели")
        if any(m.provider == "openrouter" and not m.name.endswith(":free") for m in self.models):
            raise ValueError("Для OpenRouter выбери конкретную бесплатную модель с суффиксом :free")

    def key(self, provider: str) -> str:
        return {"deepseek": self.deepseek_key, "openrouter": self.openrouter_key}[provider]

def load_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    return Settings(
        models=(
            Model("weak", "openrouter", os.getenv("OPENROUTER_MODEL_WEAK", "nvidia/nemotron-3.5-lightning:free")),
            Model("medium", "deepseek", os.getenv("DEEPSEEK_MODEL_MEDIUM", "deepseek-v4-flash")),
            Model("strong", "deepseek", os.getenv("DEEPSEEK_MODEL_STRONG", "deepseek-v4-pro")),
        ),
        deepseek_key=os.getenv("DEEPSEEK_API_KEY", ""),
        openrouter_key=os.getenv("OPENROUTER_API_KEY", ""),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def payload(model: Model, messages: list[dict]) -> dict:
    request = {
        "model": model.name,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": ANSWER_MAX_TOKENS,
        "stream": False,
    }
    if model.provider == "deepseek":
        request["thinking"] = {"type": "disabled"}
    elif model.provider == "openrouter":
        request["reasoning"] = {"enabled": False}
        request["provider"] = {"require_parameters": True, "max_price": {"prompt": 0, "completion": 0}}
    return request


def integer(value):
    return value if type(value) is int and value >= 0 else None


def usage_metrics(raw: dict) -> dict:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    inp = integer(usage.get("prompt_tokens"))
    out = integer(usage.get("completion_tokens"))
    total = integer(usage.get("total_tokens"))
    if total is None and inp is not None and out is not None:
        total = inp + out
    details = usage.get("prompt_tokens_details")
    cached = integer(usage.get("prompt_cache_hit_tokens"))
    if cached is None and isinstance(details, dict):
        cached = integer(details.get("cached_tokens"))
    miss = integer(usage.get("prompt_cache_miss_tokens"))
    if cached is None and inp is not None and miss is not None and miss <= inp:
        cached = inp - miss
    if cached is not None and (inp is None or cached > inp):
        cached = None
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total, "cached_tokens": cached}


def cost_estimate(model: Model, metrics: dict, started_at: str, groq_plan: str = "free") -> dict:
    result = {"usd": None, "as_of": PRICE_DATE, "basis": "Тариф этой модели не задан"}
    if model.provider == "openrouter":
        result["source"] = "https://openrouter.ai/" + model.name
        if model.name.endswith(":free"):
            result.update(usd=0.0, basis="OpenRouter :free: бесплатные входные и выходные токены")
        return result
    if model.provider == "groq":
        result["source"] = "https://console.groq.com/docs/model/qwen/qwen3.8-27b"
        if groq_plan == "free":
            result.update(usd=0.0, basis="Groq Free: предполагается бесплатный план")
            return result
        if model.name != "qwen/qwen3.8-27b":
            return result
        rates = {"input": 0.8, "cached": 0.8, "output": 4.0}
        result["basis"] = "Groq Paid: оценка по опубликованному тарифу"
    else:
        result["source"] = "https://api-docs.deepseek.com/quick_start/pricing/"
        multiplier = {"deepseek-v4-flash": 1, "deepseek-v4-pro": 3}.get(model.name)
        if multiplier is None:
            return result
        dt = datetime.fromisoformat(started_at).astimezone(timezone.utc)
        peak = dt.weekday() < 5 and (1 <= dt.hour < 4 or 6 <= dt.hour < 10)
        factor = multiplier * (2 if peak else 1)
        cached_rate = .007 if model.name == "deepseek-v4-flash" else .022
        rates = {"input": .22 * factor, "cached": cached_rate * (2 if peak else 1), "output": .66 * factor}
        result["basis"] = "Оценка: " + ("peak" if peak else "off-peak") + " по UTC старта запроса"
    result["rates_usd_per_million"] = rates
    inp, out = metrics["input_tokens"], metrics["output_tokens"]
    if inp is None or out is None:
        result["basis"] += "; usage отсутствует"
        return result
    cached = metrics["cached_tokens"]
    if cached is None:
        cached = 0
        if model.provider == "deepseek":
            result["basis"] += "; кеш неизвестен, весь ввод по обычной цене (верхняя оценка)"
    result["usd"] = (cached * rates["cached"] + (inp - cached) * rates["input"] + out * rates["output"]) / 1_000_000
    return result


async def request_model(client: httpx.AsyncClient, model: Model, request: dict, settings: Settings) -> dict:
    started_at = utc_now()
    start = time.perf_counter()
    record = {
        "model": model.name, "provider": model.provider, "slot": model.slot,
        "started_at": started_at, "request": request, "status": "error", "answer": "",
        "finish_reason": None, "error": None,
    }
    url = {"deepseek": "https://api.deepseek.com/chat/completions", "openrouter": "https://openrouter.ai/api/v1/chat/completions"}[model.provider]
    raw = {}
    try:
        response = await client.post(url, headers={"Authorization": "Bearer " + settings.key(model.provider)}, json=request)
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError
        record["response"] = raw
        choice = raw["choices"][0]
        answer = choice["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError
        record["answer"] = answer
        record["finish_reason"] = choice.get("finish_reason", "unknown")
        record["status"] = "ok" if record["finish_reason"] == "stop" else "incomplete"
    except httpx.HTTPStatusError as exc:
        record["error"] = f"HTTP {exc.response.status_code}; повтор не выполнялся"
        if exc.response.status_code == 429:
            # Only numeric quota metadata, never arbitrary headers or error bodies.
            record["rate_limit"] = {
                name: value for name in (
                    "retry-after", "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
                    "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
                )
                if (value := exc.response.headers.get(name, "")) and re.fullmatch(r"[0-9]{1,12}(?:\.[0-9]{1,6})?", value)
            }
            record["error"] = "HTTP 429: превышен лимит запросов или токенов API; повтор не выполнялся"
            try:
                error = exc.response.json().get("error", {})
                message = error.get("message", "") if isinstance(error, dict) else ""
                metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
                if isinstance(metadata, dict) and metadata.get("limit_source") == "upstream_provider_shared_pool":
                    record["rate_limit"]["source"] = "upstream_provider_shared_pool"
                    record["error"] = "HTTP 429: общий бесплатный пул провайдера OpenRouter временно ограничен; повтор не выполнялся"
                # Extract a known reason and numbers, never retain arbitrary error text
                # (it may include organization IDs, prompts, or echoed credentials).
                if isinstance(message, str) and "Request too large" in message:
                    match = re.search(r"output tokens per minute \(OTPM\):\s*Limit ([0-9]+),\s*Requested ([0-9]+)", message)
                    if match:
                        limit, requested = map(int, match.groups())
                        record["rate_limit"].update(kind="OTPM", limit=limit, requested=requested, request_too_large=True)
                        record["error"] = f"HTTP 429: ожидается {requested} выходных токенов при лимите OTPM {limit}. Уменьши max_tokens; пауза не уменьшает размер запроса. Повтор не выполнялся"
            except (ValueError, AttributeError, TypeError):
                pass
            if "retry-after" in record["rate_limit"]:
                record["error"] += "; повтор допустим не раньше чем через " + record["rate_limit"]["retry-after"] + " с"
    except httpx.TimeoutException:
        record["error"] = "Истекло время ожидания; сервер мог обработать запрос"
    except httpx.RequestError:
        record["error"] = "Сбой соединения; сервер мог обработать запрос"
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        record["error"] = "API не вернул ожидаемый текстовый ответ"
    record["elapsed_seconds"] = time.perf_counter() - start
    record["completed_at"] = utc_now()
    record["metrics"] = usage_metrics(raw if isinstance(raw, dict) else {})
    record["cost"] = cost_estimate(model, record["metrics"], started_at)
    if record["status"] == "error":
        record["cost"]["usd"] = None
        record["cost"]["basis"] = "Стоимость не подтверждена после ошибки"
    return record


def validate_prompts(prompts: list[str]) -> None:
    if not isinstance(prompts, list) or len(prompts) != 2 or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Введи две непустые задачи перед запуском")
    if prompts[0].strip() == prompts[1].strip():
        raise ValueError("Для эксперимента нужны два разных промпта")


def all_records(session: dict) -> list[dict]:
    return [record for results in session["results"].values() for record in results.values()]


def aggregate_metrics(session: dict) -> dict:
    summary = {}
    expected = len(session["prompts"])
    for model in session["models"]:
        slot = model["slot"]
        records = [session["results"][p["id"]][slot] for p in session["prompts"] if slot in session["results"][p["id"]]]
        values = {
            "input_tokens": [r["metrics"]["input_tokens"] for r in records],
            "output_tokens": [r["metrics"]["output_tokens"] for r in records],
            "total_tokens": [r["metrics"]["total_tokens"] for r in records],
            "estimated_cost_usd": [r["cost"]["usd"] for r in records],
            "sum_elapsed_seconds": [r["elapsed_seconds"] for r in records],
        }
        summary[slot] = {
            "expected": expected, "completed": len(records),
            "successful": sum(r["status"] == "ok" for r in records),
            **{key: sum(items) if len(items) == expected and all(v is not None for v in items) else None for key, items in values.items()},
        }
    return summary


def save_session(session: dict, path: Path) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_session(path: Path) -> dict:
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
        if session["schema_version"] == 1:
            # Adapt historical sessions in memory; never rewrite the source file.
            session["prompts"] = [{"id": "task-1", "text": session.pop("prompt")}]
            session["results"] = {"task-1": session["results"]}
            session["schema_version"] = SCHEMA_VERSION
        if session["schema_version"] != SCHEMA_VERSION:
            raise ValueError
        if not isinstance(session["prompts"], list) or len(session["prompts"]) not in (1, 2):
            raise ValueError
        for index, prompt in enumerate(session["prompts"], 1):
            if prompt["id"] != f"task-{index}" or not isinstance(prompt["text"], str) or not isinstance(session["results"][prompt["id"]], dict):
                raise ValueError
        if {m["slot"] for m in session["models"]} != {"weak", "medium", "strong"}:
            raise ValueError
        session["summary"] = aggregate_metrics(session)
        session["verification"] = verify_session(session)
        return session
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("Не удалось прочитать сессию дня 5") from None


async def run_experiment(
    settings: Settings, prompts: list[str], on_change: Callable[[dict, Path], None],
    *, results_dir: Path = RESULTS, transport=None,
) -> tuple[dict, Path]:
    settings.check()
    validate_prompts(prompts)
    tasks = [{"id": f"task-{i}", "text": text.strip()} for i, text in enumerate(prompts, 1)]
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / (session_id + ".json")
    session = {
        "schema_version": SCHEMA_VERSION, "id": session_id, "created_at": utc_now(),
        "prompts": tasks, "status": "running", "free_provider": "openrouter",
        "models": [{"slot": m.slot, "provider": m.provider, "name": m.name} for m in settings.models],
        "results": {p["id"]: {} for p in tasks},
        "price_date": PRICE_DATE, "execution_mode": "sequential", "active_request": None,
        "answer_max_tokens": ANSWER_MAX_TOKENS,
    }

    def publish():
        session["summary"] = aggregate_metrics(session)
        save_session(session, path)
        on_change(session, path)

    publish()  # Fail before any API calls if the session cannot be saved.
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(150, connect=20), transport=transport, follow_redirects=False) as client:
            for prompt in tasks:
                for model in settings.models:
                    session["active_request"] = {"task_id": prompt["id"], "slot": model.slot}
                    publish()
                    request = payload(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt["text"]}])
                    record = await request_model(client, model, request, settings)
                    session["results"][prompt["id"]][model.slot] = record
                    session["active_request"] = None
                    publish()
            session["status"] = "checking"
            publish()
            # Own the worker lifetime; don't leave a default executor on the UI loop.
            with ThreadPoolExecutor(max_workers=1) as verifier:
                pending_checks = verifier.submit(verify_session, session)
                # Polling also works where sandboxed thread-to-loop wakeups are unavailable.
                while not pending_checks.done():
                    await asyncio.sleep(.025)
                session["verification"] = pending_checks.result()
            if all(r["status"] == "ok" for r in all_records(session)):
                session["status"] = "complete"
            elif any(r["answer"] for r in all_records(session)):
                session["status"] = "partial"
                session["comparison_note"] = "incomplete_answers"
            else:
                session["status"] = "failed"
            session["completed_at"] = utc_now()
            publish()
    except asyncio.CancelledError:
        session["status"] = "interrupted"
        publish()
        raise
    return session, path
