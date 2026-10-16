"""FastAPI-приложение локального сервера поддержки тюнера.

Один loopback-origin отдаёт страницу тюнера, статические ресурсы и API
поддержки. Приложение не включает permissive CORS, обращается к внешнему
сервису DeepSeek только при генерации и публикует только явно предназначенные для браузера файлы.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import (
    assistant_agent,
    assistant_staging,
    assistant_tools,
    config,
    responder,
    retrieval_index,
    generation_gate,
)
from .db import Database
from .errors import SupportError
from .mcp_client import McpError, fetch_ticket_context
from .snapshot import validate_snapshot

# Явный allowlist статических файлов. Всё вне списка (корень репозитория,
# SQLite, .env, tasks/, intakes/, server/, тесты и README) недоступно по HTTP.
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/src/app.js": ("src/app.js", "text/javascript; charset=utf-8"),
    "/src/capture.js": ("src/capture.js", "text/javascript; charset=utf-8"),
    "/src/display.js": ("src/display.js", "text/javascript; charset=utf-8"),
    "/src/pitch.js": ("src/pitch.js", "text/javascript; charset=utf-8"),
    "/src/diagnostics.js": ("src/diagnostics.js", "text/javascript; charset=utf-8"),
    "/src/support.js": ("src/support.js", "text/javascript; charset=utf-8"),
    "/src/assistant.js": ("src/assistant.js", "text/javascript; charset=utf-8"),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_hash(text: str, snapshot: Any) -> str:
    material = text.strip() + "\n" + _canonical_json(snapshot)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _message_payload(row: Any, snapshot: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": row["id"],
        "role": row["role"],
        "text": row["text"],
        "state": row["state"],
        "createdAt": row["created_at"],
        "errorCode": row["error_code"],
    }
    if row['role'] == 'user':
        payload['idempotencyKey'] = row['idempotency_key']
    if snapshot is not None:
        payload["snapshot"] = snapshot
    return payload


def create_app(
    *,
    db_path: str | Path | None = None,
    max_text_length: int = config.MAX_TEXT_LENGTH,
    max_body_bytes: int = config.MAX_BODY_BYTES,
    max_snapshot_bytes: int = config.MAX_SNAPSHOT_BYTES,
    max_history_messages: int = config.MAX_HISTORY_MESSAGES,
    mcp_fetch: Any = None,
    mcp_server_module: str = "server.mcp_server",
    mcp_max_messages: int = config.MCP_MAX_MESSAGES,
    mcp_max_text_chars: int = config.MCP_MAX_TEXT_CHARS,
    mcp_initialize_timeout: float = config.MCP_INITIALIZE_TIMEOUT_SECONDS,
    mcp_list_tools_timeout: float = config.MCP_LIST_TOOLS_TIMEOUT_SECONDS,
    mcp_call_tool_timeout: float = config.MCP_CALL_TOOL_TIMEOUT_SECONDS,
    mcp_total_timeout: float = config.MCP_TOTAL_TIMEOUT_SECONDS,
    mcp_extra_env: dict[str, str] | None = None,
    deepseek_settings: config.DeepSeekSettings | None = None,
    deepseek_transport: Any = None,
    assistant_settings: config.AssistantSettings | None = None,
    assistant_transport: Any = None,
    assistant_project_root: str | Path | None = None,
) -> FastAPI:
    database = Database(db_path or config.db_path_from_env())
    settings = deepseek_settings or config.DeepSeekSettings.from_env()
    assistant_conf = assistant_settings or config.AssistantSettings()
    assistant_root = Path(assistant_project_root) if assistant_project_root is not None else None
    workers = set()

    def message_payload(row, snapshot=None):
        payload = _message_payload(row, snapshot)
        if row["role"] == "assistant":
            stored_retrieval = database.retrieval_for(row["id"])
            if stored_retrieval is not None:
                selected = database.selected_sources_for(row['id'])
                payload['sourcesLegacy'] = selected is None
                payload["sources"] = [
                    {"id": source["id"], "title": source["title"],
                     "headings": source["headings"], "text": source["text"],
                     "url": f"/api/support/conversations/{row['conversation_id']}/messages/{row['id']}/sources/{source['id']}"}
                    for source in stored_retrieval["sources"]
                    if selected is None or source['id'] in selected
                ]
        return payload

    async def _obtain_context(conversation_id: int, message_id: int) -> dict[str, Any]:
        """Получает контекст обращения через MCP дочерним процессом.

        Прямого чтения того же контекста в обход MCP и скрытого fallback здесь
        нет: любой сбой становится ``McpError``. Параметр ``mcp_fetch``
        позволяет тестам подставить отказоустойчивый источник; рабочий путь его
        не задаёт.
        """

        if mcp_fetch is not None:
            return await mcp_fetch(database.path, conversation_id, message_id)
        return await fetch_ticket_context(
            db_path=database.path,
            conversation_id=conversation_id,
            message_id=message_id,
            server_module=mcp_server_module,
            extra_env=mcp_extra_env,
            max_messages=min(mcp_max_messages, settings.history_messages),
            max_text_chars=min(mcp_max_text_chars, settings.history_chars),
            initialize_timeout=mcp_initialize_timeout,
            list_tools_timeout=mcp_list_tools_timeout,
            call_tool_timeout=mcp_call_tool_timeout,
            total_timeout=mcp_total_timeout,
        )

    async def _run_mcp_and_store(conversation_id: int, message_id: int):
        # Called directly after the user transaction (also for an explicit retry).
        started = time.monotonic()
        attempt_id = database.start_attempt(message_id, settings.model)
        lease = generation_gate.admit()
        external_complete = False
        stage = 'context'

        def elapsed():
            return round((time.monotonic() - started) * 1000)

        try:
            if lease is None:
                raise SupportError('generation_busy', 'Сервис занят другим ответом. Повторите явно позже.', 409)
            async with asyncio.timeout_at(started + settings.total_seconds):
                context = await _obtain_context(conversation_id, message_id)
                if context['messageId'] != message_id or context['conversationId'] != conversation_id:
                    raise McpError('Context binding mismatch')
                stage = 'storage'
                database.update_attempt(attempt_id, context_binding={
                    'messageId': message_id, 'historyIds': [m['id'] for m in context['history']],
                    'historyTruncated': context['historyTruncated'],
                    'historyOmittedCount': context['historyOmittedCount'],
                    'snapshotReceivedAt': context['snapshotReceivedAt']})
                stage = 'retrieval'
                worker = lease.submit(retrieval_index.search, context)
                workers.add(worker)
                worker.add_done_callback(workers.discard)
                # Cancelling the await never releases admission before the worker ends.
                waiting = asyncio.wrap_future(worker)
                waiting.add_done_callback(lambda f: None if f.cancelled() else f.exception())
                found = await asyncio.shield(waiting)
                stage = 'storage'
                database.update_attempt(attempt_id, retrieval=found, index_version=found['indexFingerprint'])
                stage = 'preparation'
                prepared = responder.prepare(context, found, settings)
                stage = 'storage'
                database.update_attempt(attempt_id, context_binding=prepared.binding,
                    messages_bytes=prepared.byte_count,
                    estimated_tokens=prepared.estimated_tokens)
                if not settings.api_key:
                    raise SupportError('provider_key_missing', 'Ключ DeepSeek не настроен на сервере. Сообщение сохранено; после настройки повторите явно.', 503)
                # No await between the durable debit and starting the single HTTP attempt.
                database.begin_external(attempt_id, settings.session_attempts, settings.session_window_seconds,
                                        passed_ids=prepared.source_ids)
                stage = 'provider'
                body = await responder.request_completion(prepared, settings, transport=deepseek_transport)
                external_complete = True
                stage = 'storage'
                observed = responder.metadata(body)
                if observed:
                    database.update_attempt(attempt_id, **observed)
                stage = 'validation'
                answer, ids = responder.validate(body, prepared.source_ids, settings)
                stage = 'storage'
                return database.create_assistant_reply(
                    conversation_id=conversation_id, user_message_id=message_id,
                    text=answer, retrieval=found, attempt_id=attempt_id,
                    selected_ids=ids, duration_ms=elapsed())
        except asyncio.CancelledError:
            database.fail_attempt(attempt_id, 'cancelled', elapsed(), unknown=not external_complete)
            raise
        except TimeoutError:
            database.fail_attempt(attempt_id, 'processing_timeout', elapsed(), unknown=not external_complete)
            raise SupportError('processing_timeout', 'Истекло общее время обработки. Внешний исход может быть неизвестен; повторите явно.', 504) from None
        except McpError:
            database.fail_attempt(attempt_id, 'mcp_error', elapsed())
            raise SupportError('mcp_error', 'Не удалось получить контекст обращения через MCP.', 502) from None
        except SupportError as exc:
            database.fail_attempt(attempt_id, exc.code, elapsed(),
                                  unknown=exc.code in {'provider_network', 'provider_timeout'})
            raise
        except Exception:
            code, message = {
                'context': ('mcp_error', 'Не удалось получить контекст обращения через MCP.'),
                'retrieval': ('retrieval_failed', 'Не удалось выполнить локальный поиск.'),
                'preparation': ('context_invalid', 'Не удалось подготовить контекст текущего вопроса.'),
                'provider': ('provider_failed', 'Не удалось завершить внешний вызов. Его исход может быть неизвестен.'),
                'validation': ('provider_contract', 'AI вернул некорректный ответ.'),
                'storage': ('storage_failed', 'Не удалось сохранить результат обработки.'),
            }[stage]
            # fail_attempt also requires the durable external_started_at marker:
            # a preflight failure never becomes an uncertain external attempt.
            database.fail_attempt(attempt_id, code, elapsed(), unknown=not external_complete)
            raise SupportError(code, message + ' Повторите явно.', 503) from None
        finally:
            if lease is not None:
                lease.close()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Незавершённая обработка после перезапуска не остаётся в вечном
        # ожидании: сообщение получает явное состояние ошибки.
        database.recover_processing()
        try:
            await asyncio.to_thread(retrieval_index.validate_startup)
            app.state.retrieval_error = None
        except SupportError as exc:
            # History remains available; each new attempt checks again.
            app.state.retrieval_error = exc.code
        try:
            yield
        finally:
            # Drain owned CPU futures on shutdown; their late results never call API.
            if workers:
                await asyncio.gather(*(asyncio.shield(asyncio.wrap_future(w)) for w in list(workers)), return_exceptions=True)

    app = FastAPI(
        title="Guitar Trainer Support",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.database = database

    @app.exception_handler(SupportError)
    async def _support_error_handler(_request: Request, exc: SupportError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    # --- сессия и защита изменяющих запросов ------------------------------

    def _current_session(request: Request) -> str | None:
        session_id = request.cookies.get(config.SESSION_COOKIE)
        if session_id and len(session_id) <= config.MAX_SESSION_ID_LENGTH:
            return session_id
        return None

    def _session_id(request: Request, response: Response) -> str:
        session_id = _current_session(request)
        if session_id is not None:
            return session_id
        session_id = secrets.token_urlsafe(32)
        response.set_cookie(
            config.SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="strict",
            path="/",
        )
        return session_id

    def _require_safe_request(request: Request) -> None:
        if request.headers.get(config.CSRF_HEADER) != config.CSRF_HEADER_VALUE:
            raise SupportError(
                "missing_request_header",
                "Изменяющий запрос должен содержать служебный заголовок.",
                403,
            )
        origin = request.headers.get("origin")
        if origin:
            host = request.headers.get("host", "")
            if urlparse(origin).netloc != host:
                raise SupportError("invalid_origin", "Недопустимый Origin запроса.", 403)

    def _owned_conversation(conversation_id: int, session_id: str):
        conversation = database.conversation_for_session(conversation_id, session_id)
        if conversation is None:
            # Чужой или несуществующий ID не даёт доступа и не различается.
            raise SupportError("not_found", "Обращение не найдено.", 404)
        return conversation

    # --- статика -----------------------------------------------------------

    def _static_endpoint(relative: str, media_type: str):
        # Статика не кэшируется: после применения изменений ассистентом
        # браузер не должен получать устаревшие JS/CSS.
        headers = {"Cache-Control": "no-store"}

        def endpoint() -> FileResponse:
            return FileResponse(config.WEB_DIR / relative, media_type=media_type, headers=headers)

        return endpoint

    for url_path, (relative, media_type) in STATIC_FILES.items():
        app.add_api_route(
            url_path,
            _static_endpoint(relative, media_type),
            methods=["GET"],
            include_in_schema=False,
        )

    # --- API поддержки ------------------------------------------------------

    @app.get("/api/support/session")
    def get_session(request: Request, response: Response) -> dict[str, Any]:
        session_id = _session_id(request, response)
        conversation = database.latest_conversation(session_id)
        return {
            "sessionId": session_id,
            "currentConversationId": conversation["id"] if conversation else None,
            "testMode": False,
        }

    @app.post("/api/support/conversations")
    def create_conversation(request: Request, response: Response) -> dict[str, Any]:
        _require_safe_request(request)
        session_id = _session_id(request, response)
        conversation = database.create_conversation(session_id)
        return {"conversationId": conversation["id"], "testMode": False}

    @app.get("/api/support/conversations/{conversation_id}/messages")
    def list_messages(
        conversation_id: int, request: Request
    ) -> dict[str, Any]:
        session_id = _current_session(request) or ""
        _owned_conversation(conversation_id, session_id)
        messages = []
        for row in database.messages(conversation_id):
            snapshot = None
            if row["role"] == "user":
                stored = database.snapshot_for(row["id"])
                if stored is not None:
                    snapshot = json.loads(stored["payload"])
            messages.append(message_payload(row, snapshot))
        return {
            "conversationId": conversation_id,
            "messages": messages,
            "testMode": False,
        }

    @app.get('/api/support/conversations/{conversation_id}/messages/{message_id}/sources/{source_id}')
    def saved_source(conversation_id: int, message_id: int, source_id: str, request: Request):
        _owned_conversation(conversation_id, _current_session(request) or '')
        row = database.get_message(message_id)
        if row is not None and row['conversation_id'] == conversation_id and row['role'] == 'assistant':
            for source in message_payload(row).get('sources', []):
                if source['id'] == source_id:
                    return PlainTextResponse(source['title'] + '\n' + ' / '.join(source['headings']) + '\n\n' + source['text'],
                                             headers={'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store'})
        raise SupportError('not_found', 'Материал не найден.', 404)

    @app.post("/api/support/conversations/{conversation_id}/messages")
    async def send_message(
        conversation_id: int, request: Request, response: Response
    ) -> dict[str, Any]:
        _require_safe_request(request)
        session_id = _session_id(request, response)
        _owned_conversation(conversation_id, session_id)

        raw = await request.body()
        if len(raw) > max_body_bytes:
            raise SupportError(
                "body_too_large", "Тело запроса превышает допустимый размер.", 413
            )
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SupportError("invalid_request", "Тело запроса должно быть JSON-объектом.")
        if not isinstance(data, dict):
            raise SupportError("invalid_request", "Тело запроса должно быть JSON-объектом.")

        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SupportError("invalid_text", "Текст сообщения не должен быть пустым.")
        try:
            text.encode('utf-8')
        except UnicodeError:
            raise SupportError('invalid_text', 'Текст содержит некорректные символы.') from None
        if len(text) > max_text_length:
            raise SupportError(
                "text_too_long", "Текст сообщения превышает допустимую длину.", 413
            )

        idempotency_key = data.get("idempotencyKey")
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > config.MAX_IDEMPOTENCY_KEY_LENGTH
        ):
            raise SupportError(
                "invalid_idempotency_key",
                "Ключ идемпотентности обязателен и ограничен по длине.",
            )

        if "snapshot" not in data:
            raise SupportError(
                "invalid_snapshot", "Диагностический снимок обязателен."
            )
        snapshot = data["snapshot"]
        try:
            snapshot_size = len(_canonical_json(snapshot).encode("utf-8"))
        except (ValueError, UnicodeError):
            raise SupportError('invalid_snapshot', 'Снимок содержит некорректные символы.') from None
        if snapshot_size > max_snapshot_bytes:
            raise SupportError(
                "snapshot_too_large",
                "Диагностический снимок превышает допустимый размер.",
                413,
            )
        validate_snapshot(snapshot)

        content_hash = _content_hash(text, snapshot)

        existing = database.find_by_idempotency(conversation_id, idempotency_key)
        if existing is not None:
            if existing["content_hash"] != content_hash:
                raise SupportError(
                    "idempotency_conflict",
                    "Ключ идемпотентности уже использован для другого содержимого.",
                    409,
                )
            reply = database.reply_for(existing["id"])
            stored = database.snapshot_for(existing["id"])
            if reply is not None:
                return {
                    "conversationId": conversation_id,
                    "userMessage": message_payload(
                        existing, json.loads(stored["payload"]) if stored else None
                    ),
                    "assistantMessage": message_payload(reply),
                    "testMode": False,
                }
            if existing["state"] == "processing":
                # Повтор во время активной обработки не запускает вторую.
                raise SupportError(
                    "already_processing",
                    "Предыдущее сообщение ещё обрабатывается.",
                    409,
                )
            # Повтор к сообщению в state=error: MCP и обработчик запускаются
            # заново по исходному сохранённому снимку, без нового сообщения,
            # нового снимка и дублей.
            database.mark_message_processing(existing["id"])
            assistant_row = await _run_mcp_and_store(conversation_id, existing["id"])
            existing = database.get_message(existing["id"]) or existing
            return {
                "conversationId": conversation_id,
                "userMessage": message_payload(
                    existing, json.loads(stored["payload"]) if stored else None
                ),
                "assistantMessage": message_payload(assistant_row),
                "testMode": False,
            }

        user_row = database.create_user_message(
            conversation_id=conversation_id,
            text=text,
            idempotency_key=idempotency_key,
            content_hash=content_hash,
            snapshot_payload=_canonical_json(snapshot),
            max_history=max_history_messages,
        )
        # MCP и обработчик запускаются после завершения транзакции записи.
        assistant_row = await _run_mcp_and_store(conversation_id, user_row["id"])
        # Перечитываем вопрос: отдельная транзакция ответа перевела его в answered.
        user_row = database.get_message(user_row["id"]) or user_row
        return {
            "conversationId": conversation_id,
            "userMessage": message_payload(user_row, snapshot),
            "assistantMessage": message_payload(assistant_row),
            "testMode": False,
        }

    # --- веб-ассистент разработки -----------------------------------------

    def _assistant_payload(row: Any) -> dict[str, Any]:
        stored: dict[str, Any] = {}
        if row["payload"]:
            try:
                stored = json.loads(row["payload"])
            except (ValueError, TypeError):
                stored = {}
        payload: dict[str, Any] = {
            "id": row["id"],
            "role": row["role"],
            "text": row["text"],
            "state": row["state"],
            "createdAt": row["created_at"],
            "errorCode": row["error_code"],
            "result": stored,
        }
        if row["role"] == "user":
            payload["idempotencyKey"] = row["idempotency_key"]
        return payload

    def _owned_assistant_conversation(conversation_id: int, session_id: str):
        conversation = database.assistant_conversation_for_session(
            conversation_id, session_id
        )
        if conversation is None:
            raise SupportError("not_found", "Обращение ассистента не найдено.", 404)
        return conversation

    def _assistant_error_payload(code: str, message: str) -> str:
        return json.dumps(
            {
                "status": "error",
                "summary": message,
                "errorCode": code,
                "changedPaths": [],
                "diff": "",
                "checks": None,
                "applied": False,
            },
            ensure_ascii=False,
        )

    async def _run_assistant(conversation_id: int, user_message_id: int, goal: str) -> dict:
        if not settings.api_key:
            raise SupportError(
                "provider_key_missing",
                "Ключ DeepSeek не настроен на сервере. Сообщение сохранено; после настройки повторите явно.",
                503,
            )
        with assistant_agent.ADMISSION.hold():
            staging = assistant_staging.Staging.create(assistant_root)
            try:
                tools = assistant_tools.AssistantTools(
                    staging, settings=assistant_conf, project_root=assistant_root
                )
                return await assistant_agent.run_agent(
                    goal,
                    settings=assistant_conf,
                    api_key=settings.api_key,
                    tools=tools,
                    transport=assistant_transport,
                )
            finally:
                staging.cleanup()

    @app.get("/api/assistant/session")
    def get_assistant_session(request: Request, response: Response) -> dict[str, Any]:
        session_id = _session_id(request, response)
        conversation = database.latest_assistant_conversation(session_id)
        return {
            "sessionId": session_id,
            "currentConversationId": conversation["id"] if conversation else None,
            "testMode": False,
        }

    @app.post("/api/assistant/conversations")
    def create_assistant_conversation(request: Request, response: Response) -> dict[str, Any]:
        _require_safe_request(request)
        session_id = _session_id(request, response)
        conversation = database.create_assistant_conversation(session_id)
        return {"conversationId": conversation["id"], "testMode": False}

    @app.get("/api/assistant/conversations/{conversation_id}/messages")
    def list_assistant_messages(conversation_id: int, request: Request) -> dict[str, Any]:
        session_id = _current_session(request) or ""
        _owned_assistant_conversation(conversation_id, session_id)
        return {
            "conversationId": conversation_id,
            "messages": [
                _assistant_payload(row)
                for row in database.assistant_messages(conversation_id)
            ],
            "testMode": False,
        }

    @app.post("/api/assistant/conversations/{conversation_id}/messages")
    async def send_assistant_message(
        conversation_id: int, request: Request, response: Response
    ) -> dict[str, Any]:
        _require_safe_request(request)
        session_id = _session_id(request, response)
        _owned_assistant_conversation(conversation_id, session_id)

        raw = await request.body()
        if len(raw) > max_body_bytes:
            raise SupportError(
                "body_too_large", "Тело запроса превышает допустимый размер.", 413
            )
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SupportError("invalid_request", "Тело запроса должно быть JSON-объектом.")
        if not isinstance(data, dict):
            raise SupportError("invalid_request", "Тело запроса должно быть JSON-объектом.")

        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SupportError("invalid_text", "Текст запроса не должен быть пустым.")
        try:
            text.encode("utf-8")
        except UnicodeError:
            raise SupportError("invalid_text", "Текст содержит некорректные символы.") from None
        if len(text) > assistant_conf.max_goal_chars:
            raise SupportError(
                "text_too_long", "Текст запроса превышает допустимую длину.", 413
            )

        idempotency_key = data.get("idempotencyKey")
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > config.MAX_IDEMPOTENCY_KEY_LENGTH
        ):
            raise SupportError(
                "invalid_idempotency_key",
                "Ключ идемпотентности обязателен и ограничен по длине.",
            )

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        existing = database.find_assistant_by_idempotency(conversation_id, idempotency_key)
        if existing is not None:
            if existing["content_hash"] != content_hash:
                raise SupportError(
                    "idempotency_conflict",
                    "Ключ идемпотентности уже использован для другого запроса.",
                    409,
                )
            if existing["state"] == "processing":
                raise SupportError(
                    "already_processing",
                    "Предыдущая операция ещё выполняется.",
                    409,
                )
            if existing["state"] == "completed":
                reply = database.assistant_reply_for(existing["id"])
                return {
                    "conversationId": conversation_id,
                    "userMessage": _assistant_payload(existing),
                    "assistantMessage": _assistant_payload(reply) if reply else None,
                    "testMode": False,
                }
            # Явный повтор к завершившейся ошибкой операции.
            database.mark_assistant_processing(existing["id"])
            try:
                result = await _run_assistant(conversation_id, existing["id"], text)
            except SupportError as exc:
                reply = database.fail_assistant_operation(
                    conversation_id=conversation_id,
                    user_message_id=existing["id"],
                    error_code=exc.code,
                    message=exc.message,
                    payload=_assistant_error_payload(exc.code, exc.message),
                )
                raise
            reply = database.complete_assistant_operation(
                conversation_id=conversation_id,
                user_message_id=existing["id"],
                payload=json.dumps(result, ensure_ascii=False),
                text=result.get("summary") or "Операция завершена.",
            )
            existing = database.get_assistant_message(existing["id"]) or existing
            return {
                "conversationId": conversation_id,
                "userMessage": _assistant_payload(existing),
                "assistantMessage": _assistant_payload(reply),
                "testMode": False,
            }

        user_row = database.create_assistant_user_message(
            conversation_id=conversation_id,
            text=text,
            idempotency_key=idempotency_key,
            content_hash=content_hash,
            max_history=max_history_messages,
        )
        try:
            result = await _run_assistant(conversation_id, user_row["id"], text)
        except SupportError as exc:
            database.fail_assistant_operation(
                conversation_id=conversation_id,
                user_message_id=user_row["id"],
                error_code=exc.code,
                message=exc.message,
                payload=_assistant_error_payload(exc.code, exc.message),
            )
            raise
        reply = database.complete_assistant_operation(
            conversation_id=conversation_id,
            user_message_id=user_row["id"],
            payload=json.dumps(result, ensure_ascii=False),
            text=result.get("summary") or "Операция завершена.",
        )
        user_row = database.get_assistant_message(user_row["id"]) or user_row
        return {
            "conversationId": conversation_id,
            "userMessage": _assistant_payload(user_row),
            "assistantMessage": _assistant_payload(reply),
            "testMode": False,
        }

    return app
