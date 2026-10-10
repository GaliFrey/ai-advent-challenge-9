"""Single-worker chat service, persistent ownership and a bounded generation queue."""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import CONTEXT, GUARD, MODEL, OUTPUT, Ollama, PromptBuilder

ROOT = Path(__file__).resolve().parent
COOKIE = "__Host-day30"
SESSION_SECONDS = 8 * 3600
WAITING = 4
RATE = 5
INPUT_BUDGET = CONTEXT - OUTPUT - GUARD
logger = logging.getLogger("day30")


@dataclass
class Settings:
    database: Path
    origin: str
    tokenizer: Path = ROOT / "tokenizer/tokenizer.json"
    ollama_url: str = "http://127.0.0.1:11434"

    @classmethod
    def environment(cls):
        origin = os.environ["PUBLIC_ORIGIN"].rstrip("/")
        if not origin.startswith("https://") or "/" in origin[8:]:
            raise ValueError("PUBLIC_ORIGIN must be an HTTPS origin")
        return cls(Path(os.environ.get("DATABASE_PATH", "chat.sqlite3")), origin)


def password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    value = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
    return salt.hex() + ":" + value.hex()


def password_matches(password: str, encoded: str) -> bool:
    salt, _ = encoded.split(":")
    return hmac.compare_digest(password_hash(password, bytes.fromhex(salt)), encoded)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, password TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, chat_id TEXT REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                chat_id TEXT REFERENCES chats(id) ON DELETE CASCADE,
                status TEXT NOT NULL, text TEXT NOT NULL, prompt TEXT NOT NULL,
                tokens INTEGER NOT NULL, created REAL NOT NULL, started REAL, finished REAL,
                result TEXT, error TEXT);
            CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created);
            CREATE TABLE IF NOT EXISTS rates (scope TEXT NOT NULL, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS rate_scope ON rates(scope, created);
        """)

    def one(self, sql, args=()):
        return self.db.execute(sql, args).fetchone()

    def all(self, sql, args=()):
        return self.db.execute(sql, args).fetchall()

    def clean(self):
        now = time.time()
        with self.db:
            self.db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
            self.db.execute("DELETE FROM rates WHERE created < ?", (now - 3600,))
            self.db.execute("DELETE FROM jobs WHERE finished < ?", (now - 86400,))

    def rate(self, scope: str, limit: int, consume: bool = True):
        rows = self.all("SELECT created FROM rates WHERE scope=? AND created>? ORDER BY created",
                        (scope, time.time() - 60))
        if len(rows) >= limit:
            retry = max(1, int(rows[0]["created"] + 61 - time.time()))
            raise HTTPException(429, {"code": "rate_limit", "message":
                "Слишком много запросов. Подождите минуту.", "retry_after": retry},
                headers={"Retry-After": str(retry)})
        if consume:
            with self.db:
                self.db.execute("INSERT INTO rates VALUES (?,?)", (scope, time.time()))

    def messages(self, chat_id):
        return [dict(row) for row in self.all(
            "SELECT role,content FROM messages WHERE chat_id=? ORDER BY id", (chat_id,))]

    def own_chat(self, chat_id, user_id):
        row = self.one("SELECT * FROM chats WHERE id=? AND user_id=?", (chat_id, user_id))
        if not row:
            raise HTTPException(404, "Диалог не найден.")
        return row


def fail(code: str, message: str, status=422, **data):
    raise HTTPException(status, {"code": code, "message": message, **data})


def create_app(settings: Settings, generator=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        store = Store(settings.database)
        app.state.store = store
        app.state.builder = PromptBuilder(settings.tokenizer)
        app.state.generator = generator or Ollama(settings.ollama_url)
        app.state.wakeup = asyncio.Event()
        app.state.dummy_hash = await asyncio.to_thread(password_hash, secrets.token_hex(32))
        with store.db:
            store.db.execute("UPDATE jobs SET status='error', error=?, finished=?, prompt='' "
                             "WHERE status IN ('queued','running')",
                             ("Сервер перезапущен. Отправьте сообщение снова.", time.time()))
        store.clean()
        task = asyncio.create_task(worker(app))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await app.state.generator.close()
            store.db.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def boundaries(request, call_next):
        if request.method in {"POST", "DELETE", "PUT", "PATCH"}:
            if request.headers.get("origin") != settings.origin:
                return JSONResponse({"detail": "Недопустимый Origin."}, status_code=403)
            if request.method == "POST":
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 65536:
                        return JSONResponse({"detail": "Запрос слишком большой."}, status_code=413)
                request._body = bytes(body)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    def user(request):
        token = request.cookies.get(COOKIE, "")
        digest = hashlib.sha256(token.encode()).hexdigest()
        row = app.state.store.one(
            "SELECT users.id,users.name FROM sessions JOIN users ON users.id=sessions.user_id "
            "WHERE token=? AND expires>?", (digest, time.time()))
        if not row:
            raise HTTPException(401, "Войдите в свою учётную запись.")
        return row

    async def payload(request):
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            fail("invalid_json", "Некорректный JSON.", 400)
        if not isinstance(body, dict):
            fail("invalid_json", "Ожидается объект JSON.", 400)
        return body

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/login")
    async def login(request: Request, response: Response):
        body = await payload(request)
        name, password = body.get("username"), body.get("password")
        if not isinstance(name, str) or not isinstance(password, str) or len(name) > 64 or len(password) > 256:
            fail("credentials", "Неверный логин или пароль.", 401)
        store = app.state.store
        store.clean()
        store.rate("login-ip:" + request.client.host, 10)
        store.rate("login-user:" + name.casefold(), 5)
        row = store.one("SELECT * FROM users WHERE name=?", (name,))
        matched = await asyncio.to_thread(password_matches, password,
                                         row["password"] if row else app.state.dummy_hash)
        if not row or not matched:
            fail("credentials", "Неверный логин или пароль.", 401)
        token = secrets.token_urlsafe(32)
        with store.db:
            store.db.execute("INSERT INTO sessions VALUES (?,?,?)", (
                hashlib.sha256(token.encode()).hexdigest(), row["id"], time.time() + SESSION_SECONDS))
            store.db.execute("DELETE FROM sessions WHERE user_id=? AND token NOT IN "
                             "(SELECT token FROM sessions WHERE user_id=? ORDER BY expires DESC LIMIT 10)",
                             (row["id"], row["id"]))
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, secure=True,
                            httponly=True, samesite="strict", path="/")
        return {"username": row["name"]}

    @app.post("/api/logout")
    async def logout(request: Request, response: Response):
        digest = hashlib.sha256(request.cookies.get(COOKIE, "").encode()).hexdigest()
        with app.state.store.db:
            app.state.store.db.execute("DELETE FROM sessions WHERE token=?", (digest,))
        response.delete_cookie(COOKIE, secure=True, httponly=True, samesite="strict", path="/")
        return {"ok": True}

    @app.get("/api/me")
    async def me(request: Request):
        current = user(request)
        return {"username": current["name"], "model": MODEL, "limits": {
            "context": CONTEXT, "input": INPUT_BUDGET, "output": OUTPUT,
            "requests_per_minute": RATE, "waiting": WAITING}}

    @app.get("/api/chats")
    async def chats(request: Request):
        current = user(request)
        return [dict(row) for row in app.state.store.all(
            "SELECT id,title,created FROM chats WHERE user_id=? ORDER BY created DESC", (current["id"],))]

    @app.post("/api/chats", status_code=201)
    async def new_chat(request: Request):
        current = user(request)
        store = app.state.store
        if store.one("SELECT count(*) AS n FROM chats WHERE user_id=?", (current["id"],))["n"] >= 20:
            fail("chats_limit", "Можно хранить 20 диалогов. Удалите ненужный.")
        chat_id = uuid4().hex
        with store.db:
            store.db.execute("INSERT INTO chats VALUES (?,?,?,?)", (
                chat_id, current["id"], "Новый диалог", time.time()))
        return {"id": chat_id}

    @app.get("/api/chats/{chat_id}")
    async def get_chat(chat_id: str, request: Request):
        current = user(request)
        store = app.state.store
        chat = store.own_chat(chat_id, current["id"])
        pending = store.one("SELECT id FROM jobs WHERE chat_id=? AND status IN ('queued','running')",
                            (chat_id,))
        return {"id": chat_id, "title": chat["title"], "messages": store.messages(chat_id),
                "pending": job_view(store, pending["id"], current["id"]) if pending else None}

    @app.delete("/api/chats/{chat_id}")
    async def delete_chat(chat_id: str, request: Request):
        current = user(request)
        store = app.state.store
        store.own_chat(chat_id, current["id"])
        if store.one("SELECT id FROM jobs WHERE chat_id=? AND status IN ('queued','running')", (chat_id,)):
            fail("busy", "Дождитесь ответа перед удалением диалога.", 409)
        with store.db:
            store.db.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/messages", status_code=202)
    async def send(chat_id: str, request: Request):
        current = user(request)
        store = app.state.store
        store.own_chat(chat_id, current["id"])
        body = await payload(request)
        text = body.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 8192:
            fail("message", "Введите сообщение длиной от 1 до 8192 символов.")
        text = text.strip()
        try:
            prompt, tokens = app.state.builder.build(store.messages(chat_id), text)
        except ValueError as error:
            fail("control_token", str(error))
        if tokens > INPUT_BUDGET:
            fail("context_limit", "Контекст заполнен. Сократите сообщение или начните новый диалог.",
                 input_tokens=tokens, input_limit=INPUT_BUDGET)
        store.clean()
        with store.db:
            store.db.execute("BEGIN IMMEDIATE")
            if store.one("SELECT id FROM jobs WHERE user_id=? AND status IN ('queued','running')",
                         (current["id"],)):
                fail("busy", "У вас уже есть запрос в обработке.", 409)
            if store.one("SELECT count(*) AS n FROM jobs WHERE status='queued'")["n"] >= WAITING:
                fail("queue_full", "Очередь заполнена. Попробуйте немного позже.", 429)
            store.rate("message:" + str(current["id"]), RATE, consume=False)
            store.db.execute("INSERT INTO rates VALUES (?,?)", (
                "message:" + str(current["id"]), time.time()))
            job_id = uuid4().hex
            store.db.execute("INSERT INTO jobs (id,user_id,chat_id,status,text,prompt,tokens,created) "
                             "VALUES (?,?,?,'queued',?,?,?,?)",
                             (job_id, current["id"], chat_id, text, prompt, tokens, time.time()))
        app.state.wakeup.set()
        return {"id": job_id, "input_tokens": tokens}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        return job_view(app.state.store, job_id, user(request)["id"])

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


def job_view(store, job_id, user_id):
    row = store.one("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user_id))
    if not row:
        raise HTTPException(404, "Запрос не найден.")
    position = 0
    if row["status"] == "queued":
        position = store.one("SELECT count(*) AS n FROM jobs WHERE status='queued' AND rowid <= "
                             "(SELECT rowid FROM jobs WHERE id=?)", (job_id,))["n"]
    return {"id": row["id"], "chat_id": row["chat_id"], "status": row["status"],
            "position": position, "text": row["text"], "input_tokens": row["tokens"],
            "created": row["created"], "started": row["started"], "finished": row["finished"],
            "result": json.loads(row["result"]) if row["result"] else None, "error": row["error"]}


async def worker(app):
    store = app.state.store
    while True:
        app.state.wakeup.clear()
        row = store.one("SELECT * FROM jobs WHERE status='queued' ORDER BY rowid LIMIT 1")
        if not row:
            await app.state.wakeup.wait()
            continue
        if time.time() - row["created"] > 300:
            with store.db:
                store.db.execute("UPDATE jobs SET status='error',error=?,finished=?,prompt='' WHERE id=?",
                                 ("Истекло время ожидания в очереди. Повторите запрос.", time.time(), row["id"]))
            continue
        with store.db:
            store.db.execute("UPDATE jobs SET status='running',started=? WHERE id=?", (time.time(), row["id"]))
        try:
            result = await asyncio.wait_for(app.state.generator.generate(row["prompt"]), timeout=125)
            # Avoid persisting model-generated control tokens into the next prompt.
            app.state.builder.validate(result["text"])
            with store.db:
                store.db.executemany("INSERT INTO messages (chat_id,role,content) VALUES (?,?,?)", [
                    (row["chat_id"], "user", row["text"]), (row["chat_id"], "assistant", result["text"])])
                store.db.execute("UPDATE chats SET title=? WHERE id=? AND title='Новый диалог'",
                                 (row["text"][:48], row["chat_id"]))
                store.db.execute("UPDATE jobs SET status='done',finished=?,result=?,prompt='' WHERE id=?",
                                 (time.time(), json.dumps(result, ensure_ascii=False), row["id"]))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Generation failed for job %s", row["id"])
            with store.db:
                store.db.execute("UPDATE jobs SET status='error',finished=?,error=?,prompt='' WHERE id=?",
                                 (time.time(), "Модель не ответила. Попробуйте ещё раз.", row["id"]))


def app_factory():
    return create_app(Settings.environment())
