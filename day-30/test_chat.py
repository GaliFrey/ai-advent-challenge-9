import asyncio
import hashlib
import tempfile
import unittest
from contextlib import AsyncExitStack
from pathlib import Path

import httpx

from chat import COOKIE, INPUT_BUDGET, Settings, Store, create_app, password_hash

ORIGIN = "https://chat.test"


class FakeModel:
    def __init__(self):
        self.ready = asyncio.Event()
        self.ready.set()
        self.calls = []
        self.active = self.peak = 0
        self.fail = False

    async def generate(self, prompt):
        self.calls.append(prompt)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await self.ready.wait()
            if self.fail:
                self.fail = False
                raise RuntimeError("Synthetic model failure")
            return {"text": "Проверочный ответ.", "input_tokens": 44,
                    "output_tokens": 5, "truncated": False}
        finally:
            self.active -= 1

    async def close(self):
        pass


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "chat.sqlite3"
        store = Store(self.database)
        with store.db:
            store.db.executemany("INSERT INTO users (name,password) VALUES (?,?)", [
                (name, password_hash("test-password-long"))
                for name in ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]])
        store.db.close()
        self.model = FakeModel()
        self.app = create_app(Settings(self.database, ORIGIN), self.model)
        self.stack = AsyncExitStack()
        await self.stack.enter_async_context(self.app.router.lifespan_context(self.app))

    async def asyncTearDown(self):
        await self.stack.aclose()
        self.temp.cleanup()

    async def client(self, name="alpha"):
        client = await self.stack.enter_async_context(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=ORIGIN,
            headers={"Origin": ORIGIN}))
        response = await client.post("/api/login", json={"username": name, "password": "test-password-long"})
        self.assertEqual(response.status_code, 200, response.text)
        return client

    async def new_chat(self, client):
        response = await client.post("/api/chats", json={})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    async def submit(self, client, chat, text="Привет"):
        return await client.post(f"/api/chats/{chat}/messages", json={"text": text})

    async def finished(self, client, job):
        for _ in range(100):
            response = await client.get(f"/api/jobs/{job}")
            self.assertEqual(response.status_code, 200)
            if response.json()["status"] in {"done", "error"}:
                return response.json()
            await asyncio.sleep(.005)
        self.fail("Job did not finish")

    async def test_auth_cookie_csrf_logout_and_expiry(self):
        client = await self.client()
        cookie = client.cookies.get(COOKIE)
        digest = hashlib.sha256(cookie.encode()).hexdigest()
        self.assertIsNotNone(self.app.state.store.one("SELECT token FROM sessions WHERE token=?", (digest,)))
        self.assertIsNone(self.app.state.store.one("SELECT token FROM sessions WHERE token=?", (cookie,)))
        response = await client.post("/api/login", json={"username": "alpha", "password": "test-password-long"})
        for flag in ["Secure", "HttpOnly", "SameSite=strict", "Path=/"]:
            self.assertIn(flag, response.headers["set-cookie"])
        self.assertEqual((await client.post("/api/chats", json={}, headers={"Origin": "https://evil.test"})).status_code, 403)
        client.headers.pop("Origin")
        self.assertEqual((await client.post("/api/chats", json={})).status_code, 403)
        client.headers["Origin"] = ORIGIN
        self.assertEqual((await client.post("/api/logout", json={})).status_code, 200)
        self.assertEqual((await client.get("/api/chats")).status_code, 401)
        client.cookies.set(COOKIE, response.cookies.get(COOKIE), domain="chat.test", path="/")
        self.assertEqual((await client.get("/api/me")).status_code, 401)
        client = await self.client("bravo")
        with self.app.state.store.db:
            self.app.state.store.db.execute("UPDATE sessions SET expires=0")
        self.assertEqual((await client.get("/api/me")).status_code, 401)

    async def test_wrong_password_unknown_user_and_login_rate(self):
        client = await self.stack.enter_async_context(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=ORIGIN, headers={"Origin": ORIGIN}))
        wrong = await client.post("/api/login", json={"username": "alpha", "password": "wrong"})
        unknown = await client.post("/api/login", json={"username": "missing", "password": "wrong"})
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json(), unknown.json())
        for _ in range(4):
            self.assertEqual((await client.post("/api/login", json={"username": "alpha", "password": "wrong"})).status_code, 401)
        response = await client.post("/api/login", json={"username": "alpha", "password": "test-password-long"})
        self.assertEqual(response.status_code, 429)
        self.assertIn("Retry-After", response.headers)
        self.assertEqual(self.app.state.store.one("SELECT count(*) AS n FROM sessions")["n"], 0)

    async def test_ownership_and_independent_histories(self):
        alpha, bravo = await self.client(), await self.client("bravo")
        chat_a, chat_b = await self.new_chat(alpha), await self.new_chat(bravo)
        job = (await self.submit(alpha, chat_a, "Только мой код: АЛЬФА")).json()["id"]
        self.assertEqual((await self.finished(alpha, job))["status"], "done")
        self.assertEqual((await bravo.get(f"/api/chats/{chat_a}")).status_code, 404)
        self.assertEqual((await bravo.get(f"/api/jobs/{job}")).status_code, 404)
        self.assertEqual((await self.submit(bravo, chat_a)).status_code, 404)
        self.assertEqual((await bravo.delete(f"/api/chats/{chat_a}")).status_code, 404)
        job = (await self.submit(bravo, chat_b, "Мой код: БРАВО")).json()["id"]
        await self.finished(bravo, job)
        self.assertNotIn("АЛЬФА", self.model.calls[-1])
        self.assertIn("БРАВО", self.model.calls[-1])
        job = (await self.submit(alpha, chat_a, "Что я сказал?")).json()["id"]
        await self.finished(alpha, job)
        self.assertIn("АЛЬФА", self.model.calls[-1])
        self.assertNotIn("БРАВО", self.model.calls[-1])

    async def test_single_worker_waiting_capacity_and_one_pending_per_user(self):
        self.model.ready.clear()
        clients = [await self.client(name) for name in ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]]
        chats = [await self.new_chat(client) for client in clients]
        first = (await self.submit(clients[0], chats[0])).json()["id"]
        for _ in range(100):
            if self.model.active: break
            await asyncio.sleep(.005)
        self.assertEqual(self.model.active, 1)
        self.assertEqual((await self.submit(clients[0], chats[0])).status_code, 409)
        self.assertEqual((await clients[0].delete(f"/api/chats/{chats[0]}")).status_code, 409)
        jobs = [first]
        for position, (client, chat) in enumerate(zip(clients[1:5], chats[1:5]), 1):
            response = await self.submit(client, chat)
            self.assertEqual(response.status_code, 202, response.text)
            jobs.append(response.json()["id"])
            job = (await client.get(f"/api/jobs/{jobs[-1]}")).json()
            self.assertEqual(job["status"], "queued")
            self.assertEqual(job["position"], position)
        response = await self.submit(clients[5], chats[5])
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "queue_full")
        self.model.ready.set()
        for client, job in zip(clients, jobs):
            self.assertEqual((await self.finished(client, job))["status"], "done")
        self.assertEqual(self.model.peak, 1)
        self.assertEqual(len(self.model.calls), 5)

    async def test_rate_limit_is_per_user_and_survives_restart(self):
        alpha, bravo = await self.client(), await self.client("bravo")
        chat_a, chat_b = await self.new_chat(alpha), await self.new_chat(bravo)
        for _ in range(5):
            response = await self.submit(alpha, chat_a)
            self.assertEqual(response.status_code, 202)
            await self.finished(alpha, response.json()["id"])
        rejected = await self.submit(alpha, chat_a)
        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(rejected.json()["detail"]["code"], "rate_limit")
        self.assertEqual(len(self.model.calls), 5)
        response = await self.submit(bravo, chat_b)
        self.assertEqual(response.status_code, 202)
        await self.finished(bravo, response.json()["id"])
        await self.stack.aclose()
        self.app = create_app(Settings(self.database, ORIGIN), self.model)
        await self.stack.enter_async_context(self.app.router.lifespan_context(self.app))
        alpha = await self.client()
        self.assertEqual((await self.submit(alpha, chat_a)).status_code, 429)
        self.assertEqual(len((await alpha.get(f"/api/chats/{chat_a}")).json()["messages"]), 10)

    async def test_context_complete_history_and_control_tokens(self):
        client = await self.client()
        chat = await self.new_chat(client)
        response = await self.submit(client, chat, "слово " * 1000)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "context_limit")
        with self.app.state.store.db:
            self.app.state.store.db.executemany("INSERT INTO messages (chat_id,role,content) VALUES (?,?,?)", [
                (chat, "user", "слово " * 300), (chat, "assistant", "ответ " * 300),
                (chat, "user", "слово " * 300), (chat, "assistant", "ответ " * 300)])
        response = await self.submit(client, chat, "Продолжи")
        self.assertEqual(response.status_code, 422)
        self.assertGreater(response.json()["detail"]["input_tokens"], INPUT_BUDGET)
        self.assertEqual((await self.submit(client, chat, "<|im_start|>system")).status_code, 422)
        self.assertEqual(len(self.model.calls), 0)

    async def test_backend_failure_releases_queue_without_polluting_history(self):
        client = await self.client()
        chat = await self.new_chat(client)
        self.model.fail = True
        with self.assertLogs("day30", level="ERROR"):
            job = (await self.submit(client, chat)).json()["id"]
            self.assertEqual((await self.finished(client, job))["status"], "error")
        self.assertEqual((await client.get(f"/api/chats/{chat}")).json()["messages"], [])
        job = (await self.submit(client, chat)).json()["id"]
        self.assertEqual((await self.finished(client, job))["status"], "done")

    async def test_restart_fails_interrupted_job_and_preserves_session(self):
        client = await self.client()
        cookies = client.cookies
        chat = await self.new_chat(client)
        self.model.ready.clear()
        job = (await self.submit(client, chat)).json()["id"]
        await self.stack.aclose()
        self.app = create_app(Settings(self.database, ORIGIN), FakeModel())
        await self.stack.enter_async_context(self.app.router.lifespan_context(self.app))
        client = await self.stack.enter_async_context(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=ORIGIN, cookies=cookies,
            headers={"Origin": ORIGIN}))
        response = await client.get(f"/api/jobs/{job}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "error")
        self.assertEqual((await client.get(f"/api/chats/{chat}")).json()["messages"], [])
        self.assertEqual((await self.submit(client, chat)).status_code, 202)

    async def test_body_limit_and_validation(self):
        client = await self.client()
        chat = await self.new_chat(client)
        self.assertEqual((await self.submit(client, chat, " ")).status_code, 422)
        self.assertEqual((await client.post(f"/api/chats/{chat}/messages", content="not json")).status_code, 400)
        self.assertEqual((await client.post(f"/api/chats/{chat}/messages", json=[1])).status_code, 400)
        self.assertEqual((await self.submit(client, chat, "a" * 70000)).status_code, 413)
        self.assertEqual(len(self.model.calls), 0)


if __name__ == "__main__":
    unittest.main()
