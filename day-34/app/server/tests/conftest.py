"""Общие фикстуры серверных тестов T007."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import create_app

# Служебный заголовок, обязательный для изменяющих запросов (защита от CSRF).
CSRF = {"X-Support-Request": "1"}


def make_client(db_path, **kwargs) -> TestClient:
    app = create_app(db_path=db_path, **kwargs)
    return TestClient(app)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "support.db"


@pytest.fixture
def client(db_path):
    with make_client(db_path) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def legacy_chat_retrieval_boundary(request, monkeypatch):
    """T007/T008 regressions isolate the newly added retrieval dependency.

    These tests still run the real stdio MCP. Retrieval itself, persistence and
    the joint production path are covered separately in test_retrieval.py.
    """
    if request.module.__name__ not in {"server.tests.test_support", "server.tests.test_mcp"}:
        return
    from server import retrieval_index

    def found(context):
        return {"version": 1, "indexFingerprint": "regression-fixture", "queryText": "fixture",
                "sources": [{"id": "fixture", "title": "Статья", "headings": ["Статья", "Раздел"],
                             "article": "docs/support/faq.md", "text": "Сохранённый фрагмент",
                             "start": 0, "end": 20, "documentVersion": "fixture", "similarity": 1.0}]}
    monkeypatch.setattr(retrieval_index, "search", found)
    monkeypatch.setattr(retrieval_index, "validate_startup", lambda: None)


MOCK_ANSWER = 'Ответ подменённого DeepSeek для офлайн-регрессии'


@pytest.fixture(autouse=True)
def legacy_chat_provider_boundary(request, monkeypatch):
    """Keep pre-provider regressions offline, exercising preparation/validation/DB.

    T010 tests separately intercept actual httpx wire requests.
    """
    if request.module.__name__ not in {'server.tests.test_support', 'server.tests.test_mcp', 'server.tests.test_retrieval'}:
        return
    import json
    from server import responder
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-regression-key')

    async def completion(prepared, settings, **kwargs):
        return {'model': 'deepseek-flash', 'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(
                    {'answer': MOCK_ANSWER, 'sourceIds': prepared.source_ids})}}]}
    monkeypatch.setattr(responder, 'request_completion', completion)
