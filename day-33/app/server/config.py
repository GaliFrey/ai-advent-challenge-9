"""Конфигурация локального сервера поддержки.

Все значения по умолчанию рассчитаны на работу только по loopback-адресу.
Путь к базе данных и порт можно переопределить переменными окружения
``GUITAR_TRAINER_DB`` и ``GUITAR_TRAINER_PORT``.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

# Корень репозитория: файл лежит в <root>/server/config.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"

# Сервер слушает только loopback. Широковещательные адреса не поддерживаются.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "support.db"

# Лимиты входных данных. Превышение любого из них даёт понятную ошибку без
# частичной записи.
MAX_TEXT_LENGTH = 4000
MAX_BODY_BYTES = 64 * 1024
MAX_SNAPSHOT_BYTES = 16 * 1024
MAX_HISTORY_MESSAGES = 200
MAX_IDEMPOTENCY_KEY_LENGTH = 200
MAX_SESSION_ID_LENGTH = 200

# Имя сессионной cookie и заголовка защиты изменяющих запросов от CSRF.
SESSION_COOKIE = "gt_session"
CSRF_HEADER = "X-Support-Request"
CSRF_HEADER_VALUE = "1"

# Лимиты контекста, который read-only инструмент get_ticket_context отдаёт
# обработчику. Лимит истории совместный: по числу сообщений и по суммарному
# объёму текста.
MCP_MAX_MESSAGES = 20
MCP_MAX_TEXT_CHARS = 16000

# Таймауты stdio-цикла MCP (initialize, tools/list, tools/call) и всей операции.
MCP_INITIALIZE_TIMEOUT_SECONDS = 5.0
MCP_LIST_TOOLS_TIMEOUT_SECONDS = 5.0
MCP_CALL_TOOL_TIMEOUT_SECONDS = 10.0
MCP_TOTAL_TIMEOUT_SECONDS = 20.0


def db_path_from_env() -> Path:
    """Путь к базе данных: переменная окружения или значение по умолчанию."""

    raw = os.environ.get("GUITAR_TRAINER_DB")
    return Path(raw) if raw else DEFAULT_DB_PATH


def port_from_env() -> int:
    """Порт сервера: переменная окружения или значение по умолчанию."""

    raw = os.environ.get("GUITAR_TRAINER_PORT")
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        return DEFAULT_PORT
    if 1 <= port <= 65535:
        return port
    return DEFAULT_PORT


# .env is read only by the explicit runtime entry point, never by MCP.
def load_project_env() -> None:
    """Small KEY=value project-only loader; process environment takes priority.

    No expansion, execution, global search or logging of values. Unsupported
    lines fail closed. Only backend configuration names are accepted.
    """
    path = PROJECT_ROOT / '.env'
    if not path.is_file():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, separator, value = line.partition('=')
        name, value = name.strip(), value.strip()
        if not separator or not (name.startswith('DEEPSEEK_') or name in {'GUITAR_TRAINER_DB', 'GUITAR_TRAINER_PORT'}):
            raise ValueError('Неподдерживаемая строка проектного .env')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


@dataclass(frozen=True)
class DeepSeekSettings:
    api_key: str = field(default='', repr=False)
    model: str = 'deepseek-flash'
    history_messages: int = 20
    history_chars: int = 16000
    rag_fragments: int = 3
    rag_chars: int = 12000
    messages_bytes: int = 65536
    max_tokens: int = 1200
    answer_chars: int = 4000
    session_attempts: int = 10
    session_window_seconds: int = 3600
    connect_seconds: float = 5
    read_seconds: float = 45
    write_seconds: float = 10
    pool_seconds: float = 5
    total_seconds: float = 90

    def __post_init__(self):
        if not isinstance(self.api_key, str) or len(self.api_key) > 4096 or any(not 33 <= ord(c) <= 126 for c in self.api_key):
            raise ValueError('Недопустимый формат DEEPSEEK_API_KEY')
        if self.model != 'deepseek-flash':
            raise ValueError('DEEPSEEK_MODEL должен быть deepseek-flash')
        # Upper bounds retain the agreed data minimization and cost envelope.
        caps = {'history_messages': 20, 'history_chars': 16000, 'rag_fragments': 3,
                'rag_chars': 12000, 'messages_bytes': 65536, 'max_tokens': 1200,
                'answer_chars': 4000, 'session_attempts': 1000,
                'session_window_seconds': 86400, 'connect_seconds': 90,
                'read_seconds': 90, 'write_seconds': 90, 'pool_seconds': 90,
                'total_seconds': 90}
        for name, cap in caps.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= cap:
                raise ValueError(f'Недопустимая настройка DEEPSEEK_{name.upper()}')
        if any(not isinstance(getattr(self, n), int) for n in caps if not n.endswith('_seconds') or n == 'session_window_seconds'):
            raise ValueError('Лимиты должны быть целыми числами')

    @classmethod
    def from_env(cls):
        values = {}
        for name in cls.__dataclass_fields__:
            raw = os.environ.get('DEEPSEEK_' + name.upper())
            if raw is None:
                continue
            if name in {'api_key', 'model'}:
                values[name] = raw
            else:
                try:
                    values[name] = float(raw) if name.endswith('_seconds') and name != 'session_window_seconds' else int(raw)
                except ValueError:
                    raise ValueError(f'Недопустимая настройка DEEPSEEK_{name.upper()}') from None
        return cls(**values)
