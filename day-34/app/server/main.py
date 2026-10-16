"""Точка входа локального сервера поддержки.

Запуск (слушает только loopback):

    uv run python -m server.main

Порт по умолчанию 8000 переопределяется переменной GUITAR_TRAINER_PORT.
"""

from __future__ import annotations

from .app import create_app
from .config import DEFAULT_HOST, port_from_env, load_project_env

load_project_env()
app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=DEFAULT_HOST, port=port_from_env(), log_level="info")


if __name__ == "__main__":
    main()
