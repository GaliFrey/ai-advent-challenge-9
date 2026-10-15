"""Тестовый MCP-сервер, который зависает до цикла initialize (таймаут, C7/C9).

Пишет свой PID в файл из переменной ``MCP_TEST_PIDFILE`` (если задана), чтобы
тест мог убедиться, что процесс завершён. Аргументы запуска не разбираются.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def main() -> None:
    pidfile = os.environ.get("MCP_TEST_PIDFILE")
    if pidfile:
        Path(pidfile).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(120)


if __name__ == "__main__":
    main()
