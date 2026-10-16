"""Тестовый сервер, пишущий в stdout посторонние строки вместо протокола (C9).

Показывает, что посторонний вывод в stdout ломает stdio-протокол: рабочий сервер
обязан направлять диагностику только в stderr.
"""

from __future__ import annotations

import sys


def main() -> None:
    for _ in range(3):
        sys.stdout.write("это не JSON-RPC\n")
    sys.stdout.flush()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
