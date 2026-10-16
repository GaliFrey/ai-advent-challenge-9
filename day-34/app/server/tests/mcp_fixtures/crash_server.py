"""Тестовый MCP-сервер, аварийно завершающийся до рукопожатия (C7/C9)."""

from __future__ import annotations

import sys


def main() -> None:
    sys.stderr.write("test crash server exiting\n")
    raise SystemExit(3)


if __name__ == "__main__":
    main()
