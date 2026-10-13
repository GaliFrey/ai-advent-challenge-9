"""MCP stdio and DeepSeek clients with bounded calls and sanitized failures."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def project_context(project):
    # MCP needs the interpreter environment but no cloud credentials.
    env = {name: value for name, value in os.environ.items()
           if name in {"PATH", "HOME", "SYSTEMROOT", "LANG", "LC_ALL", "TMPDIR"}}
    env["DAY31_PROJECT"] = str(Path(project).resolve())
    params = StdioServerParameters(command=sys.executable,
                                  args=[str(Path(__file__).with_name("project_mcp.py"))], env=env)
    try:
        async with asyncio.timeout(35):
            with open(os.devnull, "w") as errors:
                async with stdio_client(params, errlog=errors) as (read, write):
                    async with ClientSession(read, write) as session:
                        initialized = await session.initialize()
                        catalog = await session.list_tools()
                        names = {tool.name for tool in catalog.tools}
                        expected = ["get_project_context", "list_project_files"]
                        if not set(expected).issubset(names):
                            raise ValueError("Неполный каталог MCP.")
                        values = []
                        for name in expected:
                            result = await session.call_tool(name, arguments={})
                            if result.isError:
                                raise ValueError("Ошибка инструмента MCP.")
                            blocks = [block.text for block in result.content if block.type == "text"]
                            if len(blocks) != 1:
                                raise ValueError("Неожиданный ответ MCP.")
                            values.append(json.loads(blocks[0]))
                        context, files = values
                        if (not isinstance(context, dict)
                                or not all(isinstance(context.get(key), str) and context[key]
                                           for key in ("branch", "head", "project", "repository"))
                                or not isinstance(files, dict)
                                or not isinstance(files.get("files"), list)
                                or not all(isinstance(path, str) for path in files["files"])):
                            raise ValueError("Неожиданный контракт MCP.")
                        return {"context": context, "files": files,
                                "trace": {"server": initialized.serverInfo.name,
                                          "tools": expected,
                                          "steps": ["initialize", "tools/list", "tools/call", "tools/call"]}}
    except Exception:
        raise RuntimeError("MCP недоступен или вернул ошибку. Проверьте проект и Git.") from None


def complete(messages, key, model, transport=None):
    if not key.strip():
        raise ValueError("Задайте DEEPSEEK_API_KEY в day-31/.env или окружении.")
    started = time.monotonic()
    try:
        with httpx.Client(timeout=90, transport=transport) as client:
            response = client.post("https://api.deepseek.com/chat/completions",
                                   headers={"Authorization": f"Bearer {key}"},
                                   json={"model": model, "messages": messages,
                                         "response_format": {"type": "json_object"},
                                         "thinking": {"type": "disabled"},
                                         "temperature": 0, "max_tokens": 1800})
        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek HTTP {response.status_code}; автоматического повтора нет.")
        data = response.json()
        choice = data["choices"][0]
        if choice["finish_reason"] != "stop":
            raise RuntimeError("Ответ DeepSeek не завершён; он не будет принят.")
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError
        return {"content": content, "model": data.get("model", model),
                "usage": data.get("usage", {}), "finish_reason": "stop",
                "seconds": round(time.monotonic() - started, 3)}
    except httpx.HTTPError:
        raise RuntimeError("Сетевая ошибка DeepSeek; автоматического повтора нет.") from None
    except (ValueError, KeyError, IndexError, TypeError):
        raise RuntimeError("DeepSeek вернул некорректный ответ.") from None
