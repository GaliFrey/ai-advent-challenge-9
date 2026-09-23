"""Verify the local STDIO MCP server and collector status on the VM."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


async def check() -> None:
    server_path = Path(__file__).with_name("server.py")
    target = StdioServerParameters(command=sys.executable, args=[str(server_path)])
    async with Client(target, mode="legacy") as client:
        tools = await client.list_tools()
        names = [tool.name for tool in tools.tools]
        if "ssh_collector_status" not in names:
            raise RuntimeError("MCP server did not expose ssh_collector_status")
        result = await client.call_tool("ssh_collector_status", {})
        if result.is_error:
            raise RuntimeError("ssh_collector_status returned an MCP error")
        if result.structured_content is not None:
            status = result.structured_content
        else:
            status = json.loads(next(item.text for item in result.content if item.type == "text"))
        print(f"MCP server: {client.server_info.name}")
        print(f"Tools: {', '.join(names)}")
        print(f"Saved logins: {status['total_saved']}")
        print(f"Last collection: {status['last_collection']} ({status['timezone']})")


if __name__ == "__main__":
    asyncio.run(check())
