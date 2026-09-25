"""Run a direct end-to-end MCP check without an LLM call."""
from __future__ import annotations

import asyncio
import json

from agent import digest, result_data
from mcp_client import connect, discover_tools
from report_download import download_report
from servers import SERVERS


async def main() -> None:
    for server in SERVERS:
        discovered = await discover_tools(server)
        assert server.tool in {tool.name for tool in discovered.tools}
        print(f"{server.role}: {discovered.server_name} / {server.tool}")
    async with connect(SERVERS[0]) as client:
        source_result = result_data(await client.call_tool(SERVERS[0].tool, {"hours": 24}))
    assert digest(source_result["snapshot"]) == source_result["snapshot_sha256"]
    async with connect(SERVERS[1]) as client:
        analysis_result = result_data(await client.call_tool(SERVERS[1].tool, {
            "snapshot": source_result["snapshot"], "snapshot_sha256": source_result["snapshot_sha256"],
        }))
    assert digest(analysis_result["analysis"]) == analysis_result["analysis_sha256"]
    assert analysis_result["analysis"]["total"] == source_result["event_count"]
    async with connect(SERVERS[2]) as client:
        saved = result_data(await client.call_tool(SERVERS[2].tool, {
            "analysis": analysis_result["analysis"], "analysis_sha256": analysis_result["analysis_sha256"],
        }))
    downloaded = await download_report(saved)
    print(json.dumps({"event_count": source_result["event_count"],
                      "unique_ips": analysis_result["analysis"]["unique_ips"],
                      "remote_path": saved["path"], **downloaded}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
