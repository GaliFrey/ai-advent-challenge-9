"""Aggregate an exact source snapshot on a separate MCP server."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


server = MCPServer("day-20-analyze")


@server.tool()
def analyze_logins(snapshot: dict, snapshot_sha256: str) -> dict:
    """Analyze a source snapshot forwarded unchanged by the client. Return a structured analysis for the report server."""
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != snapshot_sha256:
        raise ToolError("Source snapshot SHA256 mismatch")
    events = snapshot.get("events")
    if not isinstance(events, list) or len(events) > 1000:
        raise ToolError("Invalid or oversized source events")
    for event in events:
        if not isinstance(event, dict) or not all(isinstance(event.get(key), str) for key in ("occurred_at", "username", "ip", "method")):
            raise ToolError("Invalid source event")
    analysis = {"snapshot_sha256": snapshot_sha256, "since": snapshot["since"],
                "until": snapshot["until"], "timezone": snapshot["timezone"],
                "total": len(events), "unique_ips": len({event["ip"] for event in events}),
                "by_user": dict(sorted(Counter(event["username"] for event in events).items())),
                "by_ip": dict(sorted(Counter(event["ip"] for event in events).items())),
                "warnings": snapshot["warnings"]}
    canonical_analysis = json.dumps(analysis, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {"analysis": analysis, "analysis_sha256": hashlib.sha256(canonical_analysis).hexdigest()}


if __name__ == "__main__":
    server.run(transport="stdio")
