"""Real MCP server that refuses a child without bytecode protection."""
import sys

if not sys.dont_write_bytecode:
    raise RuntimeError("Child bytecode writes enabled")

from server.mcp_server import main

if __name__ == "__main__":
    main()
