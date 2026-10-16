"""Actual stdio MCP fixture rejecting any leaked backend provider variable."""
import os
from server.mcp_server import create_server, _parse_args
import sys

if any(name.startswith('DEEPSEEK_') for name in os.environ):
    # No values or environment dumps, even in the fixture's failure branch.
    raise RuntimeError('Provider configuration leaked to MCP')
args = _parse_args(sys.argv[1:])
create_server(db_path=args.db_path, conversation_id=args.conversation_id,
              message_id=args.message_id, max_messages=args.max_messages,
              max_text_chars=args.max_text_chars).run(transport='stdio')
