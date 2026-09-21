from __future__ import annotations

import unittest

from mcp import types

from tool_metrics import compact_json, format_bytes, format_metrics, measure_text, measure_tool, measure_tools


class ToolMetricsTests(unittest.TestCase):
    def test_measure_text_distinguishes_characters_and_utf8_bytes(self):
        metrics = measure_text("abcЯ")
        self.assertEqual(metrics.characters, 4)
        self.assertEqual(metrics.utf8_bytes, 5)
        self.assertEqual(metrics.estimated_tokens, 1)

    def test_measure_tool_uses_compact_wire_format_without_null_fields(self):
        tool = types.Tool(
            name="search",
            description="Поиск",
            input_schema={"type": "object", "properties": {}},
        )
        payload = compact_json(tool.model_dump(mode="json", by_alias=True, exclude_none=True))
        self.assertNotIn("input_schema", payload)
        self.assertIn('"inputSchema"', payload)
        self.assertNotIn(": ", payload)
        self.assertEqual(measure_tool(tool), measure_text(payload))

    def test_total_includes_json_list_overhead(self):
        tools = (
            types.Tool(name="a", input_schema={"type": "object"}),
            types.Tool(name="b", input_schema={"type": "object"}),
        )
        total = measure_tools(tools)
        self.assertGreater(total.characters, sum(measure_tool(tool).characters for tool in tools))

    def test_human_readable_format(self):
        self.assertEqual(format_bytes(512), "512 Б")
        self.assertEqual(format_bytes(1536), "1.5 КиБ")
        self.assertEqual(format_metrics(measure_text("x" * 4000)), "4 000 симв. · 3.9 КиБ · ≈1 000 токенов")


if __name__ == "__main__":
    unittest.main()
