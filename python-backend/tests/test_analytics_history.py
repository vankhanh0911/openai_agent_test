from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class AnalyticsHistoryTests(unittest.TestCase):
    def test_build_persistent_input_items_keeps_only_messages_between_turns(self) -> None:
        from analytics.chat_server import AnalyticsServer

        server = AnalyticsServer()
        items = [
            {"role": "user", "content": [{"type": "input_text", "text": "show revenue"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "read_data",
                "arguments": "{\"query\":{}}",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "{\"rows\":1}"},
        ]

        persisted = server._build_persistent_input_items(items, "Revenue is 123")

        self.assertEqual(
            persisted,
            [
                {"role": "user", "content": [{"type": "input_text", "text": "show revenue"}]},
                {"role": "assistant", "content": "Revenue is 123"},
            ],
        )

    def test_compact_input_items_truncates_large_tool_output_and_drops_reasoning(self) -> None:
        from analytics.chat_server import AnalyticsServer

        server = AnalyticsServer()
        items = [
            {"role": "user", "content": [{"type": "input_text", "text": "show revenue"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "x" * 9000}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "y" * 9000},
        ]

        compacted = server._compact_input_items(items)

        self.assertEqual(len(compacted), 2)
        self.assertNotIn("reasoning", [item.get("type") for item in compacted if isinstance(item, dict)])
        tool_output = compacted[-1]
        self.assertIsInstance(tool_output, dict)
        self.assertLess(len(tool_output["output"]), len(items[-1]["output"]))
        self.assertIn("[truncated", tool_output["output"])

    def test_compact_input_items_keeps_only_latest_history_window(self) -> None:
        from analytics.chat_server import AnalyticsServer

        server = AnalyticsServer()
        items = [
            {"role": "user", "content": [{"type": "input_text", "text": f"message-{idx}"}]}
            for idx in range(server._MAX_HISTORY_ITEMS + 5)
        ]

        compacted = server._compact_input_items(items)

        self.assertEqual(len(compacted), server._MAX_HISTORY_ITEMS)
        self.assertEqual(compacted[0]["content"][0]["text"], "message-5")
        self.assertEqual(compacted[-1]["content"][0]["text"], f"message-{server._MAX_HISTORY_ITEMS + 4}")


if __name__ == "__main__":
    unittest.main()
