from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class AnalyticsLoggingTests(unittest.TestCase):
    def test_log_openai_request_emits_summary(self) -> None:
        from analytics.workflow import OPENAI_REQUEST_LOGGER, log_openai_request

        input_items = [
            {"role": "user", "content": [{"type": "input_text", "text": "show yesterday revenue"}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "x" * 5000},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "analytics_openai.log"
            previous = os.environ.get("ARISTINO_OPENAI_LOG_PATH")
            os.environ["ARISTINO_OPENAI_LOG_PATH"] = str(log_path)
            try:
                with self.assertLogs("analytics.openai", level="INFO") as logs:
                    summary = log_openai_request(
                        model="gpt-5.2",
                        trace_id="trace_123",
                        run_kind="streamed",
                        input_items=input_items,
                        instruction_text="Use MCP tools.",
                        schema_text="cube schema",
                        tool_names=("get_metadata", "read_data"),
                    )
            finally:
                if previous is None:
                    os.environ.pop("ARISTINO_OPENAI_LOG_PATH", None)
                else:
                    os.environ["ARISTINO_OPENAI_LOG_PATH"] = previous

            joined = "\n".join(logs.output)
            file_text = log_path.read_text()
            self.assertIn("trace_123", joined)
            self.assertIn("gpt-5.2", joined)
            self.assertIn("get_metadata", joined)
            self.assertIn("show yesterday revenue", joined)
            self.assertIn("show yesterday revenue", file_text)
            self.assertGreater(summary["approx_input_chars"], 1000)
            self.assertEqual(summary["item_count"], 2)
            self.assertIn("instruction_preview", summary)
            self.assertIn("input_items", summary)
            for handler in list(OPENAI_REQUEST_LOGGER.handlers):
                if hasattr(handler, "close"):
                    handler.close()


if __name__ == "__main__":
    unittest.main()
