from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

from agents.mcp import MCPServerSse


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class AnalyticsMcpTests(unittest.TestCase):
    def test_chat_server_imports_cleanly(self) -> None:
        module = importlib.import_module("analytics.chat_server")
        self.assertTrue(hasattr(module, "AnalyticsServer"))

    def test_remote_sse_server_can_be_attached_to_agent(self) -> None:
        from analytics.workflow import AnalyticsSettings, _build_network_mcp_server, build_analytics_agent

        settings = AnalyticsSettings(
            analysis_model="gpt-5.2",
            mcp_server_url="https://dev-cube-mcp.ants.tech/sse",
            mcp_server_label="CUBE",
            mcp_authorization=None,
            mcp_allowed_tools=("get_metadata", "read_data"),
            trace_name="trace",
            trace_workflow_id=None,
            prompt_injection_model="gpt-4.1-mini",
        )

        server = _build_network_mcp_server(settings)

        self.assertIsInstance(server, MCPServerSse)

        agent = build_analytics_agent(settings, mcp_servers=[server])

        self.assertEqual(agent.mcp_servers, [server])
        self.assertEqual(agent.tools, [])


if __name__ == "__main__":
    unittest.main()
