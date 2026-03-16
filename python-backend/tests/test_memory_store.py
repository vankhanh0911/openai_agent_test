from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_thread_recreates_missing_thread_for_current_session(self) -> None:
        from memory_store import MemoryStore

        store = MemoryStore()
        context = {
            "analysis_portal_id": "aristino",
            "analysis_account_id": "acct_1",
        }

        thread = await store.load_thread("thr_25876b14", context)

        self.assertEqual(thread.id, "thr_25876b14")
        self.assertEqual((thread.metadata or {}).get("session_key"), "aristino:acct_1")

        items_page = await store.load_thread_items("thr_25876b14", None, 20, "asc", context)
        self.assertEqual(items_page.data, [])


if __name__ == "__main__":
    unittest.main()
