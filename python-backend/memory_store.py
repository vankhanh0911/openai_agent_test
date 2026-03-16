from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, List

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, Thread, ThreadItem, ThreadMetadata


@dataclass
class _ThreadState:
    thread: ThreadMetadata
    items: List[ThreadItem]


class MemoryStore(Store[dict[str, Any]]):
    """Simple in-memory store compatible with the ChatKit server interface."""

    def __init__(self) -> None:
        self._threads: Dict[str, _ThreadState] = {}
        self._attachments: Dict[str, Attachment] = {}

    def generate_attachment_id(self, mime_type: str, context: dict[str, Any]) -> str:
        """Return a new identifier for an attachment."""
        return f"atc_{uuid4().hex[:8]}"

    @staticmethod
    def _normalize_session_value(value: Any, default: str) -> str:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
        return default

    @classmethod
    def _session_key_from_context(cls, context: dict[str, Any]) -> str:
        explicit = context.get("session_key")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()

        portal_id = cls._normalize_session_value(
            context.get("analysis_portal_id") or context.get("portal_id"),
            "default-portal",
        )
        account_id = cls._normalize_session_value(
            context.get("analysis_account_id") or context.get("account_id"),
            "default-account",
        )
        return f"{portal_id}:{account_id}"

    @classmethod
    def _decorate_thread_metadata(
        cls,
        thread: ThreadMetadata | Thread,
        context: dict[str, Any],
    ) -> ThreadMetadata:
        metadata = cls._get_thread_metadata(thread)
        session_key = cls._session_key_from_context(context)
        portal_id, account_id = session_key.split(":", 1)
        metadata.metadata = {
            **(metadata.metadata or {}),
            "session_key": session_key,
            "portal_id": portal_id,
            "account_id": account_id,
        }
        return metadata

    @classmethod
    def _thread_session_key(cls, thread: ThreadMetadata) -> str:
        metadata = thread.metadata or {}
        session_key = metadata.get("session_key")
        if isinstance(session_key, str) and session_key.strip():
            return session_key.strip()

        portal_id = cls._normalize_session_value(metadata.get("portal_id"), "default-portal")
        account_id = cls._normalize_session_value(
            metadata.get("account_id"), "default-account"
        )
        return f"{portal_id}:{account_id}"

    @classmethod
    def _thread_matches_context(
        cls,
        thread: ThreadMetadata,
        context: dict[str, Any],
    ) -> bool:
        return cls._thread_session_key(thread) == cls._session_key_from_context(context)

    def _state_for_thread(self, thread_id: str) -> _ThreadState:
        state = self._threads.get(thread_id)
        if state is None:
            raise NotFoundError(f"Thread {thread_id} not found")
        return state

    def _restore_missing_thread(
        self,
        thread_id: str,
        context: dict[str, Any],
    ) -> _ThreadState:
        metadata = self._decorate_thread_metadata(
            ThreadMetadata(id=thread_id, created_at=datetime.utcnow()),
            context,
        )
        state = _ThreadState(thread=metadata, items=[])
        self._threads[thread_id] = state
        return state

    def _assert_thread_access(self, thread_id: str, context: dict[str, Any]) -> _ThreadState:
        state = self._state_for_thread(thread_id)
        if not self._thread_matches_context(state.thread, context):
            raise NotFoundError(f"Thread {thread_id} not found")
        return state

    @staticmethod
    def _get_thread_metadata(thread: ThreadMetadata | Thread) -> ThreadMetadata:
        """Return thread metadata without any embedded items (openai-chatkit>=1.0)."""
        has_items = isinstance(thread, Thread) or "items" in getattr(
            thread, "model_fields_set", set()
        )
        if not has_items:
            return thread.model_copy(deep=True)

        data = thread.model_dump()
        data.pop("items", None)
        return ThreadMetadata(**data).model_copy(deep=True)

    # -- Thread metadata -------------------------------------------------
    async def load_thread(self, thread_id: str, context: dict[str, Any]) -> ThreadMetadata:
        try:
            state = self._assert_thread_access(thread_id, context)
        except NotFoundError:
            state = self._restore_missing_thread(thread_id, context)
        return self._get_thread_metadata(state.thread)

    async def save_thread(self, thread: ThreadMetadata, context: dict[str, Any]) -> None:
        metadata = self._decorate_thread_metadata(thread, context)
        state = self._threads.get(thread.id)
        if state:
            state.thread = metadata
        else:
            self._threads[thread.id] = _ThreadState(
                thread=metadata,
                items=[],
            )

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: dict[str, Any],
    ) -> Page[ThreadMetadata]:
        threads = sorted(
            (
                self._get_thread_metadata(state.thread)
                for state in self._threads.values()
                if self._thread_matches_context(state.thread, context)
            ),
            key=lambda t: t.created_at or datetime.min,
            reverse=(order == "desc"),
        )

        if after:
            index_map = {thread.id: idx for idx, thread in enumerate(threads)}
            start = index_map.get(after, -1) + 1
        else:
            start = 0

        slice_threads = threads[start : start + limit + 1]
        has_more = len(slice_threads) > limit
        slice_threads = slice_threads[:limit]
        next_after = slice_threads[-1].id if has_more and slice_threads else None
        return Page(
            data=slice_threads,
            has_more=has_more,
            after=next_after,
        )

    async def delete_thread(self, thread_id: str, context: dict[str, Any]) -> None:
        self._assert_thread_access(thread_id, context)
        self._threads.pop(thread_id, None)

    # -- Thread items ----------------------------------------------------
    def _items(self, thread_id: str) -> List[ThreadItem]:
        state = self._threads.get(thread_id)
        if state is None:
            state = _ThreadState(
                thread=ThreadMetadata(id=thread_id, created_at=datetime.utcnow()),
                items=[],
            )
            self._threads[thread_id] = state
        return state.items

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: dict[str, Any],
    ) -> Page[ThreadItem]:
        try:
            self._assert_thread_access(thread_id, context)
        except NotFoundError:
            self._restore_missing_thread(thread_id, context)
        items = [item.model_copy(deep=True) for item in self._items(thread_id)]
        items.sort(
            key=lambda item: getattr(item, "created_at", datetime.utcnow()),
            reverse=(order == "desc"),
        )

        if after:
            index_map = {item.id: idx for idx, item in enumerate(items)}
            start = index_map.get(after, -1) + 1
        else:
            start = 0

        slice_items = items[start : start + limit + 1]
        has_more = len(slice_items) > limit
        slice_items = slice_items[:limit]
        next_after = slice_items[-1].id if has_more and slice_items else None
        return Page(data=slice_items, has_more=has_more, after=next_after)

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: dict[str, Any]
    ) -> None:
        self._assert_thread_access(thread_id, context)
        self._items(thread_id).append(item.model_copy(deep=True))

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict[str, Any]) -> None:
        self._assert_thread_access(thread_id, context)
        items = self._items(thread_id)
        for idx, existing in enumerate(items):
            if existing.id == item.id:
                items[idx] = item.model_copy(deep=True)
                return
        items.append(item.model_copy(deep=True))

    async def load_item(self, thread_id: str, item_id: str, context: dict[str, Any]) -> ThreadItem:
        self._assert_thread_access(thread_id, context)
        for item in self._items(thread_id):
            if item.id == item_id:
                return item.model_copy(deep=True)
        raise NotFoundError(f"Item {item_id} not found")

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: dict[str, Any]
    ) -> None:
        self._assert_thread_access(thread_id, context)
        items = self._items(thread_id)
        self._threads[thread_id].items = [item for item in items if item.id != item_id]

    # -- Files -----------------------------------------------------------

    async def save_attachment(
        self,
        attachment: Attachment,
        context: dict[str, Any],
    ) -> None:
        self._attachments[attachment.id] = attachment.model_copy(deep=True)

    async def load_attachment(
        self,
        attachment_id: str,
        context: dict[str, Any],
    ) -> Attachment:
        att = self._attachments.get(attachment_id)
        if not att:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return att.model_copy(deep=True)

    async def delete_attachment(self, attachment_id: str, context: dict[str, Any]) -> None:
        self._attachments.pop(attachment_id, None)
