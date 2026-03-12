from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from agents import (
    ItemHelpers,
    MessageOutputItem,
    RunConfig,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
)
from chatkit.agents import AgentContext, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.store import NotFoundError
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ClientEffectEvent,
    ProgressUpdateEvent,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)
from pydantic import BaseModel

from analytics import (
    AnalysisContext,
    AnalyticsSettings,
    build_analytics_agent,
    build_analytics_guardrails_config,
    build_trace_identity,
    run_and_apply_guardrails,
)
from memory_store import MemoryStore


def _normalize_override(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float


def _user_message_to_text(message: UserMessageItem) -> str:
    parts: List[str] = []
    for part in message.content:
        text = getattr(part, "text", "")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _parse_tool_args(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return raw_args
    return raw_args


def _guardrail_reasoning(info: dict[str, Any]) -> str:
    for key in ("observation", "reasoning", "message", "error"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _guardrail_name(result: Any) -> str:
    info = getattr(result, "info", None) or {}
    if isinstance(info, dict):
        for key in ("guardrail_name", "guardrailName"):
            value = info.get(key)
            if isinstance(value, str) and value:
                return value
    return "Unknown Guardrail"


@dataclass
class ConversationState:
    input_items: List[Any] = field(default_factory=list)
    current_agent_name: str = "Analysis"
    events: List[AgentEvent] = field(default_factory=list)
    guardrails: List[GuardrailCheck] = field(default_factory=list)
    session_key: str | None = None
    context: Dict[str, Any] = field(
        default_factory=lambda: {
            "portal": "aristino",
            "mode": "analytics",
            "mcp_tool": "read_query",
        }
    )


class AnalyticsServer(ChatKitServer[dict[str, Any]]):
    def __init__(self) -> None:
        self.store = MemoryStore()
        super().__init__(self.store)
        self._state: Dict[str, ConversationState] = {}
        self._listeners: Dict[str, list[asyncio.Queue]] = {}
        self._last_event_index: Dict[str, int] = {}
        self._last_snapshot: Dict[str, str] = {}

    def _state_for_thread(self, thread_id: str) -> ConversationState:
        if thread_id not in self._state:
            self._state[thread_id] = ConversationState()
        return self._state[thread_id]

    async def _ensure_thread(
        self, thread_id: Optional[str], context: dict[str, Any]
    ) -> ThreadMetadata:
        if thread_id:
            try:
                return await self.store.load_thread(thread_id, context)
            except NotFoundError:
                pass
        new_thread = ThreadMetadata(
            id=self.store.generate_thread_id(context),
            created_at=datetime.now(),
        )
        await self.store.save_thread(new_thread, context)
        self._state_for_thread(new_thread.id)
        return new_thread

    async def ensure_thread(
        self, thread_id: Optional[str], context: dict[str, Any]
    ) -> ThreadMetadata:
        return await self._ensure_thread(thread_id, context)

    @staticmethod
    def _truncate(val: Any, limit: int = 300) -> Any:
        if isinstance(val, str) and len(val) > limit:
            return val[:limit] + "..."
        return val

    def _build_agents_list(self, settings: AnalyticsSettings) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Analysis",
                "description": "Answers Aristino marketing analytics questions using hosted MCP queries.",
                "handoffs": [],
                "tools": list(settings.mcp_allowed_tools),
                "input_guardrails": ["Moderation", "Prompt Injection Detection"],
            }
        ]

    def _record_events(self, run_items: List[Any]) -> List[AgentEvent]:
        events: List[AgentEvent] = []
        for item in run_items:
            now_ms = time.time() * 1000
            if isinstance(item, MessageOutputItem):
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="message",
                        agent=item.agent.name,
                        content=self._truncate(ItemHelpers.text_message_output(item)),
                        timestamp=now_ms,
                    )
                )
            elif isinstance(item, ToolCallItem):
                tool_name = getattr(item.raw_item, "name", None)
                raw_args = getattr(item.raw_item, "arguments", None)
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="tool_call",
                        agent=item.agent.name,
                        content=self._truncate(tool_name or ""),
                        metadata={"tool_args": self._truncate(_parse_tool_args(raw_args))},
                        timestamp=now_ms,
                    )
                )
            elif isinstance(item, ToolCallOutputItem):
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="tool_output",
                        agent=item.agent.name,
                        content=self._truncate(str(item.output)),
                        metadata={"tool_result": self._truncate(item.output)},
                        timestamp=now_ms,
                    )
                )
        return events

    def _record_guardrails(
        self,
        input_text: str,
        guardrails_config: dict[str, Any],
        results: list[Any],
    ) -> List[GuardrailCheck]:
        checks: List[GuardrailCheck] = []
        timestamp = time.time() * 1000
        by_name = {_guardrail_name(result): result for result in results or []}
        for guardrail in guardrails_config.get("guardrails", []):
            name = (guardrail or {}).get("name", "Unknown Guardrail")
            result = by_name.get(name)
            info = (getattr(result, "info", None) or {}) if result is not None else {}
            if not isinstance(info, dict):
                info = {}
            checks.append(
                GuardrailCheck(
                    id=uuid4().hex,
                    name=name,
                    input=input_text,
                    reasoning=_guardrail_reasoning(info),
                    passed=not bool(getattr(result, "tripwire_triggered", False)),
                    timestamp=timestamp,
                )
            )
        return checks

    async def _broadcast_state(self, thread: ThreadMetadata, context: dict[str, Any]) -> None:
        listeners = self._listeners.get(thread.id, [])
        if not listeners:
            return
        snap = await self.snapshot(thread.id, context)
        last_idx = self._last_event_index.get(thread.id, 0)
        total_events = len(snap.get("events", []))
        delta = snap.get("events", [])[last_idx:] if total_events >= last_idx else snap.get("events", [])
        self._last_event_index[thread.id] = total_events
        payload_obj = {**snap, "events_delta": delta}
        payload = json.dumps(payload_obj, default=str)
        self._last_snapshot[thread.id] = payload
        for q in list(listeners):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def _register_listener(self, thread_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.setdefault(thread_id, []).append(q)
        last = self._last_snapshot.get(thread_id)
        if last:
            try:
                q.put_nowait(last)
            except asyncio.QueueFull:
                pass
        return q

    def register_listener(self, thread_id: str) -> asyncio.Queue:
        return self._register_listener(thread_id)

    def _unregister_listener(self, thread_id: str, queue: asyncio.Queue) -> None:
        listeners = self._listeners.get(thread_id, [])
        if queue in listeners:
            listeners.remove(queue)
        if not listeners and thread_id in self._listeners:
            self._listeners.pop(thread_id, None)

    def unregister_listener(self, thread_id: str, queue: asyncio.Queue) -> None:
        self._unregister_listener(thread_id, queue)

    async def _emit_assistant_message(
        self,
        thread: ThreadMetadata,
        content: str,
        context: dict[str, Any],
        state: ConversationState,
        user_text: str,
        event_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ThreadStreamEvent]:
        timestamp = time.time() * 1000
        event = AgentEvent(
            id=uuid4().hex,
            type="message",
            agent=state.current_agent_name,
            content=self._truncate(content),
            metadata=event_metadata,
            timestamp=timestamp,
        )
        state.events.append(event)
        state.context["last_user_message"] = user_text
        state.context["last_response_preview"] = self._truncate(content, 120)
        state.input_items.append({"role": "assistant", "content": content})
        await self._broadcast_state(thread, context)
        yield ClientEffectEvent(
            name="runner_state_update",
            data={"thread_id": thread.id, "ts": time.time()},
        )
        yield ClientEffectEvent(
            name="runner_event_delta",
            data={"thread_id": thread.id, "ts": time.time(), "events": [event.model_dump()]},
        )
        yield ThreadItemDoneEvent(
            item=AssistantMessageItem(
                id=self.store.generate_item_id("message", thread, context),
                thread_id=thread.id,
                created_at=datetime.now(),
                content=[AssistantMessageContent(text=content)],
            )
        )

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        state = self._state_for_thread(thread.id)
        settings = AnalyticsSettings.from_env()
        user_text = ""
        instruction_override = _normalize_override(context.get("analysis_instruction"))
        schema_override = _normalize_override(context.get("analysis_schema"))
        portal_id = _normalize_override(context.get("analysis_portal_id"))
        account_id = _normalize_override(context.get("analysis_account_id"))
        session_key = f"{portal_id or 'default-portal'}:{account_id or 'default-account'}"

        if state.session_key and state.session_key != session_key:
            state.input_items = []
            state.events = []
            state.guardrails = []
            state.context = {
                "portal": "aristino",
                "mode": "analytics",
                "mcp_tool": "read_query",
            }
        state.session_key = session_key

        if input_user_message is not None:
            user_text = _user_message_to_text(input_user_message)
            state.input_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            )
            state.context["last_user_message"] = user_text

        state.context["instruction_configured"] = bool(instruction_override)
        state.context["schema_configured"] = bool(schema_override)
        state.context["portal_id"] = portal_id or ""
        state.context["account_id"] = account_id or ""
        state.context["portal_id_configured"] = bool(portal_id)
        state.context["account_id_configured"] = bool(account_id)
        state.context["session_key"] = session_key
        if instruction_override:
            state.context["instruction_preview"] = self._truncate(instruction_override, 180)
            state.context["instruction_length"] = len(instruction_override)
        else:
            state.context.pop("instruction_preview", None)
            state.context.pop("instruction_length", None)
        if schema_override:
            state.context["schema_preview"] = self._truncate(schema_override, 180)
            state.context["schema_length"] = len(schema_override)
        else:
            state.context.pop("schema_preview", None)
            state.context.pop("schema_length", None)

        streamed_items_seen = 0
        yield ClientEffectEvent(
            name="runner_bind_thread",
            data={"thread_id": thread.id, "ts": time.time()},
        )

        guardrails_config = build_analytics_guardrails_config(settings)
        workflow = {"input_as_text": user_text}
        conversation_history = list(state.input_items)
        trace_id, group_id = build_trace_identity(portal_id, account_id)
        trace_metadata = {
            "__trace_source__": "agent-builder",
            "portal_id": portal_id or "",
            "account_id": account_id or "",
        }
        if settings.trace_workflow_id:
            trace_metadata["workflow_id"] = settings.trace_workflow_id

        try:
            guardrails_result = await run_and_apply_guardrails(
                user_text,
                guardrails_config,
                conversation_history,
                workflow,
            )
            state.guardrails = self._record_guardrails(
                input_text=user_text,
                guardrails_config=guardrails_config,
                results=guardrails_result["results"],
            )
            state.input_items = conversation_history
            state.context["last_safe_text"] = guardrails_result["safe_text"]

            if guardrails_result["has_tripwire"]:
                failed = [check.name for check in state.guardrails if not check.passed]
                blocked_message = (
                    "I can't process that analytics request because it was blocked by "
                    + ", ".join(failed)
                    + ". Please rephrase the query."
                )
                async for event in self._emit_assistant_message(
                    thread,
                    blocked_message,
                    context,
                    state,
                    user_text,
                    {"guardrails": guardrails_result["fail_output"]},
                ):
                    yield event
                return

            chat_context = AgentContext[dict[str, Any]](
                thread=thread,
                store=self.store,
                request_context=context,
            )
            result = Runner.run_streamed(
                build_analytics_agent(settings),
                conversation_history,
                context=AnalysisContext(
                    workflow_input_as_text=workflow["input_as_text"],
                    instruction_text=instruction_override,
                    schema_text=schema_override,
                ),
                run_config=RunConfig(
                    trace_id=trace_id,
                    group_id=group_id,
                    trace_metadata=trace_metadata,
                ),
            )
            async for event in stream_agent_response(chat_context, result):
                if isinstance(event, ProgressUpdateEvent) or getattr(event, "type", "") == "progress_update_event":
                    continue
                if hasattr(event, "item"):
                    run_item = getattr(event, "item")
                    new_events = self._record_events([run_item])
                    if new_events:
                        state.events.extend(new_events)
                        await self._broadcast_state(thread, context)
                        yield ClientEffectEvent(
                            name="runner_state_update",
                            data={"thread_id": thread.id, "ts": time.time()},
                        )
                        yield ClientEffectEvent(
                            name="runner_event_delta",
                            data={
                                "thread_id": thread.id,
                                "ts": time.time(),
                                "events": [e.model_dump() for e in new_events],
                            },
                        )
                yield event
                new_items = result.new_items[streamed_items_seen:]
                if new_items:
                    new_events = self._record_events(new_items)
                    if new_events:
                        state.events.extend(new_events)
                    streamed_items_seen += len(new_items)
                    await self._broadcast_state(thread, context)
                    yield ClientEffectEvent(
                        name="runner_state_update",
                        data={"thread_id": thread.id, "ts": time.time()},
                    )
                    if new_events:
                        yield ClientEffectEvent(
                            name="runner_event_delta",
                            data={
                                "thread_id": thread.id,
                                "ts": time.time(),
                                "events": [e.model_dump() for e in new_events],
                            },
                        )
        except RuntimeError as exc:
            async for event in self._emit_assistant_message(
                thread,
                f"Analytics chat is not configured: {exc}",
                context,
                state,
                user_text,
            ):
                yield event
            return

        state.input_items = result.to_input_list()
        remaining_items = result.new_items[streamed_items_seen:]
        new_events = self._record_events(remaining_items)
        if new_events:
            state.events.extend(new_events)
        state.current_agent_name = "Analysis"
        try:
            state.context["last_response_preview"] = self._truncate(
                result.final_output_as(str), 120
            )
        except Exception:
            pass
        await self._broadcast_state(thread, context)
        yield ClientEffectEvent(
            name="runner_state_update",
            data={"thread_id": thread.id, "ts": time.time()},
        )
        if new_events:
            yield ClientEffectEvent(
                name="runner_event_delta",
                data={
                    "thread_id": thread.id,
                    "ts": time.time(),
                    "events": [e.model_dump() for e in new_events],
                },
            )

    async def action(
        self,
        thread: ThreadMetadata,
        action: Any,
        sender: Any,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        if False:
            yield

    async def snapshot(
        self, thread_id: Optional[str], context: dict[str, Any]
    ) -> Dict[str, Any]:
        thread = await self._ensure_thread(thread_id, context)
        state = self._state_for_thread(thread.id)
        settings = AnalyticsSettings.from_env()
        return {
            "thread_id": thread.id,
            "current_agent": state.current_agent_name,
            "context": state.context,
            "agents": self._build_agents_list(settings),
            "events": [e.model_dump() for e in state.events],
            "guardrails": [g.model_dump() for g in state.guardrails],
        }
