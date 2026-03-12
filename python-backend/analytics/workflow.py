from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from agents import (
    Agent,
    HostedMCPTool,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    trace,
)
from openai import AsyncOpenAI, OpenAIError
from openai.types.shared import Reasoning
from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, ConfigDict, Field

try:
    from guardrails.runtime import instantiate_guardrails, load_config_bundle, run_guardrails
except ImportError:  # pragma: no cover - dependency is optional until installed
    instantiate_guardrails = None
    load_config_bundle = None
    run_guardrails = None

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH, override=True)
ENV_DEFAULTS = {
    key: value
    for key, value in dotenv_values(ENV_PATH).items()
    if isinstance(value, str) and value.strip()
}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if isinstance(value, str) and value.strip():
        return value.strip()

    fallback = ENV_DEFAULTS.get(name)
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()

    return default


ARISTINO_SCHEMA = """
cubes:
  - name: transaction_event_details
    sql_table: aristino.transaction_event_detail
    joins:
      - name: customers
        sql: "{CUBE}.customer_id = {customers}.id"
        relationship: many_to_one
      - name: stores
        sql: "{CUBE}.store_id = {stores}.id"
        relationship: many_to_one
      - name: products
        sql: "{CUBE}.product_id = {products}.item_id"
        relationship: many_to_one
    dimensions:
      - name: id
        sql: event_id
        type: string
        primary_key: true
      - name: transaction_id
        sql: transaction_id
        type: string
      - name: transaction_code
        sql: transaction_code
        type: string
      - name: transaction_source
        sql: transaction_source
        type: string
      - name: line_item_status
        sql: line_item_status
        type: string
      - name: tracked_time
        sql: tracked_time
        type: time
      - name: line_item_promotion_code
        sql: line_item_promotion_code
        type: string
      - name: showroom_chuong_trinh
        sql: showroom_chuong_trinh
        type: string
    measures:
      - name: count_transaction
        type: count
      - name: total_discount_amount
        sql: discount_amount
        type: sum
      - name: total_line_item_discount
        sql: line_item_discount_amount
        type: sum
      - name: total_quantity
        sql: line_item_quantity
        type: sum
      - name: total_item_revenue
        sql: line_item_rounded_amount
        type: sum
      - name: avg_revenue_per_transaction
        type: number
        sql: "{total_item_revenue} / NULLIF({count_transaction}, 0)"
  - name: products
    sql_table: aristino.product
    dimensions:
      - name: item_id
        sql: item_id
        type: string
        primary_key: true
      - name: name
        sql: name
        type: string
      - name: brand
        sql: brand
        type: string
      - name: color
        sql: color
        type: string
      - name: size
        sql: size
        type: string
      - name: status
        sql: status
        type: string
      - name: category_label
        sql: category_label
        type: string
      - name: category_level_1
        sql: category_level_1
        type: string
      - name: category_level_2
        sql: category_level_2
        type: string
      - name: main_category
        sql: main_category
        type: string
      - name: parent_item_id
        sql: parent_item_id
        type: string
      - name: is_parent
        sql: is_parent
        type: number
      - name: season
        sql: season
        type: string
      - name: source_website
        sql: source_website
        type: string
    measures:
      - name: count
        type: count
      - name: avg_price
        sql: price
        type: avg
      - name: avg_original_price
        sql: original_price
        type: avg
  - name: stores
    sql_table: aristino.store
    dimensions:
      - name: id
        sql: id
        type: string
        primary_key: true
      - name: item_name
        sql: item_name
        type: string
      - name: brand
        sql: brand
        type: string
      - name: province
        sql: province
        type: string
      - name: district
        sql: district
        type: string
      - name: national_channel_name
        sql: national_channel_name
        type: string
      - name: regional_channel_name
        sql: regional_channel_name
        type: string
      - name: warehouse_name
        sql: warehouse_name
        type: string
    measures:
      - name: count
        type: count
""".strip()

DEFAULT_GUARDRAILS_CONFIG = {
    "guardrails": [
        {
            "name": "Moderation",
            "config": {
                "categories": [
                    "sexual",
                    "sexual/minors",
                    "hate",
                    "hate/threatening",
                    "harassment",
                    "harassment/threatening",
                    "self-harm",
                    "self-harm/intent",
                    "self-harm/instructions",
                    "violence",
                    "violence/graphic",
                    "illicit",
                    "illicit/violent",
                ]
            },
        },
        {
            "name": "Prompt Injection Detection",
            "config": {"model": "gpt-4.1-mini", "confidence_threshold": 0.7},
        },
    ]
}

DEFAULT_ANALYSIS_INSTRUCTION = (
    "You are a senior data analyst for Aristino's multichannel marketing reporting workflow. "
    "Use the Aristino database and the available MCP tools to answer the user's request accurately. "
    "Always ground the answer in the available schema, explain assumptions when data is ambiguous, "
    "and avoid inventing fields or tables."
)


class AnalyticsWorkflowInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input_as_text: str
    instruction_text: str | None = Field(default=None, alias="instruction")
    schema_text: str | None = Field(default=None, alias="schema")
    portal_id: str | None = Field(default=None, alias="portalId")
    account_id: str | None = Field(default=None, alias="accountId")


class AnalysisContext:
    def __init__(
        self,
        workflow_input_as_text: str,
        instruction_text: str | None = None,
        schema_text: str | None = None,
    ):
        self.workflow_input_as_text = workflow_input_as_text
        self.instruction_text = instruction_text
        self.schema_text = schema_text


@dataclass(frozen=True)
class AnalyticsSettings:
    analysis_model: str
    mcp_server_url: str
    mcp_server_label: str
    mcp_authorization: str | None
    mcp_allowed_tools: tuple[str, ...]
    trace_name: str
    trace_workflow_id: str | None
    prompt_injection_model: str

    @classmethod
    def from_env(cls) -> "AnalyticsSettings":
        allowed_tools = tuple(
            tool.strip()
            for tool in (_env("ARISTINO_MCP_ALLOWED_TOOLS", "read_query") or "read_query").split(",")
            if tool.strip()
        )
        return cls(
            analysis_model=_env("ARISTINO_ANALYSIS_MODEL", "gpt-5.2") or "gpt-5.2",
            mcp_server_url=_env("ARISTINO_MCP_SERVER_URL", "") or "",
            mcp_server_label=_env("ARISTINO_MCP_SERVER_LABEL", "Dimension_Metrics_Identified")
            or "Dimension_Metrics_Identified",
            mcp_authorization=_env("ARISTINO_MCP_AUTHORIZATION"),
            mcp_allowed_tools=allowed_tools or ("read_query",),
            trace_name=_env("ARISTINO_TRACE_NAME", "AR Data Analytics") or "AR Data Analytics",
            trace_workflow_id=_env("ARISTINO_TRACE_WORKFLOW_ID"),
            prompt_injection_model=_env("ARISTINO_PROMPT_INJECTION_MODEL", "gpt-4.1-mini")
            or "gpt-4.1-mini",
        )


def _guardrails_runtime_available() -> bool:
    return all((instantiate_guardrails, load_config_bundle, run_guardrails))


def _guardrails_bundle(config: dict[str, Any]) -> list[Any]:
    if not _guardrails_runtime_available():
        raise RuntimeError(
            "openai-guardrails is not installed. Run `pip install -r requirements.txt` in python-backend."
        )
    return instantiate_guardrails(load_config_bundle(config))


def _guardrails_context() -> SimpleNamespace:
    try:
        return SimpleNamespace(guardrail_llm=AsyncOpenAI())
    except OpenAIError as exc:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Configure it before using analytics chat."
        ) from exc


def _build_mcp_tool(settings: AnalyticsSettings) -> HostedMCPTool:
    if not settings.mcp_server_url:
        raise RuntimeError(
            "Missing ARISTINO_MCP_SERVER_URL. Set the MCP endpoint before calling /analytics/run."
        )

    headers: dict[str, str] = {}
    if settings.mcp_authorization:
        headers["authorization"] = settings.mcp_authorization

    return HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": settings.mcp_server_label,
            "allowed_tools": list(settings.mcp_allowed_tools),
            "headers": headers,
            "require_approval": "never",
            "server_url": settings.mcp_server_url,
        }
    )


def analysis_instructions(
    run_context: RunContextWrapper[AnalysisContext], _agent: Agent[AnalysisContext]
) -> str:
    instruction_text = run_context.context.instruction_text or DEFAULT_ANALYSIS_INSTRUCTION
    schema_text = run_context.context.schema_text or ARISTINO_SCHEMA
    return (
        f"{instruction_text}\n\n"
        f"User request: {run_context.context.workflow_input_as_text}\n\n"
        "Available cube schema:\n"
        f"{schema_text}"
    )


def build_trace_identity(
    portal_id: str | None,
    account_id: str | None,
) -> tuple[str, str]:
    normalized_portal = (portal_id or "default-portal").strip() or "default-portal"
    normalized_account = (account_id or "default-account").strip() or "default-account"
    group_id = f"{normalized_portal}:{normalized_account}"
    trace_id = f"analytics:{group_id}:{uuid4().hex}"
    return trace_id, group_id


def _build_guardrails_config(settings: AnalyticsSettings) -> dict[str, Any]:
    return {
        "guardrails": [
            {
                **guardrail,
                "config": {
                    **((guardrail.get("config") or {})),
                    **(
                        {"model": settings.prompt_injection_model}
                        if guardrail.get("name") == "Prompt Injection Detection"
                        else {}
                    ),
                },
            }
            for guardrail in DEFAULT_GUARDRAILS_CONFIG["guardrails"]
        ]
    }


def _build_analysis_agent(settings: AnalyticsSettings) -> Agent[AnalysisContext]:
    return Agent(
        name="Analysis",
        instructions=analysis_instructions,
        model=settings.analysis_model,
        tools=[_build_mcp_tool(settings)],
        model_settings=ModelSettings(
            store=True,
            reasoning=Reasoning(effort="low", summary="auto"),
        ),
    )


def build_analytics_guardrails_config(settings: AnalyticsSettings) -> dict[str, Any]:
    return _build_guardrails_config(settings)


def build_analytics_agent(settings: AnalyticsSettings) -> Agent[AnalysisContext]:
    return _build_analysis_agent(settings)


def guardrails_has_tripwire(results: list[Any] | None) -> bool:
    return any(bool(getattr(result, "tripwire_triggered", False)) for result in (results or []))


def get_guardrail_safe_text(results: list[Any] | None, fallback_text: str) -> str:
    for result in results or []:
        info = getattr(result, "info", None) or {}
        if isinstance(info, dict) and "checked_text" in info:
            return info.get("checked_text") or fallback_text
    pii_info = next(
        (
            getattr(result, "info", None) or {}
            for result in results or []
            if isinstance(getattr(result, "info", None) or {}, dict)
            and "anonymized_text" in (getattr(result, "info", None) or {})
        ),
        None,
    )
    if isinstance(pii_info, dict) and "anonymized_text" in pii_info:
        return pii_info.get("anonymized_text") or fallback_text
    return fallback_text


async def scrub_conversation_history(
    history: list[TResponseInputItem], config: dict[str, Any], ctx: Any
) -> None:
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii:
            return

        pii_only = {"guardrails": [pii]}
        for msg in history or []:
            content = (msg or {}).get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    res = await run_guardrails(
                        ctx,
                        part["text"],
                        "text/plain",
                        _guardrails_bundle(pii_only),
                        suppress_tripwire=True,
                        raise_guardrail_errors=True,
                    )
                    part["text"] = get_guardrail_safe_text(res, part["text"])
    except Exception:
        pass


async def scrub_workflow_input(workflow: dict[str, Any], input_key: str, config: dict[str, Any], ctx: Any) -> None:
    try:
        guardrails = (config or {}).get("guardrails") or []
        pii = next((g for g in guardrails if (g or {}).get("name") == "Contains PII"), None)
        if not pii or not isinstance(workflow, dict):
            return

        value = workflow.get(input_key)
        if not isinstance(value, str):
            return

        pii_only = {"guardrails": [pii]}
        res = await run_guardrails(
            ctx,
            value,
            "text/plain",
            _guardrails_bundle(pii_only),
            suppress_tripwire=True,
            raise_guardrail_errors=True,
        )
        workflow[input_key] = get_guardrail_safe_text(res, value)
    except Exception:
        pass


def build_guardrail_fail_output(results: list[Any] | None) -> dict[str, Any]:
    def get_result(name: str) -> Any:
        for result in results or []:
            info = getattr(result, "info", None) or {}
            if not isinstance(info, dict):
                continue
            guardrail_name = info.get("guardrail_name") or info.get("guardrailName")
            if guardrail_name == name:
                return result
        return None

    def tripwire(result: Any) -> bool:
        return bool(getattr(result, "tripwire_triggered", False))

    def info(result: Any) -> dict[str, Any]:
        value = getattr(result, "info", None) or {}
        return value if isinstance(value, dict) else {}

    pii, mod, jb, hal, nsfw, url, custom, pid = map(
        get_result,
        [
            "Contains PII",
            "Moderation",
            "Jailbreak",
            "Hallucination Detection",
            "NSFW Text",
            "URL Filter",
            "Custom Prompt Check",
            "Prompt Injection Detection",
        ],
    )
    pii_info = info(pii)
    mod_info = info(mod)
    hal_info = info(hal)
    detected_entities = pii_info.get("detected_entities") if isinstance(pii_info, dict) else {}
    pii_counts: list[str] = []
    if isinstance(detected_entities, dict):
        for entity_name, entity_matches in detected_entities.items():
            if isinstance(entity_matches, list):
                pii_counts.append(f"{entity_name}:{len(entity_matches)}")

    flagged_categories = mod_info.get("flagged_categories") or []

    return {
        "pii": {"failed": bool(pii_counts) or tripwire(pii), "detected_counts": pii_counts},
        "moderation": {"failed": tripwire(mod) or bool(flagged_categories), "flagged_categories": flagged_categories},
        "jailbreak": {"failed": tripwire(jb)},
        "hallucination": {
            "failed": tripwire(hal),
            "reasoning": hal_info.get("reasoning"),
            "hallucination_type": hal_info.get("hallucination_type"),
            "hallucinated_statements": hal_info.get("hallucinated_statements"),
            "verified_statements": hal_info.get("verified_statements"),
        },
        "nsfw": {"failed": tripwire(nsfw)},
        "url_filter": {"failed": tripwire(url)},
        "custom_prompt_check": {"failed": tripwire(custom)},
        "prompt_injection": {"failed": tripwire(pid)},
    }


async def run_and_apply_guardrails(
    input_text: str,
    config: dict[str, Any],
    history: list[TResponseInputItem],
    workflow: dict[str, Any],
) -> dict[str, Any]:
    guardrails_bundle = _guardrails_bundle(config)
    ctx = _guardrails_context()
    results = await run_guardrails(
        ctx,
        input_text,
        "text/plain",
        guardrails_bundle,
        suppress_tripwire=True,
        raise_guardrail_errors=True,
    )
    guardrails = (config or {}).get("guardrails") or []
    mask_pii = any(
        (g or {}).get("name") == "Contains PII" and ((g or {}).get("config") or {}).get("block") is False
        for g in guardrails
    )
    if mask_pii:
        await scrub_conversation_history(history, config, ctx)
        await scrub_workflow_input(workflow, "input_as_text", config, ctx)
        await scrub_workflow_input(workflow, "input_text", config, ctx)

    has_tripwire = guardrails_has_tripwire(results)
    safe_text = get_guardrail_safe_text(results, input_text)
    return {
        "results": results,
        "has_tripwire": has_tripwire,
        "safe_text": safe_text,
        "fail_output": build_guardrail_fail_output(results),
    }


async def run_analytics_workflow(workflow_input: AnalyticsWorkflowInput) -> dict[str, Any]:
    settings = AnalyticsSettings.from_env()
    trace_id, group_id = build_trace_identity(
        workflow_input.portal_id,
        workflow_input.account_id,
    )
    trace_metadata = {
        "__trace_source__": "agent-builder",
        "portal_id": workflow_input.portal_id or "",
        "account_id": workflow_input.account_id or "",
    }
    
    trace_metadata["workflow_id"] = trace_id

    with trace(
        settings.trace_name,
        trace_id=trace_id,
        group_id=group_id,
        metadata=trace_metadata,
    ):
        workflow = workflow_input.model_dump()
        conversation_history: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": workflow["input_as_text"]}],
            }
        ]
        guardrails_config = _build_guardrails_config(settings)
        guardrails_result = await run_and_apply_guardrails(
            workflow["input_as_text"],
            guardrails_config,
            conversation_history,
            workflow,
        )
        if guardrails_result["has_tripwire"]:
            return {
                "status": "blocked",
                "safe_text": guardrails_result["safe_text"],
                "guardrails": guardrails_result["fail_output"],
                "output_text": None,
            }

        analysis_agent = _build_analysis_agent(settings)

        analysis_result_temp = await Runner.run(
            analysis_agent,
            input=conversation_history,
            run_config=RunConfig(
                trace_id=trace_id,
                group_id=group_id,
                trace_metadata=trace_metadata,
            ),
            context=AnalysisContext(
                workflow_input_as_text=workflow["input_as_text"],
                instruction_text=workflow.get("instruction_text"),
                schema_text=workflow.get("schema_text"),
            ),
        )

        return {
            "status": "completed",
            "safe_text": guardrails_result["safe_text"],
            "guardrails": None,
            "output_text": analysis_result_temp.final_output_as(str),
        }
