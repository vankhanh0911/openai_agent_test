from __future__ import annotations as _annotations

import json
import os
from typing import Any, Dict

from chatkit.server import StreamingResult
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from analytics import AnalyticsWorkflowInput, run_analytics_workflow
from analytics.chat_server import AnalyticsServer

app = FastAPI()

# Disable tracing for zero data retention orgs
os.environ.setdefault("OPENAI_TRACING_DISABLED", "1")

# CORS configuration (adjust as needed for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_server = AnalyticsServer()


def get_server() -> AnalyticsServer:
    return chat_server


def _request_context_from_payload(request: Request, payload: bytes) -> Dict[str, Any]:
    context: Dict[str, Any] = {"request": request}
    try:
        parsed = json.loads(payload)
    except Exception:
        return context

    metadata = parsed.get("metadata") or {}
    if isinstance(metadata, dict):
        instruction = metadata.get("analysis_instruction")
        schema = metadata.get("analysis_schema")
        if isinstance(instruction, str):
            context["analysis_instruction"] = instruction
        if isinstance(schema, str):
            context["analysis_schema"] = schema
    return context


@app.post("/chatkit")
async def chatkit_endpoint(
    request: Request, server: AnalyticsServer = Depends(get_server)
) -> Response:
    payload = await request.body()
    result = await server.process(payload, _request_context_from_payload(request, payload))
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "json"):
        return Response(content=result.json, media_type="application/json")
    return Response(content=result)


@app.get("/chatkit/state")
async def chatkit_state(
    thread_id: str = Query(...),
    server: AnalyticsServer = Depends(get_server),
) -> Dict[str, Any]:
    return await server.snapshot(thread_id, {"request": None})


@app.get("/chatkit/bootstrap")
async def chatkit_bootstrap(
    server: AnalyticsServer = Depends(get_server),
) -> Dict[str, Any]:
    return await server.snapshot(None, {"request": None})


@app.get("/chatkit/state/stream")
async def chatkit_state_stream(
    thread_id: str = Query(...),
    server: AnalyticsServer = Depends(get_server),
):
    thread = await server.ensure_thread(thread_id, {"request": None})
    queue = server.register_listener(thread.id)

    async def event_generator():
        try:
            initial = await server.snapshot(thread.id, {"request": None})
            yield f"data: {json.dumps(initial, default=str)}\n\n"
            while True:
                data = await queue.get()
                yield f"data: {data}\n\n"
        finally:
            server.unregister_listener(thread.id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/analytics/run")
async def analytics_run(workflow_input: AnalyticsWorkflowInput) -> Dict[str, Any]:
    try:
        return await run_analytics_workflow(workflow_input)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


__all__ = [
    "AnalyticsWorkflowInput",
    "AnalyticsServer",
    "app",
    "analytics_run",
    "chat_server",
]
