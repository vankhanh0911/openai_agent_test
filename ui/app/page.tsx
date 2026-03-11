"use client";

import { useCallback, useEffect, useState } from "react";
import { AgentPanel } from "@/components/agent-panel";
import { AnalyticsConfigPanel } from "@/components/analytics-config-panel";
import { ChatKitPanel } from "@/components/chatkit-panel";
import type { Agent, AgentEvent, GuardrailCheck } from "@/lib/types";
import { fetchBootstrapState, fetchThreadState } from "@/lib/api";

const ANALYTICS_INSTRUCTION_STORAGE_KEY = "analytics_chat_instruction";
const ANALYTICS_SCHEMA_STORAGE_KEY = "analytics_chat_schema";

export default function Home() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [currentAgent, setCurrentAgent] = useState<string>("");
  const [guardrails, setGuardrails] = useState<GuardrailCheck[]>([]);
  const [context, setContext] = useState<Record<string, any>>({});
  const [threadId, setThreadId] = useState<string | null>(null);
  const [initialThreadId, setInitialThreadId] = useState<string | null>(null);
  const [instruction, setInstruction] = useState("");
  const [schema, setSchema] = useState("");

  const normalizeEvents = useCallback((items: AgentEvent[]) => {
    if (!items.length) return items;
    const now = Date.now();
    const latestNonProgress = items
      .filter((e) => e.type !== "progress_update")
      .reduce((max, e) => Math.max(max, e.timestamp.getTime()), 0);
    const pruned = items.filter((e) => {
      if (e.type !== "progress_update") return true;
      const ts = e.timestamp.getTime();
      // Drop old progress once a newer non-progress exists, or after 15s
      if (latestNonProgress && ts < latestNonProgress) return false;
      if (now - ts > 15000) return false;
      return true;
    });
    return pruned;
  }, []);

  const hydrateState = useCallback(async (id: string | null) => {
    if (!id) return;
    const data = await fetchThreadState(id);
    if (!data) return;

    setCurrentAgent(data.current_agent || "");
    setContext(data.context || {});
    if (Array.isArray(data.agents)) setAgents(data.agents);
    if (Array.isArray(data.events)) {
      setEvents(
        data.events.map((e: any) => ({
          ...e,
          timestamp: new Date(e.timestamp ?? Date.now()),
        }))
      );
    }
    if (Array.isArray(data.guardrails)) {
      setGuardrails(
        data.guardrails.map((g: any) => ({
          ...g,
          timestamp: new Date(g.timestamp ?? Date.now()),
        }))
      );
    }
  }, []);

  useEffect(() => {
    if (threadId) {
      void hydrateState(threadId);
    }
  }, [threadId, hydrateState]);

  useEffect(() => {
    (async () => {
      const bootstrap = await fetchBootstrapState();
      if (!bootstrap) return;
      setInitialThreadId(bootstrap.thread_id || null);
      setThreadId(bootstrap.thread_id || null);
      if (bootstrap.current_agent) setCurrentAgent(bootstrap.current_agent);
      if (Array.isArray(bootstrap.agents)) setAgents(bootstrap.agents);
      if (bootstrap.context) setContext(bootstrap.context);
      if (Array.isArray(bootstrap.events)) {
        setEvents(
          normalizeEvents(
            bootstrap.events.map((e: any) => ({
              ...e,
              timestamp: new Date(e.timestamp ?? Date.now()),
            }))
          )
        );
      }
      if (Array.isArray(bootstrap.guardrails)) {
        setGuardrails(
          bootstrap.guardrails.map((g: any) => ({
            ...g,
            timestamp: new Date(g.timestamp ?? Date.now()),
          }))
        );
      }
    })();
  }, []);

  useEffect(() => {
    try {
      const savedInstruction = window.localStorage.getItem(
        ANALYTICS_INSTRUCTION_STORAGE_KEY
      );
      const savedSchema = window.localStorage.getItem(
        ANALYTICS_SCHEMA_STORAGE_KEY
      );
      if (savedInstruction !== null) {
        setInstruction(savedInstruction);
      }
      if (savedSchema !== null) {
        setSchema(savedSchema);
      }
    } catch (error) {
      console.error("Failed to load analytics config from localStorage:", error);
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        ANALYTICS_INSTRUCTION_STORAGE_KEY,
        instruction
      );
    } catch (error) {
      console.error("Failed to persist instruction to localStorage:", error);
    }
  }, [instruction]);

  useEffect(() => {
    try {
      window.localStorage.setItem(ANALYTICS_SCHEMA_STORAGE_KEY, schema);
    } catch (error) {
      console.error("Failed to persist schema to localStorage:", error);
    }
  }, [schema]);

  const handleThreadChange = useCallback((id: string | null) => {
    setThreadId(id);
  }, []);

  const handleBindThread = useCallback((id: string) => {
    setThreadId(id);
  }, []);

  const handleResponseEnd = useCallback(() => {
    void hydrateState(threadId);
  }, [hydrateState, threadId]);

  return (
    <main className="flex h-screen gap-2 bg-gray-100 p-2">
      <div className="flex h-full w-3/5 flex-col gap-2">
        <AnalyticsConfigPanel
          instruction={instruction}
          schema={schema}
          onInstructionChange={setInstruction}
          onSchemaChange={setSchema}
        />
        <AgentPanel
          agents={agents}
          currentAgent={currentAgent}
          events={events}
          guardrails={guardrails}
          context={context}
        />
      </div>
      <ChatKitPanel
        initialThreadId={initialThreadId}
        instruction={instruction}
        schema={schema}
        onThreadChange={handleThreadChange}
        onResponseEnd={handleResponseEnd}
        onRunnerBindThread={handleBindThread}
      />
    </main>
  );
}
