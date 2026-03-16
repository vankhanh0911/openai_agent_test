"use client";

import { useCallback, useEffect, useState } from "react";
import { AgentPanel } from "@/components/agent-panel";
import { AnalyticsConfigPanel } from "@/components/analytics-config-panel";
import { ChatKitPanel } from "@/components/chatkit-panel";
import type { Agent, AgentEvent, GuardrailCheck } from "@/lib/types";
import { fetchBootstrapState, fetchThreadState } from "@/lib/api";

const ANALYTICS_INSTRUCTION_STORAGE_KEY = "analytics_chat_instruction";
const ANALYTICS_SCHEMA_STORAGE_KEY = "analytics_chat_schema";
const ANALYTICS_PORTAL_ID_STORAGE_KEY = "analytics_chat_portal_id";
const ANALYTICS_ACCOUNT_ID_STORAGE_KEY = "analytics_chat_account_id";

function buildSessionContext(
  baseContext: Record<string, any> | null | undefined,
  portalId: string,
  accountId: string
) {
  const normalizedPortalId = portalId.trim();
  const normalizedAccountId = accountId.trim();
  return {
    portal: "aristino",
    mode: "analytics",
    mcp_tool: "read_data",
    ...(baseContext ?? {}),
    portal_id: normalizedPortalId,
    account_id: normalizedAccountId,
    portal_id_configured: Boolean(normalizedPortalId),
    account_id_configured: Boolean(normalizedAccountId),
    session_key: `${
      normalizedPortalId || "default-portal"
    }_${normalizedAccountId || "default-account"}`,
  };
}

export default function Home() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [currentAgent, setCurrentAgent] = useState<string>("");
  const [guardrails, setGuardrails] = useState<GuardrailCheck[]>([]);
  const [context, setContext] = useState<Record<string, any>>({});
  const [threadId, setThreadId] = useState<string | null>(null);
  const [initialThreadId, setInitialThreadId] = useState<string | null>(null);
  const [chatResetKey, setChatResetKey] = useState(0);
  const [instruction, setInstruction] = useState("");
  const [schema, setSchema] = useState("");
  const [draftPortalId, setDraftPortalId] = useState("");
  const [draftAccountId, setDraftAccountId] = useState("");
  const [activePortalId, setActivePortalId] = useState("");
  const [activeAccountId, setActiveAccountId] = useState("");
  const [isSavingPortalConfig, setIsSavingPortalConfig] = useState(false);

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

  const applySnapshot = useCallback(
    (data: any, portalId: string, accountId: string) => {
      setCurrentAgent(data?.current_agent || "Analysis");
      setContext(buildSessionContext(data?.context || {}, portalId, accountId));
      if (Array.isArray(data?.agents)) {
        setAgents(data.agents);
      }
      if (Array.isArray(data?.events)) {
        setEvents(
          normalizeEvents(
            data.events.map((e: any) => ({
              ...e,
              timestamp: new Date(e.timestamp ?? Date.now()),
            }))
          )
        );
      } else {
        setEvents([]);
      }
      if (Array.isArray(data?.guardrails)) {
        setGuardrails(
          data.guardrails.map((g: any) => ({
            ...g,
            timestamp: new Date(g.timestamp ?? Date.now()),
          }))
        );
      } else {
        setGuardrails([]);
      }
    },
    [normalizeEvents]
  );

  const resetConversationView = useCallback((portalId: string, accountId: string) => {
    setThreadId(null);
    setInitialThreadId(null);
    setCurrentAgent("Analysis");
    setEvents([]);
    setGuardrails([]);
    setContext(buildSessionContext({}, portalId, accountId));
  }, []);

  const hydrateState = useCallback(
    async (id: string | null, portalId: string, accountId: string) => {
      if (!id) return;
      const data = await fetchThreadState(id, portalId, accountId);
      if (!data) return;

      applySnapshot(data, portalId, accountId);
    },
    [applySnapshot]
  );

  const reloadBootstrapState = useCallback(
    async (portalId: string, accountId: string) => {
      const bootstrap = await fetchBootstrapState(portalId, accountId);
      if (!bootstrap) {
        resetConversationView(portalId, accountId);
        return;
      }

      const nextThreadId = bootstrap.thread_id || null;
      setInitialThreadId(nextThreadId);
      setThreadId(nextThreadId);
      applySnapshot(bootstrap, portalId, accountId);
    },
    [applySnapshot, resetConversationView]
  );

  useEffect(() => {
    if (threadId) {
      void hydrateState(threadId, activePortalId, activeAccountId);
    }
  }, [threadId, hydrateState, activePortalId, activeAccountId]);

  useEffect(() => {
    let savedInstruction = "";
    let savedSchema = "";
    let savedPortalId = "";
    let savedAccountId = "";

    try {
      savedInstruction =
        window.localStorage.getItem(ANALYTICS_INSTRUCTION_STORAGE_KEY) ?? "";
      savedSchema = window.localStorage.getItem(ANALYTICS_SCHEMA_STORAGE_KEY) ?? "";
      savedPortalId =
        window.localStorage.getItem(ANALYTICS_PORTAL_ID_STORAGE_KEY) ?? "";
      savedAccountId =
        window.localStorage.getItem(ANALYTICS_ACCOUNT_ID_STORAGE_KEY) ?? "";
    } catch (error) {
      console.error("Failed to load analytics config from localStorage:", error);
    }

    setInstruction(savedInstruction);
    setSchema(savedSchema);
    setDraftPortalId(savedPortalId);
    setDraftAccountId(savedAccountId);
    setActivePortalId(savedPortalId);
    setActiveAccountId(savedAccountId);
    resetConversationView(savedPortalId, savedAccountId);
    void reloadBootstrapState(savedPortalId, savedAccountId);
  }, [reloadBootstrapState, resetConversationView]);

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

  const handleSavePortalConfig = useCallback(async () => {
    const nextPortalId = draftPortalId.trim();
    const nextAccountId = draftAccountId.trim();

    setIsSavingPortalConfig(true);
    resetConversationView(nextPortalId, nextAccountId);
    setActivePortalId(nextPortalId);
    setActiveAccountId(nextAccountId);

    try {
      window.localStorage.setItem(ANALYTICS_PORTAL_ID_STORAGE_KEY, nextPortalId);
      window.localStorage.setItem(
        ANALYTICS_ACCOUNT_ID_STORAGE_KEY,
        nextAccountId
      );
    } catch (error) {
      console.error("Failed to persist portal/account config:", error);
    }

    await reloadBootstrapState(nextPortalId, nextAccountId);
    setChatResetKey((value) => value + 1);
    setIsSavingPortalConfig(false);
  }, [draftPortalId, draftAccountId, reloadBootstrapState, resetConversationView]);

  const portalConfigDirty =
    draftPortalId.trim() !== activePortalId || draftAccountId.trim() !== activeAccountId;

  const handleThreadChange = useCallback((id: string | null) => {
    setThreadId(id);
  }, []);

  const handleBindThread = useCallback((id: string) => {
    setThreadId(id);
  }, []);

  const handleResponseEnd = useCallback(() => {
    void hydrateState(threadId, activePortalId, activeAccountId);
  }, [hydrateState, threadId, activePortalId, activeAccountId]);

  return (
    <main className="flex h-screen gap-2 bg-gray-100 p-2">
      <div className="flex h-full w-3/5 flex-col gap-2">
        <AnalyticsConfigPanel
          instruction={instruction}
          schema={schema}
          portalId={draftPortalId}
          accountId={draftAccountId}
          portalConfigDirty={portalConfigDirty}
          portalConfigSaving={isSavingPortalConfig}
          onInstructionChange={setInstruction}
          onSchemaChange={setSchema}
          onPortalIdChange={setDraftPortalId}
          onAccountIdChange={setDraftAccountId}
          onSavePortalConfig={handleSavePortalConfig}
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
        key={`chat-session-${chatResetKey}-${initialThreadId ?? "none"}`}
        initialThreadId={initialThreadId}
        instruction={instruction}
        schema={schema}
        portalId={activePortalId}
        accountId={activeAccountId}
        onThreadChange={handleThreadChange}
        onResponseEnd={handleResponseEnd}
        onRunnerBindThread={handleBindThread}
      />
    </main>
  );
}
