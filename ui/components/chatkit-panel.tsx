"use client";

import { ChatKit, useChatKit } from "@openai/chatkit-react";
import React from "react";

type ChatKitPanelProps = {
  initialThreadId?: string | null;
  instruction: string;
  schema: string;
  portalId: string;
  accountId: string;
  onThreadChange?: (threadId: string | null) => void;
  onResponseEnd?: () => void;
  onRunnerUpdate?: () => void;
  onRunnerEventDelta?: (events: any[]) => void;
  onRunnerBindThread?: (threadId: string) => void;
};

const CHATKIT_DOMAIN_KEY =
  process.env.NEXT_PUBLIC_CHATKIT_DOMAIN_KEY ?? "domain_pk_localhost_dev";

export function ChatKitPanel({
  initialThreadId,
  instruction,
  schema,
  portalId,
  accountId,
  onThreadChange,
  onResponseEnd,
  onRunnerUpdate,
  onRunnerEventDelta,
  onRunnerBindThread,
}: ChatKitPanelProps) {
  const chatkit = useChatKit({
    api: {
      url: "/chatkit",
      domainKey: CHATKIT_DOMAIN_KEY,
      fetch: async (input, init) => {
        if (!init?.body || typeof init.body !== "string") {
          return fetch(input, init);
        }

        let nextBody = init.body;
        try {
          const payload = JSON.parse(init.body);
          payload.metadata = {
            ...(payload.metadata ?? {}),
            analysis_instruction: instruction,
            analysis_schema: schema,
            analysis_portal_id: portalId,
            analysis_account_id: accountId,
          };
          nextBody = JSON.stringify(payload);
        } catch {
          return fetch(input, init);
        }

        const headers = new Headers(init.headers);
        headers.set("content-type", "application/json");
        return fetch(input, {
          ...init,
          headers,
          body: nextBody,
        });
      },
    },
    composer: {
      placeholder: "Ask an analytics question...",
    },
    history: {
      enabled: true,
    },
    theme: {
      colorScheme: "light",
      radius: "round",
      density: "normal",
      color: {
        accent: {
          primary: "#2563eb",
          level: 1,
        },
      },
    },
    initialThread: initialThreadId ?? null,
    startScreen: {
      greeting: "Ask about Aristino marketing performance, revenue, channels, stores, or products.",
      prompts: [
        {
          label: "Revenue by channel",
          prompt: "Tong doanh thu theo transaction_source trong 30 ngay qua la gi?",
        },
        {
          label: "Top stores",
          prompt: "Top 10 cua hang co doanh thu cao nhat thang nay la gi?",
        },
        {
          label: "Product discount",
          prompt: "Nhom san pham nao dang co tong discount cao nhat trong quy nay?",
        },
      ],
    },
    threadItemActions: {
      feedback: false,
    },
    onThreadChange: ({ threadId }) => onThreadChange?.(threadId ?? null),
    onResponseEnd: () => onResponseEnd?.(),
    onError: ({ error }) => {
      console.error("ChatKit error", error);
    },
    onEffect: async ({ name }) => {
      if (name === "runner_state_update") {
        onRunnerUpdate?.();
      }
      if (name === "runner_event_delta") {
        onRunnerEventDelta?.((arguments as any)?.[0]?.data?.events ?? []);
      }
      if (name === "runner_bind_thread") {
        const tid = (arguments as any)?.[0]?.data?.thread_id;
        if (tid) {
          onRunnerBindThread?.(tid);
        }
      }
    },
  });

  return (
    <div className="flex flex-col h-full flex-1 bg-white shadow-sm border border-gray-200 border-t-0 rounded-xl">
      <div className="bg-blue-600 text-white h-12 px-4 flex items-center rounded-t-xl">
        <h2 className="font-semibold text-sm sm:text-base lg:text-lg">
          Analytics Chat
        </h2>
      </div>
      <div className="flex-1 overflow-hidden pb-1.5">
        <ChatKit
          control={chatkit.control}
          className="block h-full w-full"
          style={{ height: "100%", width: "100%" }}
        />
      </div>
    </div>
  );
}
