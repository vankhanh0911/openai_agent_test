"use client";

interface AnalyticsConfigPanelProps {
  instruction: string;
  schema: string;
  portalId: string;
  accountId: string;
  portalConfigDirty: boolean;
  portalConfigSaving: boolean;
  onInstructionChange: (value: string) => void;
  onSchemaChange: (value: string) => void;
  onPortalIdChange: (value: string) => void;
  onAccountIdChange: (value: string) => void;
  onSavePortalConfig: () => void;
}

export function AnalyticsConfigPanel({
  instruction,
  schema,
  portalId,
  accountId,
  portalConfigDirty,
  portalConfigSaving,
  onInstructionChange,
  onSchemaChange,
  onPortalIdChange,
  onAccountIdChange,
  onSavePortalConfig,
}: AnalyticsConfigPanelProps) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-t-xl border-b border-gray-200 bg-blue-600 px-4 py-3 text-white">
        <h2 className="font-semibold text-sm sm:text-base lg:text-lg">
          Analytics Config
        </h2>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm font-medium">
            <span>PortalId</span>
            <input
              value={portalId}
              onChange={(event) => onPortalIdChange(event.target.value)}
              placeholder="portal-001"
              className="w-40 rounded-md border border-white/40 bg-white/10 px-3 py-1.5 text-sm text-white outline-none transition placeholder:text-white/60 focus:border-white focus:bg-white/15 focus:ring-2 focus:ring-white/30"
            />
          </label>
          <label className="flex items-center gap-2 text-sm font-medium">
            <span>AccountId</span>
            <input
              value={accountId}
              onChange={(event) => onAccountIdChange(event.target.value)}
              placeholder="account-001"
              className="w-40 rounded-md border border-white/40 bg-white/10 px-3 py-1.5 text-sm text-white outline-none transition placeholder:text-white/60 focus:border-white focus:bg-white/15 focus:ring-2 focus:ring-white/30"
            />
          </label>
          <button
            type="button"
            onClick={onSavePortalConfig}
            disabled={!portalConfigDirty || portalConfigSaving}
            className="rounded-md border border-white/50 bg-white px-3 py-1.5 text-sm font-semibold text-blue-700 transition hover:bg-blue-50 disabled:cursor-not-allowed disabled:border-white/20 disabled:bg-white/30 disabled:text-white/70"
          >
            {portalConfigSaving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
      <div className="grid gap-3 p-4">
        <div className="grid gap-1.5">
          <label className="text-xs font-medium uppercase tracking-wide text-gray-600">
            Instruction
          </label>
          <textarea
            value={instruction}
            onChange={(event) => onInstructionChange(event.target.value)}
            placeholder="Leave blank to use the default analytics instruction."
            className="min-h-24 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>
        <div className="grid gap-1.5">
          <label className="text-xs font-medium uppercase tracking-wide text-gray-600">
            Schema
          </label>
          <textarea
            value={schema}
            onChange={(event) => onSchemaChange(event.target.value)}
            placeholder="Leave blank to use the default Aristino schema."
            className="min-h-32 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>
      </div>
    </div>
  );
}
