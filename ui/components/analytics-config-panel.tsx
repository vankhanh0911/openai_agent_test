"use client";

interface AnalyticsConfigPanelProps {
  instruction: string;
  schema: string;
  onInstructionChange: (value: string) => void;
  onSchemaChange: (value: string) => void;
}

export function AnalyticsConfigPanel({
  instruction,
  schema,
  onInstructionChange,
  onSchemaChange,
}: AnalyticsConfigPanelProps) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-200 bg-blue-600 px-4 py-3 text-white rounded-t-xl">
        <h2 className="font-semibold text-sm sm:text-base lg:text-lg">
          Analytics Config
        </h2>
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
