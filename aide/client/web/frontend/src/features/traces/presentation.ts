import type { TraceRecordDTO } from "../../shared/service/protocol";

export function usageLabel(record: TraceRecordDTO, t: (key: string, options?: Record<string, unknown>) => string, language: string): string {
  if (record.usage.state === "not_applicable") return t("traces.notApplicable");
  if (record.usage.state === "unknown" || record.usage.total_tokens === null) return t("traces.usageUnknown");
  return `${record.usage.total_tokens.toLocaleString(language)} ${t("traces.tokensUnit")}`;
}

export function formatDateTime(value: string, language: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString(language, { dateStyle: "medium", timeStyle: "medium" });
}

export function formatDuration(milliseconds: number, language: string): string {
  if (milliseconds >= 1000) return `${(milliseconds / 1000).toLocaleString(language, { maximumFractionDigits: 1 })} s`;
  return `${Math.round(milliseconds).toLocaleString(language)} ms`;
}
