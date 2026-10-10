import type { ToolActivity } from "./run.ts";

export function safeMarkdownUrl(url: string): string {
  try {
    const parsed = new URL(url, window.location.origin);
    if (["http:", "https:", "mailto:", "tel:"].includes(parsed.protocol)) return url;
  } catch {
    return "";
  }
  return "";
}

export function historyToolActivities(message: Record<string, unknown>): ToolActivity[] {
  if (!Array.isArray(message.tool_calls)) return [];
  return message.tool_calls.flatMap((candidate, index) => {
    if (candidate === null || typeof candidate !== "object" || Array.isArray(candidate)) return [];
    const toolCall = candidate as Record<string, unknown>;
    const rawArguments = toolCall.arguments;
    let argumentsText = "";
    if (typeof rawArguments === "string") {
      argumentsText = rawArguments;
    } else if (rawArguments !== undefined) {
      try {
        argumentsText = JSON.stringify(rawArguments) ?? "";
      } catch {
        argumentsText = "";
      }
    }
    return [{
      toolCallId: typeof toolCall.id === "string" ? toolCall.id : `tool-call-${index}`,
      name: typeof toolCall.name === "string" ? toolCall.name : "",
      arguments: argumentsText,
      status: "unknown" as const,
    }];
  });
}

export function historyRoleLabel(
  role: unknown,
  t: (key: string) => string,
): string {
  if (role === "user") return t("sessions.userMessage");
  if (role === "assistant") return t("sessions.assistantMessage");
  if (role === "tool") return t("sessions.toolMessage");
  return t("sessions.systemMessage");
}

export function historyMessageText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
