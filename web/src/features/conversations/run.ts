
export type RunStatus = "submitting" | "accepted" | "running" | "completed" | "failed" | "canceled";

export type ToolStatus = "running" | "completed" | "failed" | "rejected" | "canceled" | "unknown";

export interface ToolActivity {
  toolCallId: string;
  name: string;
  arguments: string;
  status: ToolStatus;
  result?: string | null;
}

export function toolStatusKey(status: ToolStatus): string {
  return `conversation.tool${status[0].toUpperCase()}${status.slice(1)}`;
}
