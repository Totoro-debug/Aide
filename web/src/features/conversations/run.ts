
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



import type {
  ServiceEvent
} from "../../protocol";

export interface LiveRun {
  localId: string;
  runId: string | null;
  prompt: string;
  responseSegments: string[];
  assistantContent: string;
  status: RunStatus;
  tools: ToolActivity[];
  error: string | null;
  cancelRequested: boolean;
  cancellable: boolean;
}

export type RunActivityStatus = Extract<RunStatus, "running" | "completed" | "failed" | "canceled">;

export type RunActivityPart =
  | { kind: "text"; key: string; content: string }
  | { kind: "tool"; tool: ToolActivity };

export interface RunActivity {
  status: RunActivityStatus | null;
  parts: RunActivityPart[];
}

export function newLiveRun(
  localId: string,
  runId: string | null,
  prompt: string,
  status: RunStatus,
): LiveRun {
  return {
    localId,
    runId,
    prompt,
    responseSegments: [],
    assistantContent: "",
    status,
    tools: [],
    error: null,
    cancelRequested: false,
    cancellable: true,
  };
}

export function reduceLiveRunEvent(runs: LiveRun[], event: ServiceEvent): LiveRun[] {
  if (event.run_id === null) return runs;
  const runId = event.run_id;
  if (event.type === "input.recalled") return runs.filter((run) => run.runId !== runId);
  const requestId = typeof event.payload.request_id === "string" ? event.payload.request_id : null;
  const index = runs.findIndex((run) => run.runId === runId
    || (event.type === "input.accepted" && requestId !== null && run.localId === requestId));
  const current = index >= 0 ? runs[index] : newLiveRun(requestId ?? `event-${runId}`, runId,
    typeof event.payload.text === "string" ? event.payload.text : "", "accepted");
  let next = { ...current, runId };
  if (event.type === "input.accepted") {
    next.prompt = typeof event.payload.text === "string" ? event.payload.text : next.prompt;
    if (next.status === "submitting") next.status = "accepted";
  } else if (event.type === "run.started") {
    next.status = "running";
    next.cancellable = true;
  } else if (event.type === "run.output") {
    const message = event.payload.message;
    if (typeof message !== "object" || message === null || Array.isArray(message)) return runs;
    const value = message as Record<string, unknown>;
    const metadata = typeof value.metadata === "object" && value.metadata !== null
      ? value.metadata as Record<string, unknown> : {};
    const content = typeof value.content === "string" ? value.content : "";
    if (!isLiveRunActive(next)) return runs;
    next.status = "running";
    next.cancellable = true;
    if (value.type === "model_response" && metadata._stream_delta === true) {
      next.assistantContent += content;
    } else if (value.type === "tool_call" && typeof metadata.tool_call_id === "string") {
      const toolId = metadata.tool_call_id;
      const tool = next.tools.find((item) => item.toolCallId === toolId);
      if (tool === undefined) {
        next.responseSegments = [...next.responseSegments, next.assistantContent];
        next.assistantContent = "";
      }
      const status: ToolStatus = metadata.status === "success" ? "completed"
        : metadata.status === "error" ? "failed" : metadata.status === "refused" ? "rejected"
          : metadata.status === "cancelled" || metadata.status === "canceled" ? "canceled" : "running";
      const result = typeof metadata.result === "string" ? metadata.result : undefined;
      next.tools = tool === undefined ? [...next.tools, { toolCallId: toolId, name: content,
        arguments: typeof metadata.arguments === "string" ? metadata.arguments : "", status, result }]
        : next.tools.map((item) => item.toolCallId === toolId
          ? { ...item, status, result: result ?? item.result } : item);
    } else if (value.type === "system_control" && metadata._streamed === true) {
      next.status = metadata.finish_reason === "cancelled" ? "canceled" : "failed";
      next.error = content || null;
    }
  } else if (["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
    const finish = event.payload.finish_reason ?? (event.type === "run.cancelled" ? "cancelled"
      : event.type === "run.failed" ? "failed" : "completed");
    next.status = current.status === "canceled" || finish === "cancelled" ? "canceled"
      : finish === "completed" ? "completed" : "failed";
    next.cancellable = false;
  } else {
    return runs;
  }
  if (next.status === "canceled") {
    next = { ...next, tools: next.tools.map((tool) => tool.status === "running"
      ? { ...tool, status: "canceled" } : tool) };
  }
  return index < 0 ? [...runs, next] : runs.map((run, i) => i === index ? next : run);
}

export function isLiveRunActive(run: LiveRun): boolean {
  return run.status === "submitting" || run.status === "accepted" || run.status === "running";
}

export function runStatusKey(status: RunStatus): string {
  return `conversation.${status}`;
}
