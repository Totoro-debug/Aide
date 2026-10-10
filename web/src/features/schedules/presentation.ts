import {
ApiError
} from "../../api";
import type {
ScheduleHistoryGroup,
ScheduleJob
} from "../../protocol";

export type ScheduleLoadState = "idle" | "loading" | "ready" | "error";

export function scheduleRouteHref(
  projectId: string | null,
  workspaceDirectory: string | null,
  sessionId: string | null,
  jobId?: string,
): string {
  const base = projectId === null
    ? "/chat/schedule"
    : `/projects/${encodeURIComponent(projectId)}/schedule`;
  const path = jobId === undefined ? base : `${base}/jobs/${encodeURIComponent(jobId)}/history`;
  const query = new URLSearchParams();
  if (projectId === null && workspaceDirectory !== null) query.set("directory", workspaceDirectory);
  if (sessionId !== null) query.set("session", sessionId);
  const queryString = query.toString();
  return queryString === "" ? path : `${path}?${queryString}`;
}

export function sessionRouteHref(projectId: string | null, workspaceDirectory: string | null, sessionId: string | null): string {
  if (projectId !== null) {
    const query = sessionId === null ? "" : `?session=${encodeURIComponent(sessionId)}`;
    return `/projects/${encodeURIComponent(projectId)}${query}`;
  }
  const query = new URLSearchParams();
  if (workspaceDirectory !== null) query.set("directory", workspaceDirectory);
  if (sessionId !== null) query.set("session", sessionId);
  return `/chat${query.size === 0 ? "" : `?${query.toString()}`}`;
}

export function scheduleErrorKey(error: unknown, fallback: string): string {
  if (error instanceof ApiError) {
    switch (error.body?.code) {
      case "not_found": return "schedule.notFound";
      case "forbidden": return "schedule.forbidden";
      case "admission_closed": return "schedule.admissionClosed";
      case "schedule_unavailable": return "schedule.unavailable";
      case "request_reused": return "schedule.requestReused";
      case "schedule_changed": return "schedule.changed";
    }
  }
  return fallback;
}

export type ScheduleHistoryLoadState = "idle" | "loading" | "ready" | "error";

export function scheduleHistoryGroupKey(group: ScheduleHistoryGroup, index: number): string {
  const firstMessage = group.messages[0];
  const timestamp = typeof firstMessage?.timestamp === "string" ? firstMessage.timestamp : "unknown";
  return `${group.started_at ?? "unknown"}-${timestamp}-${index}`;
}

export function scheduleHistoryTime(
  value: string | null,
  language: string,
  t: (key: string) => string,
): string {
  if (value === null) return t("schedule.historyUnknownTime");
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? t("schedule.historyUnknownTime")
    : parsed.toLocaleString(language);
}

export function scheduleJobRule(
  schedule: ScheduleJob["schedule"],
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  if (schedule.kind === "at") return t("schedule.ruleAt", { value: schedule.at_time });
  if (schedule.kind === "every") return t("schedule.ruleEvery", { value: schedule.every_seconds });
  return t("schedule.ruleCron", { expression: schedule.cron_expr, timezone: schedule.timezone });
}

export function scheduleJobLastResult(
  job: ScheduleJob,
  language: string,
  t: (key: string) => string,
): string {
  if (job.active) return t("schedule.runningNow");
  if (job.state.last_finished_at_ms === null) return t("schedule.neverRun");
  return new Date(job.state.last_finished_at_ms).toLocaleString(language);
}
