import type { NavigationSessionAction } from "../../app/NavigationSidebar";
import {
  ApiError
} from "../../shared/service/api";
import type {
  SessionClaim,
  SessionSummary
} from "../../shared/service/protocol";

export interface PendingSessionAction extends NavigationSessionAction {
  requestId: string;
}

export function restoreSessionActionFocus(trigger: HTMLElement | null): boolean {
  if (!trigger?.isConnected || (trigger instanceof HTMLButtonElement && trigger.disabled)) return false;
  const navigationCollapsed = trigger.closest("#app-sidebar")?.getAttribute("data-open") === "false"
    && window.matchMedia("(max-width: 1023px)").matches;
  const target = navigationCollapsed ? document.getElementById("app-sidebar-toggle") : trigger;
  target?.focus();
  return target !== null;
}

export type SessionLoadState = "idle" | "loading" | "ready" | "error";

export function mergeSessionSummaries(
  current: Record<string, SessionSummary>,
  summaries: SessionSummary[],
): Record<string, SessionSummary> {
  const next = { ...current };
  for (const summary of summaries) {
    const previous = next[summary.id];
    if (previous === undefined || summary.metadata_version > previous.metadata_version
      || (summary.metadata_version === previous.metadata_version
        && Date.parse(summary.updated_at) >= Date.parse(previous.updated_at))) {
      next[summary.id] = summary;
    }
  }
  return next;
}

export interface PendingSubmission {
  localId: string;
  sessionId: string;
  claim: SessionClaim;
  text: string;
}

export interface PendingSessionDeletion {
  claim: SessionClaim;
  requestId: string;
  attempted: boolean;
  title?: string;
}

export function readPendingDeletion(projectId: string): PendingSessionDeletion | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(`aide.session-delete.${projectId}`) ?? "null") as Partial<PendingSessionDeletion> | null;
    if (value?.attempted !== true || typeof value.requestId !== "string"
      || typeof value.claim?.session_id !== "string" || typeof value.claim.workspace_id !== "string"
      || typeof value.claim.reconnect_credential !== "string"
      || !Number.isInteger(value.claim.claim_version) || value.claim.claim_version < 1) return null;
    return value as PendingSessionDeletion;
  } catch {
    return null;
  }
}

export function formatSessionTime(value: string, language: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString(language);
}

export function sessionErrorKey(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.body?.code) {
      case "session_claimed": return "sessions.claimedError";
      case "stale_claim": return "sessions.claimExpiredError";
      case "not_found": return "sessions.notFoundError";
      case "admission_closed": return "sessions.admissionClosedError";
      case "metadata_conflict": return "sessions.renameConflict";
      case "session_not_persisted": return "sessions.notPersistedError";
      case "session_deleting": return "sessions.deletionInProgressError";
      case "session_busy": return "sessions.busyError";
      case "restore_pending": return "sessions.restorePendingError";
      case "validation_error": return "sessions.invalidTitle";
    }
  }
  return "sessions.actionError";
}

export function sessionDeleteErrorKey(error: unknown): string {
  if (error instanceof ApiError && error.body?.code === "persistence_error") {
    return "sessions.deletePersistenceError";
  }
  return sessionErrorKey(error);
}
