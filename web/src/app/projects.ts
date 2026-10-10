import {
  ApiError
} from "../shared/service/api";
import type {
  RegisteredProject
} from "../shared/service/protocol";

export function scheduleText(job: RegisteredProject["saved_jobs"][number], t: (key: string, options?: Record<string, unknown>) => string): string {
  if (job.schedule.kind === "every") return t("projects.everySchedule", { seconds: job.schedule.every_seconds });
  if (job.schedule.kind === "cron") {
    return t("projects.cronSchedule", { expression: job.schedule.cron_expr, timezone: job.schedule.timezone });
  }
  return t("projects.atSchedule", { time: job.schedule.at_time });
}

export function projectPathError(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null;
  const value = error.body?.field_errors.path;
  if (typeof value !== "string" || !value) return null;
  if (value === "must be an absolute directory") return "projects.pathAbsoluteError";
  if (value === "must not overlap Agent Home") return "projects.pathOverlapError";
  if (value === "must name an existing directory") return "projects.pathMissingError";
  return "projects.pathInvalidError";
}

export function projectErrorKey(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.body?.code === "directory_picker_busy") return "projects.pickerBusyError";
    if (error.body?.code === "directory_picker_unavailable") return "projects.pickerUnavailableError";
    if (error.body?.code === "stale_schedule_review") return "projects.staleReviewError";
    if (error.body?.code === "persistence_error") return "projects.persistenceError";
    if (error.body?.code === "admission_closed") return "projects.removalInProgressError";
    if (error.body?.code === "project_removal_failed") return "projects.removalFailedError";
    if (error.body?.code === "project_removal_blocked") return "projects.removalFailedError";
  }
  return "projects.actionError";
}
