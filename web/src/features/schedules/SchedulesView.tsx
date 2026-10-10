import * as Dialog from "@radix-ui/react-dialog";
import {
  Activity,
  ArrowLeft,
  Ban,
  BookOpen,
  CalendarClock,
  CircleAlert,
  CircleCheck,
  Clock3,
  Eye,
  FolderOpen,
  Info,
  MessageSquare,
  Plus,
  RefreshCw,
  Trash2,
  TriangleAlert,
  X
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  ApiError,
  createRequestId,
  createScheduleJob,
  deleteScheduleJob,
  enterChatWorkspace,
  getProjectSessions,
  getScheduleJob,
  getScheduleJobHistory,
  getScheduleJobs
} from "../../api";
import commonStyles from "../../App.module.css";
import type {
  ProjectScheduleKind,
  RegisteredProject,
  ScheduleHistoryGroup,
  ScheduleHistoryResultState,
  ScheduleJob,
  ScheduleJobHistoryResponse,
  ScheduleJobInput,
  ScheduleJobStatus,
  ScheduleJobsResponse,
  ScheduleStatus
} from "../../protocol";
import type { AuthState, ConnectionState } from "../../shared/service/types.ts";
import { HistoryMessageView } from "../conversations/history.tsx";
import type { ScheduleHistoryLoadState, ScheduleLoadState } from "./presentation.ts";
import { scheduleErrorKey, scheduleHistoryGroupKey, scheduleHistoryTime, scheduleJobLastResult, scheduleJobRule, scheduleRouteHref, sessionRouteHref } from "./presentation.ts";
import schedulesStyles from "./Schedules.module.css";

const styles = { ...commonStyles, ...schedulesStyles };

interface ScheduleJobsViewProps {
  authState: AuthState;
  connectionState: ConnectionState;
  projects: RegisteredProject[];
}

export function ScheduleJobsView({ authState, connectionState, projects }: ScheduleJobsViewProps) {
  const { projectId: routeProjectId } = useParams();
  const [searchParams] = useSearchParams();
  const projectId = routeProjectId ?? null;
  const workspaceDirectory = projectId === null ? searchParams.get("directory") : null;
  const sessionId = searchParams.get("session");
  return (
    <ScheduleJobsContent
      key={`${projectId ?? "chat"}:${workspaceDirectory ?? ""}`}
      authState={authState}
      connectionState={connectionState}
      projects={projects}
      projectId={projectId}
      workspaceDirectory={workspaceDirectory}
      sessionId={sessionId}
    />
  );
}

function ScheduleJobsContent({
  authState,
  connectionState,
  projects,
  projectId,
  workspaceDirectory,
  sessionId,
}: ScheduleJobsViewProps & {
  projectId: string | null;
  workspaceDirectory: string | null;
  sessionId: string | null;
}) {
  const { i18n, t } = useTranslation();
  const project = projectId === null ? undefined : projects.find((item) => item.project_id === projectId);
  const scopeMissing = projectId === null ? workspaceDirectory === null : project === undefined;
  const projectAvailable = projectId === null
    ? workspaceDirectory !== null && workspaceDirectory.trim() !== ""
    : project?.available === true;
  const [workspaceId, setWorkspaceId] = useState<string | null>(null);
  const [jobs, setJobs] = useState<ScheduleJob[]>([]);
  const jobsRef = useRef<ScheduleJob[]>([]);
  const [scheduleStatus, setScheduleStatus] = useState<ScheduleStatus | null>(null);
  const [loadState, setLoadState] = useState<ScheduleLoadState>("idle");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [kind, setKind] = useState<ProjectScheduleKind>("at");
  const [message, setMessage] = useState("");
  const [title, setTitle] = useState("");
  const [atTime, setAtTime] = useState("");
  const [everySeconds, setEverySeconds] = useState("60");
  const [cronExpr, setCronExpr] = useState("0 * * * *");
  const [timezone, setTimezone] = useState("UTC");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [createBusy, setCreateBusy] = useState(false);
  const [deleteJob, setDeleteJob] = useState<ScheduleJob | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [detailJob, setDetailJob] = useState<ScheduleJob | null>(null);
  const [detailBusy, setDetailBusy] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const scopeVersionRef = useRef(0);
  const lifecycleVersionRef = useRef(0);
  const loadPendingRef = useRef(false);
  const workspaceIdRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const createRequestIdRef = useRef<string | null>(null);
  const createInputRef = useRef<ScheduleJobInput | null>(null);
  const deleteRequestIdRef = useRef<string | null>(null);
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null);
  const detailRequestVersionRef = useRef(0);
  const detailJobIdRef = useRef<string | null>(null);
  const detailTriggerRef = useRef<HTMLButtonElement | null>(null);
  const errorSummaryRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (notice === null) return;
    const timer = window.setTimeout(() => setNotice(null), 10_000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      scopeVersionRef.current += 1;
    };
  }, []);

  const loadSchedule = useCallback(async (initial: boolean) => {
    if (authState !== "ready" || connectionState !== "online" || !projectAvailable) return;
    if (!initial && loadPendingRef.current) return;
    loadPendingRef.current = true;
    const lifecycleVersion = lifecycleVersionRef.current;
    const version = scopeVersionRef.current + 1;
    scopeVersionRef.current = version;
    if (initial || jobsRef.current.length === 0) setLoadState("loading");
    setLoadError(null);
    try {
      let nextWorkspaceId = workspaceIdRef.current;
      if (nextWorkspaceId === null) {
        const sessions = projectId === null
          ? await enterChatWorkspace(workspaceDirectory ?? undefined)
          : await getProjectSessions(projectId, { limit: 1 });
        if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion || scopeVersionRef.current !== version) return;
        nextWorkspaceId = sessions.workspace_id;
        workspaceIdRef.current = nextWorkspaceId;
        setWorkspaceId(nextWorkspaceId);
      }
      const response: ScheduleJobsResponse = await getScheduleJobs(nextWorkspaceId);
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion || scopeVersionRef.current !== version) return;
      jobsRef.current = response.jobs;
      setJobs(response.jobs);
      setScheduleStatus(response.status);
      if (detailJobIdRef.current !== null) {
        const currentDetail = response.jobs.find((job) => job.job_id === detailJobIdRef.current);
        detailRequestVersionRef.current += 1;
        setDetailBusy(false);
        if (currentDetail === undefined) {
          detailJobIdRef.current = null;
        }
        setDetailJob(currentDetail ?? null);
      }
      setLoadState("ready");
    } catch (error) {
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion || scopeVersionRef.current !== version) return;
      setLoadState("error");
      setLoadError(scheduleErrorKey(error, "schedule.loadError"));
    } finally {
      if (mountedRef.current && lifecycleVersionRef.current === lifecycleVersion && scopeVersionRef.current === version) {
        loadPendingRef.current = false;
      }
    }
  }, [authState, connectionState, projectAvailable, projectId, workspaceDirectory]);

  useEffect(() => {
    lifecycleVersionRef.current += 1;
    scopeVersionRef.current += 1;
    loadPendingRef.current = false;
    detailRequestVersionRef.current += 1;
    detailJobIdRef.current = null;
    setDetailJob(null);
    setDetailBusy(false);
    setDetailError(null);
    setCreateBusy(false);
    setDeleteBusy(false);
    if (createRequestIdRef.current !== null) setActionError("schedule.resultUnknown");
    if (deleteRequestIdRef.current !== null) setDeleteError("schedule.resultUnknown");
    if (authState !== "ready" || !projectAvailable) setDeleteJob(null);
    if (authState === "ready" && projectAvailable && connectionState !== "online") {
      setLoadState("idle");
      return;
    }
    workspaceIdRef.current = null;
    setWorkspaceId(null);
    jobsRef.current = [];
    setJobs([]);
    setScheduleStatus(null);
    setLoadState("idle");
    if (authState !== "ready" || connectionState !== "online" || !projectAvailable) return;
    void loadSchedule(true);
    const timer = window.setInterval(() => void loadSchedule(false), 5000);
    return () => {
      lifecycleVersionRef.current += 1;
      scopeVersionRef.current += 1;
      window.clearInterval(timer);
    };
  }, [authState, connectionState, loadSchedule, projectAvailable, projectId, workspaceDirectory]);

  function clearCreateAttempt() {
    createRequestIdRef.current = null;
    createInputRef.current = null;
    setActionError(null);
  }

  function focusErrorSummary() {
    window.setTimeout(() => errorSummaryRef.current?.focus(), 0);
  }

  function validateCreateForm(): Record<string, string> {
    const errors: Record<string, string> = {};
    if (!message.trim()) errors.message = "schedule.fieldRequired";
    if (title.trim() === "" && title.length > 0) errors.title = "schedule.fieldRequired";
    if (kind === "at" && !atTime.trim()) errors.at_time = "schedule.fieldRequired";
    if (kind === "every") {
      const parsed = Number(everySeconds);
      if (!Number.isInteger(parsed) || parsed < 1) errors.every_seconds = "schedule.everyPositive";
    }
    if (kind === "cron") {
      if (!cronExpr.trim()) errors.cron_expr = "schedule.fieldRequired";
      if (!timezone.trim()) errors.timezone = "schedule.fieldRequired";
    }
    return errors;
  }

  function createInput(): ScheduleJobInput {
    const input: ScheduleJobInput = {
      message: message.trim(),
      kind,
    };
    if (title.trim()) input.title = title.trim();
    if (kind === "at") input.at_time = atTime.trim();
    if (kind === "every") input.every_seconds = Number(everySeconds);
    if (kind === "cron") {
      input.cron_expr = cronExpr.trim();
      input.timezone = timezone.trim();
    }
    return input;
  }

  async function submitCreate(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (workspaceId === null || createBusy || connectionState !== "online") return;
    const localErrors = validateCreateForm();
    if (Object.keys(localErrors).length > 0) {
      setFieldErrors(localErrors);
      setActionError("schedule.validationSummary");
      focusErrorSummary();
      return;
    }
    const requestId = createRequestIdRef.current ?? createRequestId();
    createRequestIdRef.current = requestId;
    const input = createInputRef.current ?? createInput();
    createInputRef.current = input;
    const lifecycleVersion = lifecycleVersionRef.current;
    setCreateBusy(true);
    setFieldErrors({});
    setActionError(null);
    setNotice(null);
    try {
      const response = await createScheduleJob(workspaceId, input, requestId);
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion || workspaceIdRef.current !== response.workspace_id) return;
      createRequestIdRef.current = null;
      createInputRef.current = null;
      setNotice("schedule.createdNotice");
      setMessage("");
      setTitle("");
      setAtTime("");
      await loadSchedule(true);
    } catch (error) {
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion) return;
      if (error instanceof ApiError) {
        createRequestIdRef.current = null;
        createInputRef.current = null;
        const serverErrors = error.body?.field_errors ?? {};
        setFieldErrors(Object.fromEntries(Object.keys(serverErrors).map((key) => [key, "schedule.fieldInvalid"])));
        setActionError(scheduleErrorKey(error, "schedule.createError"));
        if (Object.keys(serverErrors).length > 0) focusErrorSummary();
      } else {
        setActionError("schedule.resultUnknown");
      }
    } finally {
      if (mountedRef.current && lifecycleVersionRef.current === lifecycleVersion) setCreateBusy(false);
    }
  }

  async function openDetail(job: ScheduleJob, event: React.MouseEvent<HTMLButtonElement>) {
    detailTriggerRef.current = event.currentTarget;
    detailJobIdRef.current = job.job_id;
    const lifecycleVersion = lifecycleVersionRef.current;
    const requestVersion = detailRequestVersionRef.current + 1;
    detailRequestVersionRef.current = requestVersion;
    setDetailJob(job);
    setDetailBusy(true);
    setDetailError(null);
    const currentWorkspaceId = workspaceIdRef.current;
    if (currentWorkspaceId === null) {
      setDetailBusy(false);
      setDetailError("schedule.unavailable");
      return;
    }
    try {
      const response = await getScheduleJob(currentWorkspaceId, job.job_id);
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion || detailRequestVersionRef.current !== requestVersion) return;
      setDetailJob(response.job);
      setJobs((current) => current.map((candidate) => (
        candidate.job_id === response.job.job_id ? response.job : candidate
      )));
      setScheduleStatus(response.status);
    } catch (error) {
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion || detailRequestVersionRef.current !== requestVersion) return;
      setDetailError(scheduleErrorKey(error, "schedule.detailError"));
    } finally {
      if (mountedRef.current && lifecycleVersionRef.current === lifecycleVersion && detailRequestVersionRef.current === requestVersion) {
        setDetailBusy(false);
      }
    }
  }

  function openDelete(job: ScheduleJob, event: React.MouseEvent<HTMLButtonElement>) {
    deleteTriggerRef.current = event.currentTarget;
    deleteRequestIdRef.current = createRequestId();
    setDeleteError(null);
    setDeleteJob(job);
  }

  async function confirmDelete() {
    if (workspaceId === null || deleteJob === null || deleteBusy || connectionState !== "online") return;
    const requestId = deleteRequestIdRef.current ?? createRequestId();
    deleteRequestIdRef.current = requestId;
    const lifecycleVersion = lifecycleVersionRef.current;
    setDeleteBusy(true);
    setDeleteError(null);
    setNotice(null);
    try {
      const response = await deleteScheduleJob(workspaceId, deleteJob.job_id, requestId);
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion || workspaceIdRef.current !== response.workspace_id) return;
      deleteRequestIdRef.current = null;
      setJobs((current) => current.filter((job) => job.job_id !== deleteJob.job_id));
      setScheduleStatus(response.status);
      setDeleteJob(null);
      setNotice("schedule.deletedNotice");
      await loadSchedule(true);
    } catch (error) {
      if (!mountedRef.current || lifecycleVersionRef.current !== lifecycleVersion) return;
      if (error instanceof ApiError) {
        deleteRequestIdRef.current = null;
        setDeleteError(scheduleErrorKey(error, "schedule.deleteError"));
      } else {
        setDeleteError("schedule.resultUnknown");
      }
    } finally {
      if (mountedRef.current && lifecycleVersionRef.current === lifecycleVersion) setDeleteBusy(false);
    }
  }

  const authUnavailable = authState !== "ready";
  const createFieldsLocked = createBusy || createRequestIdRef.current !== null;
  const summaryFields = Object.keys(fieldErrors);
  return (
    <section className={styles.schedulePage} aria-labelledby="schedule-heading">
      <div className={styles.pageHeading}>
        <div>
          <Link className={styles.backLink} to={sessionRouteHref(projectId, workspaceDirectory, sessionId)}>
            <ArrowLeft size={15} aria-hidden="true" />
            {t("controls.backToSessions")}
          </Link>
          <p className={styles.eyebrow}>{t("nav.schedule")}</p>
          <h1 id="schedule-heading" tabIndex={-1}>{t("schedule.title")}</h1>
          <p className={styles.pageDescription}>
            {project?.name ?? workspaceDirectory ?? t("schedule.title")}
            {project !== undefined ? ` · ${project.path}` : ""}
          </p>
        </div>
        <div className={styles.pageActions}>
          <Link className={styles.secondaryButton} to={sessionRouteHref(projectId, workspaceDirectory, sessionId)}>
            <MessageSquare size={15} aria-hidden="true" />
            {t("controls.openSessions")}
          </Link>
          <button
            className={styles.iconButton}
            type="button"
            aria-label={t("controls.refreshSchedule")}
            title={t("controls.refreshSchedule")}
            disabled={authUnavailable || connectionState !== "online" || loadState === "loading" || !projectAvailable}
            onClick={() => void loadSchedule(true)}
          >
            <RefreshCw size={16} className={loadState === "loading" ? styles.spin : undefined} aria-hidden="true" />
          </button>
        </div>
      </div>

      {!authUnavailable && connectionState !== "online" ? (
        <div className={styles.actionError} role="status">
          <CircleAlert size={16} aria-hidden="true" />
          <span>{t("schedule.disconnected")}</span>
        </div>
      ) : null}

      {notice !== null ? (
        <div className={styles.restoreResultNotice} role="status" aria-live="polite">
          <div className={styles.restoreResultContent}>
            <CircleCheck size={16} aria-hidden="true" />
            <span>{t(notice)}</span>
          </div>
          <div className={styles.restoreResultActions}>
            <button
              className={styles.iconButton}
              type="button"
              aria-label={t("controls.close")}
              title={t("controls.close")}
              onClick={() => setNotice(null)}
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      ) : null}
      {actionError !== null && summaryFields.length === 0 ? (
        <div className={styles.errorBanner} role="alert">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{t(actionError)}</span>
        </div>
      ) : null}

      {authUnavailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><Info size={22} /></div>
          <div><h2>{t("schedule.authenticationRequired")}</h2><p>{t("status.unavailable")}</p></div>
        </div>
      ) : scopeMissing ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true"><CircleAlert size={22} /></div>
          <div><h2>{t(projectId === null ? "schedule.workspaceNotFound" : "schedule.notFound")}</h2><Link className={styles.secondaryButton} to={sessionRouteHref(projectId, workspaceDirectory, sessionId)}>{t("controls.backToSessions")}</Link></div>
        </div>
      ) : projectId !== null && !projectAvailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><FolderOpen size={22} /></div>
          <div><h2>{t("schedule.projectUnavailable")}</h2><p>{project?.path}</p></div>
        </div>
      ) : (
        <>
          {scheduleStatus !== null ? (
            <dl className={styles.scheduleStatusGrid} aria-label={t("schedule.statusTitle")}>
              <div className={styles.scheduleStatusMetric}>
                <dt>{t("schedule.admission")}</dt>
                <dd data-state={scheduleStatus.admitted ? "active" : "paused"}>
                  <span className={styles.statusDot} aria-hidden="true" />
                  {scheduleStatus.admitted ? t("schedule.admitted") : t("schedule.paused")}
                </dd>
              </div>
              <div className={styles.scheduleStatusMetric}>
                <dt>{t("schedule.health")}</dt>
                <dd>{scheduleStatus.status === "available" ? t("schedule.available") : t("schedule.faulted")}</dd>
              </div>
              <div className={styles.scheduleStatusMetric}>
                <dt>{t("schedule.activeJobCount")}</dt>
                <dd>{scheduleStatus.active_job_count}</dd>
              </div>
            </dl>
          ) : null}

          {summaryFields.length > 0 ? (
            <div
              ref={errorSummaryRef}
              className={styles.errorSummary}
              role="alert"
              tabIndex={-1}
              aria-labelledby="schedule-error-summary-title"
            >
              <strong id="schedule-error-summary-title">{t(actionError ?? "schedule.validationSummary")}</strong>
              <ul>
                {summaryFields.map((field) => (
                  <li key={field}>
                    <a href={`#schedule-field-${field}`}>{t(`schedule.fields.${field}`)}</a>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className={styles.scheduleLayout}>
            <form className={styles.scheduleForm} onSubmit={(event) => void submitCreate(event)} noValidate>
              <div className={styles.schedulePanelHeader}>
                <div>
                  <p className={styles.eyebrow}>{t("schedule.createEyebrow")}</p>
                  <h2>{t("schedule.createTitle")}</h2>
                </div>
                <CalendarClock size={21} aria-hidden="true" />
              </div>
              <label className={styles.fieldLabel} htmlFor="schedule-field-message">{t("schedule.fields.message")}</label>
              <textarea
                id="schedule-field-message"
                className={styles.scheduleTextarea}
                rows={4}
                value={message}
                disabled={createFieldsLocked}
                aria-invalid={fieldErrors.message !== undefined}
                aria-describedby={fieldErrors.message !== undefined ? "schedule-error-message" : undefined}
                onChange={(event) => { clearCreateAttempt(); setMessage(event.target.value); }}
              />
              {fieldErrors.message !== undefined ? <p className={styles.fieldError} id="schedule-error-message">{t(fieldErrors.message)}</p> : null}
              <label className={styles.fieldLabel} htmlFor="schedule-field-title">{t("schedule.fields.title")}</label>
              <input
                id="schedule-field-title"
                className={styles.textInput}
                type="text"
                value={title}
                disabled={createFieldsLocked}
                aria-invalid={fieldErrors.title !== undefined}
                aria-describedby={fieldErrors.title !== undefined ? "schedule-error-title" : undefined}
                onChange={(event) => { clearCreateAttempt(); setTitle(event.target.value); }}
              />
              {fieldErrors.title !== undefined ? <p className={styles.fieldError} id="schedule-error-title">{t(fieldErrors.title)}</p> : null}
              <fieldset className={styles.scheduleKindFieldset} disabled={createFieldsLocked}>
                <legend className={styles.fieldLabel}>{t("schedule.fields.kind")}</legend>
                <div className={styles.scheduleKindOptions} role="group" aria-label={t("schedule.fields.kind")}>
                  {(["at", "every", "cron"] as ProjectScheduleKind[]).map((option) => (
                    <button
                      className={kind === option ? styles.scheduleKindActive : styles.scheduleKindButton}
                      key={option}
                      type="button"
                      aria-pressed={kind === option}
                      onClick={() => { clearCreateAttempt(); setKind(option); setFieldErrors({}); }}
                    >
                      {t(`schedule.kinds.${option}`)}
                    </button>
                  ))}
                </div>
              </fieldset>
              {kind === "at" ? (
                <>
                  <label className={styles.fieldLabel} htmlFor="schedule-field-at_time">{t("schedule.fields.at_time")}</label>
                  <input
                    id="schedule-field-at_time"
                    className={styles.textInput}
                    type="text"
                    placeholder={t("schedule.atPlaceholder")}
                    value={atTime}
                    disabled={createFieldsLocked}
                    aria-invalid={fieldErrors.at_time !== undefined}
                    aria-describedby={fieldErrors.at_time !== undefined ? "schedule-error-at_time" : undefined}
                    onChange={(event) => { clearCreateAttempt(); setAtTime(event.target.value); }}
                  />
                  {fieldErrors.at_time !== undefined ? <p className={styles.fieldError} id="schedule-error-at_time">{t(fieldErrors.at_time)}</p> : null}
                </>
              ) : null}
              {kind === "every" ? (
                <>
                  <label className={styles.fieldLabel} htmlFor="schedule-field-every_seconds">{t("schedule.fields.every_seconds")}</label>
                  <input
                    id="schedule-field-every_seconds"
                    className={styles.textInput}
                    type="number"
                    min={1}
                    step={1}
                    value={everySeconds}
                    disabled={createFieldsLocked}
                    aria-invalid={fieldErrors.every_seconds !== undefined}
                    aria-describedby={fieldErrors.every_seconds !== undefined ? "schedule-error-every_seconds" : undefined}
                    onChange={(event) => { clearCreateAttempt(); setEverySeconds(event.target.value); }}
                  />
                  {fieldErrors.every_seconds !== undefined ? <p className={styles.fieldError} id="schedule-error-every_seconds">{t(fieldErrors.every_seconds)}</p> : null}
                </>
              ) : null}
              {kind === "cron" ? (
                <>
                  <label className={styles.fieldLabel} htmlFor="schedule-field-cron_expr">{t("schedule.fields.cron_expr")}</label>
                  <input
                    id="schedule-field-cron_expr"
                    className={styles.textInput}
                    type="text"
                    value={cronExpr}
                    disabled={createFieldsLocked}
                    aria-invalid={fieldErrors.cron_expr !== undefined}
                    aria-describedby={fieldErrors.cron_expr !== undefined ? "schedule-error-cron_expr" : undefined}
                    onChange={(event) => { clearCreateAttempt(); setCronExpr(event.target.value); }}
                  />
                  {fieldErrors.cron_expr !== undefined ? <p className={styles.fieldError} id="schedule-error-cron_expr">{t(fieldErrors.cron_expr)}</p> : null}
                  <label className={styles.fieldLabel} htmlFor="schedule-field-timezone">{t("schedule.fields.timezone")}</label>
                  <input
                    id="schedule-field-timezone"
                    className={styles.textInput}
                    type="text"
                    value={timezone}
                    disabled={createFieldsLocked}
                    aria-invalid={fieldErrors.timezone !== undefined}
                    aria-describedby={fieldErrors.timezone !== undefined ? "schedule-error-timezone" : undefined}
                    onChange={(event) => { clearCreateAttempt(); setTimezone(event.target.value); }}
                  />
                  {fieldErrors.timezone !== undefined ? <p className={styles.fieldError} id="schedule-error-timezone">{t(fieldErrors.timezone)}</p> : null}
                </>
              ) : null}
              <p className={styles.fieldHint}>{t("schedule.formHint")}</p>
              <button
                className={styles.primaryButton}
                type="submit"
                disabled={createBusy || connectionState !== "online" || workspaceId === null || loadState === "loading"}
              >
                <Plus size={16} aria-hidden="true" />
                {createBusy ? t("schedule.creating") : createRequestIdRef.current !== null ? t("schedule.retryCreate") : t("schedule.create")}
              </button>
            </form>

            <section className={styles.scheduleJobsPanel} aria-labelledby="schedule-jobs-heading">
              <div className={styles.schedulePanelHeader}>
                <div>
                  <p className={styles.eyebrow}>{t("schedule.listEyebrow")}</p>
                  <h2 id="schedule-jobs-heading">{t("schedule.listTitle")}</h2>
                </div>
                <span className={styles.scheduleCount}>{jobs.length}</span>
              </div>
              {loadState === "loading" && jobs.length === 0 ? (
                <div className={styles.scheduleEmptyState} role="status"><RefreshCw size={18} className={styles.spin} aria-hidden="true" />{t("schedule.loading")}</div>
              ) : loadState === "error" && jobs.length === 0 ? (
                <div className={styles.scheduleEmptyState} role="alert"><CircleAlert size={18} aria-hidden="true" /><span>{t(loadError ?? "schedule.loadError")}</span><button className={styles.secondaryButton} type="button" onClick={() => void loadSchedule(true)}>{t("controls.retry")}</button></div>
              ) : jobs.length === 0 ? (
                <div className={styles.scheduleEmptyState} role="status"><Clock3 size={18} aria-hidden="true" /><span>{t(projectId === null ? "schedule.emptyWorkspace" : "schedule.empty")}</span></div>
              ) : (
                <ul className={styles.scheduleJobList}>
                  {jobs.map((job) => (
                    <li className={styles.scheduleJobItem} key={job.job_id}>
                      <div className={styles.scheduleJobHeader}>
                        <div className={styles.scheduleJobTitle}>
                          <h3>{job.title}</h3>
                          <p>{job.message}</p>
                        </div>
                        <span className={styles.scheduleJobStatus} data-state={job.status} role="status">
                          {scheduleJobStatusIcon(job.status)}
                          {t(`schedule.jobStatus.${job.status}`)}
                        </span>
                      </div>
                      <dl className={styles.scheduleJobDetails}>
                        <div><dt>{t("schedule.fields.kind")}</dt><dd>{scheduleJobRule(job.schedule, t)}</dd></div>
                        <div><dt>{t("schedule.fields.session")}</dt><dd>{job.session_id}</dd></div>
                        <div><dt>{t("schedule.fields.lastResult")}</dt><dd>{scheduleJobLastResult(job, i18n.language, t)}</dd></div>
                      </dl>
                      {job.state.last_error !== null ? <p className={styles.scheduleJobError}>{job.state.last_error}</p> : null}
                      <div className={styles.scheduleJobActions}>
                        <Link
                          className={styles.secondaryButton}
                          to={scheduleRouteHref(projectId, workspaceDirectory, sessionId, job.job_id)}
                        >
                          <BookOpen size={15} aria-hidden="true" />
                          {t("schedule.history")}
                        </Link>
                        <button
                          className={styles.secondaryButton}
                          type="button"
                          disabled={detailBusy}
                          onClick={(event) => void openDetail(job, event)}
                        >
                          <Eye size={15} aria-hidden="true" />
                          {t("schedule.inspect")}
                        </button>
                        <button
                          className={styles.dangerButton}
                          type="button"
                          disabled={deleteBusy}
                          onClick={(event) => openDelete(job, event)}
                        >
                          <Trash2 size={15} aria-hidden="true" />
                          {t("schedule.delete")}
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </>
      )}

      <Dialog.Root
        open={deleteJob !== null}
        onOpenChange={(open) => {
          if (!open && !deleteBusy) {
            deleteRequestIdRef.current = null;
            setDeleteJob(null);
            setDeleteError(null);
          }
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={styles.dialogContent}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              if (deleteTriggerRef.current?.isConnected) deleteTriggerRef.current.focus();
              else document.getElementById("schedule-heading")?.focus();
            }}
          >
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{t("schedule.deleteTitle")}</Dialog.Title>
                <Dialog.Description className={styles.dialogDescription}>
                  {t("schedule.deleteDescription", { title: deleteJob?.title ?? "" })}
                </Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button className={styles.iconButton} type="button" aria-label={t("controls.close")} disabled={deleteBusy}>
                  <X size={17} aria-hidden="true" />
                </button>
              </Dialog.Close>
            </div>
            {deleteJob?.active ? <p className={styles.dialogWarning}><TriangleAlert size={17} aria-hidden="true" />{t("schedule.deleteActiveWarning")}</p> : null}
            {deleteError !== null ? <p className={styles.fieldError} role="alert">{t(deleteError)}</p> : null}
            <div className={styles.dialogActions}>
              <Dialog.Close asChild>
                <button className={styles.secondaryButton} type="button" disabled={deleteBusy}>{t("controls.cancel")}</button>
              </Dialog.Close>
              <button className={styles.dangerButton} type="button" disabled={deleteBusy || connectionState !== "online"} onClick={() => void confirmDelete()}>
                <Trash2 size={15} aria-hidden="true" />
                {deleteBusy ? t("schedule.deleting") : deleteRequestIdRef.current !== null && deleteError !== null ? t("schedule.retryDelete") : t("schedule.confirmDelete")}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>

      <Dialog.Root
        open={detailJob !== null}
        onOpenChange={(open) => {
          if (!open) {
            detailJobIdRef.current = null;
            detailRequestVersionRef.current += 1;
            setDetailJob(null);
            setDetailBusy(false);
            setDetailError(null);
          }
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={styles.dialogContent}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              if (detailTriggerRef.current?.isConnected) detailTriggerRef.current.focus();
              else document.getElementById("schedule-heading")?.focus();
            }}
          >
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{t("schedule.detailTitle")}</Dialog.Title>
                <Dialog.Description className={styles.dialogDescription}>
                  {t("schedule.detailDescription", { title: detailJob?.title ?? "" })}
                </Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                  <X size={17} aria-hidden="true" />
                </button>
              </Dialog.Close>
            </div>
            {detailBusy ? (
              <div className={styles.managementLoading} role="status">
                <RefreshCw size={16} className={styles.spin} aria-hidden="true" />
                {t("schedule.detailLoading")}
              </div>
            ) : null}
            {detailError !== null ? <p className={styles.fieldError} role="alert">{t(detailError)}</p> : null}
            {detailJob !== null ? (
              <dl className={styles.detailList}>
                <div><dt>{t("schedule.fields.message")}</dt><dd>{detailJob.message}</dd></div>
                <div><dt>{t("schedule.fields.kind")}</dt><dd>{scheduleJobRule(detailJob.schedule, t)}</dd></div>
                <div><dt>{t("schedule.fields.status")}</dt><dd>{t(`schedule.jobStatus.${detailJob.status}`)}</dd></div>
                <div><dt>{t("schedule.fields.source")}</dt><dd>{detailJob.source}</dd></div>
                <div><dt>{t("schedule.fields.session")}</dt><dd>{detailJob.session_id}</dd></div>
                <div><dt>{t("schedule.fields.createdAt")}</dt><dd>{new Date(detailJob.created_at_ms).toLocaleString(i18n.language)}</dd></div>
                <div><dt>{t("schedule.fields.updatedAt")}</dt><dd>{new Date(detailJob.updated_at_ms).toLocaleString(i18n.language)}</dd></div>
              </dl>
            ) : null}
            <div className={styles.dialogActions}>
              {detailJob !== null ? (
                <Link
                  className={styles.secondaryButton}
                  to={scheduleRouteHref(projectId, workspaceDirectory, sessionId, detailJob.job_id)}
                >
                  <BookOpen size={15} aria-hidden="true" />
                  {t("schedule.history")}
                </Link>
              ) : null}
              <Dialog.Close asChild>
                <button className={styles.secondaryButton} type="button">{t("controls.close")}</button>
              </Dialog.Close>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </section>
  );
}

export function ScheduleJobHistoryView({
  authState,
  connectionState,
  projects,
}: ScheduleJobsViewProps) {
  const { projectId: routeProjectId, jobId = "" } = useParams();
  const [searchParams] = useSearchParams();
  const projectId = routeProjectId ?? null;
  const workspaceDirectory = projectId === null ? searchParams.get("directory") : null;
  const sessionId = searchParams.get("session");
  return (
    <ScheduleJobHistoryContent
      key={`${projectId ?? "chat"}:${workspaceDirectory ?? ""}:${jobId}`}
      authState={authState}
      connectionState={connectionState}
      projects={projects}
      projectId={projectId}
      workspaceDirectory={workspaceDirectory}
      sessionId={sessionId}
      jobId={jobId}
    />
  );
}

function ScheduleJobHistoryContent({
  authState,
  connectionState,
  projects,
  projectId,
  workspaceDirectory,
  sessionId,
  jobId,
}: ScheduleJobsViewProps & {
  projectId: string | null;
  workspaceDirectory: string | null;
  sessionId: string | null;
  jobId: string;
}) {
  const { i18n, t } = useTranslation();
  const project = projectId === null ? undefined : projects.find((item) => item.project_id === projectId);
  const scopeMissing = projectId === null ? workspaceDirectory === null : project === undefined;
  const projectAvailable = projectId === null
    ? workspaceDirectory !== null && workspaceDirectory.trim() !== ""
    : project?.available === true;
  const [job, setJob] = useState<ScheduleJob | null>(null);
  const [scheduleStatus, setScheduleStatus] = useState<ScheduleStatus | null>(null);
  const [groups, setGroups] = useState<ScheduleHistoryGroup[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loadState, setLoadState] = useState<ScheduleHistoryLoadState>("idle");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const failedPageRef = useRef<{ cursor: string | null; append: boolean } | null>(null);
  const scopeVersionRef = useRef(0);
  const requestVersionRef = useRef(0);
  const loadPendingRef = useRef(false);
  const workspaceIdRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const refreshButtonRef = useRef<HTMLButtonElement | null>(null);
  const restoreRefreshFocusRef = useRef(false);
  const loadMoreButtonRef = useRef<HTMLButtonElement | null>(null);
  const restoreMoreFocusRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      scopeVersionRef.current += 1;
    };
  }, []);

  useEffect(() => {
    if (!restoreRefreshFocusRef.current || !["ready", "error"].includes(loadState)) return;
    restoreRefreshFocusRef.current = false;
    refreshButtonRef.current?.focus();
  }, [loadState]);

  useEffect(() => {
    if (!restoreMoreFocusRef.current || loadingMore) return;
    restoreMoreFocusRef.current = false;
    (loadMoreButtonRef.current ?? refreshButtonRef.current)?.focus();
  }, [loadingMore]);

  const loadHistory = useCallback(async (cursor: string | null, append: boolean) => {
    if (
      authState !== "ready"
      || connectionState !== "online"
      || !jobId
      || !projectAvailable
      || loadPendingRef.current
    ) return;
    loadPendingRef.current = true;
    failedPageRef.current = null;
    setLoadError(null);
    const scopeVersion = scopeVersionRef.current;
    const requestVersion = requestVersionRef.current + 1;
    requestVersionRef.current = requestVersion;
    if (append) {
      setLoadingMore(true);
    } else {
      setLoadState("loading");
      setLoadError(null);
    }
    try {
      let nextWorkspaceId = workspaceIdRef.current;
      if (nextWorkspaceId === null) {
        const sessions = projectId === null
          ? await enterChatWorkspace(workspaceDirectory ?? undefined)
          : await getProjectSessions(projectId, { limit: 1 });
        if (
          !mountedRef.current
          || scopeVersionRef.current !== scopeVersion
          || requestVersionRef.current !== requestVersion
        ) return;
        nextWorkspaceId = sessions.workspace_id;
        workspaceIdRef.current = nextWorkspaceId;
      }
      const response: ScheduleJobHistoryResponse = await getScheduleJobHistory(
        nextWorkspaceId,
        jobId,
        { limit: 20, ...(cursor === null ? {} : { cursor }) },
      );
      if (
        !mountedRef.current
        || scopeVersionRef.current !== scopeVersion
        || requestVersionRef.current !== requestVersion
        || workspaceIdRef.current !== response.workspace_id
      ) return;
      setJob(response.job);
      setScheduleStatus(response.status);
      setGroups((current) => append ? [...current, ...response.groups] : response.groups);
      setNextCursor(response.next_cursor);
      setLoadState("ready");
    } catch (error) {
      if (
        !mountedRef.current
        || scopeVersionRef.current !== scopeVersion
        || requestVersionRef.current !== requestVersion
      ) return;
      setLoadState("error");
      failedPageRef.current = { cursor, append };
      setLoadError(scheduleErrorKey(error, "schedule.historyLoadError"));
    } finally {
      if (
        mountedRef.current
        && scopeVersionRef.current === scopeVersion
        && requestVersionRef.current === requestVersion
      ) {
        loadPendingRef.current = false;
        setLoadingMore(false);
      }
    }
  }, [authState, connectionState, jobId, projectAvailable, projectId, workspaceDirectory]);

  useEffect(() => {
    scopeVersionRef.current += 1;
    requestVersionRef.current += 1;
    loadPendingRef.current = false;
    workspaceIdRef.current = null;
    setJob(null);
    setScheduleStatus(null);
    setGroups([]);
    setNextCursor(null);
    setLoadError(null);
    setLoadingMore(false);
    setLoadState("idle");
    if (
      authState !== "ready"
      || connectionState !== "online"
      || !jobId
      || !projectAvailable
    ) return;
    void loadHistory(null, false);
  }, [
    authState,
    connectionState,
    jobId,
    loadHistory,
    projectAvailable,
    projectId,
    workspaceDirectory,
  ]);

  const authUnavailable = authState !== "ready";
  const heading = job?.title || t("schedule.historyTitle");
  return (
    <section className={styles.schedulePage} aria-labelledby="schedule-history-heading">
      <div className={styles.pageHeading}>
        <div>
          <Link className={styles.backLink} to={scheduleRouteHref(projectId, workspaceDirectory, sessionId)}>
            <ArrowLeft size={15} aria-hidden="true" />
            {t("schedule.backToJobs")}
          </Link>
          <p className={styles.eyebrow}>{t("schedule.historyEyebrow")}</p>
          <h1 id="schedule-history-heading" tabIndex={-1}>{heading}</h1>
          <p className={styles.pageDescription}>
            {project?.name ?? workspaceDirectory ?? t("schedule.historyTitle")}
            {project !== undefined ? ` · ${project.path}` : ""}
          </p>
        </div>
        <div className={styles.pageActions}>
          <Link className={styles.secondaryButton} to={scheduleRouteHref(projectId, workspaceDirectory, sessionId)}>
            <CalendarClock size={15} aria-hidden="true" />
            {t("schedule.backToJobs")}
          </Link>
          <button
            ref={refreshButtonRef}
            className={styles.iconButton}
            type="button"
            aria-label={t("controls.refreshScheduleHistory")}
            title={t("controls.refreshScheduleHistory")}
            disabled={authUnavailable || connectionState !== "online" || loadState === "loading" || loadingMore || !projectAvailable}
            onClick={() => {
              restoreRefreshFocusRef.current = document.activeElement === refreshButtonRef.current;
              void loadHistory(null, false);
            }}
          >
            <RefreshCw size={16} className={loadState === "loading" ? styles.spin : undefined} aria-hidden="true" />
          </button>
        </div>
      </div>

      {!authUnavailable && connectionState !== "online" ? (
        <div className={styles.actionError} role="status">
          <CircleAlert size={16} aria-hidden="true" />
          <span>{t("schedule.disconnected")}</span>
        </div>
      ) : null}

      {authUnavailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><Info size={22} /></div>
          <div><h2>{t("schedule.authenticationRequired")}</h2><p>{t("status.unavailable")}</p></div>
        </div>
      ) : scopeMissing ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true"><CircleAlert size={22} /></div>
          <div><h2>{t(projectId === null ? "schedule.workspaceNotFound" : "schedule.notFound")}</h2><Link className={styles.secondaryButton} to={sessionRouteHref(projectId, workspaceDirectory, sessionId)}>{t("controls.backToSessions")}</Link></div>
        </div>
      ) : projectId !== null && !projectAvailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><FolderOpen size={22} /></div>
          <div><h2>{t("schedule.projectUnavailable")}</h2><p>{project?.path}</p></div>
        </div>
      ) : (
        <>
          {job !== null ? (
            <section className={styles.scheduleHistorySummary} aria-labelledby="schedule-history-job-heading">
              <div className={styles.schedulePanelHeader}>
                <div>
                  <p className={styles.eyebrow}>{t("schedule.historyJobEyebrow")}</p>
                  <h2 id="schedule-history-job-heading">{job.title}</h2>
                </div>
                <span className={styles.scheduleJobStatus} data-state={job.status} role="status">
                  {scheduleJobStatusIcon(job.status)}
                  {t(`schedule.jobStatus.${job.status}`)}
                </span>
              </div>
              <dl className={styles.scheduleJobDetails}>
                <div><dt>{t("schedule.fields.message")}</dt><dd>{job.message}</dd></div>
                <div><dt>{t("schedule.fields.kind")}</dt><dd>{scheduleJobRule(job.schedule, t)}</dd></div>
                <div><dt>{t("schedule.fields.session")}</dt><dd>{job.session_id}</dd></div>
              </dl>
            </section>
          ) : null}

          {scheduleStatus !== null ? (
            <dl className={styles.scheduleStatusGrid} aria-label={t("schedule.statusTitle")}>
              <div className={styles.scheduleStatusMetric}>
                <dt>{t("schedule.admission")}</dt>
                <dd data-state={scheduleStatus.admitted ? "active" : "paused"}>
                  <span className={styles.statusDot} aria-hidden="true" />
                  {scheduleStatus.admitted ? t("schedule.admitted") : t("schedule.paused")}
                </dd>
              </div>
              <div className={styles.scheduleStatusMetric}>
                <dt>{t("schedule.activeJobCount")}</dt>
                <dd>{scheduleStatus.active_job_count}</dd>
              </div>
              <div className={styles.scheduleStatusMetric}>
                <dt>{t("schedule.historyGroups")}</dt>
                <dd>{groups.length}</dd>
              </div>
            </dl>
          ) : null}

          <section className={styles.scheduleHistoryPanel} aria-labelledby="schedule-history-list-heading">
            <div className={styles.schedulePanelHeader}>
              <div>
                <p className={styles.eyebrow}>{t("schedule.historyEyebrow")}</p>
                <h2 id="schedule-history-list-heading">{t("schedule.historyListTitle")}</h2>
              </div>
              <span className={styles.scheduleCount}>{groups.length}</span>
            </div>
            {loadState === "error" && groups.length > 0 ? (
              <div className={styles.actionError} role="alert">
                <CircleAlert size={16} aria-hidden="true" />
                <span>{t(loadError ?? "schedule.historyLoadError")}</span>
                <button className={styles.secondaryButton} type="button" onClick={() => {
                  const failed = failedPageRef.current;
                  if (failed !== null) {
                    restoreRefreshFocusRef.current = !failed.append;
                    restoreMoreFocusRef.current = failed.append;
                    void loadHistory(failed.cursor, failed.append);
                  }
                }}>
                  {t("controls.retry")}
                </button>
              </div>
            ) : null}
            {loadState === "loading" && groups.length === 0 ? (
              <div className={styles.scheduleEmptyState} role="status">
                <RefreshCw size={18} className={styles.spin} aria-hidden="true" />
                {t("schedule.historyLoading")}
              </div>
            ) : loadState === "error" && groups.length === 0 ? (
              <div className={styles.scheduleEmptyState} role="alert">
                <CircleAlert size={18} aria-hidden="true" />
                <span>{t(loadError ?? "schedule.historyLoadError")}</span>
                <button className={styles.secondaryButton} type="button" onClick={() => {
                  restoreRefreshFocusRef.current = true;
                  void loadHistory(null, false);
                }}>
                  {t("controls.retry")}
                </button>
              </div>
            ) : groups.length === 0 ? (
              <div className={styles.scheduleEmptyState} role="status">
                <Clock3 size={18} aria-hidden="true" />
                <span>{t("schedule.historyEmpty")}</span>
              </div>
            ) : (
              <div className={styles.scheduleHistoryList}>
                {groups.map((group, index) => (
                  <article
                    className={styles.scheduleHistoryGroup}
                    data-state={group.result_state}
                    key={scheduleHistoryGroupKey(group, index)}
                  >
                    <header className={styles.scheduleHistoryGroupHeader}>
                      <div>
                        <p className={styles.eyebrow}>{t("schedule.historyRun", { count: index + 1 })}</p>
                        <div className={styles.scheduleHistoryTimes}>
                          <time dateTime={group.started_at ?? undefined}>
                            {scheduleHistoryTime(group.started_at, i18n.language, t)}
                          </time>
                          <span aria-hidden="true">→</span>
                          <time dateTime={group.finished_at ?? undefined}>
                            {scheduleHistoryTime(group.finished_at, i18n.language, t)}
                          </time>
                        </div>
                      </div>
                      <span className={styles.scheduleHistoryState} data-state={group.result_state} role="status">
                        {scheduleHistoryStateIcon(group.result_state)}
                        {t(`schedule.historyState.${group.result_state}`)}
                      </span>
                    </header>
                    <div className={styles.messageHistory} role="log" aria-label={t("schedule.historyMessages")}>
                      {group.messages.map((message, messageIndex) => (
                        <HistoryMessageView
                          key={`${scheduleHistoryGroupKey(group, index)}-${messageIndex}`}
                          message={message}
                          index={messageIndex}
                          t={t}
                          scheduleHistory
                        />
                      ))}
                    </div>
                  </article>
                ))}
                {nextCursor !== null ? (
                  <div className={styles.scheduleHistoryMore}>
                    <button
                      ref={loadMoreButtonRef}
                      className={styles.secondaryButton}
                      type="button"
                      disabled={loadingMore || loadState === "loading" || connectionState !== "online"}
                      onClick={() => {
                        restoreMoreFocusRef.current = document.activeElement === loadMoreButtonRef.current;
                        void loadHistory(nextCursor, true);
                      }}
                    >
                      {loadingMore ? t("schedule.historyLoadingMore") : t("schedule.historyLoadMore")}
                    </button>
                  </div>
                ) : null}
              </div>
            )}
          </section>
        </>
      )}
    </section>
  );
}

function scheduleHistoryStateIcon(state: ScheduleHistoryResultState) {
  if (state === "success") return <CircleCheck size={14} aria-hidden="true" />;
  if (state === "failure") return <CircleAlert size={14} aria-hidden="true" />;
  if (state === "canceled") return <Ban size={14} aria-hidden="true" />;
  return <Info size={14} aria-hidden="true" />;
}

function scheduleJobStatusIcon(status: ScheduleJobStatus) {
  if (status === "running") return <Activity size={14} aria-hidden="true" />;
  if (status === "ok") return <CircleCheck size={14} aria-hidden="true" />;
  if (status === "error") return <CircleAlert size={14} aria-hidden="true" />;
  return <Clock3 size={14} aria-hidden="true" />;
}
