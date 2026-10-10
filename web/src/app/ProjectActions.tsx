import * as Dialog from "@radix-ui/react-dialog";
import {
  Play,
  Trash2,
  TriangleAlert,
  X
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLocation, useNavigate } from "react-router-dom";
import {
  getProjectRemoval,
  pickProjectDirectory,
  registerProject,
  removeProject,
  resumeProjectSchedule
} from "../shared/service/api";
import type {
  RegisteredProject
} from "../shared/service/protocol";
import type { AuthState, ConnectionState, ServiceEventListener } from "../shared/service/types.ts";
import commonStyles from "../shared/styles/controls.module.css";
import moduleStyles from "./App.module.css";
import type { NavigationProjectAction } from "./NavigationSidebar";
import { projectErrorKey, projectPathError, scheduleText } from "./projects.ts";

const styles = { ...commonStyles, ...moduleStyles };

interface ProjectActionsProps {
  authState: AuthState;
  connectionState: ConnectionState;
  openRegistrationRequest: number;
  onRegistrationRequestConsumed: () => void;
  onAddingChange: (adding: boolean) => void;
  onRetryRegistration: () => void;
  actionRequest: NavigationProjectAction | null;
  onActionRequestConsumed: () => void;
  onRefresh: () => Promise<void>;
  projects: RegisteredProject[];
  subscribeServiceEvents: (listener: ServiceEventListener) => () => void;
}

export function ProjectActions({
  authState, connectionState, openRegistrationRequest, onRegistrationRequestConsumed,
  onAddingChange, onRetryRegistration, actionRequest, onActionRequestConsumed,
  onRefresh, projects, subscribeServiceEvents,
}: ProjectActionsProps) {
  const { i18n, t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [registrationFailed, setRegistrationFailed] = useState(false);
  const registrationAbortRef = useRef<AbortController | null>(null);
  const [resumingProjectId, setResumingProjectId] = useState<string | null>(null);
  const [reviewProjectId, setReviewProjectId] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [removalProjectId, setRemovalProjectId] = useState<string | null>(null);
  const [removingProjectId, setRemovingProjectId] = useState<string | null>(null);
  const [removalOperationId, setRemovalOperationId] = useState<string | null>(null);
  const [activeRemoval, setActiveRemoval] = useState<{ projectId: string; operationId: string } | null>(null);
  const reviewTriggerRef = useRef<HTMLElement | null>(null);
  const removalTriggerRef = useRef<HTMLElement | null>(null);
  const reviewProject = projects.find((project) => project.project_id === reviewProjectId);
  const removalProject = projects.find((project) => project.project_id === removalProjectId);

  const returnFromRemovedProject = useCallback((projectId: string) => {
    const projectPath = `/projects/${encodeURIComponent(projectId)}`;
    if (location.pathname === projectPath || location.pathname.startsWith(`${projectPath}/`)) {
      navigate("/", { replace: true });
    }
  }, [location.pathname, navigate]);

  const startRegistration = useCallback(async () => {
    if (registrationAbortRef.current !== null || authState !== "ready" || connectionState !== "online") return;
    const controller = new AbortController();
    registrationAbortRef.current = controller;
    onAddingChange(true);
    setActionError(null);
    setNotice(null);
    setRegistrationFailed(false);
    try {
      const selection = await pickProjectDirectory(controller.signal);
      if (controller.signal.aborted || selection.path === null) return;
      const result = await registerProject(selection.path, controller.signal);
      if (controller.signal.aborted) return;
      await onRefresh();
      setNotice(result.saved_jobs.length > 0 ? "projects.registeredPausedNotice" : "projects.registeredNotice");
    } catch (error) {
      if (!controller.signal.aborted) {
        setActionError(projectPathError(error) ?? projectErrorKey(error));
        setRegistrationFailed(true);
      }
    } finally {
      registrationAbortRef.current = null;
      onAddingChange(false);
      document.getElementById("add-project-button")?.focus();
    }
  }, [authState, connectionState, onAddingChange, onRefresh]);

  useEffect(() => {
    if (openRegistrationRequest === 0) return;
    onRegistrationRequestConsumed();
    void startRegistration();
  }, [openRegistrationRequest, onRegistrationRequestConsumed, startRegistration]);

  useEffect(() => {
    if (authState !== "ready" || connectionState !== "online") registrationAbortRef.current?.abort();
  }, [authState, connectionState]);
  useEffect(() => () => registrationAbortRef.current?.abort(), []);

  useEffect(() => {
    if (actionRequest === null) return;
    onActionRequestConsumed();
    setActionError(null);
    setNotice(null);
    setRegistrationFailed(false);
    if (actionRequest.action === "resume") {
      reviewTriggerRef.current = actionRequest.trigger;
      setReviewError(null);
      setReviewProjectId(actionRequest.projectId);
    } else {
      removalTriggerRef.current = actionRequest.trigger;
      setRemovalProjectId(actionRequest.projectId);
      setRemovalOperationId(null);
    }
  }, [actionRequest, onActionRequestConsumed]);

  useEffect(() => {
    if (notice === null) return;
    const timer = window.setTimeout(() => setNotice(null), 10000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    return subscribeServiceEvents((event) => {
      if (
        event.type === "project.removal.started"
        || event.type === "project.removal.failed"
        || event.type === "project.removal.completed"
        || event.type === "project.removed"
      ) {
        void onRefresh();
        if (event.type === "project.removed" || event.type === "project.removal.completed") {
          returnFromRemovedProject(String(event.payload.project_id));
        }
      }
      if (
        activeRemoval !== null
        && (event.type === "project.removal.completed" || event.type === "project.removal.failed")
        && event.payload.project_id === activeRemoval.projectId
        && event.payload.operation_id === activeRemoval.operationId
      ) {
        setNotice(event.type === "project.removal.completed"
          ? "projects.removalCompletedNotice" : "projects.removalFailedError");
        setActiveRemoval(null);
      }
    });
  }, [activeRemoval, onRefresh, subscribeServiceEvents, returnFromRemovedProject]);

  useEffect(() => {
    if (activeRemoval === null) return;
    let active = true;
    async function refreshRemovalStatus() {
      if (activeRemoval === null) return;
      try {
        const result = await getProjectRemoval(activeRemoval.projectId, activeRemoval.operationId);
        if (!active || result.status === "removing") return;
        setNotice(result.status === "completed"
          ? "projects.removalCompletedNotice" : "projects.removalFailedError");
        setActiveRemoval(null);
        if (result.status === "completed") returnFromRemovedProject(activeRemoval.projectId);
        void onRefresh();
      } catch {
        // The event stream or the next status check may still deliver the outcome.
      }
    }
    void refreshRemovalStatus();
    const timer = window.setInterval(() => void refreshRemovalStatus(), 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [activeRemoval, onRefresh, returnFromRemovedProject]);

  async function handleResume(project: RegisteredProject) {
    setResumingProjectId(project.project_id);
    setReviewError(null);
    setNotice(null);
    try {
      await resumeProjectSchedule(
        project.project_id,
        project.saved_jobs.map((job) => job.job_id),
      );
      await onRefresh();
      setReviewProjectId(null);
      setNotice("projects.scheduleReady");
    } catch (error) {
      setReviewError(projectErrorKey(error));
    } finally {
      setResumingProjectId(null);
    }
  }

  async function handleRemoval() {
    if (removalProject === undefined) return;
    const projectId = removalProject.project_id;
    setRemovingProjectId(projectId);
    setActionError(null);
    setNotice(null);
    try {
      const result = await removeProject(projectId);
      setRemovalOperationId(result.operation_id);
      await onRefresh();
      setRemovalProjectId(null);
      if (result.status === "completed") returnFromRemovedProject(projectId);
      if (result.status === "completed" || result.status === "failed") {
        setNotice(result.status === "completed"
          ? "projects.removalCompletedNotice" : "projects.removalFailedError");
      } else {
        setActiveRemoval({ projectId, operationId: result.operation_id });
        setNotice("projects.removalStartedNotice");
      }
    } catch (error) {
      setActionError(projectErrorKey(error));
    } finally {
      setRemovingProjectId(null);
    }
  }

  return (
    <>
      {notice !== null || (actionError !== null && removalProject === undefined) ? (
        <div className={styles.projectFeedback} role={actionError !== null ? "alert" : "status"} aria-live="polite">
          <span>{t(actionError ?? notice!, { operationId: removalOperationId ?? undefined })}</span>
          {registrationFailed ? (
            <button className={styles.sidebarTextButton} type="button" onClick={onRetryRegistration}>{t("controls.retry")}</button>
          ) : null}
          <button className={styles.iconButton} type="button" aria-label={t("controls.close")}
            onClick={() => { setNotice(null); setActionError(null); }}><X size={15} aria-hidden="true" /></button>
        </div>
      ) : null}
      <Dialog.Root
        open={removalProject !== undefined}
        onOpenChange={(open) => { if (!open && removingProjectId === null) setRemovalProjectId(null); }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={styles.dialogContent}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              if (removalTriggerRef.current?.isConnected) removalTriggerRef.current.focus();
              else document.getElementById("projects-heading")?.focus();
            }}
          >
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{t("projects.removeTitle")}</Dialog.Title>
                <Dialog.Description className={styles.dialogDescription}>
                  {t("projects.removeDescription", { name: removalProject?.name })}
                </Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                  <X size={17} aria-hidden="true" />
                </button>
              </Dialog.Close>
            </div>
            {actionError !== null ? <p className={styles.fieldError} role="alert">{t(actionError)}</p> : null}
            <div className={styles.removalWarning}>
              <TriangleAlert size={18} aria-hidden="true" />
              <p>{t("projects.removeDataNotice")}</p>
            </div>
            <div className={styles.dialogActions}>
              <Dialog.Close asChild>
                <button className={styles.secondaryButton} type="button" disabled={removingProjectId !== null}>
                  {t("controls.cancel")}
                </button>
              </Dialog.Close>
              <button
                className={styles.dangerButton}
                type="button"
                disabled={removalProject === undefined || removingProjectId !== null}
                onClick={() => void handleRemoval()}
              >
                <Trash2 size={15} aria-hidden="true" />
                {removingProjectId !== null ? t("controls.removingProject") : t("controls.confirmRemoval")}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root
        open={reviewProject !== undefined}
        onOpenChange={(open) => { if (!open) setReviewProjectId(null); }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={`${styles.dialogContent} ${styles.scheduleReviewDialog}`}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              if (reviewTriggerRef.current?.isConnected) reviewTriggerRef.current.focus();
              else document.getElementById("projects-heading")?.focus();
            }}
          >
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{t("projects.reviewTitle")}</Dialog.Title>
                <Dialog.Description className={styles.dialogDescription}>
                  {t("projects.reviewDescription", { name: reviewProject?.name })}
                </Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                  <X size={17} aria-hidden="true" />
                </button>
              </Dialog.Close>
            </div>
            {reviewProject !== undefined ? (
              <ul className={styles.reviewJobList}>
                {reviewProject.saved_jobs.map((job) => (
                  <li key={job.job_id}>
                    <strong>{job.title}</strong>
                    <span>{scheduleText(job, t)}</span>
                    <span>{t(`projects.reviewStatus.${job.review_status}`)}</span>
                    {job.due_at !== null ? (
                      <time dateTime={job.due_at}>
                        {new Date(job.due_at).toLocaleString(i18n.language)}
                      </time>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : null}
            {reviewError !== null ? <p className={styles.fieldError} role="alert">{t(reviewError)}</p> : null}
            <div className={styles.dialogActions}>
              <Dialog.Close asChild>
                <button className={styles.secondaryButton} type="button">{t("controls.cancel")}</button>
              </Dialog.Close>
              <button
                className={styles.primaryButton}
                type="button"
                disabled={reviewProject === undefined || resumingProjectId !== null}
                onClick={() => { if (reviewProject !== undefined) void handleResume(reviewProject); }}
              >
                <Play size={15} aria-hidden="true" />
                {resumingProjectId !== null ? t("controls.resuming") : t("controls.resumeSchedule")}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
}
