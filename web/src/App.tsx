import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowLeft,
  Activity,
  Check,
  CircleAlert,
  ChevronRight,
  FolderOpen,
  Info,
  LockKeyhole,
  Languages,
  LogOut,
  MessageSquare,
  Moon,
  Monitor,
  Play,
  Plus,
  RefreshCw,
  Sun,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, NavLink, Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";

import {
  ApiError,
  claimProjectSession,
  createProjectSession,
  exchangeTicket,
  getProjectSessions,
  getProjects,
  getServiceStatus,
  openEventStream,
  releaseProjectSession,
  registerProject,
  registerWebClient,
  resumeProjectSchedule,
  restoreBrowserSession,
} from "./api";
import type {
  ProjectSessionsResponse,
  RegisteredProject,
  RegisteredClient,
  ServiceState,
  ServiceStatus,
  SessionClaim,
  SessionSnapshot,
} from "./protocol";
import styles from "./App.module.css";

type AuthState = "checking" | "ready" | "required" | "error";
type ConnectionState = "checking" | "online" | "offline" | "recovering";
type Theme = "system" | "light" | "dark";
type ProjectsLoadState = "idle" | "loading" | "ready" | "error";

const THEME_KEY = "myclaw.theme";
const initialLaunchTicket = readAndClearTicket();

export default function App() {
  const { i18n, t } = useTranslation();
  const location = useLocation();
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [connectionState, setConnectionState] = useState<ConnectionState>("checking");
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [registeredClient, setRegisteredClient] = useState<RegisteredClient | null>(null);
  const [projects, setProjects] = useState<RegisteredProject[]>([]);
  const [projectsLoadState, setProjectsLoadState] = useState<ProjectsLoadState>("idle");
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(() => readThemePreference());
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [sessionEventVersion, setSessionEventVersion] = useState(0);
  const bootstrapPromise = useRef<Promise<RegisteredClient> | null>(null);
  const consumeRegisteredClient = useCallback(() => setRegisteredClient(null), []);

  const refreshProjects = useCallback(async () => {
    setProjectsLoadState("loading");
    setProjectsError(null);
    try {
      const response = await getProjects();
      setProjects(response.projects);
      setProjectsLoadState("ready");
    } catch (error) {
      setProjectsLoadState("error");
      setProjectsError(projectErrorKey(error));
    }
  }, []);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    if (authState === "ready") void refreshProjects();
  }, [authState, refreshProjects]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: number | null = null;
    let statusTimer: number | null = null;
    let active = true;

    async function authenticate(): Promise<RegisteredClient> {
      if (initialLaunchTicket !== null) {
        await exchangeTicket(initialLaunchTicket);
      } else if ((await restoreBrowserSession()) === null) {
        throw new ApiError(401, null);
      }
      return registerWebClient();
    }

    function scheduleReconnect(recovering: boolean) {
      if (!active || retryTimer !== null) return;
      if (recovering) setConnectionState("recovering");
      retryTimer = window.setTimeout(() => {
        retryTimer = null;
        void connect(false);
      }, recovering ? 1000 : 3000);
    }

    async function refreshStatus() {
      try {
        const current = await getServiceStatus();
        if (active) setServiceStatus(current);
      } catch {
        // The socket lifecycle reports connection failures separately.
      }
    }

    async function connect(initial: boolean) {
      try {
        let client: RegisteredClient;
        if (initial) {
          bootstrapPromise.current ??= authenticate();
          client = await bootstrapPromise.current;
        } else {
          client = await registerWebClient();
        }
        const current = await getServiceStatus();
        if (!active) {
          return;
        }
        setServiceStatus(current);
        setRegisteredClient(client);
        setAuthState("ready");
        setConnectionState("online");
        setSessionEventVersion((version) => version + 1);
        statusTimer ??= window.setInterval(() => void refreshStatus(), 5000);
        socket = openEventStream(
          () => {
            setConnectionState("online");
            setSessionEventVersion((version) => version + 1);
          },
          () => {
            setSessionEventVersion((version) => version + 1);
            scheduleReconnect(true);
          },
          (event) => {
            if (event.type === "session.claimed" || event.type === "session.released") {
              setSessionEventVersion((version) => version + 1);
            }
            void refreshStatus();
          },
        );
      } catch (error) {
        if (!active) {
          return;
        }
        if (initial || (error instanceof ApiError && error.status === 401)) {
          setAuthState(error instanceof ApiError && error.status === 401 ? "required" : "error");
        }
        setConnectionState("offline");
        if (!initial && !(error instanceof ApiError && error.status === 401)) {
          scheduleReconnect(false);
        }
      }
    }

    void connect(true);
    return () => {
      active = false;
      if (retryTimer !== null) window.clearTimeout(retryTimer);
      if (statusTimer !== null) window.clearInterval(statusTimer);
      socket?.close();
    };
  }, []);

  const language = i18n.language.toLowerCase().startsWith("zh") ? "zh-CN" : "en";
  const connectionLabel = useMemo(() => {
    if (connectionState === "online") return t("status.online");
    if (connectionState === "recovering") return t("status.reconnecting");
    if (connectionState === "checking") return t("status.checking");
    return t("status.offline");
  }, [connectionState, t]);

  return (
    <div className={styles.appShell}>
      <a className={styles.skipLink} href="#main-content">
        {t("nav.status")}
      </a>
      <aside className={styles.sidebar} aria-label={t("app.name")}>
        <div className={styles.brandBlock}>
          <div className={styles.brandMark} aria-hidden="true">
            <Activity size={18} strokeWidth={2.2} />
          </div>
          <div>
            <div className={styles.brandName}>{t("app.name")}</div>
            <div className={styles.brandSubtitle}>{t("app.subtitle")}</div>
          </div>
        </div>
        <nav className={styles.navigation} aria-label={t("app.name")}>
          <NavLink className={({ isActive }) => isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink} to="/status">
            <Activity size={16} aria-hidden="true" />
            <span>{t("nav.status")}</span>
          </NavLink>
          <NavLink className={({ isActive }) => isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink} to="/projects">
            <FolderOpen size={16} aria-hidden="true" />
            <span>{t("nav.projects")}</span>
          </NavLink>
        </nav>
        {authState === "ready" && projects.length > 0 ? (
          <div className={styles.projectNavigation} aria-label={t("nav.projects")}>
            {projects.map((project) => (
              <Link
                className={styles.projectNavigationLink}
                key={project.project_id}
                to={`/projects/${project.project_id}`}
              >
                <span className={styles.projectNavigationDot} data-available={project.available} />
                <span>{project.name || project.path}</span>
              </Link>
            ))}
          </div>
        ) : null}
        <div className={styles.sidebarFooter}>{t("footer.localOnly")}</div>
      </aside>

      <div className={styles.mainColumn}>
        <header className={styles.topbar}>
          <div className={styles.breadcrumb}>
            <span className={styles.breadcrumbMuted}>{t("app.name")}</span>
            <span className={styles.breadcrumbDivider} aria-hidden="true">
              /
            </span>
            <span>
              {location.pathname.startsWith("/projects/")
                ? t("nav.sessions")
                : t(location.pathname === "/projects" ? "nav.projects" : "nav.status")}
            </span>
          </div>
          <div className={styles.toolbar}>
            <div className={styles.toolbarGroup} aria-label={t("controls.language")}>
              <Languages size={15} aria-hidden="true" />
              <button
                className={language === "zh-CN" ? styles.segmentActive : styles.segmentButton}
                type="button"
                aria-pressed={language === "zh-CN"}
                onClick={() => void i18n.changeLanguage("zh-CN")}
              >
                中文
              </button>
              <button
                className={language === "en" ? styles.segmentActive : styles.segmentButton}
                type="button"
                aria-pressed={language === "en"}
                onClick={() => void i18n.changeLanguage("en")}
              >
                EN
              </button>
            </div>
            <div className={styles.toolbarGroup} aria-label={t("controls.theme")}>
              <button
                className={theme === "system" ? styles.iconButtonActive : styles.iconButton}
                type="button"
                aria-label={t("controls.system")}
                aria-pressed={theme === "system"}
                onClick={() => setTheme("system")}
              >
                <Monitor size={16} aria-hidden="true" />
              </button>
              <button
                className={theme === "light" ? styles.iconButtonActive : styles.iconButton}
                type="button"
                aria-label={t("controls.light")}
                aria-pressed={theme === "light"}
                onClick={() => setTheme("light")}
              >
                <Sun size={16} aria-hidden="true" />
              </button>
              <button
                className={theme === "dark" ? styles.iconButtonActive : styles.iconButton}
                type="button"
                aria-label={t("controls.dark")}
                aria-pressed={theme === "dark"}
                onClick={() => setTheme("dark")}
              >
                <Moon size={16} aria-hidden="true" />
              </button>
            </div>
          </div>
        </header>

        <main id="main-content" className={styles.mainContent}>
          <Routes>
            <Route
              path="/"
              element={<Navigate replace to="/status" />}
            />
            <Route
              path="/status"
              element={
                <StatusView
                  authState={authState}
                  connectionLabel={connectionLabel}
                  connectionState={connectionState}
                  detailsOpen={detailsOpen}
                  onDetailsOpenChange={setDetailsOpen}
                  serviceStatus={serviceStatus}
                />
              }
            />
            <Route
              path="/projects"
              element={
                <ProjectsView
                  authState={authState}
                  error={projectsError}
                  loadState={projectsLoadState}
                  onRefresh={refreshProjects}
                  projects={projects}
                />
              }
            />
            <Route
              path="/projects/:projectId"
              element={
                <ProjectSessionsView
                  authState={authState}
                  connectionState={connectionState}
                  projects={projects}
                  registeredClient={registeredClient}
                  onRestoreConsumed={consumeRegisteredClient}
                  refreshVersion={sessionEventVersion}
                />
              }
            />
            <Route path="*" element={<Navigate replace to="/status" />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

interface StatusViewProps {
  authState: AuthState;
  connectionLabel: string;
  connectionState: ConnectionState;
  detailsOpen: boolean;
  onDetailsOpenChange: (open: boolean) => void;
  serviceStatus: ServiceStatus | null;
}

function StatusView({
  authState,
  connectionLabel,
  connectionState,
  detailsOpen,
  onDetailsOpenChange,
  serviceStatus,
}: StatusViewProps) {
  const { t } = useTranslation();
  const serviceState = serviceStatus?.state ?? "starting";
  const serviceLabel = serviceStatus === null ? t("status.checking") : serviceStateLabel(serviceState, t);
  const authMessage = authState === "required" ? t("status.authenticationRequired") : t("status.unavailable");

  return (
    <section className={styles.statusPage} aria-labelledby="status-heading">
      <div className={styles.pageHeading}>
        <div>
          <p className={styles.eyebrow}>{t("nav.status")}</p>
          <h1 id="status-heading">{t("status.title")}</h1>
        </div>
        <div className={styles.connectionBadge} data-state={connectionState} role="status" aria-live="polite">
          <span className={styles.statusDot} aria-hidden="true" />
          {connectionLabel}
        </div>
      </div>

      {authState === "ready" ? (
        <>
          <div className={styles.statusGrid}>
            <StatusMetric label={t("status.connection")} value={connectionLabel} state={connectionState} />
            <StatusMetric label={t("status.title")} value={serviceLabel} state={serviceState} />
            <StatusMetric
              label={t("status.workspaces")}
              value={serviceStatus?.active_workspace_count.toString() ?? "-"}
            />
            <StatusMetric
              label={t("status.protocol")}
              value={`v${serviceStatus?.protocol_version ?? "-"}`}
            />
          </div>
          <div className={styles.actionRow}>
            <Dialog.Root open={detailsOpen} onOpenChange={onDetailsOpenChange}>
              <Dialog.Trigger asChild>
                <button className={styles.secondaryButton} type="button">
                  <Info size={16} aria-hidden="true" />
                  {t("controls.details")}
                </button>
              </Dialog.Trigger>
              <Dialog.Portal>
                <Dialog.Overlay className={styles.dialogOverlay} />
                <Dialog.Content className={styles.dialogContent}>
                  <div className={styles.dialogHeader}>
                    <div>
                      <Dialog.Title className={styles.dialogTitle}>{t("status.detailsTitle")}</Dialog.Title>
                      <Dialog.Description className={styles.dialogDescription}>
                        {t("status.detailsDescription")}
                      </Dialog.Description>
                    </div>
                    <Dialog.Close asChild>
                      <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                        <X size={17} aria-hidden="true" />
                      </button>
                    </Dialog.Close>
                  </div>
                  <dl className={styles.detailList}>
                    <div>
                      <dt>{t("status.instance")}</dt>
                      <dd>{serviceStatus?.service_instance_id ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>{t("status.protocol")}</dt>
                      <dd>v{serviceStatus?.protocol_version ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>{t("status.connection")}</dt>
                      <dd>{connectionLabel}</dd>
                    </div>
                  </dl>
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
          </div>
        </>
      ) : (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true">
            {authState === "required" ? <Info size={22} /> : <CircleAlert size={22} />}
          </div>
          <div>
            <h2>{authMessage}</h2>
            <p>{connectionLabel}</p>
          </div>
        </div>
      )}
    </section>
  );
}

function StatusMetric({
  label,
  value,
  state,
}: {
  label: string;
  value: string;
  state?: string;
}) {
  return (
    <div className={styles.metric}>
      <div className={styles.metricLabel}>{label}</div>
      <div className={styles.metricValue} data-state={state}>
        {state === "ready" || state === "online" ? <Check size={15} aria-hidden="true" /> : null}
        {value}
      </div>
    </div>
  );
}

interface ProjectsViewProps {
  authState: AuthState;
  error: string | null;
  loadState: ProjectsLoadState;
  onRefresh: () => Promise<void>;
  projects: RegisteredProject[];
}

function ProjectsView({
  authState,
  error,
  loadState,
  onRefresh,
  projects,
}: ProjectsViewProps) {
  const { i18n, t } = useTranslation();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [path, setPath] = useState("");
  const [pathError, setPathError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [resumingProjectId, setResumingProjectId] = useState<string | null>(null);
  const [reviewProjectId, setReviewProjectId] = useState<string | null>(null);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const registrationTriggerRef = useRef<HTMLButtonElement | null>(null);
  const reviewTriggerRef = useRef<HTMLButtonElement | null>(null);
  const reviewProject = projects.find((project) => project.project_id === reviewProjectId);

  function openRegistration(event: React.MouseEvent<HTMLButtonElement>) {
    registrationTriggerRef.current = event.currentTarget;
    setPath("");
    setPathError(null);
    setActionError(null);
    setDialogOpen(true);
  }

  async function submitRegistration(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const submittedPath = path.trim();
    if (!submittedPath) {
      setPathError("projects.pathHint");
      return;
    }
    setSubmitting(true);
    setPathError(null);
    setActionError(null);
    setNotice(null);
    try {
      const result = await registerProject(submittedPath);
      await onRefresh();
      setPath("");
      setDialogOpen(false);
      setNotice(
        result.saved_jobs.length > 0
          ? "projects.registeredPausedNotice"
          : "projects.registeredNotice",
      );
    } catch (error) {
      const fieldError = projectPathError(error);
      if (fieldError !== null) {
        setPathError(fieldError);
      } else {
        setActionError(projectErrorKey(error));
      }
    } finally {
      setSubmitting(false);
    }
  }

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

  const authUnavailable = authState !== "ready";
  return (
    <section className={styles.projectsPage} aria-labelledby="projects-heading">
      <div className={styles.pageHeading}>
        <div>
          <p className={styles.eyebrow}>{t("nav.projects")}</p>
          <h1 id="projects-heading" tabIndex={-1}>{t("projects.title")}</h1>
          <p className={styles.pageDescription}>{t("projects.description")}</p>
        </div>
        <div className={styles.pageActions}>
          <button
            className={styles.iconButton}
            type="button"
            aria-label={t("controls.refresh")}
            title={t("controls.refresh")}
            disabled={authUnavailable || loadState === "loading"}
            onClick={() => void onRefresh()}
          >
            <RefreshCw size={16} aria-hidden="true" />
          </button>
          <button
            className={styles.primaryButton}
            type="button"
            disabled={authUnavailable}
            onClick={openRegistration}
          >
            <Plus size={16} aria-hidden="true" />
            {t("controls.addProject")}
          </button>
        </div>
      </div>

      {notice !== null ? (
        <div className={styles.notice} role="status" aria-live="polite">
          <Check size={16} aria-hidden="true" />
          {t(notice)}
        </div>
      ) : null}
      {actionError !== null ? (
        <div className={styles.errorBanner} role="alert">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{t(actionError)}</span>
        </div>
      ) : null}

      {authUnavailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true">
            <Info size={22} />
          </div>
          <div>
            <h2>{t("projects.authenticationRequired")}</h2>
            <p>{t("status.unavailable")}</p>
          </div>
        </div>
      ) : loadState === "loading" && projects.length === 0 ? (
        <div className={styles.emptyState} role="status" aria-live="polite">
          <div className={styles.emptyIcon} aria-hidden="true">
            <RefreshCw size={22} className={styles.spin} />
          </div>
          <div>
            <h2>{t("projects.loading")}</h2>
          </div>
        </div>
      ) : loadState === "error" && projects.length === 0 ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true">
            <CircleAlert size={22} />
          </div>
          <div>
            <h2>{t("projects.loadError")}</h2>
            <p>{t(error ?? "projects.actionError")}</p>
            <button className={styles.secondaryButton} type="button" onClick={() => void onRefresh()}>
              {t("controls.retry")}
            </button>
          </div>
        </div>
      ) : projects.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon} aria-hidden="true">
            <FolderOpen size={22} />
          </div>
          <div>
            <h2>{t("projects.emptyTitle")}</h2>
            <p>{t("projects.emptyDescription")}</p>
            <button className={styles.secondaryButton} type="button" onClick={openRegistration}>
              <Plus size={16} aria-hidden="true" />
              {t("controls.addProject")}
            </button>
          </div>
        </div>
      ) : (
        <>
          {loadState === "error" ? (
            <div className={styles.errorBanner} role="alert">
              <CircleAlert size={17} aria-hidden="true" />
              <span>{t(error ?? "projects.loadError")}</span>
            </div>
          ) : null}
          <ul className={styles.projectList} aria-label={t("nav.projects")}>
            {projects.map((project) => (
              <li className={styles.projectItem} id={`project-${project.project_id}`} key={project.project_id}>
                <div className={styles.projectItemHeader}>
                  <div className={styles.projectTitleBlock}>
                    <FolderOpen size={19} aria-hidden="true" />
                    <div>
                      <h2>{project.name || project.path}</h2>
                      <p className={styles.projectPath}>{project.path}</p>
                    </div>
                  </div>
                  <span
                    className={styles.availabilityBadge}
                    data-available={project.available}
                    role="status"
                  >
                    <span className={styles.statusDot} aria-hidden="true" />
                    {project.available ? t("projects.available") : t("projects.unavailable")}
                  </span>
                </div>
                <div className={styles.projectDetails}>
                  <Link className={styles.secondaryButton} to={`/projects/${project.project_id}`}>
                    <MessageSquare size={15} aria-hidden="true" />
                    {t("controls.openSessions")}
                  </Link>
                  <span
                    className={styles.scheduleBadge}
                    data-paused={project.schedule_state === "awaiting_resume"}
                  >
                    {!project.available
                      ? t("projects.scheduleUnavailable")
                      : t(`projects.scheduleState.${project.schedule_state}`)}
                  </span>
                  {project.schedule_state === "awaiting_resume" ? (
                    <div className={styles.scheduleReview}>
                      <div>
                        <h3>{t("projects.savedJobs")}</h3>
                        {project.saved_jobs.length > 0 ? (
                          <ul className={styles.jobList}>
                            {project.saved_jobs.map((job) => (
                              <li key={job.job_id}>
                                <span>{job.title}</span>
                                <span className={styles.jobSchedule}>{scheduleText(job, t)}</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className={styles.jobEmpty}>{t("projects.noSavedJobs")}</p>
                        )}
                      </div>
                      <button
                        className={styles.secondaryButton}
                        type="button"
                        disabled={!project.available || resumingProjectId !== null}
                        onClick={(event) => {
                          reviewTriggerRef.current = event.currentTarget;
                          setReviewError(null);
                          setReviewProjectId(project.project_id);
                        }}
                      >
                        <Play size={15} aria-hidden="true" />
                        {resumingProjectId === project.project_id
                          ? t("controls.resuming")
                          : t("controls.resumeSchedule")}
                      </button>
                    </div>
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
        </>
      )}

      <Dialog.Root open={dialogOpen} onOpenChange={setDialogOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={styles.dialogContent}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              registrationTriggerRef.current?.focus();
            }}
          >
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{t("projects.addTitle")}</Dialog.Title>
                <Dialog.Description className={styles.dialogDescription}>
                  {t("projects.pathHint")}
                </Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                  <X size={17} aria-hidden="true" />
                </button>
              </Dialog.Close>
            </div>
            <form className={styles.projectForm} onSubmit={(event) => void submitRegistration(event)}>
              <label className={styles.fieldLabel} htmlFor="project-path">
                {t("projects.pathLabel")}
              </label>
              <input
                autoFocus
                className={styles.textInput}
                id="project-path"
                inputMode="text"
                placeholder={t("projects.pathPlaceholder")}
                type="text"
                value={path}
                aria-describedby="project-path-hint project-path-error"
                aria-invalid={pathError !== null}
                onChange={(event) => {
                  setPath(event.target.value);
                  if (pathError !== null) setPathError(null);
                }}
              />
              <p className={styles.fieldHint} id="project-path-hint">
                {t("projects.pathHint")}
              </p>
              {pathError !== null ? (
                <p className={styles.fieldError} id="project-path-error" role="alert">
                  {t(pathError)}
                </p>
              ) : null}
              <div className={styles.dialogActions}>
                <Dialog.Close asChild>
                  <button className={styles.secondaryButton} type="button">
                    {t("controls.cancel")}
                  </button>
                </Dialog.Close>
                <button className={styles.primaryButton} type="submit" disabled={submitting || !path.trim()}>
                  <FolderOpen size={16} aria-hidden="true" />
                  {submitting ? t("controls.registering") : t("controls.register")}
                </button>
              </div>
            </form>
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
            className={styles.dialogContent}
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
    </section>
  );
}

type SessionLoadState = "idle" | "loading" | "ready" | "error";

interface ProjectSessionsViewProps {
  authState: AuthState;
  connectionState: ConnectionState;
  projects: RegisteredProject[];
  registeredClient: RegisteredClient | null;
  onRestoreConsumed: () => void;
  refreshVersion: number;
}

function ProjectSessionsView({
  authState,
  connectionState,
  projects,
  registeredClient,
  onRestoreConsumed,
  refreshVersion,
}: ProjectSessionsViewProps) {
  const { projectId = "" } = useParams();
  return (
    <ProjectSessionsContent
      key={projectId}
      authState={authState}
      connectionState={connectionState}
      projects={projects}
      registeredClient={registeredClient}
      onRestoreConsumed={onRestoreConsumed}
      refreshVersion={refreshVersion}
      projectId={projectId}
    />
  );
}

function ProjectSessionsContent({
  authState,
  connectionState,
  projects,
  registeredClient,
  onRestoreConsumed,
  refreshVersion,
  projectId,
}: ProjectSessionsViewProps & { projectId: string }) {
  const { t, i18n } = useTranslation();
  const project = projects.find((item) => item.project_id === projectId);
  const [sessions, setSessions] = useState<ProjectSessionsResponse | null>(null);
  const [loadState, setLoadState] = useState<SessionLoadState>("idle");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [claim, setClaim] = useState<SessionClaim | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [draft, setDraft] = useState(false);
  const [busySessionId, setBusySessionId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const claimRef = useRef<SessionClaim | null>(null);
  const snapshotRef = useRef<SessionSnapshot | null>(null);
  const needsReclaimRef = useRef(false);
  const attemptedRestoreRef = useRef<string | null>(null);
  const mountedRef = useRef(true);

  const releaseOrphanClaim = useCallback((orphan: SessionClaim) => {
    void releaseProjectSession(
      projectId,
      orphan.session_id,
      orphan.claim_version,
      orphan.reconnect_credential,
    ).catch(() => {});
  }, [projectId]);

  useEffect(() => {
    claimRef.current = claim;
  }, [claim]);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  const clearClaimState = useCallback(() => {
    claimRef.current = null;
    snapshotRef.current = null;
    setClaim(null);
    setSnapshot(null);
    setSelectedSessionId(null);
    setDraft(false);
  }, []);

  const refreshSessions = useCallback(async () => {
    if (authState !== "ready" || !projectId) return;
    setLoadState((state) => (state === "ready" ? state : "loading"));
    try {
      const response = await getProjectSessions(projectId);
      if (!mountedRef.current) return;
      let restoreError: string | null = null;
      if (
        registeredClient !== null &&
        attemptedRestoreRef.current !== registeredClient.web_control_credential &&
        registeredClient.current_workspace_id === response.workspace_id &&
        registeredClient.current_session_id !== null
      ) {
        attemptedRestoreRef.current = registeredClient.web_control_credential;
        onRestoreConsumed();
        if (claimRef.current === null) {
          try {
            const restored = await claimProjectSession(projectId, registeredClient.current_session_id);
            if (!mountedRef.current) {
              releaseOrphanClaim(restored.claim);
              return;
            }
            claimRef.current = restored.claim;
            snapshotRef.current = restored.snapshot;
            setClaim(restored.claim);
            setSnapshot(restored.snapshot);
            setSelectedSessionId(restored.claim.session_id);
            setDraft(!response.sessions.some((item) => item.id === restored.claim.session_id));
          } catch (error) {
            restoreError = sessionErrorKey(error);
          }
        }
      }
      const shouldReclaim = connectionState === "online" && needsReclaimRef.current;
      if (shouldReclaim) needsReclaimRef.current = false;
      const currentClaim = claimRef.current;
      if (currentClaim !== null && shouldReclaim) {
        try {
          const restored = await claimProjectSession(projectId, currentClaim.session_id);
          if (!mountedRef.current) {
            releaseOrphanClaim(restored.claim);
            return;
          }
          if (claimRef.current === currentClaim) {
            claimRef.current = restored.claim;
            snapshotRef.current = restored.snapshot;
            setClaim(restored.claim);
            setSnapshot(restored.snapshot);
          }
        } catch (error) {
          if (claimRef.current === currentClaim) clearClaimState();
          throw error;
        }
      }
      setSessions(response);
      setLoadState("ready");
      setActionError(restoreError);
    } catch (error) {
      setLoadState("error");
      setActionError(sessionErrorKey(error));
    }
  }, [authState, clearClaimState, connectionState, onRestoreConsumed, projectId, registeredClient, releaseOrphanClaim]);

  useEffect(() => {
    if (connectionState !== "online") needsReclaimRef.current = true;
  }, [connectionState]);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions, refreshVersion]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      const current = claimRef.current;
      if (current !== null && projectId) {
        void releaseProjectSession(
          projectId,
          current.session_id,
          current.claim_version,
          current.reconnect_credential,
        );
      }
    };
  }, [projectId]);

  async function openSession(sessionId: string, isDraft: boolean, allowBusy = false) {
    if (busySessionId !== null && !allowBusy) return;
    setBusySessionId(sessionId);
    setActionError(null);
    try {
      const response = await claimProjectSession(projectId, sessionId);
      if (!mountedRef.current) {
        releaseOrphanClaim(response.claim);
        return;
      }
      claimRef.current = response.claim;
      snapshotRef.current = response.snapshot;
      setClaim(response.claim);
      setSnapshot(response.snapshot);
      setSelectedSessionId(sessionId);
      setDraft(isDraft);
      await refreshSessions();
    } catch (error) {
      setActionError(sessionErrorKey(error));
    } finally {
      setBusySessionId(null);
    }
  }

  async function createDraft() {
    if (busySessionId !== null) return;
    setBusySessionId("new");
    setActionError(null);
    try {
      const created = await createProjectSession(projectId);
      setBusySessionId(null);
      await openSession(created.session_id, true, true);
    } catch (error) {
      setActionError(sessionErrorKey(error));
    } finally {
      setBusySessionId(null);
    }
  }

  async function releaseCurrent() {
    const current = claimRef.current;
    if (current === null) return;
    setBusySessionId(current.session_id);
    setActionError(null);
    try {
      await releaseProjectSession(
        projectId,
        current.session_id,
        current.claim_version,
        current.reconnect_credential,
      );
      claimRef.current = null;
      snapshotRef.current = null;
      setClaim(null);
      setSnapshot(null);
      setSelectedSessionId(null);
      setDraft(false);
      await refreshSessions();
    } catch (error) {
      setActionError(sessionErrorKey(error));
    } finally {
      setBusySessionId(null);
    }
  }

  const selectedSummary = sessions?.sessions.find((item) => item.id === selectedSessionId);
  const authUnavailable = authState !== "ready";
  return (
    <section className={styles.sessionsPage} aria-labelledby="sessions-heading">
      <div className={styles.pageHeading}>
        <div>
          <Link className={styles.backLink} to="/projects">
            <ArrowLeft size={15} aria-hidden="true" />
            {t("controls.backToProjects")}
          </Link>
          <p className={styles.eyebrow}>{t("nav.sessions")}</p>
          <h1 id="sessions-heading" tabIndex={-1}>
            {project?.name || t("sessions.title")}
          </h1>
          {project !== undefined ? <p className={styles.pageDescription}>{project.path}</p> : null}
        </div>
        <div className={styles.pageActions}>
          <button
            className={styles.iconButton}
            type="button"
            aria-label={t("controls.refreshSessions")}
            title={t("controls.refreshSessions")}
            disabled={authUnavailable || loadState === "loading"}
            onClick={() => void refreshSessions()}
          >
            <RefreshCw size={16} className={loadState === "loading" ? styles.spin : undefined} aria-hidden="true" />
          </button>
          <button
            className={styles.primaryButton}
            type="button"
            disabled={authUnavailable || project?.available !== true || busySessionId !== null}
            onClick={() => void createDraft()}
          >
            <Plus size={16} aria-hidden="true" />
            {t("controls.newSession")}
          </button>
        </div>
      </div>

      {connectionState !== "online" ? (
        <div className={styles.connectionNotice} role="status" aria-live="polite">
          <CircleAlert size={16} aria-hidden="true" />
          {connectionState === "recovering" ? t("sessions.reconnecting") : t("sessions.disconnected")}
        </div>
      ) : null}
      {actionError !== null ? (
        <div className={styles.errorBanner} role="alert">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{t(actionError)}</span>
        </div>
      ) : null}

      {authUnavailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><Info size={22} /></div>
          <div><h2>{t("sessions.authenticationRequired")}</h2><p>{t("status.unavailable")}</p></div>
        </div>
      ) : project === undefined ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true"><CircleAlert size={22} /></div>
          <div><h2>{t("sessions.notFound")}</h2><Link className={styles.secondaryButton} to="/projects">{t("controls.backToProjects")}</Link></div>
        </div>
      ) : project.available !== true ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><FolderOpen size={22} /></div>
          <div><h2>{t("sessions.projectUnavailable")}</h2><p>{project.path}</p></div>
        </div>
      ) : loadState === "error" && sessions === null ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true"><CircleAlert size={22} /></div>
          <div><h2>{t("sessions.loadError")}</h2><button className={styles.secondaryButton} type="button" onClick={() => void refreshSessions()}>{t("controls.retry")}</button></div>
        </div>
      ) : (
        <div className={styles.sessionsLayout}>
          <aside className={styles.sessionListPanel} aria-label={t("sessions.listLabel")}>
            <div className={styles.sessionListHeader}>
              <h2>{t("sessions.listTitle")}</h2>
              <span>{sessions?.sessions.length ?? 0}</span>
            </div>
            {draft && claim !== null ? (
              <div className={styles.draftRow} aria-current="true">
                <div className={styles.sessionRowMain}>
                  <MessageSquare size={15} aria-hidden="true" />
                  <strong>{t("sessions.draft")}</strong>
                </div>
                <span className={styles.sessionMeta}>{t("sessions.notPersisted")}</span>
              </div>
            ) : null}
            {sessions?.sessions.length === 0 && !draft ? (
              <div className={styles.sessionListEmpty}>
                <MessageSquare size={20} aria-hidden="true" />
                <p>{t("sessions.empty")}</p>
              </div>
            ) : (
              <ul className={styles.sessionList} aria-label={t("sessions.listLabel")}>
                {sessions?.sessions.map((item) => {
                  const isSelected = item.id === selectedSessionId;
                  const occupiedByOther = item.occupied && item.occupied_by === "client";
                  return (
                    <li key={item.id}>
                      <button
                        className={isSelected ? styles.sessionRowActive : styles.sessionRow}
                        type="button"
                        aria-current={isSelected ? "true" : undefined}
                        disabled={busySessionId !== null}
                        onClick={() => void openSession(item.id, false)}
                      >
                        <span className={styles.sessionRowMain}>
                          {occupiedByOther ? <LockKeyhole size={15} aria-hidden="true" /> : <MessageSquare size={15} aria-hidden="true" />}
                          <strong>{item.title}</strong>
                          <ChevronRight size={14} aria-hidden="true" />
                        </span>
                        <span className={styles.sessionRowMeta}>
                          <time dateTime={item.updated_at}>{formatSessionTime(item.updated_at, i18n.language)}</time>
                          {item.occupied ? <span className={styles.occupiedBadge}>{occupiedByOther ? t("sessions.occupied") : t("sessions.occupiedHere")}</span> : null}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </aside>

          <section className={styles.sessionContentPanel} aria-live="polite">
            {claim !== null && snapshot !== null ? (
              <>
                <div className={styles.sessionContentHeader}>
                  <div>
                    <p className={styles.eyebrow}>{draft ? t("sessions.draft") : t("sessions.readOnly")}</p>
                    <h2>{draft ? t("sessions.draftTitle") : selectedSummary?.title ?? t("sessions.title")}</h2>
                  </div>
                  <button
                    className={styles.secondaryButton}
                    type="button"
                    disabled={busySessionId !== null}
                    onClick={() => void releaseCurrent()}
                  >
                    <LogOut size={15} aria-hidden="true" />
                    {t("controls.releaseSession")}
                  </button>
                </div>
                {draft ? (
                  <div className={styles.readOnlyNotice} role="status">
                    <Info size={16} aria-hidden="true" />
                    {t("sessions.draftNotice")}
                  </div>
                ) : snapshot.messages.length === 0 ? (
                  <div className={styles.emptyState}><div className={styles.emptyIcon} aria-hidden="true"><MessageSquare size={22} /></div><div><h2>{t("sessions.noMessages")}</h2></div></div>
                ) : (
                  <div className={styles.messageHistory} aria-label={t("sessions.historyLabel")}>
                    {snapshot.messages.map((message, index) => (
                      <article className={styles.historyMessage} data-role={typeof message.role === "string" ? message.role : "system"} key={`${index}-${String(message.role)}`}>
                        <div className={styles.historyMessageRole}>{historyRoleLabel(message.role, t)}</div>
                        <p>{historyMessageText(message.content)}</p>
                      </article>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className={styles.sessionPrompt}>
                <MessageSquare size={22} aria-hidden="true" />
                <h2>{t("sessions.selectTitle")}</h2>
                <p>{t("sessions.selectDescription")}</p>
              </div>
            )}
          </section>
        </div>
      )}
    </section>
  );
}

function formatSessionTime(value: string, language: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString(language);
}

function historyRoleLabel(
  role: unknown,
  t: (key: string) => string,
): string {
  if (role === "user") return t("sessions.userMessage");
  if (role === "assistant") return t("sessions.assistantMessage");
  if (role === "tool") return t("sessions.toolMessage");
  return t("sessions.systemMessage");
}

function historyMessageText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function sessionErrorKey(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.body?.code) {
      case "session_claimed": return "sessions.claimedError";
      case "stale_claim": return "sessions.claimExpiredError";
      case "not_found": return "sessions.notFoundError";
      case "admission_closed": return "sessions.admissionClosedError";
    }
  }
  return "sessions.actionError";
}

function scheduleText(job: RegisteredProject["saved_jobs"][number], t: (key: string, options?: Record<string, unknown>) => string): string {
  if (job.schedule.kind === "every") return t("projects.everySchedule", { seconds: job.schedule.every_seconds });
  if (job.schedule.kind === "cron") {
    return t("projects.cronSchedule", { expression: job.schedule.cron_expr, timezone: job.schedule.timezone });
  }
  return t("projects.atSchedule", { time: job.schedule.at_time });
}

function projectPathError(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null;
  const value = error.body?.field_errors.path;
  if (typeof value !== "string" || !value) return null;
  if (value === "must be an absolute directory") return "projects.pathAbsoluteError";
  if (value === "must not overlap Agent Home") return "projects.pathOverlapError";
  if (value === "must name an existing directory") return "projects.pathMissingError";
  return "projects.pathInvalidError";
}

function projectErrorKey(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.body?.code === "stale_schedule_review") return "projects.staleReviewError";
    if (error.body?.code === "persistence_error") return "projects.persistenceError";
  }
  return "projects.actionError";
}

function serviceStateLabel(state: ServiceState, translate: (key: string) => string): string {
  return translate(`status.${state}`);
}

function readAndClearTicket(): string | null {
  const rawHash = window.location.hash.slice(1);
  const ticket = new URLSearchParams(rawHash).get("ticket");
  if (ticket !== null) {
    window.history.replaceState({}, document.title, `${window.location.pathname}${window.location.search}`);
  }
  return ticket;
}

function readThemePreference(): Theme {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    if (value === "light" || value === "dark") return value;
  } catch {
    // Use the system default when storage is unavailable.
  }
  return "system";
}

function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    if (theme === "system") {
      window.localStorage.removeItem(THEME_KEY);
    } else {
      window.localStorage.setItem(THEME_KEY, theme);
    }
  } catch {
    // Theme preference is optional when storage is unavailable.
  }
}
