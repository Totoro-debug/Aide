import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowLeft,
  Activity,
  Ban,
  BookOpen,
  Brain,
  Check,
  CircleAlert,
  CircleCheck,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  Gauge,
  Info,
  LockKeyhole,
  Languages,
  LogOut,
  MessageSquare,
  Moon,
  Monitor,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  Send,
  ShieldX,
  Square,
  Sun,
  Trash2,
  TriangleAlert,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { Link, NavLink, Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";

import {
  ApiError,
  claimProjectSession,
  createRequestId,
  createProjectSession,
  deleteProjectSession,
  getProjectSessionDeletionStatus,
  claimProjectSessionDeletion,
  exchangeTicket,
  getProjectRemoval,
  getProjectSession,
  getRestoreResult,
  getProjectSessions,
  getProjects,
  getRuntimeMemory,
  getRuntimeStatus,
  getServiceStatus,
  openEventStream,
  releaseProjectSession,
  registerProject,
  registerWebClient,
  reloadRuntimeSkills,
  renameProjectSession,
  removeProject,
  resumeProjectSchedule,
  acknowledgeRestore,
  cancelRestore,
  executeRestore,
  inspectRestore,
  restoreBrowserSession,
  ServiceCommandError,
  triggerRuntimeDream,
  updateRuntimeEffort,
  updateRuntimePermission,
} from "./api";
import type {
  ClientCommand,
  ConfirmationRequest,
  ConfirmationOrigin,
  DreamResult,
  ProjectSessionsResponse,
  RegisteredProject,
  RegisteredClient,
  ReasoningEffort,
  RuntimeStatus,
  ServiceState,
  ServiceCommandResult,
  ServiceEvent,
  ServiceStatus,
  SessionClaim,
  SessionSnapshot,
  SessionSummary,
  SkillMetadata,
  ToolPermissionLevel,
  RestoreMode,
  RestorePlan,
  RestoreResult,
} from "./protocol";
import styles from "./App.module.css";

type AuthState = "checking" | "ready" | "required" | "error";
type ConnectionState = "checking" | "online" | "offline" | "recovering";
type Theme = "system" | "light" | "dark";
type ProjectsLoadState = "idle" | "loading" | "ready" | "error";
type ServiceEventListener = (event: ServiceEvent) => void;

interface PendingConfirmation {
  token: string;
  origin: ConfirmationOrigin;
  request: ConfirmationRequest;
  workspaceId: string;
  projectId: string | null;
  sessionId: string | null;
  runId: string | null;
  jobId: string | null;
  title: string | null;
}

const THEME_KEY = "myclaw.theme";
const initialLaunchTicket = readAndClearTicket();
const PERMISSION_LEVELS: ToolPermissionLevel[] = ["read-only", "workspace-write", "full-access"];
const REASONING_EFFORTS: ReasoningEffort[] = ["low", "medium", "high", "xhigh", "max"];

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
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
  const [confirmationNotice, setConfirmationNotice] = useState<string | null>(null);
  const bootstrapPromise = useRef<Promise<RegisteredClient> | null>(null);
  const eventStreamRef = useRef<ReturnType<typeof openEventStream> | null>(null);
  const eventListenersRef = useRef(new Set<ServiceEventListener>());
  const eventCursorRef = useRef<{
    serviceInstanceId: string; clientId: string; streamId: string; seq: number;
  } | null>(null);
  const pendingConfirmationRef = useRef<PendingConfirmation | null>(null);
  const confirmationTriggerRef = useRef<HTMLElement | null>(null);
  const resolvingConfirmationTokenRef = useRef<string | null>(null);
  const confirmationNoticeTimerRef = useRef<number | null>(null);
  const consumeRegisteredClient = useCallback(() => setRegisteredClient(null), []);
  const subscribeServiceEvents = useCallback((listener: ServiceEventListener) => {
    eventListenersRef.current.add(listener);
    return () => eventListenersRef.current.delete(listener);
  }, []);
  const sendServiceCommand = useCallback(
    (command: ClientCommand): Promise<ServiceCommandResult> => {
      const connection = eventStreamRef.current;
      if (connection === null) {
        return Promise.reject(new ServiceCommandError(null, false));
      }
      return connection.sendCommand(command);
    },
    [],
  );

  const showConfirmationNotice = useCallback((message: string) => {
    if (pendingConfirmationRef.current !== null) return;
    setConfirmationNotice(message);
    if (confirmationNoticeTimerRef.current !== null) {
      window.clearTimeout(confirmationNoticeTimerRef.current);
    }
    confirmationNoticeTimerRef.current = window.setTimeout(() => {
      confirmationNoticeTimerRef.current = null;
      setConfirmationNotice(null);
    }, 4500);
  }, []);

  const decideConfirmation = useCallback((decision: "approved" | "declined") => {
    const confirmation = pendingConfirmationRef.current;
    if (confirmation === null) return;
    resolvingConfirmationTokenRef.current = confirmation.token;
    pendingConfirmationRef.current = null;
    setPendingConfirmation(null);
    void sendServiceCommand({
      request_id: createRequestId(),
      type: "confirmation_decide",
      workspace_id: null,
      session_id: null,
      claim_version: null,
      payload: { token: confirmation.token, decision },
    }).catch((error: unknown) => {
      resolvingConfirmationTokenRef.current = null;
      if (error instanceof ServiceCommandError && error.body?.code === "confirmation_resolved") {
        showConfirmationNotice("confirmation.resolvedElsewhere");
        return;
      }
      showConfirmationNotice("confirmation.decisionFailed");
    });
  }, [sendServiceCommand, showConfirmationNotice]);

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
    if (authState !== "ready") return;
    const projectsTimer = window.setInterval(() => void refreshProjects(), 5000);
    return () => window.clearInterval(projectsTimer);
  }, [authState, refreshProjects]);

  useEffect(() => {
    const unsubscribe = subscribeServiceEvents((event) => {
      if (event.type === "confirmation.requested") {
        const next = parseConfirmationEvent(event);
        if (next === null || pendingConfirmationRef.current?.token === next.token) return;
        if (confirmationNoticeTimerRef.current !== null) {
          window.clearTimeout(confirmationNoticeTimerRef.current);
          confirmationNoticeTimerRef.current = null;
        }
        setConfirmationNotice(null);
        const activeElement = document.activeElement;
        if (next.origin === "background") {
          confirmationTriggerRef.current = null;
        } else if (
          activeElement instanceof HTMLElement
          && activeElement !== document.body
          && activeElement !== document.documentElement
          && !activeElement.closest("[role='dialog']")
          && (
            confirmationTriggerRef.current === null
            || !confirmationTriggerRef.current.isConnected
          )
        ) {
          confirmationTriggerRef.current = activeElement;
        }
        pendingConfirmationRef.current = next;
        setPendingConfirmation(next);
        return;
      }
      if (event.type !== "confirmation.resolved") return;
      const token = event.payload.token;
      const current = pendingConfirmationRef.current;
      if (typeof token !== "string" || current === null || current.token !== token) return;
      const wasLocalDecision = resolvingConfirmationTokenRef.current === token;
      resolvingConfirmationTokenRef.current = null;
      pendingConfirmationRef.current = null;
      setPendingConfirmation(null);
      if (!wasLocalDecision) showConfirmationNotice("confirmation.resolvedElsewhere");
    });
    return () => {
      unsubscribe();
    };
  }, [showConfirmationNotice, subscribeServiceEvents]);

  useEffect(() => {
    return () => {
      if (confirmationNoticeTimerRef.current !== null) {
        window.clearTimeout(confirmationNoticeTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
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
        if (
          eventCursorRef.current !== null
          && (eventCursorRef.current.serviceInstanceId !== current.service_instance_id
            || eventCursorRef.current.clientId !== client.client_id)
        ) {
          eventCursorRef.current = null;
        }
        setServiceStatus(current);
        setRegisteredClient(client);
        setAuthState("ready");
        setSessionEventVersion((version) => version + 1);
        statusTimer ??= window.setInterval(() => void refreshStatus(), 5000);
        const connection = openEventStream(
          () => {
            setConnectionState("online");
            setSessionEventVersion((version) => version + 1);
            const activeConnection = eventStreamRef.current;
            if (activeConnection !== null) {
              void activeConnection.sendCommand({
                request_id: createRequestId(),
                type: "subscribe",
                workspace_id: null,
                session_id: null,
                claim_version: null,
                payload: {
                  last_seq: eventCursorRef.current?.seq ?? null,
                  stream_id: eventCursorRef.current?.streamId ?? null,
                },
              }).catch(() => {
                activeConnection.close();
              });
            }
          },
          () => {
            pendingConfirmationRef.current = null;
            resolvingConfirmationTokenRef.current = null;
            setPendingConfirmation(null);
            setSessionEventVersion((version) => version + 1);
            scheduleReconnect(true);
          },
          (event) => {
            const cursor = eventCursorRef.current;
            const sameStream = cursor !== null
              && cursor.serviceInstanceId === event.service_instance_id
              && cursor.clientId === client.client_id
              && cursor.streamId === event.stream_id;
            if (
              sameStream
              && event.seq <= cursor.seq
            ) {
              return;
            }
            if (
              sameStream
              && event.seq > cursor.seq + 1
              && event.type !== "snapshot.required"
            ) {
              const resync = { ...event, type: "snapshot.required", workspace_id: null, session_id: null, run_id: null };
              for (const listener of eventListenersRef.current) listener(resync);
            }
            eventCursorRef.current = {
              serviceInstanceId: event.service_instance_id,
              clientId: client.client_id,
              streamId: event.stream_id,
              seq: event.seq,
            };
            for (const listener of eventListenersRef.current) listener(event);
            if (event.type === "session.claimed" || event.type === "session.released" || event.type === "session.deleted") {
              setSessionEventVersion((version) => version + 1);
            }
            void refreshStatus();
          },
        );
        eventStreamRef.current = connection;
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
      eventStreamRef.current?.close();
      eventStreamRef.current = null;
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

        <main id="main-content" className={styles.mainContent} tabIndex={-1}>
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
                  subscribeServiceEvents={subscribeServiceEvents}
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
                  sendServiceCommand={sendServiceCommand}
                  subscribeServiceEvents={subscribeServiceEvents}
                  confirmationTriggerRef={confirmationTriggerRef}
                />
              }
            />
            <Route path="*" element={<Navigate replace to="/status" />} />
          </Routes>
        </main>
        {confirmationNotice !== null && pendingConfirmation === null ? (
          <div className={styles.confirmationNotice} role="status" aria-live="polite">
            <CircleCheck size={16} aria-hidden="true" />
            {t(confirmationNotice)}
          </div>
        ) : null}
      </div>
      <ConfirmationDialog
        confirmation={pendingConfirmation}
        onOpenChange={(open) => { if (!open) decideConfirmation("declined"); }}
        onDecide={decideConfirmation}
        triggerRef={confirmationTriggerRef}
        projects={projects}
      />
    </div>
  );
}

interface ConfirmationDialogProps {
  confirmation: PendingConfirmation | null;
  onOpenChange: (open: boolean) => void;
  onDecide: (decision: "approved" | "declined") => void;
  triggerRef: { current: HTMLElement | null };
  projects: RegisteredProject[];
}

function ConfirmationDialog({
  confirmation,
  onOpenChange,
  onDecide,
  triggerRef,
  projects,
}: ConfirmationDialogProps) {
  const { t } = useTranslation();
  const declineRef = useRef<HTMLButtonElement | null>(null);
  const project = confirmation?.projectId === null
    ? undefined
    : projects.find((item) => item.project_id === confirmation?.projectId);

  function restoreFocus() {
    const target = triggerRef.current ?? document.getElementById("main-content");
    if (!(target instanceof HTMLElement) || !target.isConnected) return;
    if (target instanceof HTMLTextAreaElement && target.disabled) {
      document.getElementById("main-content")?.focus();
      return;
    }
    target.focus();
  }

  const projectSource = confirmation === null
    ? ""
    : project === undefined
      ? confirmation.projectId ?? confirmation.workspaceId
      : `${project.name} · ${project.project_id}`;

  return (
    <Dialog.Root open={confirmation !== null} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.dialogOverlay} />
        <Dialog.Content
          className={styles.confirmationDialogContent}
          data-confirmation-origin={confirmation?.origin}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            declineRef.current?.focus();
          }}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            restoreFocus();
          }}
        >
          {confirmation !== null ? (
            <>
              <div className={styles.dialogHeader}>
                <div>
                  <Dialog.Title className={styles.dialogTitle}>
                    {confirmation.origin === "background"
                      ? t("confirmation.backgroundTitle")
                      : t("confirmation.title")}
                  </Dialog.Title>
                  <Dialog.Description className={styles.dialogDescription}>
                    {t("confirmation.description")}
                  </Dialog.Description>
                </div>
                <Dialog.Close asChild>
                  <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                    <X size={17} aria-hidden="true" />
                  </button>
                </Dialog.Close>
              </div>

              <div className={styles.confirmationDialogBody}>
                <dl className={styles.confirmationSource}>
                  <div>
                    <dt>{t("confirmation.project")}</dt>
                    <dd>{projectSource}</dd>
                  </div>
                  {confirmation.origin === "background" ? (
                    <div>
                      <dt>{t("confirmation.job")}</dt>
                      <dd>
                        {confirmation.jobId ?? "-"}
                        {confirmation.title ? ` · ${confirmation.title}` : ""}
                      </dd>
                    </div>
                  ) : (
                    <div>
                      <dt>{t("confirmation.session")}</dt>
                      <dd>{confirmation.sessionId ?? "-"}</dd>
                    </div>
                  )}
                  {confirmation.runId !== null ? (
                    <div>
                      <dt>{t("confirmation.run")}</dt>
                      <dd>{confirmation.runId}</dd>
                    </div>
                  ) : null}
                </dl>

                <section className={styles.confirmationCall} aria-labelledby="confirmation-call-heading">
                  <h3 id="confirmation-call-heading">{t("confirmation.call")}</h3>
                  <p className={styles.confirmationToolName}>{confirmation.request.tool_name}</p>
                  <p className={styles.confirmationSummary}>{confirmation.request.summary}</p>
                  {confirmation.request.reason ? (
                    <p className={styles.confirmationReason}>
                      <strong>{t("confirmation.reason")}:</strong> {confirmation.request.reason}
                    </p>
                  ) : null}
                  <p className={styles.confirmationCallId}>
                    <strong>{t("confirmation.callId")}:</strong> {confirmation.request.tool_call_id}
                  </p>
                </section>

                {confirmation.request.warnings.length > 0 ? (
                  <section className={styles.confirmationWarnings} aria-labelledby="confirmation-warnings-heading">
                    <h3 id="confirmation-warnings-heading">{t("confirmation.warnings")}</h3>
                    <ul>
                      {confirmation.request.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                    </ul>
                  </section>
                ) : null}

                <section className={styles.confirmationDetails} aria-labelledby="confirmation-details-heading">
                  <h3 id="confirmation-details-heading">{t("confirmation.parameters")}</h3>
                  <pre>{formatConfirmationDetails(confirmation.request.details)}</pre>
                </section>
              </div>

              <div className={styles.confirmationActions}>
                <button
                  ref={declineRef}
                  className={styles.secondaryButton}
                  type="button"
                  onClick={() => onDecide("declined")}
                >
                  <ShieldX size={15} aria-hidden="true" />
                  {t("confirmation.decline")}
                </button>
                <button
                  className={styles.primaryButton}
                  type="button"
                  onClick={() => onDecide("approved")}
                >
                  <Check size={15} aria-hidden="true" />
                  {t("confirmation.approve")}
                </button>
              </div>
            </>
          ) : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
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

interface RuntimeManagementDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  claim: SessionClaim;
  sessionTitle: string;
  activeRuns: LiveRun[];
  connectionState: ConnectionState;
  triggerRef: { current: HTMLButtonElement | null };
}

function RuntimeManagementDialog({
  open,
  onOpenChange,
  claim,
  sessionTitle,
  activeRuns,
  connectionState,
  triggerRef,
}: RuntimeManagementDialogProps) {
  const { i18n, t } = useTranslation();
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [permission, setPermission] = useState<ToolPermissionLevel>("workspace-write");
  const [effort, setEffort] = useState<ReasoningEffort>("medium");
  const [loadState, setLoadState] = useState<"idle" | "loading" | "ready">("idle");
  const [saving, setSaving] = useState<"permission" | "effort" | null>(null);
  const [operation, setOperation] = useState<"memory" | "dream" | "skills" | null>(null);
  const [memoryContent, setMemoryContent] = useState<string | null>(null);
  const [dreamResult, setDreamResult] = useState<DreamResult | null>(null);
  const [skillMetadata, setSkillMetadata] = useState<SkillMetadata[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const requestEpoch = useRef(0);
  const dreamEpoch = useRef(0);
  const dreamInFlight = useRef(false);

  useEffect(() => {
    dreamEpoch.current += 1;
    dreamInFlight.current = false;
    setDreamResult(null);
    return () => { dreamEpoch.current += 1; };
  }, [claim, connectionState]);

  useEffect(() => {
    requestEpoch.current += 1;
    setSaving(null);
    setOperation(dreamInFlight.current ? "dream" : null);
    setMemoryContent(null);
    setSkillMetadata(null);
    if (!open || connectionState !== "online") return;
    if (dreamInFlight.current) return;
    let active = true;
    setLoadState("loading");
    setStatus(null);
    setError(null);
    setNotice(null);
    void getRuntimeStatus(
      claim.workspace_id,
      claim.session_id,
      claim.claim_version,
      claim.reconnect_credential,
    ).then((result) => {
      if (!active) return;
      if (result.status_view === undefined) {
        setError("management.invalidStatus");
        setLoadState("ready");
        return;
      }
      setStatus(result.status_view);
      setPermission(result.status_view.current_permission_level);
      setEffort(result.status_view.chat_reasoning_effort);
      setLoadState("ready");
    }).catch((reason: unknown) => {
      if (!active) return;
      setError(managementErrorKey(reason));
      setLoadState("ready");
    });
    return () => { active = false; requestEpoch.current += 1; };
  }, [claim, open, connectionState]);

  async function savePermission(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (status === null || saving !== null || connectionState !== "online") return;
    const epoch = requestEpoch.current;
    setSaving("permission");
    setError(null);
    setNotice(null);
    try {
      const result = await updateRuntimePermission(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
        permission,
      );
      if (epoch !== requestEpoch.current) return;
      const published = result.published_permission_level;
      if (published == null || !PERMISSION_LEVELS.includes(published)) {
        setError("management.invalidSelection");
        return;
      }
      setPermission(published);
      setStatus((current) => current === null ? current : {
        ...current,
        current_permission_level: published,
      });
      setNotice("management.permissionSaved");
    } catch (reason: unknown) {
      if (epoch !== requestEpoch.current) return;
      setError(managementErrorKey(reason));
    } finally {
      if (epoch === requestEpoch.current) setSaving(null);
    }
  }

  async function saveEffort(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (status === null || saving !== null || connectionState !== "online") return;
    const epoch = requestEpoch.current;
    setSaving("effort");
    setError(null);
    setNotice(null);
    try {
      const result = await updateRuntimeEffort(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
        effort,
      );
      if (epoch !== requestEpoch.current) return;
      const published = result.published_effort;
      if (published == null || !REASONING_EFFORTS.includes(published)) {
        setError("management.invalidSelection");
        return;
      }
      setEffort(published);
      setStatus((current) => current === null ? current : {
        ...current,
        chat_reasoning_effort: published,
      });
      setNotice("management.effortSaved");
    } catch (reason: unknown) {
      if (epoch !== requestEpoch.current) return;
      setError(managementErrorKey(reason));
    } finally {
      if (epoch === requestEpoch.current) setSaving(null);
    }
  }

  async function viewMemory() {
    if (operation !== null || connectionState !== "online") return;
    const epoch = requestEpoch.current;
    setOperation("memory");
    setError(null);
    setNotice(null);
    try {
      const result = await getRuntimeMemory(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
      );
      if (epoch !== requestEpoch.current) return;
      if (result.management_error !== undefined || typeof result.memory_content !== "string") {
        setError("management.memoryError");
        return;
      }
      setMemoryContent(result.memory_content);
      setNotice("management.memoryLoaded");
    } catch (reason: unknown) {
      if (epoch !== requestEpoch.current) return;
      setError(managementErrorKey(reason));
    } finally {
      if (epoch === requestEpoch.current) setOperation(null);
    }
  }

  async function runDream() {
    if (operation !== null || connectionState !== "online") return;
    const epoch = dreamEpoch.current;
    dreamInFlight.current = true;
    setOperation("dream");
    setDreamResult(null);
    setError(null);
    setNotice(null);
    try {
      const result = await triggerRuntimeDream(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
      );
      if (epoch !== dreamEpoch.current) return;
      if (result.management_error !== undefined || result.dream_result === undefined) {
        setError("management.dreamError");
        return;
      }
      const nextDream = result.dream_result;
      setDreamResult(nextDream);
      setNotice(nextDream.error === null
        ? nextDream.status === "No pending summaries"
          ? "management.dreamNoPending"
          : "management.dreamCompleted"
        : nextDream.error.code === "memory_task_running"
          ? "management.dreamAlreadyRunning"
          : "management.dreamFailed");
    } catch (reason: unknown) {
      if (epoch !== dreamEpoch.current) return;
      setError(managementErrorKey(reason));
    } finally {
      if (epoch === dreamEpoch.current) {
        dreamInFlight.current = false;
        setOperation(null);
      }
    }
  }

  async function reloadSkills() {
    if (operation !== null || connectionState !== "online") return;
    const epoch = requestEpoch.current;
    setOperation("skills");
    setError(null);
    setNotice(null);
    try {
      const result = await reloadRuntimeSkills(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
      );
      if (epoch !== requestEpoch.current) return;
      if (result.management_error !== undefined || !Array.isArray(result.skill_metadata)) {
        setError("management.skillsError");
        return;
      }
      setSkillMetadata(result.skill_metadata);
    } catch (reason: unknown) {
      if (epoch !== requestEpoch.current) return;
      setError(managementErrorKey(reason));
    } finally {
      if (epoch === requestEpoch.current) setOperation(null);
    }
  }

  const activeWorkCount = activeRuns.filter(isLiveRunActive).length;
  const numberFormat = new Intl.NumberFormat(i18n.language);
  const usedTokens = status?.projected_next_request_tokens ?? 0;
  const availableTokens = status?.available_context ?? 0;
  const usedPercent = status === null ? 0 : Math.max(0, status.input_budget_used_percent);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.dialogOverlay} />
        <Dialog.Content
          className={`${styles.dialogContent} ${styles.managementDialog}`}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            const trigger = triggerRef.current;
            if (trigger?.isConnected && !trigger.disabled) trigger.focus();
          }}
        >
          <div className={styles.dialogHeader}>
            <div>
              <Dialog.Title className={styles.dialogTitle}>{t("management.title")}</Dialog.Title>
              <Dialog.Description className={styles.dialogDescription}>
                {t("management.scope", { session: sessionTitle })}
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                <X size={17} aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>

          {loadState === "loading" ? (
            <div className={styles.managementLoading} role="status" aria-live="polite">
              <RefreshCw size={16} className={styles.spin} aria-hidden="true" />
              {t("management.loading")}
            </div>
          ) : null}
          {error !== null ? (
            <div className={styles.errorBanner} role="alert">
              <CircleAlert size={16} aria-hidden="true" />
              <span>{t(error)}</span>
            </div>
          ) : null}
          {notice !== null ? (
            <div className={styles.notice} role="status" aria-live="polite">
              <Check size={16} aria-hidden="true" />
              {t(notice)}
            </div>
          ) : null}

          {status !== null ? (
            <>
              <dl className={styles.managementStatusGrid} aria-label={t("management.statusTitle")}>
                <div className={styles.managementMetric}>
                  <dt>{t("management.model")}</dt>
                  <dd>{status.chat_model || "-"}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.context")}</dt>
                  <dd>
                    {numberFormat.format(usedTokens)} / {numberFormat.format(availableTokens)}
                    <span>{Math.round(usedPercent)}%</span>
                  </dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.activeWork")}</dt>
                  <dd>{activeWorkCount > 0 ? t("management.activeWorkCount", { count: activeWorkCount }) : t("management.idle")}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.permission")}</dt>
                  <dd>{t(`management.permissionLevels.${status.current_permission_level}`)}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.effort")}</dt>
                  <dd>{t(`management.effortLevels.${status.chat_reasoning_effort}`)}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.messages")}</dt>
                  <dd>{numberFormat.format(status.session_message_count)}</dd>
                </div>
              </dl>

              <div className={styles.managementControls}>
                <form className={styles.managementControl} onSubmit={(event) => void savePermission(event)}>
                  <label className={styles.fieldLabel} htmlFor="runtime-permission">
                    {t("management.permissionLabel")}
                  </label>
                  <select
                    id="runtime-permission"
                    className={styles.textInput}
                    value={permission}
                    disabled={saving !== null || connectionState !== "online"}
                    onChange={(event) => setPermission(event.target.value as ToolPermissionLevel)}
                  >
                    {PERMISSION_LEVELS.map((level) => (
                      <option key={level} value={level}>{t(`management.permissionLevels.${level}`)}</option>
                    ))}
                  </select>
                  <button className={styles.secondaryButton} type="submit" disabled={saving !== null || connectionState !== "online"}>
                    <Check size={15} aria-hidden="true" />
                    {saving === "permission" ? t("management.saving") : t("controls.save")}
                  </button>
                </form>
                <form className={styles.managementControl} onSubmit={(event) => void saveEffort(event)}>
                  <label className={styles.fieldLabel} htmlFor="runtime-effort">
                    {t("management.effortLabel")}
                  </label>
                  <select
                    id="runtime-effort"
                    className={styles.textInput}
                    value={effort}
                    disabled={saving !== null || connectionState !== "online"}
                    onChange={(event) => setEffort(event.target.value as ReasoningEffort)}
                  >
                    {REASONING_EFFORTS.map((level) => (
                      <option key={level} value={level}>{t(`management.effortLevels.${level}`)}</option>
                    ))}
                  </select>
                  <button className={styles.secondaryButton} type="submit" disabled={saving !== null || connectionState !== "online"}>
                    <Check size={15} aria-hidden="true" />
                    {saving === "effort" ? t("management.saving") : t("controls.save")}
                  </button>
                </form>
              </div>

              <div className={styles.managementTools}>
                <section className={styles.managementTool} aria-labelledby="management-memory-title">
                  <div className={styles.managementToolHeader}>
                    <div>
                      <h3 id="management-memory-title">{t("management.memoryTitle")}</h3>
                    </div>
                    <button
                      className={styles.secondaryButton}
                      type="button"
                      disabled={operation !== null || connectionState !== "online"}
                      onClick={() => void viewMemory()}
                    >
                      <BookOpen size={15} aria-hidden="true" />
                      {operation === "memory" ? t("management.loading") : t("management.viewMemory")}
                    </button>
                  </div>
                  {memoryContent !== null ? (
                    <div className={styles.managementMemory} role="region" aria-label={t("management.memoryRegion")}>
                      <h4>{t("management.memoryRegion")}</h4>
                      <pre>{memoryContent || t("management.memoryEmpty")}</pre>
                    </div>
                  ) : null}
                </section>

                <section className={styles.managementTool} aria-labelledby="management-dream-title">
                  <div className={styles.managementToolHeader}>
                    <div>
                      <h3 id="management-dream-title">{t("management.dreamTitle")}</h3>
                    </div>
                    <button
                      className={styles.secondaryButton}
                      type="button"
                      disabled={operation !== null || connectionState !== "online"}
                      onClick={() => void runDream()}
                    >
                      <Brain size={15} aria-hidden="true" />
                      {operation === "dream" ? t("management.dreamRunning") : t("management.runDream")}
                    </button>
                  </div>
                  {dreamResult !== null ? (
                    <div className={styles.managementOperationStatus} role="status" aria-live="polite">
                      <strong>
                        {dreamResult.error !== null
                          ? t(dreamResult.error.code === "memory_task_running"
                            ? "management.dreamAlreadyRunning" : "management.dreamFailed")
                          : dreamResult.status === "No pending summaries"
                            ? t("management.dreamNoPending")
                            : t("management.dreamCompleted")}
                      </strong>
                      <span>{t("management.dreamProcessed", { count: dreamResult.processed_count })}</span>
                      <span>{dreamResult.memory_updated ? t("management.dreamUpdated") : t("management.dreamUnchanged")}</span>
                      <span>{t("management.dreamCursor", { cursor: dreamResult.cursor })}</span>
                      {dreamResult.error !== null ? (
                        <span>{dreamResult.error.code}: {dreamResult.error.message}</span>
                      ) : null}
                    </div>
                  ) : null}
                </section>

                <section className={styles.managementTool} aria-labelledby="management-skills-title">
                  <div className={styles.managementToolHeader}>
                    <div>
                      <h3 id="management-skills-title">{t("management.skillsTitle")}</h3>
                    </div>
                    <button
                      className={styles.secondaryButton}
                      type="button"
                      disabled={operation !== null || connectionState !== "online"}
                      onClick={() => void reloadSkills()}
                    >
                      <RefreshCw size={15} aria-hidden="true" />
                      {operation === "skills" ? t("management.skillsReloading") : t("management.reloadSkills")}
                    </button>
                  </div>
                  {skillMetadata !== null ? (
                    <div className={styles.managementOperationStatus} role="status" aria-live="polite">
                      <strong>{t("management.skillsReloaded", { count: skillMetadata.length })}</strong>
                      {skillMetadata.length > 0 ? (
                        <ul aria-label={t("management.skillsList")}>
                          {skillMetadata.map((skill) => (
                            <li key={skill.name}>
                              <strong>{skill.name}</strong>
                              <span>{skill.description}</span>
                            </li>
                          ))}
                        </ul>
                      ) : <span>{t("management.skillsEmpty")}</span>}
                    </div>
                  ) : null}
                </section>
              </div>
            </>
          ) : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function managementErrorKey(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.body?.code) {
      case "stale_claim":
      case "session_claimed":
        return "sessions.staleClaimError";
      case "admission_closed":
        return "sessions.admissionClosedError";
      case "validation_error":
      case "config_invalid":
        return "management.invalidSelection";
      default:
        return "management.actionError";
    }
  }
  return "management.actionError";
}

interface ProjectsViewProps {
  authState: AuthState;
  error: string | null;
  loadState: ProjectsLoadState;
  onRefresh: () => Promise<void>;
  projects: RegisteredProject[];
  subscribeServiceEvents: (listener: ServiceEventListener) => () => void;
}

function ProjectsView({
  authState,
  error,
  loadState,
  onRefresh,
  projects,
  subscribeServiceEvents,
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
  const [removalProjectId, setRemovalProjectId] = useState<string | null>(null);
  const [removingProjectId, setRemovingProjectId] = useState<string | null>(null);
  const [removalOperationId, setRemovalOperationId] = useState<string | null>(null);
  const [activeRemoval, setActiveRemoval] = useState<{ projectId: string; operationId: string } | null>(null);
  const registrationTriggerRef = useRef<HTMLButtonElement | null>(null);
  const reviewTriggerRef = useRef<HTMLButtonElement | null>(null);
  const removalTriggerRef = useRef<HTMLButtonElement | null>(null);
  const reviewProject = projects.find((project) => project.project_id === reviewProjectId);
  const removalProject = projects.find((project) => project.project_id === removalProjectId);

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
  }, [activeRemoval, onRefresh, subscribeServiceEvents]);

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
        void onRefresh();
      } catch {
        // The event stream or the next status check may still deliver the outcome.
      }
    }
    void refreshRemovalStatus();
    const timer = window.setInterval(() => void refreshRemovalStatus(), 2000);
    return () => { active = false; window.clearInterval(timer); };
  }, [activeRemoval, onRefresh]);

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

  function openRemoval(
    project: RegisteredProject,
    event: React.MouseEvent<HTMLButtonElement>,
  ) {
    removalTriggerRef.current = event.currentTarget;
    setRemovalProjectId(project.project_id);
    setActionError(null);
    setNotice(null);
    setRemovalOperationId(null);
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
          {t(notice, { operationId: removalOperationId ?? undefined })}
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
            {projects.map((project) => {
              const removalPending = project.schedule_state === "removing"
                && project.removal_error === undefined;
              const removalBlocked = project.schedule_state === "failed"
                || project.removal_error !== undefined;
              const admissionClosed = removalPending || removalBlocked;
              return (
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
                  <div className={styles.projectActionRow}>
                    {admissionClosed ? (
                      <button className={styles.secondaryButton} type="button" disabled>
                        <MessageSquare size={15} aria-hidden="true" />
                        {t("controls.openSessions")}
                      </button>
                    ) : (
                      <Link className={styles.secondaryButton} to={`/projects/${project.project_id}`}>
                        <MessageSquare size={15} aria-hidden="true" />
                        {t("controls.openSessions")}
                      </Link>
                    )}
                    <button
                      className={styles.dangerButton}
                      type="button"
                      disabled={authUnavailable || removingProjectId !== null || removalPending}
                      onClick={(event) => openRemoval(project, event)}
                    >
                      {removalBlocked ? (
                        <RefreshCw size={15} aria-hidden="true" />
                      ) : (
                        <Trash2 size={15} aria-hidden="true" />
                      )}
                      {removingProjectId === project.project_id
                        ? t("controls.removingProject")
                        : removalBlocked
                          ? t("controls.retryRemoval")
                          : t("controls.removeProject")}
                    </button>
                  </div>
                  <span
                    className={styles.scheduleBadge}
                    data-paused={project.schedule_status?.admitted !== true}
                  >
                    {projectScheduleLabel(project, t)}
                  </span>
                  {project.removal_error !== undefined ? (
                    <div className={styles.projectRemovalError} role="alert">
                      <CircleAlert size={16} aria-hidden="true" />
                      <span>{t("projects.removalFailed", { message: project.removal_error })}</span>
                    </div>
                  ) : null}
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
              );
            })}
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
    </section>
  );
}

type SessionLoadState = "idle" | "loading" | "ready" | "error";

function mergeSessionSummaries(
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

type RunStatus = "submitting" | "accepted" | "running" | "completed" | "failed" | "canceled";
type ToolStatus = "running" | "completed" | "failed" | "rejected" | "canceled";

interface ToolActivity {
  toolCallId: string;
  name: string;
  arguments: string;
  status: ToolStatus;
}

interface LiveRun {
  localId: string;
  runId: string | null;
  prompt: string;
  assistantContent: string;
  status: RunStatus;
  tools: ToolActivity[];
  error: string | null;
  cancelRequested: boolean;
}

interface PendingSubmission {
  localId: string;
  sessionId: string;
  command: ClientCommand;
}

interface PendingSessionDeletion {
  claim: SessionClaim;
  requestId: string;
  attempted: boolean;
  title?: string;
}

function readPendingDeletion(projectId: string): PendingSessionDeletion | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(`myclaw.session-delete.${projectId}`) ?? "null") as Partial<PendingSessionDeletion> | null;
    if (value?.attempted !== true || typeof value.requestId !== "string"
      || typeof value.claim?.session_id !== "string" || typeof value.claim.workspace_id !== "string"
      || typeof value.claim.reconnect_credential !== "string"
      || !Number.isInteger(value.claim.claim_version) || value.claim.claim_version < 1) return null;
    return value as PendingSessionDeletion;
  } catch {
    return null;
  }
}

function newLiveRun(
  localId: string,
  runId: string | null,
  prompt: string,
  status: RunStatus,
): LiveRun {
  return {
    localId,
    runId,
    prompt,
    assistantContent: "",
    status,
    tools: [],
    error: null,
    cancelRequested: false,
  };
}

function isLiveRunActive(run: LiveRun): boolean {
  return run.status === "submitting" || run.status === "accepted" || run.status === "running";
}

function statusIcon(status: RunStatus | ToolStatus, size = 14) {
  if (status === "completed") return <CircleCheck size={size} aria-hidden="true" />;
  if (status === "failed") return <TriangleAlert size={size} aria-hidden="true" />;
  if (status === "rejected") return <ShieldX size={size} aria-hidden="true" />;
  if (status === "canceled") return <Ban size={size} aria-hidden="true" />;
  return <Activity size={size} aria-hidden="true" />;
}

function runStatusKey(status: RunStatus): string {
  return `conversation.${status}`;
}

function toolStatusKey(status: ToolStatus): string {
  return `conversation.tool${status[0].toUpperCase()}${status.slice(1)}`;
}

function safeMarkdownUrl(url: string): string {
  try {
    const parsed = new URL(url, window.location.origin);
    if (["http:", "https:", "mailto:", "tel:"].includes(parsed.protocol)) return url;
  } catch {
    return "";
  }
  return "";
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className={styles.markdownContent}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={safeMarkdownUrl}
        components={{
          img: ({ alt }) => alt ? <span className={styles.blockedMedia}>{alt}</span> : null,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function ToolActivityGroup({
  tools,
  t,
}: {
  tools: ToolActivity[];
  t: (key: string) => string;
}) {
  return (
    <details className={styles.toolActivity}>
      <summary className={styles.toolActivityHeader}>
        <span className={styles.toolActivityTitle}>
          <Activity size={14} aria-hidden="true" />
          {t("conversation.toolActivity")}
        </span>
        <span className={styles.toolActivityCount}>{tools.length}</span>
        <ChevronDown className={styles.toolActivityChevron} size={14} aria-hidden="true" />
      </summary>
      <ul className={styles.toolActivityList}>
        {tools.map((tool) => (
          <li className={styles.toolActivityItem} key={tool.toolCallId}>
            <div className={styles.toolActivityItemHeader}>
              <span className={styles.toolName}>{tool.name || t("conversation.unknownTool")}</span>
              <span className={`${styles.statusBadge} ${styles[`status${tool.status}`]}`}>
                {statusIcon(tool.status, 12)}
                {t(toolStatusKey(tool.status))}
              </span>
            </div>
            {tool.arguments ? (
              <details className={styles.toolArguments}>
                <summary>{t("conversation.toolArguments")}</summary>
                <pre>{tool.arguments}</pre>
              </details>
            ) : null}
          </li>
        ))}
      </ul>
    </details>
  );
}

function HistoryMessageView({
  message,
  index,
  t,
}: {
  message: Record<string, unknown>;
  index: number;
  t: (key: string) => string;
}) {
  const role = message.role;
  const messageStatus = message.status;
  if (role === "tool") {
    const rawStatus = typeof messageStatus === "string" ? messageStatus : "error";
    const status: ToolStatus = rawStatus === "success"
      ? "completed"
      : rawStatus === "refused"
        ? "rejected"
        : rawStatus === "error" && message.content === "Tool call interrupted because the turn was cancelled."
          ? "canceled"
          : "failed";
    const tool = {
      toolCallId: typeof message.tool_call_id === "string" ? message.tool_call_id : `tool-${index}`,
      name: typeof message.name === "string" ? message.name : t("conversation.unknownTool"),
      arguments: "",
      status,
    } satisfies ToolActivity;
    return (
      <article className={styles.historyMessage} data-role="tool" key={`${index}-tool`}>
        <div className={styles.historyMessageRole}>{historyRoleLabel(role, t)}</div>
        <ToolActivityGroup tools={[tool]} t={t} />
        <MarkdownContent content={historyMessageText(message.content)} />
      </article>
    );
  }
  return (
    <article className={styles.historyMessage} data-role={typeof role === "string" ? role : "system"} key={`${index}-${String(role)}`}>
      <div className={styles.historyMessageRole}>{historyRoleLabel(role, t)}</div>
      <MarkdownContent content={historyMessageText(message.content)} />
    </article>
  );
}

function LiveRunView({
  run,
  t,
  onCancel,
}: {
  run: LiveRun;
  t: (key: string) => string;
  onCancel: (run: LiveRun) => void;
}) {
  const active = isLiveRunActive(run);
  return (
    <article className={`${styles.liveRun} ${styles[`liveRun${run.status}`]}`} data-run-id={run.runId ?? run.localId}>
      <div className={styles.liveRunHeader}>
        <span className={styles.historyMessageRole}>{t("sessions.userMessage")}</span>
        <span className={`${styles.statusBadge} ${styles[`status${run.status}`]}`} role="status" aria-live="polite">
          {statusIcon(run.status)}
          {t(runStatusKey(run.status))}
        </span>
      </div>
      <div className={styles.livePrompt}>{run.prompt}</div>
      {run.tools.length > 0 ? <ToolActivityGroup tools={run.tools} t={t} /> : null}
      {run.assistantContent ? <MarkdownContent content={run.assistantContent} /> : active ? <p className={styles.pendingAnswer}>{t("conversation.assistantPending")}</p> : null}
      {run.error ? <p className={styles.runError}>{run.error}</p> : null}
      {active && run.runId !== null ? (
        <button
          className={styles.cancelRunButton}
          type="button"
          aria-label={t("controls.cancelRun")}
          disabled={run.cancelRequested}
          onClick={() => onCancel(run)}
        >
          <Square size={14} aria-hidden="true" />
          {run.cancelRequested ? t("controls.cancelingRun") : t("controls.cancelRun")}
        </button>
      ) : null}
    </article>
  );
}

interface ProjectSessionsViewProps {
  authState: AuthState;
  connectionState: ConnectionState;
  projects: RegisteredProject[];
  registeredClient: RegisteredClient | null;
  onRestoreConsumed: () => void;
  refreshVersion: number;
  sendServiceCommand: (command: ClientCommand) => Promise<ServiceCommandResult>;
  subscribeServiceEvents: (listener: ServiceEventListener) => () => void;
  confirmationTriggerRef: { current: HTMLElement | null };
}

function ProjectSessionsView({
  authState,
  connectionState,
  projects,
  registeredClient,
  onRestoreConsumed,
  refreshVersion,
  sendServiceCommand,
  subscribeServiceEvents,
  confirmationTriggerRef,
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
      sendServiceCommand={sendServiceCommand}
      subscribeServiceEvents={subscribeServiceEvents}
      confirmationTriggerRef={confirmationTriggerRef}
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
  sendServiceCommand,
  subscribeServiceEvents,
  confirmationTriggerRef,
  projectId,
}: ProjectSessionsViewProps & { projectId: string }) {
  const { t, i18n } = useTranslation();
  const project = projects.find((item) => item.project_id === projectId);
  const [sessions, setSessions] = useState<ProjectSessionsResponse | null>(null);
  const [sessionSummaries, setSessionSummaries] = useState<Record<string, SessionSummary>>({});
  const [sessionSearch, setSessionSearch] = useState("");
  const deferredSessionSearch = useDeferredValue(sessionSearch);
  const [sessionNextCursor, setSessionNextCursor] = useState<string | null>(null);
  const [loadState, setLoadState] = useState<SessionLoadState>("idle");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [claim, setClaim] = useState<SessionClaim | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [draft, setDraft] = useState(false);
  const [draftSessionIds, setDraftSessionIds] = useState<string[]>([]);
  const [busySessionId, setBusySessionId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameTitle, setRenameTitle] = useState("");
  const [renameBusy, setRenameBusy] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [restoreOpen, setRestoreOpen] = useState(false);
  const [restoreAnchorId, setRestoreAnchorId] = useState<number | null>(null);
  const [restorePlan, setRestorePlan] = useState<RestorePlan | null>(null);
  const [restoreMode, setRestoreMode] = useState<RestoreMode>("conversation-only");
  const [restoreBusy, setRestoreBusy] = useState(false);
  const [restoreError, setRestoreError] = useState<string | null>(null);
  const [restoreNotice, setRestoreNotice] = useState<RestoreResult | null>(null);
  const [pendingRestoreFailure, setPendingRestoreFailure] = useState<RestoreResult | null>(null);
  const [managementOpen, setManagementOpen] = useState(false);
  const [pendingDeletion, setPendingDeletion] = useState<PendingSessionDeletion | null>(() => readPendingDeletion(projectId));
  const pendingDeletionRef = useRef(pendingDeletion);
  const deleteBusyRef = useRef(false);
  const [liveRunsBySession, setLiveRunsBySession] = useState<Record<string, LiveRun[]>>({});
  const [inputText, setInputText] = useState("");
  const [composerError, setComposerError] = useState<string | null>(null);
  const claimRef = useRef<SessionClaim | null>(null);
  const snapshotRef = useRef<SessionSnapshot | null>(null);
  const claimsBySessionRef = useRef<Record<string, SessionClaim>>({});
  const snapshotsBySessionRef = useRef<Record<string, SessionSnapshot>>({});
  const liveRunsRef = useRef<Record<string, LiveRun[]>>({});
  const selectedSessionRef = useRef<string | null>(null);
  const workspaceIdRef = useRef<string | null>(null);
  const sessionRequestRef = useRef(0);
  const refreshSessionsRef = useRef<((cursor?: string | null, append?: boolean) => Promise<void>) | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const renameTriggerRef = useRef<HTMLButtonElement | null>(null);
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null);
  const restoreTriggerRef = useRef<HTMLButtonElement | null>(null);
  const managementTriggerRef = useRef<HTMLButtonElement | null>(null);
  const restoreBusyRef = useRef(false);
  const restoreFocusPendingRef = useRef(false);
  const restorePlanClaimRef = useRef<SessionClaim | null>(null);
  const restoreCompletedClaimRef = useRef<SessionClaim | null>(null);
  const draftsBySessionRef = useRef<Record<string, string>>({});
  const pendingSubmissionsRef = useRef<PendingSubmission[]>([]);
  const pendingClientIdRef = useRef<string | null>(null);
  const needsReclaimRef = useRef(false);
  const attemptedRestoreRef = useRef<string | null>(null);
  const mountedRef = useRef(true);

  const clearPendingDeletion = useCallback(() => {
    pendingDeletionRef.current = null;
    setPendingDeletion(null);
    try { sessionStorage.removeItem(`myclaw.session-delete.${projectId}`); } catch { /* Storage can be unavailable. */ }
  }, [projectId]);

  const rememberPendingDeletion = useCallback((operation: PendingSessionDeletion) => {
    pendingDeletionRef.current = operation;
    setPendingDeletion(operation);
    try {
      sessionStorage.setItem(`myclaw.session-delete.${projectId}`, JSON.stringify(operation));
    } catch { /* In-memory retries remain available when browser storage is unavailable. */ }
  }, [projectId]);

  const releaseOrphanClaim = useCallback((orphan: SessionClaim) => {
    void releaseProjectSession(
      projectId,
      orphan.session_id,
      orphan.claim_version,
      orphan.reconnect_credential,
    ).catch(() => {});
  }, [projectId]);

  const releaseClaims = useCallback(() => {
    if (!projectId) return;
    for (const current of Object.values(claimsBySessionRef.current)) {
      if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === current.session_id) continue;
      void releaseProjectSession(
        projectId,
        current.session_id,
        current.claim_version,
        current.reconnect_credential,
      ).catch(() => {});
    }
  }, [projectId]);

  useEffect(() => {
    claimRef.current = claim;
    if (claim !== null) claimsBySessionRef.current[claim.session_id] = claim;
  }, [claim]);

  useEffect(() => {
    snapshotRef.current = snapshot;
    if (snapshot !== null) snapshotsBySessionRef.current[snapshot.session_id] = snapshot;
  }, [snapshot]);

  useEffect(() => {
    if (connectionState !== "online" || claim === null || draft) return;
    if (restoreCompletedClaimRef.current === claim) return;
    let cancelled = false;
    setRestoreNotice(null);
    setPendingRestoreFailure(null);
    void cancelRestore(claim).then(() => getRestoreResult(
      claim.workspace_id,
      claim.session_id,
      claim.claim_version,
      claim.reconnect_credential,
    )).then((result) => {
      if (!cancelled && mountedRef.current && claimRef.current === claim && result !== null) {
        setRestoreNotice(result);
        setPendingRestoreFailure(result.file_results.some((item) => item.status === "failed")
          && !result.failure_notification_acknowledged ? result : null);
      }
    }).catch(() => {
      // A missing result is normal; Claim recovery remains owned by the Session flow.
    });
    return () => { cancelled = true; };
  }, [claim, connectionState, draft, projectId]);

  useEffect(() => {
    if (restorePlanClaimRef.current === null || restoreBusyRef.current) return;
    if (connectionState !== "online" || restorePlanClaimRef.current !== claim) {
      const previous = restorePlanClaimRef.current;
      restorePlanClaimRef.current = null;
      void cancelRestore(previous).catch(() => {});
      setRestoreOpen(false);
      setRestorePlan(null);
    }
  }, [claim, connectionState, restoreBusy]);

  useEffect(() => {
    if (restoreNotice === null) return;
    const timer = window.setTimeout(() => setRestoreNotice(null), 10_000);
    return () => window.clearTimeout(timer);
  }, [restoreNotice]);

  useEffect(() => {
    if (restoreOpen || !restoreFocusPendingRef.current) return;
    restoreFocusPendingRef.current = false;
    const trigger = restoreTriggerRef.current;
    if (trigger?.isConnected && !trigger.disabled) trigger.focus();
    else document.getElementById("sessions-heading")?.focus();
  }, [restoreOpen]);

  useEffect(() => {
    selectedSessionRef.current = selectedSessionId;
  }, [selectedSessionId]);

  const updateLiveRuns = useCallback(
    (sessionId: string, update: (runs: LiveRun[]) => LiveRun[]) => {
      const nextRuns = update(liveRunsRef.current[sessionId] ?? []);
      const next = { ...liveRunsRef.current, [sessionId]: nextRuns };
      liveRunsRef.current = next;
      setLiveRunsBySession(next);
    },
    [],
  );

  const sendPendingSubmission = useCallback(async (pending: PendingSubmission) => {
    try {
      const result = await sendServiceCommand(pending.command);
      const runId = result.run_id;
      if (typeof runId !== "string" || !runId) throw new ServiceCommandError(null, false);
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (item) => item.localId !== pending.localId,
      );
      updateLiveRuns(pending.sessionId, (runs) => runs.map((run) => run.localId === pending.localId
        ? { ...run, runId, status: run.status === "submitting" ? "accepted" : run.status }
        : run));
    } catch (error) {
      if (error instanceof ServiceCommandError && error.resultUnknown) return;
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (item) => item.localId !== pending.localId,
      );
      const message = error instanceof ServiceCommandError && error.body !== null
        ? error.body.message
        : null;
      updateLiveRuns(pending.sessionId, (runs) => runs.map((run) => run.localId === pending.localId
        ? { ...run, status: "failed", error: message }
        : run));
      if (selectedSessionRef.current === pending.sessionId) {
        setComposerError(message ?? "conversation.submitFailed");
      }
    }
  }, [sendServiceCommand, updateLiveRuns]);

  useEffect(() => {
    if (connectionState !== "online" || registeredClient === null) return;
    const previousClientId = pendingClientIdRef.current;
    pendingClientIdRef.current = registeredClient.client_id;
    if (previousClientId !== null && previousClientId !== registeredClient.client_id) {
      for (const pending of pendingSubmissionsRef.current) {
        updateLiveRuns(pending.sessionId, (runs) => runs.map((run) => run.localId === pending.localId
          ? { ...run, status: "failed", error: null } : run));
      }
      pendingSubmissionsRef.current = [];
      return;
    }
    for (const pending of [...pendingSubmissionsRef.current]) {
      void sendPendingSubmission(pending);
    }
  }, [connectionState, registeredClient, sendPendingSubmission, updateLiveRuns]);

  const clearClaimState = useCallback(() => {
    const previous = claimRef.current;
    if (previous !== null) delete claimsBySessionRef.current[previous.session_id];
    claimRef.current = null;
    snapshotRef.current = null;
    setClaim(null);
    setSnapshot(null);
    setSelectedSessionId(null);
    setDraft(false);
    setRestoreOpen(false);
    setRestorePlan(null);
    setRestoreNotice(null);
  }, []);

  const adoptSnapshot = useCallback((nextSnapshot: SessionSnapshot) => {
    const sessionId = nextSnapshot.session_id;
    const previousCount = snapshotsBySessionRef.current[sessionId]?.messages.length ?? 0;
    const committedPrompts = new Set(nextSnapshot.messages.slice(previousCount)
      .filter((message) => message.role === "user" && typeof message.content === "string")
      .map((message) => message.content as string));
    snapshotsBySessionRef.current[sessionId] = nextSnapshot;
    if (selectedSessionRef.current === sessionId) {
      snapshotRef.current = nextSnapshot;
      setSnapshot(nextSnapshot);
    }
    if (committedPrompts.size > 0) {
      updateLiveRuns(sessionId, (runs) => runs.filter((run) => !committedPrompts.has(run.prompt)));
    }
  }, [updateLiveRuns]);

  const rememberSession = useCallback((nextClaim: SessionClaim, nextSnapshot: SessionSnapshot) => {
    claimsBySessionRef.current[nextClaim.session_id] = nextClaim;
    adoptSnapshot(nextSnapshot);
    claimRef.current = nextClaim;
    setClaim(nextClaim);
    setSnapshot(nextSnapshot);
  }, [adoptSnapshot]);

  const refreshSessions = useCallback(async (cursor: string | null = null, append = false) => {
    if (authState !== "ready" || !projectId) return;
    const requestNumber = sessionRequestRef.current + 1;
    sessionRequestRef.current = requestNumber;
    setLoadState("loading");
    try {
      const response = await getProjectSessions(projectId, {
        title: deferredSessionSearch.trim() || undefined,
        cursor: cursor ?? undefined,
        limit: 100,
      });
      if (!mountedRef.current || requestNumber !== sessionRequestRef.current) return;
      setSessionSummaries((current) => mergeSessionSummaries(current, response.sessions));
      workspaceIdRef.current = response.workspace_id;
      let restoreError: string | null = null;
      const deletionOperation = pendingDeletionRef.current;
      if (deletionOperation?.attempted && deletionOperation.claim.workspace_id !== response.workspace_id) {
        const status = await getProjectSessionDeletionStatus(projectId, deletionOperation.claim.session_id);
        if (!mountedRef.current || requestNumber !== sessionRequestRef.current) return;
        if (status.state === "deleted" || status.state === "present") {
          clearPendingDeletion();
          setDeleteOpen(false);
          if (claimRef.current?.session_id === deletionOperation.claim.session_id) clearClaimState();
        } else {
          const recovered = await claimProjectSessionDeletion(projectId, deletionOperation.claim.session_id);
          if (!mountedRef.current || requestNumber !== sessionRequestRef.current) return;
          rememberPendingDeletion({ ...deletionOperation, claim: recovered.claim });
          setDeleteError("sessions.deletionInProgressError");
        }
      }
      if (registeredClient !== null
        && pendingDeletionRef.current?.claim.session_id === registeredClient.current_session_id) {
        attemptedRestoreRef.current = registeredClient.web_control_credential;
        onRestoreConsumed();
      }
      if (
        registeredClient !== null &&
        attemptedRestoreRef.current !== registeredClient.web_control_credential &&
        registeredClient.current_workspace_id === response.workspace_id &&
        registeredClient.current_session_id !== null
        && pendingDeletionRef.current?.claim.session_id !== registeredClient.current_session_id
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
            rememberSession(restored.claim, restored.snapshot);
            setSelectedSessionId(restored.claim.session_id);
            selectedSessionRef.current = restored.claim.session_id;
            const restoredDraft = restored.snapshot.messages.length === 0;
            setDraft(restoredDraft);
            if (restoredDraft) {
              setDraftSessionIds((ids) => ids.includes(restored.claim.session_id)
                ? ids : [...ids, restored.claim.session_id]);
            } else if (!response.sessions.some((item) => item.id === restored.claim.session_id)) {
              const metadata = await getProjectSessions(projectId);
              if (!mountedRef.current) return;
              setSessionSummaries((current) => mergeSessionSummaries(current, metadata.sessions));
            }
          } catch (error) {
            restoreError = sessionErrorKey(error);
          }
        }
      }
      const shouldReclaim = connectionState === "online" && needsReclaimRef.current;
      if (shouldReclaim) needsReclaimRef.current = false;
      const currentClaim = claimRef.current;
      if (currentClaim !== null && shouldReclaim
        && pendingDeletionRef.current?.claim.session_id !== currentClaim.session_id) {
        try {
          const restored = await claimProjectSession(projectId, currentClaim.session_id);
          if (!mountedRef.current) {
            releaseOrphanClaim(restored.claim);
            return;
          }
          if (claimRef.current === currentClaim) {
            rememberSession(restored.claim, restored.snapshot);
          }
        } catch (error) {
          if (claimRef.current === currentClaim) clearClaimState();
          throw error;
        }
      }
      if (!mountedRef.current || requestNumber !== sessionRequestRef.current) return;
      setSessions((current) => {
        if (!append || current === null || current.workspace_id !== response.workspace_id) {
          return response;
        }
        return {
          ...response,
          sessions: [...current.sessions, ...response.sessions],
        };
      });
      setSessionNextCursor(response.next_cursor);
      setDraftSessionIds((ids) => ids.filter((id) => !response.sessions.some((item) => item.id === id)));
      if (response.sessions.some((item) => item.id === selectedSessionRef.current)) setDraft(false);
      setLoadState("ready");
      setActionError(restoreError);
    } catch (error) {
      if (!mountedRef.current || requestNumber !== sessionRequestRef.current) return;
      setLoadState("error");
      setActionError(sessionErrorKey(error));
    }
  }, [authState, clearClaimState, clearPendingDeletion, connectionState, deferredSessionSearch, onRestoreConsumed, projectId, registeredClient, releaseOrphanClaim, rememberPendingDeletion, rememberSession]);

  useEffect(() => {
    if (connectionState !== "online") needsReclaimRef.current = true;
  }, [connectionState]);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions, refreshVersion]);

  useEffect(() => {
    refreshSessionsRef.current = refreshSessions;
  }, [refreshSessions]);

  const loadMoreSessions = useCallback(() => {
    if (sessionNextCursor === null || loadState === "loading") return;
    void refreshSessions(sessionNextCursor, true);
  }, [loadState, refreshSessions, sessionNextCursor]);

  const refreshRunSnapshot = useCallback(async (sessionId: string, runId: string) => {
    const currentClaim = claimsBySessionRef.current[sessionId];
    if (currentClaim === undefined) return;
    try {
      const nextSnapshot = await getProjectSession(
        projectId,
        sessionId,
        currentClaim.claim_version,
        currentClaim.reconnect_credential,
      );
      if (!mountedRef.current || claimsBySessionRef.current[sessionId] !== currentClaim) return;
      adoptSnapshot(nextSnapshot);
      updateLiveRuns(sessionId, (runs) => runs.filter((run) => run.runId !== runId));
      void refreshSessionsRef.current?.();
    } catch {
      // Keep the terminal live projection visible when persistence is still settling.
    }
  }, [adoptSnapshot, projectId, updateLiveRuns]);

  const handleServiceEvent = useCallback((event: ServiceEvent) => {
    if (event.type === "snapshot.required") {
      const restored = new Set<string>();
      const snapshotPayload = event.payload.snapshot;
      const sessions = typeof snapshotPayload === "object" && snapshotPayload !== null
        ? (snapshotPayload as { sessions?: unknown }).sessions : null;
      if (Array.isArray(sessions)) {
        for (const item of sessions) {
          if (typeof item !== "object" || item === null) continue;
          const entry = item as { workspace_id?: unknown; claim_version?: unknown; snapshot?: unknown };
          const nextSnapshot = entry.snapshot as Partial<SessionSnapshot> | null;
          if (
            typeof entry.workspace_id !== "string"
            || typeof nextSnapshot !== "object" || nextSnapshot === null
            || typeof nextSnapshot.session_id !== "string"
            || !Array.isArray(nextSnapshot.messages)
          ) continue;
          const currentClaim = claimsBySessionRef.current[nextSnapshot.session_id];
          if (currentClaim?.workspace_id !== entry.workspace_id
            || currentClaim.claim_version !== entry.claim_version) continue;
          adoptSnapshot(nextSnapshot as SessionSnapshot);
          restored.add(nextSnapshot.session_id);
        }
      }
      for (const currentClaim of Object.values(claimsBySessionRef.current)) {
        if (restored.has(currentClaim.session_id)) continue;
        void getProjectSession(
          projectId,
          currentClaim.session_id,
          currentClaim.claim_version,
          currentClaim.reconnect_credential,
        ).then((nextSnapshot) => {
          if (mountedRef.current && claimsBySessionRef.current[currentClaim.session_id] === currentClaim) {
            adoptSnapshot(nextSnapshot);
          }
        }).catch(() => {
          // The normal Claim recovery path handles an expired Claim.
        });
      }
      void refreshSessionsRef.current?.();
      return;
    }
    if (event.workspace_id !== workspaceIdRef.current || event.session_id === null) return;
    const sessionId = event.session_id;
    if (event.type === "session.released") {
      delete claimsBySessionRef.current[sessionId];
      return;
    }
    if (event.type === "session.metadata_updated") {
      const title = event.payload.title;
      const version = event.payload.metadata_version;
      if (typeof title === "string" && typeof version === "number") {
        setSessionSummaries((current) => current[sessionId] === undefined
          || current[sessionId].metadata_version > version ? current : {
          ...current,
          [sessionId]: { ...current[sessionId], title, metadata_version: version },
        });
      }
      void refreshSessionsRef.current?.();
      return;
    }
    if (event.type === "input.accepted") {
      if (event.run_id === null) return;
      const acceptedText = typeof event.payload.text === "string" ? event.payload.text : "";
      updateLiveRuns(sessionId, (runs) => {
        const directIndex = runs.findIndex((run) => run.runId === event.run_id);
        if (directIndex >= 0) {
          return runs.map((run, index) => index === directIndex
            ? { ...run, status: run.status === "submitting" ? "accepted" : run.status }
            : run);
        }
        const pendingIndex = runs.findIndex(
          (run) => run.runId === null && run.status === "submitting" && run.prompt === acceptedText,
        );
        if (pendingIndex >= 0) {
          const pendingLocalId = runs[pendingIndex].localId;
          pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
            (item) => item.localId !== pendingLocalId,
          );
          return runs.map((run, index) => index === pendingIndex
            ? { ...run, runId: event.run_id, status: "accepted" }
            : run);
        }
        return [...runs, newLiveRun(`event-${event.run_id}`, event.run_id, acceptedText, "accepted")];
      });
      return;
    }
    if (event.run_id === null) return;
    const runId = event.run_id;
    if (event.type === "run.output") {
      const message = event.payload.message;
      if (typeof message !== "object" || message === null || Array.isArray(message)) return;
      const messageValue = message as Record<string, unknown>;
      const metadata = messageValue.metadata;
      const metadataValue = typeof metadata === "object" && metadata !== null && !Array.isArray(metadata)
        ? metadata as Record<string, unknown>
        : {};
      const messageType = messageValue.type;
      const content = typeof messageValue.content === "string" ? messageValue.content : "";
      updateLiveRuns(sessionId, (runs) => {
        const index = runs.findIndex((run) => run.runId === runId);
        const current = index >= 0 ? runs[index] : newLiveRun(`event-${runId}`, runId, "", "running");
        const next = { ...current, status: current.status === "canceled" ? "canceled" : "running" as RunStatus };
        if (messageType === "model_response" && metadataValue._stream_delta === true) {
          next.assistantContent = `${next.assistantContent}${content}`;
        } else if (messageType === "tool_call") {
          const toolCallId = typeof metadataValue.tool_call_id === "string" ? metadataValue.tool_call_id : "";
          if (!toolCallId) return runs;
          const toolIndex = next.tools.findIndex((tool) => tool.toolCallId === toolCallId);
          const rawStatus = metadataValue.status;
          const mappedStatus = rawStatus === "success"
            ? "completed"
            : rawStatus === "error"
              ? "failed"
              : rawStatus === "refused"
                ? "rejected"
                : null;
          if (toolIndex < 0 && mappedStatus === null) {
            next.tools = [...next.tools, {
              toolCallId,
              name: content,
              arguments: typeof metadataValue.arguments === "string" ? metadataValue.arguments : "",
              status: "running",
            }];
          } else if (toolIndex >= 0 && mappedStatus !== null) {
            next.tools = next.tools.map((tool, toolIndexValue) => toolIndexValue === toolIndex
              ? { ...tool, status: mappedStatus }
              : tool);
          }
        } else if (messageType === "system_control" && metadataValue._streamed === true) {
          next.status = metadataValue.finish_reason === "cancelled" ? "canceled" : "failed";
          next.error = content || null;
          if (next.status === "canceled") {
            next.tools = next.tools.map((tool) => tool.status === "running" ? { ...tool, status: "canceled" } : tool);
          }
        }
        if (index < 0) return [...runs, next];
        return runs.map((run, runIndex) => runIndex === index ? next : run);
      });
      return;
    }
    if (event.type === "run.cancelled" || event.type === "run.failed" || event.type === "run.completed") {
      const finishReason = typeof event.payload.finish_reason === "string"
        ? event.payload.finish_reason
        : event.type === "run.cancelled" ? "cancelled" : event.type === "run.failed" ? "failed" : "completed";
      updateLiveRuns(sessionId, (runs) => {
        const index = runs.findIndex((run) => run.runId === runId);
        const current = index >= 0 ? runs[index] : newLiveRun(`event-${runId}`, runId, "", "running");
        const status: RunStatus = current.status === "canceled" || finishReason === "cancelled"
          ? "canceled"
          : finishReason === "completed" ? "completed" : "failed";
        const next: LiveRun = {
          ...current,
          status,
          error: status === "failed" && typeof event.payload.message === "string" ? event.payload.message : current.error,
          tools: status === "canceled"
            ? current.tools.map((tool) => tool.status !== "running"
              ? tool
              : { ...tool, status: "canceled" })
            : current.tools,
        };
        if (index < 0) return [...runs, next];
        return runs.map((run, runIndex) => runIndex === index ? next : run);
      });
      if (event.type !== "run.cancelled" && finishReason !== "cancelled") {
        void refreshRunSnapshot(sessionId, runId);
      }
    }
  }, [adoptSnapshot, projectId, refreshRunSnapshot, updateLiveRuns]);

  useEffect(() => subscribeServiceEvents(handleServiceEvent), [handleServiceEvent, subscribeServiceEvents]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      releaseClaims();
    };
  }, [projectId, releaseClaims]);

  async function openSession(sessionId: string, isDraft: boolean, allowBusy = false) {
    if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === sessionId) return;
    if (busySessionId !== null && !allowBusy) return;
    setManagementOpen(false);
    const previousSessionId = claimRef.current?.session_id;
    setBusySessionId(sessionId);
    setActionError(null);
    try {
      const response = await claimProjectSession(projectId, sessionId);
      if (!mountedRef.current) {
        releaseOrphanClaim(response.claim);
        return;
      }
      rememberSession(response.claim, response.snapshot);
      setSelectedSessionId(sessionId);
      selectedSessionRef.current = sessionId;
      setInputText(draftsBySessionRef.current[sessionId] ?? "");
      setComposerError(null);
      setDraft(isDraft);
      if (previousSessionId !== undefined && previousSessionId !== sessionId
        && !(liveRunsRef.current[previousSessionId] ?? []).some(isLiveRunActive)) {
        setDraftSessionIds((ids) => ids.filter((id) => id !== previousSessionId));
      }
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
      setDraftSessionIds((ids) => [...ids, created.session_id]);
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
    if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === current.session_id) return;
    setManagementOpen(false);
    setBusySessionId(current.session_id);
    setActionError(null);
    try {
      await releaseProjectSession(
        projectId,
        current.session_id,
        current.claim_version,
        current.reconnect_credential,
      );
      delete claimsBySessionRef.current[current.session_id];
      delete snapshotsBySessionRef.current[current.session_id];
      claimRef.current = null;
      snapshotRef.current = null;
      setClaim(null);
      setSnapshot(null);
      setSelectedSessionId(null);
      selectedSessionRef.current = null;
      setRestoreOpen(false);
      setRestorePlan(null);
      setRestoreNotice(null);
      delete draftsBySessionRef.current[current.session_id];
      setInputText("");
      setDraft(false);
      setDraftSessionIds((ids) => ids.filter((id) => id !== current.session_id));
      await refreshSessions();
    } catch (error) {
      setActionError(sessionErrorKey(error));
    } finally {
      setBusySessionId(null);
    }
  }

  async function submitInput(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const currentClaim = claimRef.current;
    const sessionId = selectedSessionRef.current;
    const text = inputText.trim();
    if (currentClaim === null || sessionId === null || !text) return;
    if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === sessionId) return;
    const activeRun = (liveRunsRef.current[sessionId] ?? []).some(isLiveRunActive);
    if (activeRun || connectionState !== "online") return;
    const localId = `local-${createRequestId()}`;
    confirmationTriggerRef.current = inputRef.current;
    const pending: PendingSubmission = {
      localId,
      sessionId,
      command: {
        request_id: createRequestId(),
        type: "input",
        workspace_id: currentClaim.workspace_id,
        session_id: currentClaim.session_id,
        claim_version: currentClaim.claim_version,
        payload: { text },
      },
    };
    pendingSubmissionsRef.current.push(pending);
    updateLiveRuns(sessionId, (runs) => [
      ...runs,
      newLiveRun(localId, null, text, "submitting"),
    ]);
    setInputText("");
    delete draftsBySessionRef.current[sessionId];
    setComposerError(null);
    inputRef.current?.focus();
    await sendPendingSubmission(pending);
  }

  function handleInputKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  async function cancelRun(run: LiveRun) {
    const currentClaim = claimRef.current;
    if (currentClaim === null || run.runId === null || run.cancelRequested) return;
    updateLiveRuns(currentClaim.session_id, (runs) => runs.map((candidate) => candidate.runId === run.runId
      ? { ...candidate, cancelRequested: true }
      : candidate));
    try {
      await sendServiceCommand({
        request_id: createRequestId(),
        type: "cancel",
        workspace_id: currentClaim.workspace_id,
        session_id: currentClaim.session_id,
        claim_version: currentClaim.claim_version,
        payload: { run_id: run.runId },
      });
    } catch (error) {
      if (error instanceof ServiceCommandError && error.resultUnknown) return;
      updateLiveRuns(currentClaim.session_id, (runs) => runs.map((candidate) => candidate.runId === run.runId
        ? { ...candidate, cancelRequested: false }
        : candidate));
      setComposerError("conversation.cancelFailed");
    }
  }

  const selectedSummary = selectedSessionId === null ? undefined : sessionSummaries[selectedSessionId];
  const selectedLiveRuns = selectedSessionId === null ? [] : liveRunsBySession[selectedSessionId] ?? [];
  const activeRun = selectedLiveRuns.find(isLiveRunActive) ?? null;
  const selectedRestoreAnchor = snapshot?.restore_anchors?.find(
    (anchor) => anchor.anchor_id === restoreAnchorId,
  );

  function beginRestore(event: React.MouseEvent<HTMLButtonElement>) {
    if (draft || claim === null || snapshot === null || activeRun !== null) return;
    const anchors = snapshot.restore_anchors ?? [];
    if (anchors.length === 0) return;
    restoreTriggerRef.current = event.currentTarget;
    setRestoreAnchorId(anchors[0].anchor_id);
    setRestorePlan(null);
    setRestoreMode("conversation-only");
    setRestoreError(null);
    setRestoreOpen(true);
  }

  async function closeRestore() {
    if (restoreBusyRef.current) return;
    const inspectedClaim = restorePlanClaimRef.current;
    if (inspectedClaim !== null) {
      restoreBusyRef.current = true;
      setRestoreBusy(true);
      try {
        await cancelRestore(inspectedClaim);
        restorePlanClaimRef.current = null;
      } catch (error) {
        if (mountedRef.current) setRestoreError(sessionErrorKey(error));
        return;
      } finally {
        restoreBusyRef.current = false;
        if (mountedRef.current) setRestoreBusy(false);
      }
    }
    if (mountedRef.current) {
      setRestoreOpen(false);
      setRestorePlan(null);
      setRestoreError(null);
    }
  }

  async function inspectSelectedRestore() {
    const currentClaim = claimRef.current;
    if (currentClaim === null || restoreAnchorId === null || restoreBusyRef.current) return;
    restoreBusyRef.current = true;
    setRestoreBusy(true);
    setRestoreError(null);
    restorePlanClaimRef.current = currentClaim;
    try {
      const plan = await inspectRestore(
        currentClaim.workspace_id,
        currentClaim.session_id,
        currentClaim.claim_version,
        currentClaim.reconnect_credential,
        restoreAnchorId,
      );
      if (!mountedRef.current || claimRef.current !== currentClaim) {
        void cancelRestore(currentClaim).catch(() => {});
        return;
      }
      setRestorePlan(plan);
      setRestoreMode(plan.available_modes.includes("files") ? "files" : "conversation-only");
    } catch (error) {
      if (mountedRef.current && claimRef.current === currentClaim) {
        setRestoreError(error instanceof ApiError && error.body?.code === "stale_claim"
          ? "sessions.staleClaimError"
          : "sessions.restoreError");
      }
    } finally {
      restoreBusyRef.current = false;
      if (mountedRef.current) setRestoreBusy(false);
    }
  }

  async function submitRestore(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const currentClaim = claimRef.current;
    const currentPlan = restorePlan;
    if (currentClaim === null || currentPlan === null || restoreBusyRef.current
      || currentPlan.session_id !== currentClaim.session_id
      || restorePlanClaimRef.current !== currentClaim) return;
    restoreBusyRef.current = true;
    setRestoreBusy(true);
    setRestoreError(null);
    try {
      const executed = await executeRestore(
        currentClaim.workspace_id,
        currentClaim.session_id,
        currentClaim.claim_version,
        currentClaim.reconnect_credential,
        currentPlan,
        restoreMode,
      );
      const nextClaim: SessionClaim = {
        ...currentClaim,
        claim_version: executed.claimVersion,
        reconnect_credential: executed.claimCredential,
      };
      if (!mountedRef.current || claimRef.current !== currentClaim) return;
      // Adopt the rotated credential before the snapshot fetch, which can be interrupted.
      claimsBySessionRef.current[nextClaim.session_id] = nextClaim;
      claimRef.current = nextClaim;
      restoreCompletedClaimRef.current = nextClaim;
      setClaim(nextClaim);
      restorePlanClaimRef.current = null;
      setRestoreNotice(executed.result);
      setPendingRestoreFailure(executed.result.file_results.some((item) => item.status === "failed")
        && !executed.result.failure_notification_acknowledged ? executed.result : null);
      const nextSnapshot = await getProjectSession(
        projectId,
        nextClaim.session_id,
        nextClaim.claim_version,
        nextClaim.reconnect_credential,
      );
      if (!mountedRef.current || claimRef.current !== nextClaim) return;
      rememberSession(nextClaim, nextSnapshot);
      restoreFocusPendingRef.current = true;
      setRestoreOpen(false);
      setRestorePlan(null);
      await refreshSessions();
    } catch (error) {
      if (mountedRef.current && claimRef.current?.session_id === currentClaim.session_id) {
        if (restoreCompletedClaimRef.current === claimRef.current) {
          setActionError(sessionErrorKey(error));
          restoreFocusPendingRef.current = true;
          setRestoreOpen(false);
          setRestorePlan(null);
        }
        else setRestoreError(error instanceof ApiError && error.body?.code === "stale_claim"
          ? "sessions.staleClaimError" : "sessions.restoreError");
      }
    } finally {
      restoreBusyRef.current = false;
      if (mountedRef.current) setRestoreBusy(false);
    }
  }

  async function acknowledgeRestoreNotice() {
    const currentClaim = claimRef.current;
    if (currentClaim === null || restoreBusyRef.current) return;
    restoreBusyRef.current = true;
    setRestoreBusy(true);
    try {
      await acknowledgeRestore(
        currentClaim.workspace_id,
        currentClaim.session_id,
        currentClaim.claim_version,
        currentClaim.reconnect_credential,
      );
      if (mountedRef.current && claimRef.current === currentClaim) {
        setRestoreNotice(null);
        setPendingRestoreFailure(null);
      }
    } catch (error) {
      if (mountedRef.current) setActionError(sessionErrorKey(error));
    } finally {
      restoreBusyRef.current = false;
      if (mountedRef.current) setRestoreBusy(false);
    }
  }

  function beginRename() {
    if (draft || claim === null || selectedSummary === undefined) return;
    setRenameTitle(selectedSummary.title);
    setRenameError(null);
    setRenameOpen(true);
  }

  async function submitRename(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const currentClaim = claimRef.current;
    const currentSummary = currentClaim === null ? undefined : sessionSummaries[currentClaim.session_id];
    const nextTitle = renameTitle.trim();
    if (currentClaim === null || currentSummary === undefined || !nextTitle) return;
    setRenameBusy(true);
    setRenameError(null);
    try {
      const response = await renameProjectSession(
        projectId,
        currentClaim.session_id,
        currentClaim.claim_version,
        currentClaim.reconnect_credential,
        nextTitle,
        currentSummary.metadata_version,
      );
      if (!mountedRef.current) return;
      setSessionSummaries((current) => mergeSessionSummaries(current, [response.session]));
      setSessions((current) => current === null ? current : {
        ...current,
        sessions: current.sessions.map((item) => item.id === response.session.id ? response.session : item),
      });
      setRenameOpen(false);
      await refreshSessions();
    } catch (error) {
      if (!mountedRef.current) return;
      setRenameError(sessionErrorKey(error));
      if (error instanceof ApiError && error.body?.code === "metadata_conflict") {
        await refreshSessions();
        if (!mountedRef.current) return;
        const metadata = await getProjectSessions(projectId);
        if (!mountedRef.current) return;
        setSessionSummaries((current) => mergeSessionSummaries(current, metadata.sessions));
      }
    } finally {
      if (mountedRef.current) setRenameBusy(false);
    }
  }

  function beginDelete(event: React.MouseEvent<HTMLButtonElement>) {
    if (draft || claim === null || selectedSummary === undefined) return;
    if (pendingDeletionRef.current?.attempted) return;
    deleteTriggerRef.current = event.currentTarget;
    if (pendingDeletionRef.current === null) {
      const operation = { claim: { ...claim }, requestId: createRequestId(), attempted: false, title: selectedSummary.title };
      pendingDeletionRef.current = operation;
      setPendingDeletion(operation);
    }
    setDeleteError(null);
    setDeleteOpen(true);
  }

  async function submitDelete(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const previousOperation = pendingDeletionRef.current;
    if (previousOperation === null || deleteBusyRef.current) return;
    const operation = { ...previousOperation, attempted: true };
    rememberPendingDeletion(operation);
    const currentClaim = operation.claim;
    deleteBusyRef.current = true;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      try {
        await deleteProjectSession(
          projectId,
          currentClaim.session_id,
          currentClaim.claim_version,
          currentClaim.reconnect_credential,
          operation.requestId,
        );
      } catch (error) {
        if (!(error instanceof ApiError) || !["stale_claim", "not_found", "stale_client", "session_busy", "restore_pending", "validation_error"].includes(error.body?.code ?? "")) throw error;
        const status = await getProjectSessionDeletionStatus(projectId, currentClaim.session_id);
        if (status.state === "deleting") {
          if (["stale_claim", "not_found", "stale_client"].includes(error.body?.code ?? "")) {
            const recovered = await claimProjectSessionDeletion(projectId, currentClaim.session_id);
            rememberPendingDeletion({ ...operation, claim: recovered.claim });
          }
          setDeleteError("sessions.deletionInProgressError");
          return;
        }
        if (status.state === "present") {
          const rejectedOperation = { ...operation, attempted: false };
          pendingDeletionRef.current = rejectedOperation;
          setPendingDeletion(rejectedOperation);
          try { sessionStorage.removeItem(`myclaw.session-delete.${projectId}`); } catch { /* Storage can be unavailable. */ }
          throw error;
        }
      }
      if (!mountedRef.current) return;
      const sessionId = currentClaim.session_id;
      clearPendingDeletion();
      delete claimsBySessionRef.current[sessionId];
      delete snapshotsBySessionRef.current[sessionId];
      delete liveRunsRef.current[sessionId];
      delete draftsBySessionRef.current[sessionId];
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (item) => item.sessionId !== sessionId,
      );
      if (selectedSessionRef.current === sessionId) {
        claimRef.current = null;
        snapshotRef.current = null;
        selectedSessionRef.current = null;
        setClaim(null);
        setSnapshot(null);
        setSelectedSessionId(null);
        setInputText("");
        setDraft(false);
        setComposerError(null);
      }
      setLiveRunsBySession((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setSessionSummaries((current) => {
        const next = { ...current };
        delete next[sessionId];
        return next;
      });
      setSessions((current) => current === null ? current : {
        ...current,
        sessions: current.sessions.filter((item) => item.id !== sessionId),
      });
      setDraftSessionIds((ids) => ids.filter((id) => id !== sessionId));
      setDeleteOpen(false);
      await refreshSessions();
    } catch (error) {
      if (mountedRef.current) setDeleteError(sessionDeleteErrorKey(error));
    } finally {
      deleteBusyRef.current = false;
      if (mountedRef.current) setDeleteBusy(false);
    }
  }

  useEffect(() => {
    if (activeRun !== null) return;
    const target = confirmationTriggerRef.current;
    if (target === null || !target.isConnected) return;
    target.focus();
    if (document.activeElement === target) confirmationTriggerRef.current = null;
  }, [activeRun, confirmationTriggerRef]);
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
          {pendingDeletion?.attempted ? (
            <button
              className={styles.dangerButton}
              type="button"
              disabled={deleteBusy || connectionState !== "online"}
              onClick={(event) => {
                deleteTriggerRef.current = event.currentTarget;
                setDeleteOpen(true);
              }}
            >
              <Trash2 size={15} aria-hidden="true" />
              {t("controls.retry")}
            </button>
          ) : null}
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
      {restoreNotice !== null ? (
        <div className={styles.restoreResultNotice} role="status" aria-live="polite">
          <div className={styles.restoreResultContent}>
            <CircleCheck size={18} aria-hidden="true" />
            <div>
              <strong>{t("sessions.restoreResultTitle")}</strong>
              <p>{t("sessions.restoreResultConversation", { count: restoreNotice.removed_messages })}</p>
              {restoreNotice.mode === "files" ? (
                <p>{t("sessions.restoreResultFiles", { count: restoreNotice.file_results.length })}</p>
              ) : null}
              {restoreNotice.file_results.some((item) => item.status === "failed") ? (
                <p className={styles.restoreResultFailure}>{t("sessions.restoreResultFailure")}</p>
              ) : null}
              {restoreNotice.file_results.some((item) => item.conflict) ? (
                <p>{t("sessions.restoreResultConflict", {
                  count: restoreNotice.file_results.filter((item) => item.conflict).length,
                })}</p>
              ) : null}
              <ul className={styles.restoreTargetList}>
                {restoreNotice.file_results.map((item) => (
                  <li key={item.operation_id}>
                    <code>{item.target}</code>
                    <span>{t(`sessions.restoreFileStatus.${item.status}`)}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <div className={styles.restoreResultActions}>
            {restoreNotice.file_results.some((item) => item.status === "failed")
              && !restoreNotice.failure_notification_acknowledged ? (
              <button
                className={styles.secondaryButton}
                type="button"
                disabled={restoreBusy}
                onClick={() => void acknowledgeRestoreNotice()}
              >
                {t("controls.acknowledgeRestore")}
              </button>
            ) : null}
            <button
              className={styles.iconButton}
              type="button"
              aria-label={t("controls.close")}
              title={t("controls.close")}
              onClick={() => setRestoreNotice(null)}
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
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
              <span>{(sessions?.sessions.length ?? 0) + draftSessionIds.length}</span>
            </div>
            <label className={styles.sessionSearch} htmlFor="session-search">
              <span className={styles.sessionSearchLabel}>{t("sessions.searchLabel")}</span>
              <span className={styles.sessionSearchControl}>
                <Search size={14} aria-hidden="true" />
                <input
                  id="session-search"
                  className={styles.sessionSearchInput}
                  type="search"
                  value={sessionSearch}
                  placeholder={t("sessions.searchPlaceholder")}
                  onChange={(event) => setSessionSearch(event.target.value)}
                />
              </span>
            </label>
            {draftSessionIds.map((draftId) => {
              const running = (liveRunsBySession[draftId] ?? []).some(isLiveRunActive);
              return (
                <button
                  className={styles.draftRow}
                  type="button"
                  key={draftId}
                  aria-current={selectedSessionId === draftId ? "true" : undefined}
                  disabled={busySessionId !== null}
                  onClick={() => void openSession(draftId, true)}
                >
                  <span className={styles.sessionRowMain}>
                    <MessageSquare size={15} aria-hidden="true" />
                    <strong>{running ? t("sessions.draftTitle") : t("sessions.draft")}</strong>
                  </span>
                  <span className={styles.sessionMeta}>{running ? t("conversation.running") : t("sessions.notPersisted")}</span>
                </button>
              );
            })}
            {sessions?.sessions.length === 0 && draftSessionIds.length === 0 ? (
              <div className={styles.sessionListEmpty}>
                <MessageSquare size={20} aria-hidden="true" />
                <p>{sessionSearch.trim() ? t("sessions.noSearchResults") : t("sessions.empty")}</p>
              </div>
            ) : (
              <ul className={styles.sessionList} aria-label={t("sessions.listLabel")}>
                {sessions?.sessions.map((item) => {
                  const isSelected = item.id === selectedSessionId;
                  const occupiedByOther = item.occupied && item.occupied_by === "client";
                  const hasActiveRun = (liveRunsBySession[item.id] ?? []).some(isLiveRunActive);
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
                          {hasActiveRun ? <span className={styles.sessionRunBadge}><Activity size={11} aria-hidden="true" />{t("conversation.running")}</span> : null}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
            {sessionNextCursor !== null ? (
              <button
                className={styles.sessionListMore}
                type="button"
                disabled={loadState === "loading"}
                onClick={loadMoreSessions}
              >
                <ChevronDown size={14} aria-hidden="true" />
                {t("sessions.loadMore")}
              </button>
            ) : null}
          </aside>

          <section className={styles.sessionContentPanel}>
            {claim !== null && snapshot !== null ? (
              <>
                <div className={styles.sessionContentHeader}>
                  <div>
                    <p className={styles.eyebrow}>{t("sessions.conversation")}</p>
                    <h2>{draft ? t("sessions.draftTitle") : selectedSummary?.title ?? t("sessions.title")}</h2>
                  </div>
                  <div className={styles.sessionHeaderActions}>
                    {claim !== null ? (
                      <button
                        className={styles.iconButton}
                        ref={managementTriggerRef}
                        type="button"
                        aria-label={t("controls.runtimeStatus")}
                        title={t("controls.runtimeStatus")}
                        disabled={connectionState !== "online" || busySessionId !== null}
                        onClick={() => setManagementOpen(true)}
                      >
                        <Gauge size={15} aria-hidden="true" />
                      </button>
                    ) : null}
                    {!draft && selectedSummary !== undefined ? (
                      <>
                        <button
                          className={styles.iconButton}
                          ref={renameTriggerRef}
                          type="button"
                          aria-label={t("controls.renameSession")}
                          title={t("controls.renameSession")}
                          disabled={busySessionId !== null || connectionState !== "online"}
                          onClick={beginRename}
                        >
                          <Pencil size={15} aria-hidden="true" />
                        </button>
                        <button
                          className={`${styles.iconButton} ${styles.dangerIconButton}`}
                          ref={deleteTriggerRef}
                          type="button"
                          aria-label={t("controls.deleteSession")}
                          title={t("controls.deleteSession")}
                          disabled={busySessionId !== null || connectionState !== "online" || pendingDeletion?.attempted === true}
                          onClick={beginDelete}
                        >
                          <Trash2 size={15} aria-hidden="true" />
                        </button>
                        {(snapshot.restore_anchors ?? []).length > 0 ? (
                          <button
                            className={styles.iconButton}
                            ref={restoreTriggerRef}
                            type="button"
                            aria-label={t("controls.restoreSession")}
                            title={t("controls.restoreSession")}
                            disabled={busySessionId !== null || activeRun !== null || connectionState !== "online"}
                            onClick={beginRestore}
                          >
                            <RotateCcw size={15} aria-hidden="true" />
                          </button>
                        ) : null}
                        {pendingRestoreFailure?.session_id === claim.session_id ? (
                          <button
                            className={styles.iconButton}
                            type="button"
                            aria-label={t("controls.reviewRestoreFailure")}
                            title={t("controls.reviewRestoreFailure")}
                            onClick={() => setRestoreNotice(pendingRestoreFailure)}
                          >
                            <Info size={15} aria-hidden="true" />
                          </button>
                        ) : null}
                      </>
                    ) : null}
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
                </div>
                <div className={styles.conversationViewport} role="log" aria-live="off" aria-label={t("sessions.historyLabel")}>
                  {snapshot.messages.length === 0 && selectedLiveRuns.length === 0 ? (
                    <div className={styles.emptyConversation} role="status">
                      <MessageSquare size={20} aria-hidden="true" />
                      <p>{draft ? t("conversation.emptyDraft") : t("sessions.noMessages")}</p>
                    </div>
                  ) : (
                    <div className={styles.messageHistory}>
                      {snapshot.messages.map((message, index) => (
                        <HistoryMessageView key={`history-${index}-${String(message.role)}`} message={message} index={index} t={t} />
                      ))}
                      {selectedLiveRuns.map((run) => (
                        <LiveRunView key={run.localId} run={run} t={t} onCancel={(candidate) => void cancelRun(candidate)} />
                      ))}
                    </div>
                  )}
                </div>
                <form className={styles.composer} onSubmit={(event) => void submitInput(event)}>
                  <label className={styles.srOnly} htmlFor="conversation-input">{t("conversation.inputLabel")}</label>
                  <textarea
                    ref={inputRef}
                    id="conversation-input"
                    aria-label={t("conversation.inputLabel")}
                    className={styles.composerInput}
                    rows={3}
                    value={inputText}
                    disabled={activeRun !== null || connectionState !== "online"}
                    placeholder={t("conversation.inputPlaceholder")}
                    onChange={(event) => {
                      draftsBySessionRef.current[claim.session_id] = event.target.value;
                      setInputText(event.target.value);
                    }}
                    onKeyDown={handleInputKeyDown}
                  />
                  <div className={styles.composerFooter}>
                    <p className={composerError !== null ? styles.composerError : styles.composerHint} role={composerError !== null ? "alert" : "status"}>
                      {composerError !== null
                        ? t(composerError)
                        : activeRun !== null
                          ? t("conversation.activeRun")
                          : t("conversation.enterHint")}
                    </p>
                    <button
                      className={styles.primaryButton}
                      type="submit"
                      disabled={!inputText.trim() || activeRun !== null || connectionState !== "online"}
                    >
                      <Send size={15} aria-hidden="true" />
                      {t("controls.send")}
                    </button>
                  </div>
                </form>
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
      {claim !== null && connectionState === "online" ? (
        <RuntimeManagementDialog
          key={`${claim.workspace_id}:${claim.session_id}:${claim.claim_version}:${claim.reconnect_credential}`}
          open={managementOpen}
          onOpenChange={setManagementOpen}
          claim={claim}
          sessionTitle={draft ? t("sessions.draftTitle") : selectedSummary?.title ?? t("sessions.title")}
          activeRuns={selectedLiveRuns}
          connectionState={connectionState}
          triggerRef={managementTriggerRef}
        />
      ) : null}
      <Dialog.Root
        open={deleteOpen}
        onOpenChange={(open) => {
          if (!open && deleteBusyRef.current) return;
          setDeleteOpen(open);
          if (!open) {
            setDeleteError(null);
            if (!pendingDeletionRef.current?.attempted) {
              pendingDeletionRef.current = null;
              setPendingDeletion(null);
            }
          }
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={`${styles.dialogContent} ${styles.sessionDeletionDialog}`}
            onEscapeKeyDown={(event) => { if (deleteBusyRef.current) event.preventDefault(); }}
            onPointerDownOutside={(event) => { if (deleteBusyRef.current) event.preventDefault(); }}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              const trigger = deleteTriggerRef.current;
              if (trigger?.isConnected && !trigger.disabled) trigger.focus();
              else document.getElementById("sessions-heading")?.focus();
            }}
          >
            <Dialog.Title className={styles.dialogTitle}>{t("sessions.deleteTitle")}</Dialog.Title>
            <Dialog.Description className={styles.dialogDescription}>
              {t("sessions.deleteDescription")}
            </Dialog.Description>
            {pendingDeletion?.title ? <p>{pendingDeletion.title}</p> : null}
            <form className={styles.dialogForm} onSubmit={(event) => void submitDelete(event)}>
              <p className={styles.dialogWarning}>
                <TriangleAlert size={16} aria-hidden="true" />
                {t("sessions.deleteDataNotice")}
              </p>
              {activeRun !== null ? (
                <p className={styles.fieldError} role="alert">{t("sessions.busyError")}</p>
              ) : null}
              {deleteError !== null ? (
                <p className={styles.fieldError} role="alert">{t(deleteError)}</p>
              ) : null}
              <div className={styles.dialogActions}>
                <button
                  className={styles.secondaryButton}
                  type="button"
                  disabled={deleteBusy}
                  onClick={() => {
                    setDeleteOpen(false);
                    if (!pendingDeletionRef.current?.attempted) {
                      pendingDeletionRef.current = null;
                      setPendingDeletion(null);
                    }
                  }}
                >
                  {t("controls.cancel")}
                </button>
                <button
                  className={styles.dangerButton}
                  type="submit"
                  disabled={deleteBusy || (activeRun !== null && pendingDeletion?.claim.session_id === selectedSessionId) || connectionState !== "online"}
                >
                  <Trash2 size={15} aria-hidden="true" />
                  {t("controls.confirmDelete")}
                </button>
              </div>
            </form>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root
        open={restoreOpen}
        onOpenChange={(open) => {
          if (!open) void closeRestore();
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={`${styles.dialogContent} ${styles.restoreDialog}`}
            onEscapeKeyDown={(event) => { if (restoreBusyRef.current) event.preventDefault(); }}
            onPointerDownOutside={(event) => { if (restoreBusyRef.current) event.preventDefault(); }}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              const trigger = restoreTriggerRef.current;
              if (trigger?.isConnected && !trigger.disabled) trigger.focus();
              else document.getElementById("sessions-heading")?.focus();
            }}
          >
            <Dialog.Title className={styles.dialogTitle}>{t("sessions.restoreTitle")}</Dialog.Title>
            <Dialog.Description className={styles.dialogDescription}>
              {t("sessions.restoreDescription")}
            </Dialog.Description>
            <form className={styles.dialogForm} onSubmit={(event) => void submitRestore(event)}>
              <label className={styles.fieldLabel} htmlFor="restore-anchor-select">
                {t("sessions.restoreAnchorLabel")}
              </label>
              {(snapshot?.restore_anchors ?? []).length > 0 ? (
                <select
                  id="restore-anchor-select"
                  className={styles.textInput}
                  value={restoreAnchorId ?? ""}
                  disabled={restoreBusy || restorePlan !== null}
                  onChange={(event) => {
                    setRestoreAnchorId(Number(event.target.value));
                    setRestorePlan(null);
                    setRestoreError(null);
                  }}
                >
                  <option value="" disabled>{t("sessions.restoreAnchorPlaceholder")}</option>
                  {(snapshot?.restore_anchors ?? []).map((anchor) => (
                    <option key={anchor.anchor_id} value={anchor.anchor_id}>
                      #{anchor.anchor_id} · {formatSessionTime(anchor.timestamp, i18n.language)}
                    </option>
                  ))}
                </select>
              ) : (
                <p className={styles.dialogDescription}>{t("sessions.restoreAnchorEmpty")}</p>
              )}
              {selectedRestoreAnchor !== undefined && restorePlan === null ? (
                <p className={styles.restoreAnchorPreview}>{selectedRestoreAnchor.content}</p>
              ) : null}
              {restorePlan !== null ? (
                <div className={styles.restorePlan}>
                  <p className={styles.restorePlanHeading}>{t("sessions.restorePreview")}</p>
                  <p>{t("sessions.restoreRemoved", { count: restorePlan.removed_messages })}</p>
                  <fieldset className={styles.restoreModeFieldset}>
                    <legend className={styles.fieldLabel}>{t("sessions.restoreMode")}</legend>
                    <label className={styles.restoreModeOption}>
                      <input
                        type="radio"
                        name="restore-mode"
                        value="conversation-only"
                        checked={restoreMode === "conversation-only"}
                        disabled={restoreBusy || !restorePlan.available_modes.includes("conversation-only")}
                        onChange={() => setRestoreMode("conversation-only")}
                      />
                      {t("sessions.restoreConversationOnly")}
                    </label>
                    <label className={styles.restoreModeOption}>
                      <input
                        type="radio"
                        name="restore-mode"
                        value="files"
                        checked={restoreMode === "files"}
                        disabled={restoreBusy || !restorePlan.available_modes.includes("files")}
                        onChange={() => setRestoreMode("files")}
                      />
                      {t("sessions.restoreFilesMode")}
                    </label>
                  </fieldset>
                  <p className={styles.restorePlanHeading}>{t("sessions.restoreFiles")}</p>
                  {restorePlan.targets.length === 0 ? (
                    <p>{t("sessions.restoreNoFiles")}</p>
                  ) : (
                    <ul className={styles.restoreTargetList}>
                      {restorePlan.targets.map((target) => (
                        <li key={`${target.operation_id}-${target.canonical_target}`}>
                          <code>{target.canonical_target}</code>
                          {restorePlan.conflict_targets.includes(target.canonical_target) ? (
                            <span className={styles.restoreConflict}>{t("sessions.restoreConflict")}</span>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : null}
              {restoreError !== null ? (
                <p className={styles.fieldError} role="alert">{t(restoreError)}</p>
              ) : null}
              <div className={styles.dialogActions}>
                <button
                  className={styles.secondaryButton}
                  type="button"
                  disabled={restoreBusy}
                  onClick={() => void closeRestore()}
                >
                  {t("controls.cancel")}
                </button>
                {restorePlan === null ? (
                  <button
                    className={styles.primaryButton}
                    type="button"
                    disabled={restoreBusy || restoreAnchorId === null || connectionState !== "online"}
                    onClick={() => void inspectSelectedRestore()}
                  >
                    <Info size={15} aria-hidden="true" />
                    {restoreBusy ? t("controls.restoring") : t("controls.inspectRestore")}
                  </button>
                ) : (
                  <button
                    className={styles.dangerButton}
                    type="submit"
                    disabled={restoreBusy || connectionState !== "online"}
                  >
                    <RotateCcw size={15} aria-hidden="true" />
                    {restoreBusy ? t("controls.restoring") : t("controls.executeRestore")}
                  </button>
                )}
              </div>
            </form>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root
        open={renameOpen}
        onOpenChange={(open) => {
          setRenameOpen(open);
          if (!open) setRenameError(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={styles.dialogContent}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              renameTriggerRef.current?.focus();
            }}
          >
            <Dialog.Title className={styles.dialogTitle}>{t("sessions.renameTitle")}</Dialog.Title>
            <Dialog.Description className={styles.dialogDescription}>
              {t("sessions.renameDescription")}
            </Dialog.Description>
            <form className={styles.dialogForm} onSubmit={(event) => void submitRename(event)}>
              <label className={styles.fieldLabel} htmlFor="session-rename-title">
                {t("sessions.titleLabel")}
              </label>
              <input
                id="session-rename-title"
                className={styles.textInput}
                type="text"
                value={renameTitle}
                maxLength={160}
                autoFocus
                onChange={(event) => setRenameTitle(event.target.value)}
              />
              {renameError !== null ? (
                <p className={styles.fieldError} role="alert">{t(renameError)}</p>
              ) : null}
              <div className={styles.dialogActions}>
                <button
                  className={styles.secondaryButton}
                  type="button"
                  disabled={renameBusy}
                  onClick={() => setRenameOpen(false)}
                >
                  {t("controls.cancel")}
                </button>
                <button
                  className={styles.primaryButton}
                  type="submit"
                  disabled={renameBusy || !renameTitle.trim() || connectionState !== "online"}
                >
                  <Check size={15} aria-hidden="true" />
                  {t("controls.save")}
                </button>
              </div>
            </form>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </section>
  );
}

function formatSessionTime(value: string, language: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString(language);
}

function parseConfirmationEvent(event: ServiceEvent): PendingConfirmation | null {
  if (event.workspace_id === null || !isRecord(event.payload)) return null;
  const payload = event.payload;
  const token = readString(payload.token);
  const origin = payload.origin === "foreground" || payload.origin === "background"
    ? payload.origin
    : null;
  const requestValue = payload.request;
  if (token === null || origin === null || !isRecord(requestValue)) return null;
  const confirmationId = readString(requestValue.confirmation_id);
  const toolCallId = readString(requestValue.tool_call_id);
  const toolName = readString(requestValue.tool_name);
  const reason = readString(requestValue.reason);
  const summary = readString(requestValue.summary);
  const details = requestValue.details;
  const warnings = requestValue.warnings;
  if (
    confirmationId === null
    || toolCallId === null
    || toolName === null
    || reason === null
    || summary === null
    || !isRecord(details)
    || !Array.isArray(warnings)
    || warnings.some((warning) => typeof warning !== "string")
  ) {
    return null;
  }
  return {
    token,
    origin,
    workspaceId: event.workspace_id,
    projectId: event.project_id,
    sessionId: event.session_id,
    runId: event.run_id,
    jobId: readString(payload.job_id) ?? (isRecord(payload.owner) ? readString(payload.owner.job_id) : null),
    title: readString(payload.title),
    request: {
      confirmation_id: confirmationId,
      tool_call_id: toolCallId,
      tool_name: toolName,
      reason,
      summary,
      details,
      warnings: warnings as string[],
    },
  };
}

function formatConfirmationDetails(details: Record<string, unknown>): string {
  try {
    return JSON.stringify(details, null, 2) || "{}";
  } catch {
    return "{}";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
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

function sessionDeleteErrorKey(error: unknown): string {
  if (error instanceof ApiError && error.body?.code === "persistence_error") {
    return "sessions.deletePersistenceError";
  }
  return sessionErrorKey(error);
}

function scheduleText(job: RegisteredProject["saved_jobs"][number], t: (key: string, options?: Record<string, unknown>) => string): string {
  if (job.schedule.kind === "every") return t("projects.everySchedule", { seconds: job.schedule.every_seconds });
  if (job.schedule.kind === "cron") {
    return t("projects.cronSchedule", { expression: job.schedule.cron_expr, timezone: job.schedule.timezone });
  }
  return t("projects.atSchedule", { time: job.schedule.at_time });
}

function projectScheduleLabel(
  project: RegisteredProject,
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  if (!project.available) {
    return t("projects.scheduleUnavailable");
  }
  if (project.schedule_state !== "available") {
    return t(`projects.scheduleState.${project.schedule_state}`);
  }
  if (project.schedule_status === null) {
    return t("projects.scheduleUnavailable");
  }
  if (project.schedule_status.status === "faulted") {
    return t("projects.scheduleUnavailable");
  }
  if (!project.schedule_status.admitted) return t("projects.schedulePaused");
  return project.schedule_status.active_job_count > 0
    ? t("projects.scheduleRunning", { count: project.schedule_status.active_job_count })
    : t("projects.scheduleState.available");
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
    if (error.body?.code === "admission_closed") return "projects.removalInProgressError";
    if (error.body?.code === "project_removal_failed") return "projects.removalFailedError";
    if (error.body?.code === "project_removal_blocked") return "projects.removalFailedError";
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
