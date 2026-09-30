import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowLeft,
  Activity,
  Ban,
  Check,
  CircleAlert,
  CircleCheck,
  ChevronDown,
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
  Send,
  ShieldX,
  Square,
  Sun,
  TriangleAlert,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, NavLink, Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";

import {
  ApiError,
  claimProjectSession,
  createRequestId,
  createProjectSession,
  exchangeTicket,
  getProjectSession,
  getProjectSessions,
  getProjects,
  getServiceStatus,
  openEventStream,
  releaseProjectSession,
  registerProject,
  registerWebClient,
  resumeProjectSchedule,
  restoreBrowserSession,
  ServiceCommandError,
} from "./api";
import type {
  ClientCommand,
  ConfirmationRequest,
  ConfirmationOrigin,
  ProjectSessionsResponse,
  RegisteredProject,
  RegisteredClient,
  ServiceState,
  ServiceCommandResult,
  ServiceEvent,
  ServiceStatus,
  SessionClaim,
  SessionSnapshot,
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
            if (event.type === "session.claimed" || event.type === "session.released") {
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
                    data-paused={project.schedule_status?.admitted !== true}
                  >
                    {projectScheduleLabel(project, t)}
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
  const [loadState, setLoadState] = useState<SessionLoadState>("idle");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [claim, setClaim] = useState<SessionClaim | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [draft, setDraft] = useState(false);
  const [draftSessionIds, setDraftSessionIds] = useState<string[]>([]);
  const [busySessionId, setBusySessionId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
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
  const refreshSessionsRef = useRef<(() => Promise<void>) | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const draftsBySessionRef = useRef<Record<string, string>>({});
  const pendingSubmissionsRef = useRef<PendingSubmission[]>([]);
  const pendingClientIdRef = useRef<string | null>(null);
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

  const releaseClaims = useCallback(() => {
    if (!projectId) return;
    for (const current of Object.values(claimsBySessionRef.current)) {
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

  const refreshSessions = useCallback(async () => {
    if (authState !== "ready" || !projectId) return;
    setLoadState((state) => (state === "ready" ? state : "loading"));
    try {
      const response = await getProjectSessions(projectId);
      if (!mountedRef.current) return;
      workspaceIdRef.current = response.workspace_id;
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
            rememberSession(restored.claim, restored.snapshot);
            setSelectedSessionId(restored.claim.session_id);
            selectedSessionRef.current = restored.claim.session_id;
            const restoredDraft = !response.sessions.some((item) => item.id === restored.claim.session_id);
            setDraft(restoredDraft);
            if (restoredDraft) {
              setDraftSessionIds((ids) => ids.includes(restored.claim.session_id)
                ? ids : [...ids, restored.claim.session_id]);
            }
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
            rememberSession(restored.claim, restored.snapshot);
          }
        } catch (error) {
          if (claimRef.current === currentClaim) clearClaimState();
          throw error;
        }
      }
      setSessions(response);
      setDraftSessionIds((ids) => ids.filter((id) => !response.sessions.some((item) => item.id === id)));
      if (response.sessions.some((item) => item.id === selectedSessionRef.current)) setDraft(false);
      setLoadState("ready");
      setActionError(restoreError);
    } catch (error) {
      setLoadState("error");
      setActionError(sessionErrorKey(error));
    }
  }, [authState, clearClaimState, connectionState, onRestoreConsumed, projectId, registeredClient, releaseOrphanClaim, rememberSession]);

  useEffect(() => {
    if (connectionState !== "online") needsReclaimRef.current = true;
  }, [connectionState]);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions, refreshVersion]);

  useEffect(() => {
    refreshSessionsRef.current = refreshSessions;
  }, [refreshSessions]);

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
      if (!mountedRef.current) return;
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
          const entry = item as { workspace_id?: unknown; snapshot?: unknown };
          const nextSnapshot = entry.snapshot as Partial<SessionSnapshot> | null;
          if (
            typeof entry.workspace_id !== "string"
            || typeof nextSnapshot !== "object" || nextSnapshot === null
            || typeof nextSnapshot.session_id !== "string"
            || !Array.isArray(nextSnapshot.messages)
          ) continue;
          const currentClaim = claimsBySessionRef.current[nextSnapshot.session_id];
          if (currentClaim?.workspace_id !== entry.workspace_id) continue;
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
    if (busySessionId !== null && !allowBusy) return;
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

  const selectedSummary = sessions?.sessions.find((item) => item.id === selectedSessionId);
  const selectedLiveRuns = selectedSessionId === null ? [] : liveRunsBySession[selectedSessionId] ?? [];
  const activeRun = selectedLiveRuns.find(isLiveRunActive) ?? null;
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
              <span>{(sessions?.sessions.length ?? 0) + draftSessionIds.length}</span>
            </div>
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
                <p>{t("sessions.empty")}</p>
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
          </aside>

          <section className={styles.sessionContentPanel}>
            {claim !== null && snapshot !== null ? (
              <>
                <div className={styles.sessionContentHeader}>
                  <div>
                    <p className={styles.eyebrow}>{t("sessions.conversation")}</p>
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
