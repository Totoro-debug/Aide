import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowLeft,
  Activity,
  Ban,
  BookOpen,
  Brain,
  CalendarClock,
  Check,
  CircleAlert,
  CircleCheck,
  Clock3,
  ChevronDown,
  ChevronRight,
  Eye,
  FolderOpen,
  Gauge,
  Info,
  LockKeyhole,
  Languages,
  LogOut,
  Menu,
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
  Settings2,
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
import { Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";

import {
  ApiError,
  claimProjectSession,
  createRequestId,
  createProjectSession,
  deleteProjectSession,
  getProjectSessionDeletionStatus,
  claimProjectSessionDeletion,
  claimWorkspaceSession,
  claimWorkspaceSessionDeletion,
  exchangeTicket,
  getProjectRemoval,
  getProjectSession,
  getRestoreResult,
  getProjectSessions,
  getChatSessions,
  getWorkspaceSession,
  getWorkspaceSessionDeletionStatus,
  getWorkspaceSessions,
  getProjects,
  getRuntimeMemory,
  getRuntimeStatus,
  getServiceStatus,
  getConfig,
  openEventStream,
  releaseProjectSession,
  releaseWorkspaceSession,
  registerProject,
  registerWebClient,
  reloadRuntimeSkills,
  renameProjectSession,
  renameWorkspaceSession,
  createWorkspaceSession,
  deleteWorkspaceSession,
  enterChatWorkspace,
  removeProject,
  resumeProjectSchedule,
  acknowledgeRestore,
  cancelRestore,
  createScheduleJob,
  deleteScheduleJob,
  executeRestore,
  getScheduleJobs,
  getScheduleJob,
  getScheduleJobHistory,
  inspectRestore,
  restoreBrowserSession,
  ServiceCommandError,
  triggerRuntimeDream,
  patchConfig,
  repairConfig,
  updateRuntimeEffort,
  updateRuntimePermission,
} from "./api";
import type {
  ClientCommand,
  ConfirmationRequest,
  ConfirmationOrigin,
  DreamResult,
  ProjectSessionsResponse,
  WorkspaceSessionsResponse,
  ChatSessionsResponse,
  ChatSessionSummary,
  RegisteredProject,
  RegisteredClient,
  ReasoningEffort,
  RuntimeStatus,
  ServiceState,
  ServiceCommandResult,
  ServiceEvent,
  ServiceStatus,
  ConfigFields,
  ConfigPatchFields,
  ConfigResponse,
  ConfigSecretChange,
  ConfigSecrets,
  SessionClaim,
  SessionClaimResponse,
  SessionSnapshot,
  SessionSummary,
  SkillMetadata,
  ToolPermissionLevel,
  RestoreMode,
  RestorePlan,
  RestoreResult,
  ProjectScheduleKind,
  ScheduleJob,
  ScheduleHistoryGroup,
  ScheduleHistoryResultState,
  ScheduleJobInput,
  ScheduleJobHistoryResponse,
  ScheduleJobStatus,
  ScheduleJobsResponse,
  ScheduleStatus,
} from "./protocol";
import styles from "./App.module.css";

type AuthState = "checking" | "ready" | "required" | "error";
type ConnectionState = "checking" | "online" | "offline" | "recovering";
type Theme = "system" | "light" | "dark";
type ProjectsLoadState = "idle" | "loading" | "ready" | "error";
type SettingsSection = "general" | "models" | "runtime" | "memory" | "mcp";
type PersistedSettingsSection = Exclude<SettingsSection, "general"> | "web";
type ServiceEventListener = (event: ServiceEvent) => void;

interface SettingsReturnLocation {
  pathname: string;
  search: string;
  hash: string;
  scrollTop: number;
  conversationScrollTop?: number;
}

interface SettingsNavigationState {
  returnTo?: SettingsReturnLocation;
}

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

const THEME_KEY = "omni.theme";
const initialLaunchTicket = readAndClearTicket();
const PERMISSION_LEVELS: ToolPermissionLevel[] = ["read-only", "workspace-write", "full-access"];
const REASONING_EFFORTS: ReasoningEffort[] = ["low", "medium", "high", "xhigh", "max"];

function usePanelKeyboard(open: boolean, setOpen: (open: boolean) => void, panelId: string, triggerId: string) {
  useEffect(() => {
    if (!open || !window.matchMedia("(max-width: 1024px)").matches) return;
    const panel = document.getElementById(panelId);
    const trigger = document.getElementById(triggerId);
    if (panel === null) return;
    const controls = () => Array.from(panel.querySelectorAll<HTMLElement>(
      'a[href], button:not(:disabled), input:not(:disabled), [tabindex="0"]',
    )).filter((element) => element.getClientRects().length > 0);
    controls()[0]?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (!window.matchMedia("(max-width: 1024px)").matches || panel.getClientRects().length === 0) return;
      if (event.defaultPrevented || document.querySelector('[role="dialog"]') !== null) return;
      if (event.key === "Escape") {
        event.preventDefault();
        trigger?.focus();
        setOpen(false);
      } else if (event.key === "Tab") {
        const items = controls();
        const first = items[0];
        const last = items.at(-1);
        if (!panel.contains(document.activeElement)
          || (event.shiftKey ? document.activeElement === first : document.activeElement === last)) {
          event.preventDefault();
          (event.shiftKey ? last : first)?.focus();
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
      if (panel.contains(document.activeElement) || document.activeElement === document.body) trigger?.focus();
    };
  }, [open, setOpen, panelId, triggerId]);
}

export default function App() {
  const { i18n, t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const [authState, setAuthState] = useState<AuthState>("checking");
  const [connectionState, setConnectionState] = useState<ConnectionState>("checking");
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [configurationNeedsSetup, setConfigurationNeedsSetup] = useState<boolean | null>(null);
  const [registeredClient, setRegisteredClient] = useState<RegisteredClient | null>(null);
  const [projects, setProjects] = useState<RegisteredProject[]>([]);
  const [projectsLoadState, setProjectsLoadState] = useState<ProjectsLoadState>("idle");
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(() => readThemePreference());
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [sessionEventVersion, setSessionEventVersion] = useState(0);
  const [newChatVersion, setNewChatVersion] = useState(0);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
  const [confirmationNotice, setConfirmationNotice] = useState<string | null>(null);
  const [settingsVisited, setSettingsVisited] = useState(location.pathname === "/settings");
  const mainContentRef = useRef<HTMLElement | null>(null);
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

  usePanelKeyboard(sidebarOpen, setSidebarOpen, "app-sidebar", "app-sidebar-toggle");

  useEffect(() => {
    if (authState === "ready") void refreshProjects();
  }, [authState, refreshProjects]);

  useEffect(() => {
    if (authState !== "ready") return;
    let active = true;
    let sequence = 0;
    const refreshConfiguration = async () => {
      const requestSequence = ++sequence;
      try {
        const response = await getConfig();
        if (active && requestSequence === sequence) {
          setConfigurationNeedsSetup(response.application.active_revision === null);
        }
      } catch {
        // SettingsView owns the detailed configuration error state.
      }
    };
    const unsubscribe = subscribeServiceEvents((event) => {
      if (event.type === "config.application") {
        ++sequence;
        setConfigurationNeedsSetup(event.payload.active_revision === null);
      } else if (event.type === "snapshot.required") {
        void refreshConfiguration();
      }
    });
    void refreshConfiguration();
    const timer = window.setInterval(() => void refreshConfiguration(), 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
      unsubscribe();
    };
  }, [authState, connectionState, subscribeServiceEvents]);

  useEffect(() => {
    if (
      configurationNeedsSetup === true
      && location.pathname === "/status"
    ) {
      navigate("/settings", { replace: true });
    }
  }, [configurationNeedsSetup, location.pathname, navigate]);

  useEffect(() => {
    if (authState !== "ready") return;
    const projectsTimer = window.setInterval(() => void refreshProjects(), 5000);
    return () => window.clearInterval(projectsTimer);
  }, [authState, refreshProjects]);

  useEffect(() => {
    const unsubscribe = subscribeServiceEvents((incoming) => {
      let event = incoming;
      if (event.type === "snapshot.required") {
        const snapshot = event.payload.snapshot;
        if (typeof snapshot !== "object" || snapshot === null) return;
        const pending = (snapshot as { pending_confirmation?: unknown }).pending_confirmation;
        if (pending === undefined) return;
        if (pending === null) {
          pendingConfirmationRef.current = null;
          resolvingConfirmationTokenRef.current = null;
          setPendingConfirmation(null);
          return;
        }
        if (typeof pending !== "object") return;
        event = { ...event, ...pending, type: "confirmation.requested" } as ServiceEvent;
      }
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
              const activeConnection = eventStreamRef.current;
              if (activeConnection !== null) {
                void activeConnection.sendCommand({
                  request_id: createRequestId(), type: "subscribe", workspace_id: null,
                  session_id: null, claim_version: null,
                  payload: { last_seq: null, stream_id: event.stream_id },
                }).catch(() => activeConnection.close());
              }
              return;
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

  const isProjectSessionRoute = /^\/projects\/[^/]+$/.test(location.pathname);
  const isChatRoute = location.pathname === "/" || location.pathname === "/chat";
  const settingsReturnLocation = (location.state as SettingsNavigationState | null)?.returnTo;
  const conversationLocation = location.pathname === "/settings"
    && settingsReturnLocation !== undefined
    && (settingsReturnLocation.pathname === "/"
      || settingsReturnLocation.pathname === "/chat"
      || /^\/projects\/[^/]+$/.test(settingsReturnLocation.pathname))
      ? settingsReturnLocation
      : location;
  const openSettings = (event: React.MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    if (location.pathname === "/settings") return;
    setSettingsVisited(true);
    navigate("/settings", {
      state: {
        returnTo: {
          pathname: location.pathname,
          search: location.search,
          hash: location.hash,
          scrollTop: mainContentRef.current?.scrollTop ?? 0,
          conversationScrollTop: mainContentRef.current?.querySelector<HTMLElement>('[role="log"]')?.scrollTop,
        } satisfies SettingsReturnLocation,
      } satisfies SettingsNavigationState,
    });
    setSidebarOpen(false);
  };
  const returnFromSettings = () => {
    const target = settingsReturnLocation ?? {
      pathname: "/",
      search: "",
      hash: "",
      scrollTop: 0,
    };
    navigate(`${target.pathname}${target.search}${target.hash}`, { replace: true });
    window.requestAnimationFrame(() => {
      if (mainContentRef.current !== null) {
        mainContentRef.current.scrollTop = target.scrollTop;
        const log = mainContentRef.current.querySelector<HTMLElement>('[role="log"]');
        if (log !== null && target.conversationScrollTop !== undefined) log.scrollTop = target.conversationScrollTop;
      }
    });
  };
  return (
    <div className={styles.appShell}>
      <a className={styles.skipLink} href="#main-content">
        {t("nav.status")}
      </a>
      {sidebarOpen ? (
        <button
          className={styles.sidebarBackdrop}
          type="button"
          aria-label={t("controls.closeNavigation")}
          onClick={() => setSidebarOpen(false)}
        />
      ) : null}
      <aside className={styles.sidebar} id="app-sidebar" aria-label={t("app.name")} data-open={sidebarOpen}>
        <div className={styles.brandBlock}>
          <div className={styles.brandMark} aria-hidden="true">
            <Activity size={18} strokeWidth={2.2} />
          </div>
          <div>
            <div className={styles.brandName}>{t("app.name")}</div>
            <div className={styles.brandSubtitle}>{t("app.subtitle")}</div>
          </div>
          <button
            className={`${styles.iconButton} ${styles.sidebarDismiss}`}
            type="button"
            aria-label={t("controls.closeNavigation")}
            onClick={() => setSidebarOpen(false)}
          >
            <X size={17} aria-hidden="true" />
          </button>
        </div>
        <nav className={styles.navigation} aria-label={t("app.name")}>
          <Link
            className={styles.navLink}
            to="/"
            onClick={() => {
              setNewChatVersion((version) => version + 1);
              setSidebarOpen(false);
            }}
          >
            <MessageSquare size={16} aria-hidden="true" />
            <span>{t("nav.newChat")}</span>
          </Link>
          <NavLink className={({ isActive }) => isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink} to="/status" onClick={() => setSidebarOpen(false)}>
            <Activity size={16} aria-hidden="true" />
            <span>{t("nav.status")}</span>
          </NavLink>
          <NavLink className={({ isActive }) => isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink} to="/projects" onClick={() => setSidebarOpen(false)}>
            <FolderOpen size={16} aria-hidden="true" />
            <span>{t("nav.projects")}</span>
          </NavLink>
          <NavLink className={({ isActive }) => isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink} to="/settings" onClick={openSettings}>
            <Settings2 size={16} aria-hidden="true" />
            <span>{t("nav.settings")}</span>
          </NavLink>
        </nav>
        {authState === "ready" && projects.length > 0 ? (
          <div className={styles.projectNavigation} aria-label={t("nav.projects")}>
            {projects.map((project) => (
              <Link
                className={styles.projectNavigationLink}
                key={project.project_id}
                to={`/projects/${project.project_id}`}
                onClick={() => setSidebarOpen(false)}
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
          <div className={styles.topbarLeading}>
            <button
              className={styles.sidebarToggle}
              id="app-sidebar-toggle"
              type="button"
              aria-label={sidebarOpen ? t("controls.closeNavigation") : t("controls.openNavigation")}
              aria-controls="app-sidebar"
              aria-expanded={sidebarOpen}
              onClick={() => setSidebarOpen((open) => !open)}
            >
              {sidebarOpen ? <X size={18} aria-hidden="true" /> : <Menu size={18} aria-hidden="true" />}
            </button>
            <div className={styles.breadcrumb}>
              <span className={styles.breadcrumbMuted}>{t("app.name")}</span>
              <span className={styles.breadcrumbDivider} aria-hidden="true">
                /
              </span>
              <span>
                {location.pathname === "/settings"
                  ? t("nav.settings")
                  : isChatRoute
                  ? t("nav.newChat")
                  : location.pathname.startsWith("/projects/")
                  ? location.pathname.includes("/schedule")
                    ? t("nav.schedule")
                    : t("nav.sessions")
                  : t(location.pathname === "/projects" ? "nav.projects" : "nav.status")}
              </span>
            </div>
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

        <main
          id="main-content"
          className={isProjectSessionRoute || isChatRoute ? `${styles.mainContent} ${styles.conversationMainContent}` : styles.mainContent}
          ref={mainContentRef}
          tabIndex={-1}
        >
          <div style={{ display: location.pathname === "/settings" ? "none" : "contents" }}>
            <Routes location={conversationLocation}>
              <Route
                path="/"
                element={
                  <ChatSessionsView
                        configurationNeedsSetup={configurationNeedsSetup}
                        authState={authState}
                        connectionState={connectionState}
                        projects={projects}
                        registeredClient={registeredClient}
                        onRestoreConsumed={consumeRegisteredClient}
                        refreshVersion={sessionEventVersion}
                        sendServiceCommand={sendServiceCommand}
                        subscribeServiceEvents={subscribeServiceEvents}
                        confirmationTriggerRef={confirmationTriggerRef}
                        newChatVersion={newChatVersion}
                      />
                }
              />
              <Route
                path="/chat"
                element={
                  <ChatSessionsView
                        configurationNeedsSetup={configurationNeedsSetup}
                        authState={authState}
                        connectionState={connectionState}
                        projects={projects}
                        registeredClient={registeredClient}
                        onRestoreConsumed={consumeRegisteredClient}
                        refreshVersion={sessionEventVersion}
                        sendServiceCommand={sendServiceCommand}
                        subscribeServiceEvents={subscribeServiceEvents}
                        confirmationTriggerRef={confirmationTriggerRef}
                        newChatVersion={newChatVersion}
                      />
                }
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
              <Route path="/settings" element={null} />
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
              <Route
                path="/projects/:projectId/schedule"
                element={
                  <ScheduleJobsView
                    authState={authState}
                    connectionState={connectionState}
                    projects={projects}
                  />
                }
              />
              <Route
                path="/projects/:projectId/schedule/jobs/:jobId/history"
                element={
                  <ScheduleJobHistoryView
                    authState={authState}
                    connectionState={connectionState}
                    projects={projects}
                  />
                }
              />
              <Route path="*" element={<Navigate replace to="/" />} />
            </Routes>
          </div>
          {settingsVisited || location.pathname === "/settings" ? (
            <div hidden={location.pathname !== "/settings"}>
              <SettingsView
                authState={authState}
                connectionState={connectionState}
                language={language}
                onBack={returnFromSettings}
                onLanguageChange={(next) => void i18n.changeLanguage(next)}
                onThemeChange={setTheme}
                serviceStatus={serviceStatus}
                theme={theme}
              />
            </div>
          ) : null}
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
  const lastConfirmationOriginRef = useRef<ConfirmationOrigin | null>(null);
  useEffect(() => {
    if (confirmation !== null) lastConfirmationOriginRef.current = confirmation.origin;
  }, [confirmation]);
  const project = confirmation?.projectId === null
    ? undefined
    : projects.find((item) => item.project_id === confirmation?.projectId);

  function restoreFocus() {
    const target = triggerRef.current
      ?? (lastConfirmationOriginRef.current === "foreground" ? document.querySelector("textarea") : null)
      ?? document.getElementById("main-content");
    if (!(target instanceof HTMLElement) || !target.isConnected) return;
    if (target instanceof HTMLTextAreaElement) triggerRef.current = target;
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

interface SettingsForm {
  runtime: {
    max_tool_result_chars: string;
    max_iterations: string;
    enable_skill_always_load: boolean;
    compact_ratio: string;
    permission_level: ToolPermissionLevel;
    exec_shell: "auto" | "powershell" | "pwsh";
  };
  memory: {
    batch_size: string;
    schedule: string;
  };
  web: {
    default_chat_workspace: string;
  };
  models: {
    providers: Record<string, ProviderForm>;
    routes: Record<string, RouteForm>;
  };
  mcp: Record<string, McpForm>;
}

type SecretAction = ConfigSecretChange["action"];

interface SecretDraft {
  configured: boolean;
  action: SecretAction;
  value: string;
}

interface ProviderForm {
  id: string;
  protocol: string;
  base_url: string;
  models: string[];
  api_key: SecretDraft;
}

interface RouteForm {
  name: string;
  provider_id: string;
  model: string;
  context_window: string;
  max_output: string;
  temperature: string;
  reasoning_effort: ReasoningEffort;
  timeout: string;
}

interface McpForm {
  name: string;
  enabled: boolean;
  transport: "stdio" | "streamable-http";
  command: string;
  args: string[];
  cwd: string;
  url: string;
  headers: Record<string, SecretDraft & { name: string }>;
  connect_timeout: string;
  call_timeout: string;
  tool_keywords: { id: string; name: string; keywords: string[] }[];
}

function secretDraft(configured: boolean): SecretDraft {
  return { configured, action: "keep", value: "" };
}

function formFromConfig(fields: ConfigFields, previous: SettingsForm | null = null): SettingsForm {
  return {
    runtime: {
      max_tool_result_chars: String(fields.runtime.max_tool_result_chars),
      max_iterations: String(fields.runtime.max_iterations),
      enable_skill_always_load: fields.runtime.enable_skill_always_load,
      compact_ratio: String(fields.runtime.compact_ratio),
      permission_level: fields.runtime.permission_level,
      exec_shell: fields.runtime.exec_shell,
    },
    memory: {
      batch_size: String(fields.memory.batch_size),
      schedule: fields.memory.schedule,
    },
    web: {
      default_chat_workspace: fields.web.default_chat_workspace,
    },
    models: {
      providers: Object.fromEntries(Object.entries(fields.models.providers).map(([id, provider]) => [id, {
        id,
        protocol: provider.protocol,
        base_url: provider.base_url,
        models: provider.models,
        api_key: secretDraft(provider.api_key.configured),
      }])),
      routes: Object.fromEntries(Object.entries(fields.models.routes).map(([name, route]) => [name, {
        name,
        provider_id: route.provider_id,
        model: route.model,
        context_window: String(route.context_window),
        max_output: String(route.max_output),
        temperature: String(route.temperature),
        reasoning_effort: route.reasoning_effort,
        timeout: String(route.timeout),
      }])),
    },
    mcp: Object.fromEntries(Object.entries(fields.mcp).map(([name, server]) => {
      const previousEntry = Object.entries(previous?.mcp ?? {}).find(([row, value]) => row === name || value.name === name);
      const row = previousEntry?.[0] ?? name;
      const previousServer = previousEntry?.[1];
      return [row, {
        name,
        enabled: server.enabled,
        transport: server.transport,
        command: server.command ?? "",
        args: server.args,
        cwd: server.cwd ?? "",
        url: server.url ?? "",
        headers: Object.fromEntries(Object.entries(server.headers).map(([header, value]) => {
          const previousHeader = Object.entries(previousServer?.headers ?? {}).find(([row, draft]) => (
            row === header || draft.name === header
          ));
          const row = previousHeader?.[0] ?? header;
          return [row, { ...secretDraft(value.configured), name: header }];
        })),
        connect_timeout: String(server.connect_timeout),
        call_timeout: String(server.call_timeout),
        tool_keywords: Object.entries(server.tool_keywords).map(([toolName, keywords]) => ({
          id: previousServer?.tool_keywords.find((tool) => tool.name === toolName)?.id ?? createRequestId(),
          name: toolName,
          keywords,
        })),
      }];
    })),
  };
}

function configFromForm(form: SettingsForm): { fields: ConfigPatchFields; secrets: ConfigSecrets } {
  const secrets: ConfigSecrets = {};
  const providers: Record<string, { protocol: string; base_url: string; models: string[] }> = {};
  for (const provider of Object.values(form.models.providers)) {
    providers[provider.id] = {
      protocol: provider.protocol,
      base_url: provider.base_url,
      models: provider.models,
    };
    secrets[`models.providers.${provider.id}.api_key`] = provider.api_key.action === "replace"
      ? { action: "replace", value: provider.api_key.value }
      : { action: provider.api_key.action };
  }
  const routes: Record<string, Record<string, string | number>> = {};
  for (const route of Object.values(form.models.routes)) {
    routes[route.name] = {
      provider_id: route.provider_id,
      model: route.model,
      context_window: Number(route.context_window),
      max_output: Number(route.max_output),
      temperature: Number(route.temperature),
      reasoning_effort: route.reasoning_effort,
      timeout: Number(route.timeout),
    };
  }
  const mcp: Record<string, Record<string, unknown>> = {};
  for (const server of Object.values(form.mcp)) {
    const headers: Record<string, { configured: boolean }> = Object.create(null) as Record<string, { configured: boolean }>;
    for (const draft of server.transport === "streamable-http" ? Object.values(server.headers) : []) {
      const header = draft.name;
      headers[header] = { configured: draft.configured };
      const path = `mcp.${server.name}.headers.${header}`;
      secrets[path] = draft.action === "replace"
        ? { action: "replace", value: draft.value }
        : { action: draft.action };
    }
    mcp[server.name] = {
      enabled: server.enabled,
      transport: server.transport,
      command: server.transport === "stdio" ? server.command : null,
      args: server.transport === "stdio" ? server.args : [],
      cwd: server.transport === "stdio" && server.cwd.trim() ? server.cwd.trim() : null,
      url: server.transport === "streamable-http" ? server.url.trim() : null,
      headers,
      connect_timeout: Number(server.connect_timeout),
      call_timeout: Number(server.call_timeout),
      tool_keywords: Object.fromEntries(server.tool_keywords.map((tool) => [tool.name, tool.keywords])),
    };
  }
  return {
    fields: {
    runtime: {
      max_tool_result_chars: Number(form.runtime.max_tool_result_chars),
      max_iterations: Number(form.runtime.max_iterations),
      enable_skill_always_load: form.runtime.enable_skill_always_load,
      compact_ratio: Number(form.runtime.compact_ratio),
      permission_level: form.runtime.permission_level,
      exec_shell: form.runtime.exec_shell,
    },
    memory: {
      batch_size: Number(form.memory.batch_size),
      schedule: form.memory.schedule.trim(),
    },
      web: {
        default_chat_workspace: form.web.default_chat_workspace.trim(),
      },
      models: { providers, routes },
      mcp,
    },
    secrets,
  };
}

interface SecretInputProps {
  id: string;
  label: string;
  secret: SecretDraft;
  disabled: boolean;
  onChange: (update: Partial<SecretDraft>) => void;
  error?: string;
}

function SecretInput({ id, label, secret, disabled, onChange, error }: SecretInputProps) {
  const { t } = useTranslation();
  return (
    <div className={styles.settingsField} id={id} tabIndex={-1}>
      <span className={styles.fieldLabel} id={`${id}-label`}>{label}</span>
      <div className={styles.settingsSecretRow}>
        <select
          className={styles.selectInput}
          id={`${id}-action`}
          aria-labelledby={`${id}-label`}
          aria-invalid={error !== undefined}
          aria-describedby={error !== undefined ? `${id}-error` : undefined}
          value={secret.action}
          disabled={disabled}
          onChange={(event) => onChange({ action: event.currentTarget.value as SecretAction, value: "" })}
        >
          <option value="keep">{t("settings.secretKeep")}</option>
          <option value="replace">{t("settings.secretReplace")}</option>
          <option value="clear">{t("settings.secretClear")}</option>
        </select>
        {secret.action === "replace" ? (
          <input
            className={styles.textInput}
            id={`${id}-value`}
            type="password"
            aria-invalid={error !== undefined}
            aria-describedby={error !== undefined ? `${id}-error` : undefined}
            autoComplete="new-password"
            aria-label={t("settings.secretValue")}
            value={secret.value}
            disabled={disabled}
            onChange={(event) => onChange({ value: event.currentTarget.value })}
          />
        ) : null}
      </div>
      {error !== undefined ? <span className={styles.fieldError} id={`${id}-error`}>{error}</span> : null}
      <small className={styles.settingsSecretState}>
        {secret.configured ? t("settings.secretConfigured") : t("settings.secretNotConfigured")}
      </small>
    </div>
  );
}

type SettingsFieldError = Record<string, string>;

interface SettingsSaveRequest {
  requestId: string;
  revision: string;
  section: PersistedSettingsSection;
  snapshot: SettingsForm;
  base: SettingsForm;
  repairing: boolean;
  fields: ConfigPatchFields;
  secrets: ConfigSecrets;
}

const PERSISTED_SETTINGS_SECTIONS: PersistedSettingsSection[] = ["models", "runtime", "memory", "mcp", "web"];

function sameSettingsValue(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (typeof left !== "object" || left === null || typeof right !== "object" || right === null) return false;
  return JSON.stringify(left) === JSON.stringify(right);
}

function isSettingsRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function settingsSectionForPath(path: string): PersistedSettingsSection | null {
  const section = path.split(".", 1)[0];
  return PERSISTED_SETTINGS_SECTIONS.includes(section as PersistedSettingsSection)
    ? section as PersistedSettingsSection
    : null;
}

function mergeSettingsDraft(
  base: unknown,
  local: unknown,
  remote: unknown,
  conflicts: string[],
  options: {
    sent?: unknown;
    savingSection?: PersistedSettingsSection;
    conflictOnSecret?: boolean;
  } = {},
  path = "",
): unknown {
  if (sameSettingsValue(local, base)) return remote;
  if (
    options.conflictOnSecret === true
    && /(?:^|\.)(?:api_key|headers)(?:\.|$)/.test(path)
  ) {
    conflicts.push(path);
    return local;
  }
  if (sameSettingsValue(remote, base) || sameSettingsValue(local, remote)) return local;

  if (
    (isSettingsRecord(base) || base === undefined)
    && isSettingsRecord(local)
    && isSettingsRecord(remote)
  ) {
    const merged: Record<string, unknown> = {};
    const keys = new Set([
      ...Object.keys(isSettingsRecord(base) ? base : {}),
      ...Object.keys(local),
      ...Object.keys(remote),
    ]);
    for (const key of keys) {
      const value = mergeSettingsDraft(
        isSettingsRecord(base) ? base[key] : undefined,
        local[key],
        remote[key],
        conflicts,
        options,
        path ? `${path}.${key}` : key,
      );
      if (value !== undefined) merged[key] = value;
    }
    return merged;
  }

  if (
    options.sent !== undefined
    && options.savingSection !== undefined
    && settingsSectionForPath(path) === options.savingSection
  ) {
    return local;
  }

  conflicts.push(path || "configuration");
  return local;
}

function settingsSectionChanged(left: SettingsForm, right: SettingsForm, section: PersistedSettingsSection): boolean {
  return !sameSettingsValue(left[section], right[section]);
}

function configPatchForSection(
  form: SettingsForm,
  section: PersistedSettingsSection,
): { fields: ConfigPatchFields; secrets: ConfigSecrets } {
  const config = configFromForm(form);
  if (section === "runtime") return { fields: { runtime: config.fields.runtime }, secrets: {} };
  if (section === "memory") return { fields: { memory: config.fields.memory }, secrets: {} };
  if (section === "web") return { fields: { web: config.fields.web! }, secrets: {} };
  if (section === "models") {
    return {
      fields: { models: config.fields.models! },
      secrets: Object.fromEntries(Object.entries(config.secrets).filter(([path]) => path.startsWith("models."))),
    };
  }
  return {
    fields: { mcp: config.fields.mcp! },
    secrets: Object.fromEntries(Object.entries(config.secrets).filter(([path]) => path.startsWith("mcp."))),
  };
}

interface SettingsViewProps {
  authState: AuthState;
  connectionState: ConnectionState;
  language: "en" | "zh-CN";
  onBack: () => void;
  onLanguageChange: (language: "en" | "zh-CN") => void;
  onThemeChange: (theme: Theme) => void;
  serviceStatus: ServiceStatus | null;
  theme: Theme;
}

function SettingsView({
  authState,
  connectionState,
  language,
  onBack,
  onLanguageChange,
  onThemeChange,
  serviceStatus,
  theme,
}: SettingsViewProps) {
  const { t } = useTranslation();
  const location = useLocation();
  const [activeSection, setActiveSection] = useState<SettingsSection>("general");
  useEffect(() => {
    if (location.hash === "#models") setActiveSection("models");
  }, [location.hash]);
  const [response, setResponse] = useState<ConfigResponse | null>(null);
  const [draft, setDraft] = useState<SettingsForm | null>(null);
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<SettingsFieldError>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [saveFailed, setSaveFailed] = useState(false);
  const [conflictedPaths, setConflictedPaths] = useState<string[]>([]);
  const dirtyRef = useRef(false);
  const draftRef = useRef<SettingsForm | null>(null);
  const savedFormRef = useRef<SettingsForm | null>(null);
  const draftRevisionRef = useRef<string | null>(null);
  const requestSequence = useRef(0);
  const mutationSequence = useRef(0);
  const pendingSectionsRef = useRef<PersistedSettingsSection[]>([]);
  const conflictPathsRef = useRef<string[]>([]);
  const [saveQueueVersion, setSaveQueueVersion] = useState(0);
  const mutationInFlight = useRef(false);
  const unresolvedSaveRef = useRef<SettingsSaveRequest | null>(null);
  const errorSummaryRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (submitError === null) return;
    const timer = window.setTimeout(() => errorSummaryRef.current?.focus(), 0);
    return () => window.clearTimeout(timer);
  }, [submitError]);

  useEffect(() => {
    dirtyRef.current = dirty;
  }, [dirty]);

  useEffect(() => {
    draftRef.current = draft;
  }, [draft]);

  const applyResponse = useCallback((next: ConfigResponse) => {
    setResponse(next);
    if (!dirtyRef.current || draftRef.current === null) {
      const nextDraft = formFromConfig(next.fields, draftRef.current);
      setDraft(nextDraft);
      draftRef.current = nextDraft;
      savedFormRef.current = nextDraft;
      draftRevisionRef.current = next.revision;
      conflictPathsRef.current = [];
      setConflictedPaths([]);
      setDirty(false);
      dirtyRef.current = false;
    }
    setLoading(false);
  }, []);

  const loadSettings = useCallback(async () => {
    if (mutationInFlight.current || connectionState !== "online") return;
    const sequence = requestSequence.current + 1;
    requestSequence.current = sequence;
    try {
      const next = await getConfig();
      if (requestSequence.current !== sequence) return;
      applyResponse(next);
      setLoadError(null);
    } catch (error) {
      if (requestSequence.current !== sequence) return;
      if (!dirtyRef.current) setLoading(false);
      setLoadError(error instanceof ApiError ? error.message : t("settings.unavailable"));
    }
  }, [applyResponse, connectionState, t]);

  useEffect(() => {
    if (authState !== "ready") {
      setLoading(false);
      return;
    }
    let active = true;
    void loadSettings();
    const timer = window.setInterval(() => {
      if (active) void loadSettings();
    }, 2500);
    return () => {
      active = false;
      requestSequence.current += 1;
      window.clearInterval(timer);
    };
  }, [authState, loadSettings]);

  useEffect(() => () => {
    mutationSequence.current += 1;
  }, []);

  const validateField = useCallback((path: string, value: string | boolean): string | null => {
    if (path === "web.default_chat_workspace" && (typeof value !== "string" || !value.trim())) {
      return t("settings.required");
    }
    const integerFields: Record<string, [number, number]> = {
      "runtime.max_tool_result_chars": [1000, 1_000_000],
      "runtime.max_iterations": [50, Number.MAX_SAFE_INTEGER],
      "memory.batch_size": [1, 1000],
    };
    const integerRange = integerFields[path];
    if (integerRange !== undefined) {
      const numeric = typeof value === "string" && /^\d+$/.test(value) ? Number(value) : NaN;
      if (!Number.isSafeInteger(numeric) || numeric < integerRange[0] || numeric > integerRange[1]) {
        return t("settings.invalidInteger");
      }
    }
    if (path.endsWith(".context_window")) {
      const numeric = typeof value === "string" && /^\d+$/.test(value) ? Number(value) : NaN;
      if (!Number.isSafeInteger(numeric) || numeric < 1024 || numeric > 10_000_000) {
        return t("settings.invalidInteger");
      }
    }
    if (path.endsWith(".max_output")) {
      const numeric = typeof value === "string" && /^\d+$/.test(value) ? Number(value) : NaN;
      if (!Number.isSafeInteger(numeric) || numeric < 1 || numeric > 9_999_999) {
        return t("settings.invalidInteger");
      }
    }
    if (path.endsWith(".timeout") || path.endsWith(".connect_timeout") || path.endsWith(".call_timeout")) {
      const numeric = typeof value === "string" && /^\d+$/.test(value) ? Number(value) : NaN;
      if (!Number.isSafeInteger(numeric) || numeric < 1 || numeric > 600) {
        return t("settings.invalidInteger");
      }
    }
    if (path.endsWith(".temperature")) {
      const numeric = typeof value === "string" ? Number(value) : NaN;
      if (!Number.isFinite(numeric) || numeric < 0 || numeric > 2) {
        return t("settings.invalidValue");
      }
    }
    if (path.endsWith(".api_key.value") || path.endsWith(".headers.value")) {
      if (typeof value !== "string" || value.length === 0) return t("settings.secretRequired");
    }
    if (path.endsWith(".url") && typeof value === "string" && value.trim() !== "") {
      try {
        const url = new URL(value);
        if (url.protocol !== "http:" && url.protocol !== "https:") return t("settings.invalidUrl");
      } catch {
        return t("settings.invalidUrl");
      }
    }
    if (path === "runtime.compact_ratio") {
      const numeric = typeof value === "string" ? Number(value) : NaN;
      if (!Number.isFinite(numeric) || numeric < 0.5 || numeric > 0.95) {
        return t("settings.invalidRatio");
      }
    }
    if (path === "memory.schedule") {
      if (typeof value !== "string" || value.trim().split(/\s+/).length !== 5) {
        return t("settings.invalidSchedule");
      }
    }
    return null;
  }, [t]);

  const updateField = useCallback((path: string, value: string | boolean) => {
    dirtyRef.current = true;
    setDraft((current) => {
      if (current === null) return current;
      const [section, field] = path.split(".");
      if (section !== "runtime" && section !== "memory" && section !== "web") return current;
      if (section === "web" && typeof value !== "string") return current;
      return {
        ...current,
        [section]: { ...current[section], [field]: value },
      } as SettingsForm;
    });
    setDirty(true);
    setSaveFailed(false);
    setNotice(null);
    if (conflictPathsRef.current.length === 0) setSubmitError(null);
  }, []);

  const updateDraft = useCallback((update: (current: SettingsForm) => SettingsForm) => {
    dirtyRef.current = true;
    setDraft((current) => current === null ? current : update(current));
    setDirty(true);
    setSaveFailed(false);
    setNotice(null);
    if (conflictPathsRef.current.length === 0) setSubmitError(null);
  }, []);

  const enqueueSave = useCallback((section: PersistedSettingsSection) => {
    if (!pendingSectionsRef.current.includes(section)) {
      pendingSectionsRef.current.push(section);
    }
    setSaveQueueVersion((version) => version + 1);
  }, []);

  const updateProvider = useCallback((id: string, update: Partial<ProviderForm>) => {
    updateDraft((current) => ({
      ...current,
      models: {
        ...current.models,
        providers: {
          ...current.models.providers,
          [id]: { ...current.models.providers[id], ...update },
        },
      },
    }));
  }, [updateDraft]);

  const updateRoute = useCallback((id: string, update: Partial<RouteForm>) => {
    updateDraft((current) => ({
      ...current,
      models: {
        ...current.models,
        routes: { ...current.models.routes, [id]: { ...current.models.routes[id], ...update } },
      },
    }));
  }, [updateDraft]);

  const updateMcp = useCallback((id: string, update: Partial<McpForm>) => {
    updateDraft((current) => ({
      ...current,
      mcp: { ...current.mcp, [id]: { ...current.mcp[id], ...update } },
    }));
  }, [updateDraft]);

  const addProvider = useCallback(() => {
    const current = draftRef.current;
    if (current === null) return;
    let id = "new-provider";
    let index = 2;
    while (current.models.providers[id] !== undefined) id = `new-provider-${index++}`;
    updateDraft((form) => ({
      ...form,
      models: {
        ...form.models,
        providers: {
          ...form.models.providers,
          [id]: { id, protocol: "openai-compatible", base_url: "", models: [], api_key: secretDraft(false) },
        },
      },
    }));
  }, [updateDraft]);

  const removeProvider = useCallback((id: string) => {
    updateDraft((current) => {
      const providers = { ...current.models.providers };
      delete providers[id];
      return { ...current, models: { ...current.models, providers } };
    });
  }, [updateDraft]);

  const addRoute = useCallback(() => {
    const current = draftRef.current;
    if (current === null) return;
    const name = (["default", "chat", "memory", "schedule"] as const).find(
      (candidate) => current.models.routes[candidate] === undefined,
    );
    if (name === undefined) return;
    updateDraft((form) => ({
      ...form,
      models: {
        ...form.models,
        routes: {
          ...form.models.routes,
          [name]: {
            name,
            provider_id: Object.values(form.models.providers)[0]?.id ?? "",
            model: "",
            context_window: "8192",
            max_output: "1024",
            temperature: "0",
            reasoning_effort: "medium",
            timeout: "60",
          },
        },
      },
    }));
  }, [updateDraft]);

  const removeRoute = useCallback((id: string) => {
    updateDraft((current) => {
      const routes = { ...current.models.routes };
      delete routes[id];
      return { ...current, models: { ...current.models, routes } };
    });
  }, [updateDraft]);

  const addMcp = useCallback(() => {
    const current = draftRef.current;
    if (current === null) return;
    let name = "new-mcp";
    let index = 2;
    while (current.mcp[name] !== undefined) name = `new-mcp-${index++}`;
    updateDraft((form) => ({
      ...form,
      mcp: {
        ...form.mcp,
        [name]: {
          name,
          enabled: false,
          transport: "stdio",
          command: "python",
          args: [],
          cwd: "",
          url: "",
          headers: {},
          connect_timeout: "30",
          call_timeout: "60",
          tool_keywords: [],
        },
      },
    }));
  }, [updateDraft]);

  const removeMcp = useCallback((id: string) => {
    updateDraft((current) => {
      const mcp = { ...current.mcp };
      delete mcp[id];
      return { ...current, mcp };
    });
  }, [updateDraft]);

  const addHeader = useCallback((serverId: string) => {
    const current = draftRef.current?.mcp[serverId];
    if (current === undefined) return;
    let header = "Authorization";
    let index = 2;
    while (current.headers[header] !== undefined) header = `X-Header-${index++}`;
    updateMcp(serverId, { headers: { ...current.headers, [header]: { ...secretDraft(false), name: header } } });
  }, [updateMcp]);

  const removeHeader = useCallback((serverId: string, header: string) => {
    const current = draftRef.current?.mcp[serverId];
    if (current === undefined) return;
    const headers = { ...current.headers };
    delete headers[header];
    updateMcp(serverId, { headers });
  }, [updateMcp]);

  const validateAll = useCallback((current: SettingsForm): SettingsFieldError => {
    const values: Record<string, string | boolean> = {
      "runtime.max_tool_result_chars": current.runtime.max_tool_result_chars,
      "runtime.max_iterations": current.runtime.max_iterations,
      "runtime.enable_skill_always_load": current.runtime.enable_skill_always_load,
      "runtime.compact_ratio": current.runtime.compact_ratio,
      "runtime.permission_level": current.runtime.permission_level,
      "runtime.exec_shell": current.runtime.exec_shell,
      "memory.batch_size": current.memory.batch_size,
      "memory.schedule": current.memory.schedule,
      "web.default_chat_workspace": current.web.default_chat_workspace,
    };
    const errors: SettingsFieldError = {};
    for (const [path, value] of Object.entries(values)) {
      const error = validateField(path, value);
      if (error !== null) errors[path] = error;
    }
    for (const [providerRow, provider] of Object.entries(current.models.providers)) {
      const providerPath = `models.providers.${provider.id}`;
      if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(provider.id)) {
        errors[`models.providers.${providerRow}.id`] = t("settings.invalidProviderId");
      }
      if (Object.values(current.models.providers).filter((entry) => entry.id === provider.id).length !== 1) errors[`models.providers.${providerRow}.id`] = t("settings.invalidProviderId");
      if (provider.models.some((model) => !model || model !== model.trim()) || new Set(provider.models).size !== provider.models.length) errors[`${providerPath}.models`] = t("settings.invalidValue");
      const baseError = validateField(`${providerPath}.url`, provider.base_url);
      if (baseError || !provider.base_url) errors[`${providerPath}.base_url`] = baseError ?? t("settings.required");
      if (provider.api_key.action === "replace" && provider.api_key.value.length === 0) {
        errors[`${providerPath}.api_key.value`] = t("settings.secretRequired");
      }
    }
    for (const route of Object.values(current.models.routes)) {
      const routePath = `models.routes.${route.name}`;
      for (const field of ["context_window", "max_output", "temperature", "timeout"] as const) {
        const error = validateField(`${routePath}.${field}`, route[field]);
        if (error !== null) errors[`${routePath}.${field}`] = error;
      }
      if (!route.provider_id.trim()) errors[`${routePath}.provider_id`] = t("settings.required");
      if (!route.model.trim()) errors[`${routePath}.model`] = t("settings.required");
    }
    for (const [serverRow, server] of Object.entries(current.mcp)) {
      const serverPath = `mcp.${server.name}`;
      if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(server.name)) {
        errors[`mcp.${serverRow}.name`] = t("settings.invalidServerName");
      }
      for (const field of ["connect_timeout", "call_timeout"] as const) {
        const error = validateField(`${serverPath}.${field}`, server[field]);
        if (error !== null) errors[`${serverPath}.${field}`] = error;
      }
      if (server.transport === "stdio" && !server.command.trim()) {
        errors[`${serverPath}.command`] = t("settings.required");
      }
      if (server.transport === "streamable-http") {
        const error = validateField(`${serverPath}.url`, server.url);
        if (error !== null) errors[`${serverPath}.url`] = error;
        else if (!server.url.trim()) errors[`${serverPath}.url`] = t("settings.required");
      }
      if (Object.values(current.mcp).filter((entry) => entry.name === server.name).length !== 1) errors[`mcp.${serverRow}.name`] = t("settings.invalidServerName");
      const toolNames = server.tool_keywords.map((tool) => tool.name);
      if (new Set(toolNames).size !== toolNames.length || toolNames.some((name) => !name)) errors[`${serverPath}.tool_keywords`] = t("settings.invalidValue");
      if (server.transport !== "streamable-http") continue;
      const headerNames = Object.values(server.headers).map((secret) => secret.name);
      if (new Set(headerNames).size !== headerNames.length) errors[`${serverPath}.headers`] = t("settings.invalidIdentifier");
      for (const [header, secret] of Object.entries(server.headers)) {
        if (!secret.name || secret.name !== secret.name.trim()) errors[`${serverPath}.headers`] = t("settings.required");
        if (secret.action === "replace" && secret.value.length === 0) {
          errors[`${serverPath}.headers.${header}.value`] = t("settings.secretRequired");
        }
      }
    }
    return errors;
  }, [t, validateField]);

  const blurField = useCallback((path: string, value: string | boolean) => {
    const error = validateField(path, value);
    setFieldErrors((current) => {
      const next = { ...current };
      if (error === null) delete next[path];
      else next[path] = error;
      return next;
    });
  }, [validateField]);

  useEffect(() => {
    if (
      authState !== "ready"
      || connectionState !== "online"
      || mutationInFlight.current
      || (unresolvedSaveRef.current !== null && saveFailed)
      || pendingSectionsRef.current.length === 0
    ) return;

    let section = pendingSectionsRef.current.shift()!;
    const unresolved = unresolvedSaveRef.current;
    const retryIncludesCompletedEdit = unresolved !== null
      && unresolved.section === section
      && !sameSettingsValue(draftRef.current?.[section], unresolved.snapshot[section]);
    if (unresolved !== null && unresolved.section !== section) {
      pendingSectionsRef.current.unshift(section);
      section = unresolved.section;
    }
    setSaveQueueVersion((version) => version + 1);
    const snapshot = unresolved?.snapshot ?? draftRef.current;
    const base = unresolved?.base ?? savedFormRef.current;
    const revision = unresolved?.revision ?? draftRevisionRef.current;
    if (snapshot === null || base === null || revision === null) return;
    if (conflictPathsRef.current.some((path) => settingsSectionForPath(path) === section)) return;
    const repairing = response?.configuration.repair_required === true;
    if (unresolved === null && !repairing && !settingsSectionChanged(snapshot, base, section)) return;

    const allErrors = validateAll(snapshot);
    const validationErrors = Object.fromEntries(Object.entries(allErrors).filter(([path]) => (
      repairing || settingsSectionForPath(path) === section
    )));
    setFieldErrors((current) => {
      const next = { ...current };
      for (const path of Object.keys(next)) {
        if (repairing || settingsSectionForPath(path) === section) delete next[path];
      }
      return { ...next, ...validationErrors };
    });
    if (unresolved === null && Object.keys(validationErrors).length > 0) {
      setSaveFailed(true);
      setSubmitError(t("settings.validationSummary"));
      return;
    }

    const sectionPatch = configPatchForSection(snapshot, section);
    const fullConfig = repairing ? configFromForm(snapshot) : null;
    const pendingRequest = unresolved ?? {
      requestId: createRequestId(), revision, section, snapshot, base, repairing,
      fields: fullConfig?.fields ?? sectionPatch.fields,
      secrets: fullConfig?.secrets ?? sectionPatch.secrets,
    };
    unresolvedSaveRef.current = pendingRequest;
    requestSequence.current += 1;
    const sequence = ++mutationSequence.current;
    mutationInFlight.current = true;
    setSaving(true);
    setSaveFailed(false);
    setNotice(null);
    if (conflictPathsRef.current.length === 0) setSubmitError(null);

    void (async () => {
      try {
        const next = pendingRequest.repairing
          ? await repairConfig(revision, pendingRequest.fields, pendingRequest.secrets, pendingRequest.requestId)
          : await patchConfig(revision, pendingRequest.fields, pendingRequest.secrets, pendingRequest.requestId);
        if (mutationSequence.current !== sequence) return;

        unresolvedSaveRef.current = null;
        const latestLocal = draftRef.current ?? snapshot;
        if (retryIncludesCompletedEdit) {
          enqueueSave(section);
        }
        const remote = formFromConfig(next.fields, sameSettingsValue(latestLocal, snapshot) ? null : snapshot);
        const mergeConflicts: string[] = [];
        const mergeBase = pendingRequest.repairing ? snapshot : { ...base, [section]: snapshot[section] };
        const merged = mergeSettingsDraft(mergeBase, latestLocal, remote, mergeConflicts, {
          sent: snapshot,
          savingSection: section,
        }) as SettingsForm;
        const nextConflicts = [...new Set([...conflictPathsRef.current, ...mergeConflicts])];
        setResponse(next);
        savedFormRef.current = remote;
        draftRevisionRef.current = next.revision;
        draftRef.current = merged;
        setDraft(merged);
        const stillDirty = !sameSettingsValue(merged, remote);
        dirtyRef.current = stillDirty;
        setDirty(stillDirty);
        conflictPathsRef.current = nextConflicts;
        setConflictedPaths(nextConflicts);
        setFieldErrors((current) => {
          const nextErrors = { ...current };
          for (const path of Object.keys(nextErrors)) {
            if (settingsSectionForPath(path) === section) delete nextErrors[path];
          }
          for (const path of mergeConflicts) nextErrors[path] = t("settings.conflictField");
          return nextErrors;
        });
        if (nextConflicts.length > 0) {
          setSaveFailed(true);
          setSubmitError(t("settings.conflict"));
        } else if (Object.keys(validateAll(merged)).some((path) => settingsSectionForPath(path) !== section)) {
          setSaveFailed(true);
          setSubmitError(t("settings.validationSummary"));
        } else {
          setSaveFailed(false);
          setSubmitError(null);
        }
        setNotice(next.application.restart_required ? t("settings.restartRequired") : t("settings.saved"));
      } catch (error) {
        if (mutationSequence.current !== sequence) return;
        if (error instanceof ApiError && error.body !== null) unresolvedSaveRef.current = null;
        if (error instanceof ApiError && error.body?.code === "config_revision_conflict") {
          try {
            const latest = await getConfig();
            if (mutationSequence.current !== sequence) return;
            const remote = formFromConfig(latest.fields, base);
            const latestLocal = draftRef.current ?? snapshot;
            const mergeConflicts: string[] = [];
            const merged = mergeSettingsDraft(
              savedFormRef.current ?? base,
              latestLocal,
              remote,
              mergeConflicts,
              { conflictOnSecret: true },
            ) as SettingsForm;
            const nextConflicts = [...new Set([...conflictPathsRef.current, ...mergeConflicts])];
            const retryThisSection = sameSettingsValue(latestLocal[section], snapshot[section])
              || pendingSectionsRef.current.includes(section);
            pendingSectionsRef.current = pendingSectionsRef.current.filter((queued) => (
              !nextConflicts.some((path) => settingsSectionForPath(path) === queued)
            ));
            if (
              nextConflicts.length === 0
              && retryThisSection
              && settingsSectionChanged(merged, remote, section)
            ) {
              pendingSectionsRef.current.unshift(section);
            }
            setResponse(latest);
            savedFormRef.current = remote;
            draftRevisionRef.current = latest.revision;
            draftRef.current = merged;
            setDraft(merged);
            const stillDirty = !sameSettingsValue(merged, remote);
            dirtyRef.current = stillDirty;
            setDirty(stillDirty);
            conflictPathsRef.current = nextConflicts;
            setConflictedPaths(nextConflicts);
            setFieldErrors((current) => {
              const nextErrors = { ...current };
              for (const path of Object.keys(nextErrors)) {
                if (settingsSectionForPath(path) === section) delete nextErrors[path];
              }
              for (const path of nextConflicts) nextErrors[path] = t("settings.conflictField");
              return nextErrors;
            });
            setSaveFailed(nextConflicts.length > 0);
            setSubmitError(nextConflicts.length > 0 ? t("settings.conflict") : null);
            setNotice(null);
          } catch (reloadError) {
            if (mutationSequence.current === sequence) {
              setSaveFailed(true);
              setSubmitError(reloadError instanceof ApiError ? reloadError.message : t("settings.unavailable"));
            }
          }
        } else if (error instanceof ApiError && error.body !== null) {
          const serverErrors = Object.fromEntries(Object.keys(error.body.field_errors).map((path) => [
            path,
            t(path === "memory.schedule" ? "settings.invalidSchedule"
              : path === "runtime.compact_ratio" ? "settings.invalidRatio" : "settings.invalidValue"),
          ]));
          setFieldErrors((current) => ({ ...current, ...serverErrors }));
          setSaveFailed(true);
          setSubmitError(error.body.code === "config_invalid" ? t("settings.validationSummary")
            : error.body.code === "persistence_error" ? t("settings.persistenceFailed") : error.body.message);
        } else {
          setSaveFailed(true);
          setSubmitError(t("settings.unavailable"));
        }
      } finally {
        if (mutationSequence.current === sequence) {
          mutationInFlight.current = false;
          setSaving(false);
          setSaveQueueVersion((version) => version + 1);
        }
      }
    })();
  }, [authState, connectionState, response?.configuration.repair_required, saveFailed, saveQueueVersion, t, validateAll, enqueueSave]);

  const reloadSaved = async () => {
    if (saving || connectionState !== "online") return;
    const sequence = ++mutationSequence.current;
    setSaving(true);
    mutationInFlight.current = true;
    try {
      const next = await getConfig();
      if (mutationSequence.current !== sequence) return;
      unresolvedSaveRef.current = null;
      pendingSectionsRef.current = [];
      dirtyRef.current = false;
      applyResponse(next);
      setFieldErrors({});
      setSubmitError(null);
      setSaveFailed(false);
      setConflictedPaths([]);
      conflictPathsRef.current = [];
      setNotice(null);
    } catch (error) {
      if (mutationSequence.current === sequence) {
        setSaveFailed(true);
        setSubmitError(error instanceof ApiError ? error.message : t("settings.unavailable"));
      }
    } finally {
      if (mutationSequence.current === sequence) {
        mutationInFlight.current = false;
        setSaving(false);
        setSaveQueueVersion((version) => version + 1);
      }
    }
  };

  const keepLocalChanges = () => {
    const sections = [...new Set(conflictPathsRef.current.map(settingsSectionForPath).filter(
      (section): section is PersistedSettingsSection => section !== null,
    ))];
    conflictPathsRef.current = [];
    setConflictedPaths([]);
    setFieldErrors((current) => Object.fromEntries(Object.entries(current).filter(([path]) => (
      !sections.some((section) => settingsSectionForPath(path) === section)
    ))));
    setSubmitError(null);
    setSaveFailed(false);
    for (const section of sections) enqueueSave(section);
  };

  const retryPendingChanges = () => {
    const current = draftRef.current;
    const saved = savedFormRef.current;
    if (current === null || saved === null) return;
    setSaveFailed(false);
    setSubmitError(null);
    if (unresolvedSaveRef.current !== null) enqueueSave(unresolvedSaveRef.current.section);
    for (const section of PERSISTED_SETTINGS_SECTIONS) {
      if (
        settingsSectionChanged(current, saved, section)
        && !conflictPathsRef.current.some((path) => settingsSectionForPath(path) === section)
      ) enqueueSave(section);
    }
  };

  const errorEntries = Object.entries(fieldErrors);
  const controlDisabled = draft === null || loading || connectionState !== "online";
  const captureSettingsBlur = (event?: React.FocusEvent<HTMLFormElement>) => {
    if (event !== undefined && !(event.target instanceof HTMLInputElement
      || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement)) return;
    if (activeSection === "general") {
      if (event?.target instanceof HTMLInputElement
        && event.target.id === fieldId("web.default_chat_workspace")) enqueueSave("web");
      return;
    }
    enqueueSave(activeSection);
  };
  const captureSettingsChange = (event: React.FormEvent<HTMLFormElement>) => {
    const target = event.target;
    if (
      target instanceof HTMLSelectElement
      || (target instanceof HTMLInputElement && (target.type === "checkbox" || target.type === "radio"))
    ) captureSettingsBlur();
  };
  const captureSettingsClick = (event: React.MouseEvent<HTMLFormElement>) => {
    if (event.target instanceof Element && event.target.closest("button") !== null
      && event.target.closest("nav") === null) captureSettingsBlur();
  };
  const canAddRoute = draft !== null && (["default", "chat", "memory", "schedule"] as const).some(
    (name) => draft.models.routes[name] === undefined,
  );
  const labelFor = (path: string): string => {
    const labels: Record<string, string> = {
      "runtime.max_tool_result_chars": t("settings.maxToolResultChars"),
      "runtime.max_iterations": t("settings.maxIterations"),
      "runtime.enable_skill_always_load": t("settings.enableSkillAlwaysLoad"),
      "runtime.compact_ratio": t("settings.compactRatio"),
      "runtime.permission_level": t("settings.permissionLevel"),
      "runtime.exec_shell": t("settings.execShell"),
      "memory.batch_size": t("settings.batchSize"),
      "memory.schedule": t("settings.schedule"),
    };
    return labels[path] ?? path;
  };
  const fieldId = (path: string) => `settings-${path.replaceAll(".", "-")}`;
  const headerInputId = (server: string, header: string) => `settings-mcp-${server}-headers-${encodeURIComponent(header).replaceAll(".", "%2E")}`;
  const fieldError = (path: string) => fieldErrors[path];
  const groupError = (prefix: string) => Object.entries(fieldErrors).find(([path]) => path === prefix || path.startsWith(`${prefix}.`))?.[1];
  const focusError = (path: string, switchSection = true) => {
    const fieldSection = settingsSectionForPath(path);
    const section = fieldSection === "web" ? "general" : fieldSection;
    if (switchSection && section !== null && section !== activeSection) {
      setActiveSection(section);
      window.setTimeout(() => focusError(path, false), 0);
      return;
    }
    let targetPath = path;
    for (const server of Object.values(draft?.mcp ?? {})) {
      const prefix = `mcp.${server.name}.headers.`;
      for (const [row, secret] of Object.entries(server.headers).sort((left, right) => right[1].name.length - left[1].name.length)) {
        const secretPath = `${prefix}${secret.name}`;
        const rowPath = `${prefix}${row}`;
        const matchedPath = path === secretPath || path.startsWith(`${secretPath}.`) ? secretPath
          : path === rowPath || path.startsWith(`${rowPath}.`) ? rowPath : null;
        if (matchedPath !== null) {
          const id = `${headerInputId(server.name, row)}${path.slice(matchedPath.length).replaceAll(".", "-")}`;
          const target = document.getElementById(id);
          (target?.matches("input, select") ? target : target?.querySelector<HTMLElement>("input, select") ?? target)?.focus();
          return;
        }
      }
      const keywordPrefix = `mcp.${server.name}.tool_keywords`;
      if (path.startsWith(`${keywordPrefix}.`)) {
        const index = server.tool_keywords.findIndex((tool) => path === `${keywordPrefix}.${tool.name}`);
        if (index >= 0) {
          document.getElementById(`${fieldId(keywordPrefix)}-${index}`)?.querySelector<HTMLTextAreaElement>("textarea")?.focus();
          return;
        }
      }
    }
    let target = document.getElementById(fieldId(targetPath));
    while (target === null && targetPath.includes(".")) {
      targetPath = targetPath.slice(0, targetPath.lastIndexOf("."));
      target = document.getElementById(fieldId(targetPath));
    }
    (target?.matches("input, select, textarea") ? target : target?.querySelector<HTMLElement>("input, select, textarea") ?? target)?.focus();
  };

  const settingsStatusState = saveFailed
    ? "error"
    : saving || pendingSectionsRef.current.length > 0
      ? "saving"
      : dirty
        ? "unsaved"
        : response?.application.status;
  const settingsStatusLabel = saveFailed
    ? t("settings.saveFailed")
    : saving || pendingSectionsRef.current.length > 0
      ? t("settings.saving")
      : dirty
        ? t("settings.unsaved")
        : response?.application.status === "restart-required"
          ? t("settings.restartRequired")
          : response?.application.status === "pending-repair"
            ? t("settings.pendingRepair")
            : t("settings.active");

  if (authState !== "ready") {
    return (
      <section className={styles.settingsPage} aria-labelledby="settings-title">
        <button className={styles.secondaryButton} type="button" onClick={onBack}>
          <ArrowLeft size={15} aria-hidden="true" />
          {t("settings.backToConversation")}
        </button>
        <div className={styles.pageHeading}>
          <div>
            <p className={styles.eyebrow}>{t("nav.settings")}</p>
            <h1 id="settings-title">{t("settings.title")}</h1>
          </div>
        </div>
        <div className={styles.errorBanner} role="status">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{t("settings.authenticationRequired")}</span>
        </div>
      </section>
    );
  }

  return (
    <section className={styles.settingsPage} aria-labelledby="settings-title">
      <div className={styles.settingsHeader}>
        <button className={styles.secondaryButton} type="button" onClick={onBack}>
          <ArrowLeft size={15} aria-hidden="true" />
          {t("settings.backToConversation")}
        </button>
        <div className={styles.pageHeading}>
          <div>
            <p className={styles.eyebrow}>{t("nav.settings")}</p>
            <h1 id="settings-title">{t("settings.title")}</h1>
            <p className={styles.pageDescription}>{t("settings.description")}</p>
          </div>
          {response !== null ? (
            <div className={styles.settingsStatus} data-state={settingsStatusState} role="status" aria-live="polite">
              {saveFailed
                ? <CircleAlert size={15} aria-hidden="true" />
                : saving || pendingSectionsRef.current.length > 0
                  ? <RefreshCw size={15} className={styles.spin} aria-hidden="true" />
                  : response.application.status === "active"
                    ? <CircleCheck size={15} aria-hidden="true" />
                    : <Clock3 size={15} aria-hidden="true" />}
              <span>{settingsStatusLabel}</span>
            </div>
          ) : null}
        </div>
      </div>

      {response !== null ? (
        <dl className={styles.settingsVersions} aria-label={t("settings.versions")}>
          <div><dt>{t("settings.savedVersion")}</dt><dd>{response.application.saved_revision}</dd></div>
          <div><dt>{t("settings.activeVersion")}</dt><dd>{response.application.active_revision ?? "-"}</dd></div>
        </dl>
      ) : null}

      {loadError !== null && response === null ? (
        <div className={styles.errorBanner} role="alert">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{loadError}</span>
          <button className={styles.secondaryButton} type="button" onClick={() => void loadSettings()}>
            <RefreshCw size={14} aria-hidden="true" />
            {t("controls.retry")}
          </button>
        </div>
      ) : null}

      {response?.configuration.repair_required ? (
        <div className={styles.errorBanner} role="alert">
          <TriangleAlert size={17} aria-hidden="true" />
          <span>
            {response.configuration.state === "malformed"
              ? t("settings.malformedBackup")
              : t(response.application.active_revision === null ? "settings.repairRequired" : "settings.savedRepairRequired")}
            {response.configuration.requires_secret_reentry ? ` ${t("settings.secretReentry")}` : ""}
          </span>
        </div>
      ) : null}

      {submitError !== null ? (
        <div
          className={styles.errorSummary}
          ref={errorSummaryRef}
          tabIndex={-1}
          role="alert"
          aria-labelledby="settings-error-title"
        >
          <strong id="settings-error-title">{t("settings.errorSummary")}</strong>
          {submitError !== null ? <p>{submitError}</p> : null}
          {errorEntries.length > 0 ? (
            <ul>
              {errorEntries.map(([path, message]) => (
                <li key={path}>
                  <a href={`#${fieldId(path)}`} onClick={(event) => {
                    event.preventDefault();
                    focusError(path);
                  }}>{labelFor(path)}: {message}</a>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      {conflictedPaths.length > 0 ? (
        <div className={styles.settingsErrorActions}>
          <button className={styles.secondaryButton} type="button" onClick={keepLocalChanges} disabled={controlDisabled}>
            {t("settings.keepChanges")}
          </button>
          <button className={styles.secondaryButton} type="button" onClick={() => void reloadSaved()} disabled={controlDisabled}>
            <RotateCcw size={14} aria-hidden="true" />
            {t("settings.reload")}
          </button>
        </div>
      ) : saveFailed ? (
        <div className={styles.settingsErrorActions}>
          <button className={styles.secondaryButton} type="button" onClick={retryPendingChanges} disabled={controlDisabled}>
            <RefreshCw size={14} aria-hidden="true" />
            {t("settings.retrySave")}
          </button>
        </div>
      ) : null}

      {draft !== null ? (
        <form
          className={styles.settingsForm}
          onSubmit={(event) => event.preventDefault()}
          onBlurCapture={captureSettingsBlur}
          onChangeCapture={captureSettingsChange}
          onClickCapture={captureSettingsClick}
          noValidate
        >
          <div className={styles.settingsLayout}>
            <nav className={styles.settingsNavigation} aria-label={t("settings.sections")}>
              <span className={styles.settingsNavigationLabel}>{t("settings.sections")}</span>
              {([
                ["general", "settings.generalAppearance"],
                ["models", "settings.models"],
                ["runtime", "settings.runtime"],
                ["memory", "settings.memory"],
                ["mcp", "settings.mcp"],
              ] as const).map(([section, label]) => (
                <button
                  className={styles.settingsNavigationItem}
                  type="button"
                  key={section}
                  aria-current={activeSection === section ? "page" : undefined}
                  onClick={() => setActiveSection(section)}
                >
                  {t(label)}
                </button>
              ))}
            </nav>
            <div className={styles.settingsDetail}>
              {activeSection === "general" ? (
                <div className={styles.settingsSection}>
                  <div className={styles.settingsSectionHeader}>
                    <div>
                      <p className={styles.eyebrow}>{t("settings.generalAppearance")}</p>
                      <h2>{t("settings.generalAppearance")}</h2>
                    </div>
                  </div>
                  <div className={styles.settingsFieldGrid}>
                    <label className={styles.settingsField} htmlFor="settings-theme">
                      <span className={styles.fieldLabel}>{t("controls.theme")}</span>
                      <select
                        className={styles.selectInput}
                        id="settings-theme"
                        value={theme}
                        onChange={(event) => onThemeChange(event.currentTarget.value as Theme)}
                      >
                        {(["system", "light", "dark"] as const).map((value) => (
                          <option key={value} value={value}>{t(`controls.${value}`)}</option>
                        ))}
                      </select>
                    </label>
                    <label className={styles.settingsField} htmlFor="settings-language">
                      <span className={styles.fieldLabel}>{t("controls.language")}</span>
                      <select
                        className={styles.selectInput}
                        id="settings-language"
                        value={language}
                        onChange={(event) => onLanguageChange(event.currentTarget.value as "en" | "zh-CN")}
                      >
                        <option value="en">English</option>
                        <option value="zh-CN">简体中文</option>
                      </select>
                    </label>
                    <label className={styles.settingsField} htmlFor={fieldId("web.default_chat_workspace")}>
                      <span className={styles.fieldLabel}>{t("settings.defaultChatWorkspace")}</span>
                      <input
                        className={styles.textInput}
                        id={fieldId("web.default_chat_workspace")}
                        value={draft?.web.default_chat_workspace ?? "~/.omni/chat"}
                        disabled={controlDisabled}
                        aria-invalid={fieldError("web.default_chat_workspace") !== undefined}
                        aria-describedby={fieldError("web.default_chat_workspace") !== undefined
                          ? `${fieldId("web.default_chat_workspace")}-error` : undefined}
                        onChange={(event) => updateField("web.default_chat_workspace", event.currentTarget.value)}
                        onBlur={() => blurField("web.default_chat_workspace", draft?.web.default_chat_workspace ?? "~/.omni/chat")}
                      />
                      <span className={styles.fieldError} id={`${fieldId("web.default_chat_workspace")}-error`}>
                        {fieldError("web.default_chat_workspace") ?? ""}
                      </span>
                    </label>
                  </div>
                </div>
              ) : null}

              {activeSection === "runtime" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.runtime")}</p>
                <h2>{t("settings.runtime")}</h2>
              </div>
              <Settings2 size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsFieldGrid}>
              <SettingsNumberField
                id={fieldId("runtime.max_tool_result_chars")}
                label={t("settings.maxToolResultChars")}
                value={draft.runtime.max_tool_result_chars}
                error={fieldError("runtime.max_tool_result_chars")}
                disabled={controlDisabled}
                onChange={(value) => updateField("runtime.max_tool_result_chars", value)}
                onBlur={() => blurField("runtime.max_tool_result_chars", draft.runtime.max_tool_result_chars)}
              />
              <SettingsNumberField
                id={fieldId("runtime.max_iterations")}
                label={t("settings.maxIterations")}
                value={draft.runtime.max_iterations}
                error={fieldError("runtime.max_iterations")}
                disabled={controlDisabled}
                onChange={(value) => updateField("runtime.max_iterations", value)}
                onBlur={() => blurField("runtime.max_iterations", draft.runtime.max_iterations)}
              />
              <SettingsNumberField
                id={fieldId("runtime.compact_ratio")}
                label={t("settings.compactRatio")}
                value={draft.runtime.compact_ratio}
                error={fieldError("runtime.compact_ratio")}
                disabled={controlDisabled}
                step="0.01"
                onChange={(value) => updateField("runtime.compact_ratio", value)}
                onBlur={() => blurField("runtime.compact_ratio", draft.runtime.compact_ratio)}
              />
              <label className={styles.settingsField} htmlFor={fieldId("runtime.permission_level")}>
                <span className={styles.fieldLabel}>{t("settings.permissionLevel")}</span>
                <select
                  className={styles.selectInput}
                  id={fieldId("runtime.permission_level")}
                  value={draft.runtime.permission_level}
                  disabled={controlDisabled}
                  onChange={(event) => updateField("runtime.permission_level", event.currentTarget.value)}
                >
                  {(["read-only", "workspace-write", "full-access"] as ToolPermissionLevel[]).map((level) => (
                    <option key={level} value={level}>{t(`settings.permissionLevels.${level}`)}</option>
                  ))}
                </select>
              </label>
              <label className={styles.settingsField} htmlFor={fieldId("runtime.exec_shell")}>
                <span className={styles.fieldLabel}>{t("settings.execShell")}</span>
                <select
                  className={styles.selectInput}
                  id={fieldId("runtime.exec_shell")}
                  value={draft.runtime.exec_shell}
                  disabled={controlDisabled}
                  onChange={(event) => updateField("runtime.exec_shell", event.currentTarget.value)}
                >
                  {(["auto", "powershell", "pwsh"] as const).map((shell) => (
                    <option key={shell} value={shell}>{t(`settings.shells.${shell}`)}</option>
                  ))}
                </select>
              </label>
              <label className={styles.settingsToggle} htmlFor={fieldId("runtime.enable_skill_always_load")}>
                <input
                  id={fieldId("runtime.enable_skill_always_load")}
                  type="checkbox"
                  checked={draft.runtime.enable_skill_always_load}
                  disabled={controlDisabled}
                  onChange={(event) => updateField("runtime.enable_skill_always_load", event.currentTarget.checked)}
                />
                <span>
                  <strong>{t("settings.enableSkillAlwaysLoad")}</strong>
                  <small>{draft.runtime.enable_skill_always_load ? t("settings.yes") : t("settings.no")}</small>
                </span>
              </label>
            </div>
            <dl className={styles.settingsServiceStatus} aria-label={t("status.title")}>
              <div>
                <dt>{t("status.title")}</dt>
                <dd>{serviceStatus === null ? t("status.checking") : serviceStateLabel(serviceStatus.state, t)}</dd>
              </div>
              <div><dt>{t("status.connection")}</dt><dd>{t(`status.${connectionState}`)}</dd></div>
              <div><dt>{t("status.workspaces")}</dt><dd>{serviceStatus?.active_workspace_count ?? "-"}</dd></div>
              <div><dt>{t("status.protocol")}</dt><dd>v{serviceStatus?.protocol_version ?? "-"}</dd></div>
            </dl>
          </div>

              : null}

              {activeSection === "memory" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.memory")}</p>
                <h2>{t("settings.memory")}</h2>
              </div>
              <Brain size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsFieldGrid}>
              <SettingsNumberField
                id={fieldId("memory.batch_size")}
                label={t("settings.batchSize")}
                value={draft.memory.batch_size}
                error={fieldError("memory.batch_size")}
                disabled={controlDisabled}
                onChange={(value) => updateField("memory.batch_size", value)}
                onBlur={() => blurField("memory.batch_size", draft.memory.batch_size)}
              />
              <label className={styles.settingsField} htmlFor={fieldId("memory.schedule")}>
                <span className={styles.fieldLabel} id={`${fieldId("memory.schedule")}-label`}>{t("settings.schedule")}</span>
                <input
                  className={styles.textInput}
                  id={fieldId("memory.schedule")}
                  aria-labelledby={`${fieldId("memory.schedule")}-label`}
                  value={draft.memory.schedule}
                  disabled={controlDisabled}
                  aria-invalid={fieldError("memory.schedule") !== undefined}
                  aria-describedby={fieldError("memory.schedule") !== undefined ? `${fieldId("memory.schedule")}-error` : undefined}
                  onChange={(event) => updateField("memory.schedule", event.currentTarget.value)}
                  onBlur={() => blurField("memory.schedule", draft.memory.schedule)}
                />
                <span className={styles.fieldError} id={`${fieldId("memory.schedule")}-error`}>{fieldError("memory.schedule") ?? ""}</span>
              </label>
            </div>
          </div>

              : null}

              {activeSection === "models" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.models")}</p>
                <h2>{t("settings.models")}</h2>
              </div>
              <Gauge size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsSubsection}>
              <div className={styles.settingsCollectionHeader}>
                <h3>{t("settings.providers")}</h3>
                <button className={styles.secondaryButton} type="button" onClick={addProvider} disabled={controlDisabled}>
                  <Plus size={14} aria-hidden="true" />{t("settings.addProvider")}
                </button>
              </div>
              <div className={styles.settingsCollection}>
                {Object.entries(draft.models.providers).map(([providerRow, provider]) => (
                  <div className={styles.settingsCollectionItem} key={providerRow} id={fieldId(`models.providers.${provider.id}`)} tabIndex={-1}>
                    <div className={styles.settingsCollectionItemHeader}>
                      <h4>{provider.id}</h4>
                      <button
                        className={styles.iconButton}
                        type="button"
                        aria-label={t("settings.removeProvider")}
                        title={t("settings.removeProvider")}
                        disabled={controlDisabled}
                        onClick={() => removeProvider(providerRow)}
                      ><Trash2 size={15} aria-hidden="true" /></button>
                    </div>
                    <div className={styles.settingsFieldGrid}>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.providers.${providerRow}.id`)}>
                        <span className={styles.fieldLabel}>{t("settings.providerId")}</span>
                        <input
                          className={styles.textInput}
                          id={fieldId(`models.providers.${providerRow}.id`)}
                          value={provider.id}
                          readOnly={response?.fields.models.providers[providerRow] !== undefined}
                          disabled={controlDisabled}
                          onChange={(event) => updateProvider(providerRow, { id: event.currentTarget.value })}
                          aria-invalid={fieldError(`models.providers.${providerRow}.id`) !== undefined}
                        />
                        <span className={styles.fieldError}>{fieldError(`models.providers.${providerRow}.id`) ?? ""}</span>
                      </label>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.providers.${provider.id}.protocol`)}>
                        <span className={styles.fieldLabel}>{t("settings.protocol")}</span>
                        <select className={styles.selectInput} id={fieldId(`models.providers.${provider.id}.protocol`)} value={provider.protocol} disabled={controlDisabled} onChange={(event) => updateProvider(providerRow, { protocol: event.currentTarget.value })}>
                          {!['openai-compatible', 'anthropic'].includes(provider.protocol) ? <option value={provider.protocol}>{provider.protocol}</option> : null}
                          <option value="openai-compatible">openai-compatible</option><option value="anthropic">anthropic</option>
                        </select>
                      </label>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.providers.${provider.id}.base_url`)}>
                        <span className={styles.fieldLabel}>{t("settings.baseUrl")}</span>
                        <input
                          className={styles.textInput}
                          id={fieldId(`models.providers.${provider.id}.base_url`)}
                          value={provider.base_url}
                          aria-invalid={fieldError(`models.providers.${provider.id}.base_url`) !== undefined}
                          aria-describedby={fieldError(`models.providers.${provider.id}.base_url`) !== undefined ? `${fieldId(`models.providers.${provider.id}.base_url`)}-error` : undefined}
                          disabled={controlDisabled}
                          onChange={(event) => updateProvider(providerRow, { base_url: event.currentTarget.value })}
                        />
                        <span className={styles.fieldError} id={`${fieldId(`models.providers.${provider.id}.base_url`)}-error`}>{fieldError(`models.providers.${provider.id}.base_url`) ?? ""}</span>
                      </label>
                      <SettingsListField id={fieldId(`models.providers.${provider.id}.models`)} label={t("settings.modelsList")} values={provider.models} error={groupError(`models.providers.${provider.id}.models`)} disabled={controlDisabled} onChange={(models) => updateProvider(providerRow, { models })} />
                      <SecretInput
                        id={fieldId(`models.providers.${provider.id}.api_key`)}
                        label={t("settings.apiKey")}
                        secret={provider.api_key}
                        error={groupError(`models.providers.${provider.id}.api_key`)}
                        disabled={controlDisabled}
                        onChange={(update) => updateProvider(providerRow, { api_key: { ...provider.api_key, ...update } as SecretDraft })}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className={styles.settingsSubsection}>
              <div className={styles.settingsCollectionHeader} id="settings-models-routes" tabIndex={-1}>
                <h3>{t("settings.routes")}</h3>
                <button className={styles.secondaryButton} type="button" onClick={addRoute} disabled={controlDisabled || !canAddRoute}>
                  <Plus size={14} aria-hidden="true" />{t("settings.addRoute")}
                </button>
              </div>
              <div className={styles.settingsCollection}>
                {Object.values(draft.models.routes).map((route) => (
                  <div className={styles.settingsCollectionItem} key={route.name} id={fieldId(`models.routes.${route.name}`)} tabIndex={-1}>
                    <div className={styles.settingsCollectionItemHeader}>
                      <h4>{route.name}</h4>
                      <button
                        className={styles.iconButton}
                        type="button"
                        aria-label={t("settings.removeRoute")}
                        title={t("settings.removeRoute")}
                        disabled={controlDisabled}
                        onClick={() => removeRoute(route.name)}
                      ><Trash2 size={15} aria-hidden="true" /></button>
                    </div>
                    <div className={styles.settingsFieldGrid}>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.routes.${route.name}.provider_id`)}>
                        <span className={styles.fieldLabel}>{t("settings.providerId")}</span>
                        <select
                          className={styles.selectInput}
                          id={fieldId(`models.routes.${route.name}.provider_id`)}
                          value={route.provider_id}
                          disabled={controlDisabled}
                          onChange={(event) => updateRoute(route.name, { provider_id: event.currentTarget.value })}
                        >
                          {!Object.values(draft.models.providers).some((provider) => provider.id === route.provider_id) ? <option value={route.provider_id}>{route.provider_id || t("settings.required")}</option> : null}
                          {Object.entries(draft.models.providers).map(([row, provider]) => <option key={row} value={provider.id}>{provider.id}</option>)}
                        </select>
                        <span className={styles.fieldError}>{fieldError(`models.routes.${route.name}.provider_id`) ?? ""}</span>
                      </label>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.routes.${route.name}.model`)}>
                        <span className={styles.fieldLabel}>{t("settings.model")}</span>
                        <input
                          className={styles.textInput}
                          id={fieldId(`models.routes.${route.name}.model`)}
                          value={route.model}
                          disabled={controlDisabled}
                          onChange={(event) => updateRoute(route.name, { model: event.currentTarget.value })}
                        />
                        <span className={styles.fieldError}>{fieldError(`models.routes.${route.name}.model`) ?? ""}</span>
                      </label>
                      <SettingsNumberField id={fieldId(`models.routes.${route.name}.context_window`)} label={t("settings.contextWindow")} value={route.context_window} error={fieldError(`models.routes.${route.name}.context_window`)} disabled={controlDisabled} onChange={(value) => updateRoute(route.name, { context_window: value })} onBlur={() => blurField(`models.routes.${route.name}.context_window`, route.context_window)} />
                      <SettingsNumberField id={fieldId(`models.routes.${route.name}.max_output`)} label={t("settings.maxOutput")} value={route.max_output} error={fieldError(`models.routes.${route.name}.max_output`)} disabled={controlDisabled} onChange={(value) => updateRoute(route.name, { max_output: value })} onBlur={() => blurField(`models.routes.${route.name}.max_output`, route.max_output)} />
                      <SettingsNumberField id={fieldId(`models.routes.${route.name}.temperature`)} label={t("settings.temperature")} value={route.temperature} error={fieldError(`models.routes.${route.name}.temperature`)} disabled={controlDisabled} step="0.1" onChange={(value) => updateRoute(route.name, { temperature: value })} onBlur={() => blurField(`models.routes.${route.name}.temperature`, route.temperature)} />
                      <label className={styles.settingsField} htmlFor={fieldId(`models.routes.${route.name}.reasoning_effort`)}>
                        <span className={styles.fieldLabel}>{t("settings.reasoningEffort")}</span>
                        <select className={styles.selectInput} id={fieldId(`models.routes.${route.name}.reasoning_effort`)} value={route.reasoning_effort} disabled={controlDisabled} onChange={(event) => updateRoute(route.name, { reasoning_effort: event.currentTarget.value as ReasoningEffort })}>
                          {(["low", "medium", "high", "xhigh", "max"] as ReasoningEffort[]).map((effort) => <option key={effort} value={effort}>{t(`settings.reasoningEfforts.${effort}`)}</option>)}
                        </select>
                      </label>
                      <SettingsNumberField id={fieldId(`models.routes.${route.name}.timeout`)} label={t("settings.timeout")} value={route.timeout} error={fieldError(`models.routes.${route.name}.timeout`)} disabled={controlDisabled} onChange={(value) => updateRoute(route.name, { timeout: value })} onBlur={() => blurField(`models.routes.${route.name}.timeout`, route.timeout)} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

              : null}

              {activeSection === "mcp" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.mcp")}</p>
                <h2>{t("settings.mcp")}</h2>
              </div>
              <ShieldX size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsCollectionHeader}>
              <h3>{t("settings.mcpServers")}</h3>
              <button className={styles.secondaryButton} type="button" onClick={addMcp} disabled={controlDisabled}>
                <Plus size={14} aria-hidden="true" />{t("settings.addMcp")}
              </button>
            </div>
            <div className={styles.settingsCollection}>
              {Object.entries(draft.mcp).map(([serverRow, server]) => (
                <div className={styles.settingsCollectionItem} key={serverRow} id={fieldId(`mcp.${server.name}`)} tabIndex={-1}>
                  <div className={styles.settingsCollectionItemHeader}>
                    <h4>{server.name}</h4>
                    <button className={styles.iconButton} type="button" aria-label={t("settings.removeMcp")} title={t("settings.removeMcp")} disabled={controlDisabled} onClick={() => removeMcp(serverRow)}><Trash2 size={15} aria-hidden="true" /></button>
                  </div>
                  <div className={styles.settingsFieldGrid}>
                    <label className={styles.settingsField} htmlFor={fieldId(`mcp.${serverRow}.name`)}><span className={styles.fieldLabel}>{t("settings.serverName")}</span><input className={styles.textInput} id={fieldId(`mcp.${serverRow}.name`)} value={server.name} aria-invalid={fieldError(`mcp.${serverRow}.name`) !== undefined} readOnly={response?.fields.mcp[serverRow] !== undefined} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { name: event.currentTarget.value })} /></label>
                    <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.transport`)}>
                      <span className={styles.fieldLabel}>{t("settings.transport")}</span>
                      <select className={styles.selectInput} id={fieldId(`mcp.${server.name}.transport`)} value={server.transport} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { transport: event.currentTarget.value as McpForm["transport"] })}>
                        <option value="stdio">stdio</option>
                        <option value="streamable-http">streamable-http</option>
                      </select>
                    </label>
                    <label className={styles.settingsToggle} htmlFor={fieldId(`mcp.${server.name}.enabled`)}>
                      <input id={fieldId(`mcp.${server.name}.enabled`)} type="checkbox" checked={server.enabled} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { enabled: event.currentTarget.checked })} />
                      <span><strong>{t("settings.enabled")}</strong><small>{server.enabled ? t("settings.yes") : t("settings.no")}</small></span>
                    </label>
                    {server.transport === "stdio" ? (
                      <>
                        <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.command`)}><span className={styles.fieldLabel}>{t("settings.command")}</span><input className={styles.textInput} id={fieldId(`mcp.${server.name}.command`)} value={server.command} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { command: event.currentTarget.value })} /><span className={styles.fieldError}>{fieldError(`mcp.${server.name}.command`) ?? ""}</span></label>
                        <SettingsListField id={fieldId(`mcp.${server.name}.args`)} label={t("settings.args")} values={server.args} error={groupError(`mcp.${server.name}.args`)} disabled={controlDisabled} onChange={(args) => updateMcp(serverRow, { args })} />
                        <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.cwd`)}><span className={styles.fieldLabel}>{t("settings.cwd")}</span><input className={styles.textInput} id={fieldId(`mcp.${server.name}.cwd`)} value={server.cwd} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { cwd: event.currentTarget.value })} /></label>
                      </>
                    ) : (
                      <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.url`)}><span className={styles.fieldLabel}>{t("settings.url")}</span><input className={styles.textInput} id={fieldId(`mcp.${server.name}.url`)} value={server.url} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { url: event.currentTarget.value })} onBlur={() => blurField(`mcp.${server.name}.url`, server.url)} /><span className={styles.fieldError}>{fieldError(`mcp.${server.name}.url`) ?? ""}</span></label>
                    )}
                    <SettingsNumberField id={fieldId(`mcp.${server.name}.connect_timeout`)} label={t("settings.connectTimeout")} value={server.connect_timeout} error={fieldError(`mcp.${server.name}.connect_timeout`)} disabled={controlDisabled} onChange={(value) => updateMcp(serverRow, { connect_timeout: value })} onBlur={() => blurField(`mcp.${server.name}.connect_timeout`, server.connect_timeout)} />
                    <SettingsNumberField id={fieldId(`mcp.${server.name}.call_timeout`)} label={t("settings.callTimeout")} value={server.call_timeout} error={fieldError(`mcp.${server.name}.call_timeout`)} disabled={controlDisabled} onChange={(value) => updateMcp(serverRow, { call_timeout: value })} onBlur={() => blurField(`mcp.${server.name}.call_timeout`, server.call_timeout)} />
                    <div className={styles.settingsField} id={fieldId(`mcp.${server.name}.tool_keywords`)} tabIndex={-1}>
                      <span className={styles.fieldLabel}>{t("settings.toolKeywords")}</span>
                      {server.tool_keywords.map((tool, index) => (
                        <div className={styles.settingsCollectionItem} key={tool.id}>
                          <label className={styles.settingsField}><span className={styles.fieldLabel}>{t("settings.toolName")}</span><input className={styles.textInput} value={tool.name} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { tool_keywords: server.tool_keywords.map((entry) => entry.id === tool.id ? { ...entry, name: event.currentTarget.value } : entry) })} /></label>
                          <SettingsListField id={`${fieldId(`mcp.${server.name}.tool_keywords`)}-${index}`} label={t("settings.toolKeywords")} values={tool.keywords} error={fieldError(`mcp.${server.name}.tool_keywords.${tool.name}`)} disabled={controlDisabled} onChange={(keywords) => updateMcp(serverRow, { tool_keywords: server.tool_keywords.map((entry) => entry.id === tool.id ? { ...entry, keywords } : entry) })} />
                          <button type="button" className={styles.secondaryButton} disabled={controlDisabled} onClick={() => updateMcp(serverRow, { tool_keywords: server.tool_keywords.filter((entry) => entry.id !== tool.id) })}>{t("settings.removeTool")}</button>
                        </div>
                      ))}
                      <span className={styles.fieldError}>{groupError(`mcp.${server.name}.tool_keywords`) ?? ""}</span>
                      <button type="button" className={styles.secondaryButton} disabled={controlDisabled} onClick={() => updateMcp(serverRow, { tool_keywords: [...server.tool_keywords, { id: createRequestId(), name: "", keywords: [] }] })}>{t("settings.addTool")}</button>
                    </div>
                  </div>
                  {server.transport === "streamable-http" ? (
                    <div className={styles.settingsSubsection} id={fieldId(`mcp.${server.name}.headers`)} tabIndex={-1}>
                      <span className={styles.fieldError}>{groupError(`mcp.${server.name}.headers`) ?? ""}</span>
                      <div className={styles.settingsCollectionHeader}><h4>{t("settings.headers")}</h4><button className={styles.secondaryButton} type="button" onClick={() => addHeader(serverRow)} disabled={controlDisabled}><Plus size={14} aria-hidden="true" />{t("settings.addHeader")}</button></div>
                      <div className={styles.settingsCollection}>
                        {Object.entries(server.headers).map(([header, secret]) => (
                          <div className={styles.settingsSecretItem} key={header}>
                            <label className={styles.settingsField}><span className={styles.fieldLabel}>{t("settings.headerName")}</span><input className={styles.textInput} value={secret.name} readOnly={response?.fields.mcp[serverRow]?.headers[header] !== undefined} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { headers: { ...server.headers, [header]: { ...secret, name: event.currentTarget.value } } })} /></label>
                            <SecretInput id={headerInputId(server.name, header)} label={t("settings.headerValue")} secret={secret} error={groupError(`mcp.${server.name}.headers.${header}`) ?? groupError(`mcp.${server.name}.headers.${secret.name}`)} disabled={controlDisabled} onChange={(update) => updateMcp(serverRow, { headers: { ...server.headers, [header]: { ...secret, ...update } } })} />
                            <button className={styles.iconButton} type="button" aria-label={t("settings.removeHeader")} title={t("settings.removeHeader")} disabled={controlDisabled} onClick={() => removeHeader(serverRow, header)}><Trash2 size={15} aria-hidden="true" /></button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

              : null}
            </div>
          </div>
          {notice !== null ? (
            <div className={styles.notice} role="status" aria-live="polite">
              <CircleCheck size={16} aria-hidden="true" />
              <span>{notice}</span>
            </div>
          ) : null}
        </form>
      ) : null}
    </section>
  );
}

interface SettingsListFieldProps {
  error?: string;
  id: string;
  label: string;
  values: string[];
  disabled: boolean;
  onChange: (values: string[]) => void;
}

function SettingsListField({ id, label, values, disabled, onChange, error }: SettingsListFieldProps) {
  const { t } = useTranslation();
  return (
    <div className={styles.settingsField} id={id} tabIndex={-1}>
      <span className={styles.fieldLabel}>{label}</span>
      {values.map((value, index) => (
        <div className={styles.settingsListRow} key={index}>
          <textarea className={styles.textArea} aria-label={`${label} ${index + 1}`} rows={2} aria-invalid={error !== undefined} aria-describedby={error !== undefined ? `${id}-error` : undefined} value={value} disabled={disabled} onChange={(event) => onChange(values.map((entry, position) => position === index ? event.currentTarget.value : entry))} />
          <button type="button" className={styles.iconButton} aria-label={`${t("settings.removeItem")} ${label} ${index + 1}`} disabled={disabled} onClick={() => { onChange(values.filter((_, position) => position !== index)); document.getElementById(id)?.focus(); }}><Trash2 size={15} aria-hidden="true" /></button>
        </div>
      ))}
      {error !== undefined ? <span className={styles.fieldError} id={`${id}-error`}>{error}</span> : null}
      <button type="button" className={styles.secondaryButton} disabled={disabled} onClick={() => onChange([...values, ""])}>{t("settings.addItem")}</button>
    </div>
  );
}

interface SettingsNumberFieldProps {
  id: string;
  label: string;
  value: string;
  error: string | undefined;
  disabled: boolean;
  step?: string;
  onChange: (value: string) => void;
  onBlur: () => void;
}

function SettingsNumberField({ id, label, value, error, disabled, step, onChange, onBlur }: SettingsNumberFieldProps) {
  return (
    <label className={styles.settingsField} htmlFor={id}>
      <span className={styles.fieldLabel} id={`${id}-label`}>{label}</span>
      <input
        className={styles.textInput}
        id={id}
        aria-labelledby={`${id}-label`}
        type="number"
        inputMode="decimal"
        step={step}
        value={value}
        disabled={disabled}
        aria-invalid={error !== undefined}
        aria-describedby={error !== undefined ? `${id}-error` : undefined}
        onChange={(event) => onChange(event.currentTarget.value)}
        onBlur={onBlur}
      />
      <span className={styles.fieldError} id={`${id}-error`}>{error ?? ""}</span>
    </label>
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
              const admissionClosed = !project.available || removalPending || removalBlocked;
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
                    {admissionClosed ? (
                      <button className={styles.secondaryButton} type="button" disabled>
                        <CalendarClock size={15} aria-hidden="true" />
                        {t("controls.openSchedule")}
                      </button>
                    ) : (
                      <Link
                        className={styles.secondaryButton}
                        to={`/projects/${project.project_id}/schedule`}
                      >
                        <CalendarClock size={15} aria-hidden="true" />
                        {t("controls.openSchedule")}
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

type ScheduleLoadState = "idle" | "loading" | "ready" | "error";

interface ScheduleJobsViewProps {
  authState: AuthState;
  connectionState: ConnectionState;
  projects: RegisteredProject[];
}

function ScheduleJobsView({ authState, connectionState, projects }: ScheduleJobsViewProps) {
  const { projectId = "" } = useParams();
  return (
    <ScheduleJobsContent
      key={projectId}
      authState={authState}
      connectionState={connectionState}
      projects={projects}
      projectId={projectId}
    />
  );
}

function ScheduleJobsContent({
  authState,
  connectionState,
  projects,
  projectId,
}: ScheduleJobsViewProps & { projectId: string }) {
  const { i18n, t } = useTranslation();
  const project = projects.find((item) => item.project_id === projectId);
  const projectAvailable = project?.available === true;
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
    if (authState !== "ready" || connectionState !== "online" || !projectId || !projectAvailable) return;
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
        const sessions = await getProjectSessions(projectId, { limit: 1 });
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
  }, [authState, connectionState, projectAvailable, projectId]);

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
  }, [authState, connectionState, loadSchedule, projectAvailable, projectId]);

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
          <Link className={styles.backLink} to={`/projects/${projectId}`}>
            <ArrowLeft size={15} aria-hidden="true" />
            {t("controls.backToSessions")}
          </Link>
          <p className={styles.eyebrow}>{t("nav.schedule")}</p>
          <h1 id="schedule-heading" tabIndex={-1}>{t("schedule.title")}</h1>
          <p className={styles.pageDescription}>
            {project?.name || t("schedule.title")}
            {project !== undefined ? ` · ${project.path}` : ""}
          </p>
        </div>
        <div className={styles.pageActions}>
          <Link className={styles.secondaryButton} to={`/projects/${projectId}`}>
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
      ) : project === undefined ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true"><CircleAlert size={22} /></div>
          <div><h2>{t("schedule.notFound")}</h2><Link className={styles.secondaryButton} to="/projects">{t("controls.backToProjects")}</Link></div>
        </div>
      ) : !projectAvailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><FolderOpen size={22} /></div>
          <div><h2>{t("schedule.projectUnavailable")}</h2><p>{project.path}</p></div>
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
                <div className={styles.scheduleEmptyState} role="status"><Clock3 size={18} aria-hidden="true" /><span>{t("schedule.empty")}</span></div>
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
                          to={`/projects/${projectId}/schedule/jobs/${job.job_id}/history`}
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
                  to={`/projects/${projectId}/schedule/jobs/${detailJob.job_id}/history`}
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

function scheduleErrorKey(error: unknown, fallback: string): string {
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

type ScheduleHistoryLoadState = "idle" | "loading" | "ready" | "error";

function ScheduleJobHistoryView({
  authState,
  connectionState,
  projects,
}: ScheduleJobsViewProps) {
  const { projectId = "", jobId = "" } = useParams();
  return (
    <ScheduleJobHistoryContent
      key={`${projectId}:${jobId}`}
      authState={authState}
      connectionState={connectionState}
      projects={projects}
      projectId={projectId}
      jobId={jobId}
    />
  );
}

function ScheduleJobHistoryContent({
  authState,
  connectionState,
  projects,
  projectId,
  jobId,
}: ScheduleJobsViewProps & { projectId: string; jobId: string }) {
  const { i18n, t } = useTranslation();
  const project = projects.find((item) => item.project_id === projectId);
  const projectAvailable = project?.available === true;
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
      || !projectId
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
        const sessions = await getProjectSessions(projectId, { limit: 1 });
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
  }, [authState, connectionState, jobId, projectAvailable, projectId]);

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
      || !projectId
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
  ]);

  const authUnavailable = authState !== "ready";
  const heading = job?.title || t("schedule.historyTitle");
  return (
    <section className={styles.schedulePage} aria-labelledby="schedule-history-heading">
      <div className={styles.pageHeading}>
        <div>
          <Link className={styles.backLink} to={`/projects/${projectId}/schedule`}>
            <ArrowLeft size={15} aria-hidden="true" />
            {t("schedule.backToJobs")}
          </Link>
          <p className={styles.eyebrow}>{t("schedule.historyEyebrow")}</p>
          <h1 id="schedule-history-heading" tabIndex={-1}>{heading}</h1>
          <p className={styles.pageDescription}>
            {project?.name || t("schedule.historyTitle")}
            {project !== undefined ? ` · ${project.path}` : ""}
          </p>
        </div>
        <div className={styles.pageActions}>
          <Link className={styles.secondaryButton} to={`/projects/${projectId}/schedule`}>
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
      ) : project === undefined ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true"><CircleAlert size={22} /></div>
          <div><h2>{t("schedule.notFound")}</h2><Link className={styles.secondaryButton} to="/projects">{t("controls.backToProjects")}</Link></div>
        </div>
      ) : !projectAvailable ? (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true"><FolderOpen size={22} /></div>
          <div><h2>{t("schedule.projectUnavailable")}</h2><p>{project.path}</p></div>
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

function scheduleHistoryGroupKey(group: ScheduleHistoryGroup, index: number): string {
  const firstMessage = group.messages[0];
  const timestamp = typeof firstMessage?.timestamp === "string" ? firstMessage.timestamp : "unknown";
  return `${group.started_at ?? "unknown"}-${timestamp}-${index}`;
}

function scheduleHistoryTime(
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

function scheduleHistoryStateIcon(state: ScheduleHistoryResultState) {
  if (state === "success") return <CircleCheck size={14} aria-hidden="true" />;
  if (state === "failure") return <CircleAlert size={14} aria-hidden="true" />;
  if (state === "canceled") return <Ban size={14} aria-hidden="true" />;
  return <Info size={14} aria-hidden="true" />;
}

function scheduleJobRule(
  schedule: ScheduleJob["schedule"],
  t: (key: string, options?: Record<string, unknown>) => string,
): string {
  if (schedule.kind === "at") return t("schedule.ruleAt", { value: schedule.at_time });
  if (schedule.kind === "every") return t("schedule.ruleEvery", { value: schedule.every_seconds });
  return t("schedule.ruleCron", { expression: schedule.cron_expr, timezone: schedule.timezone });
}

function scheduleJobStatusIcon(status: ScheduleJobStatus) {
  if (status === "running") return <Activity size={14} aria-hidden="true" />;
  if (status === "ok") return <CircleCheck size={14} aria-hidden="true" />;
  if (status === "error") return <CircleAlert size={14} aria-hidden="true" />;
  return <Clock3 size={14} aria-hidden="true" />;
}

function scheduleJobLastResult(
  job: ScheduleJob,
  language: string,
  t: (key: string) => string,
): string {
  if (job.active) return t("schedule.runningNow");
  if (job.state.last_finished_at_ms === null) return t("schedule.neverRun");
  return new Date(job.state.last_finished_at_ms).toLocaleString(language);
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
type ToolStatus = "running" | "completed" | "failed" | "rejected" | "canceled" | "unknown";

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
  cancellable: boolean;
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
    const value = JSON.parse(sessionStorage.getItem(`omni.session-delete.${projectId}`) ?? "null") as Partial<PendingSessionDeletion> | null;
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
    cancellable: true,
  };
}

function reduceLiveRunEvent(runs: LiveRun[], event: ServiceEvent): LiveRun[] {
  if (event.run_id === null) return runs;
  const runId = event.run_id;
  const requestId = typeof event.payload.request_id === "string" ? event.payload.request_id : null;
  const index = runs.findIndex((run) => run.runId === runId
    || (event.type === "input.accepted" && requestId !== null && run.localId === requestId));
  const current = index >= 0 ? runs[index] : newLiveRun(requestId ?? `event-${runId}`, runId,
    typeof event.payload.text === "string" ? event.payload.text : "", "accepted");
  let next = { ...current, runId };
  if (event.type === "input.accepted") {
    next.prompt = typeof event.payload.text === "string" ? event.payload.text : next.prompt;
    if (next.status === "submitting") next.status = "accepted";
  } else if (event.type === "run.output") {
    const message = event.payload.message;
    if (typeof message !== "object" || message === null || Array.isArray(message)) return runs;
    const value = message as Record<string, unknown>;
    const metadata = typeof value.metadata === "object" && value.metadata !== null
      ? value.metadata as Record<string, unknown> : {};
    const content = typeof value.content === "string" ? value.content : "";
    if (!isLiveRunActive(next)) return runs;
    next.status = "running";
    if (value.type === "model_response" && metadata._stream_delta === true) {
      next.assistantContent += content;
    } else if (value.type === "tool_call" && typeof metadata.tool_call_id === "string") {
      const toolId = metadata.tool_call_id;
      const tool = next.tools.find((item) => item.toolCallId === toolId);
      const status: ToolStatus = metadata.status === "success" ? "completed"
        : metadata.status === "error" ? "failed" : metadata.status === "refused" ? "rejected"
          : metadata.status === "cancelled" || metadata.status === "canceled" ? "canceled" : "running";
      next.tools = tool === undefined ? [...next.tools, { toolCallId: toolId, name: content,
        arguments: typeof metadata.arguments === "string" ? metadata.arguments : "", status }]
        : next.tools.map((item) => item.toolCallId === toolId ? { ...item, status } : item);
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

function isLiveRunActive(run: LiveRun): boolean {
  return run.status === "submitting" || run.status === "accepted" || run.status === "running";
}

function statusIcon(status: RunStatus | ToolStatus, size = 14) {
  if (status === "unknown") return <Info size={size} aria-hidden="true" />;
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
  showStatus = true,
}: {
  tools: ToolActivity[];
  t: (key: string) => string;
  showStatus?: boolean;
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
              {showStatus ? <span className={`${styles.statusBadge} ${styles[`status${tool.status}`]}`}>
                {statusIcon(tool.status, 12)}
                {t(toolStatusKey(tool.status))}
              </span> : null}
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
  scheduleHistory = false,
}: {
  message: Record<string, unknown>;
  index: number;
  t: (key: string) => string;
  scheduleHistory?: boolean;
}) {
  const role = message.role;
  const messageStatus = message.status;
  if (role === "tool") {
    const rawStatus = typeof messageStatus === "string" ? messageStatus : "error";
    const status: ToolStatus = scheduleHistory && !["success", "refused", "error"].includes(String(messageStatus))
      ? "unknown"
      : rawStatus === "success"
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
  const toolActivities = scheduleHistory && role === "assistant" ? historyToolActivities(message) : [];
  return (
    <article className={styles.historyMessage} data-role={typeof role === "string" ? role : "system"} key={`${index}-${String(role)}`}>
      <div className={styles.historyMessageRole}>{historyRoleLabel(role, t)}</div>
      {toolActivities.length > 0 ? <ToolActivityGroup tools={toolActivities} t={t} showStatus={false} /> : null}
      <MarkdownContent content={historyMessageText(message.content)} />
    </article>
  );
}

function historyToolActivities(message: Record<string, unknown>): ToolActivity[] {
  if (!Array.isArray(message.tool_calls)) return [];
  return message.tool_calls.flatMap((candidate, index) => {
    if (candidate === null || typeof candidate !== "object" || Array.isArray(candidate)) return [];
    const toolCall = candidate as Record<string, unknown>;
    const rawArguments = toolCall.arguments;
    let argumentsText = "";
    if (typeof rawArguments === "string") {
      argumentsText = rawArguments;
    } else if (rawArguments !== undefined) {
      try {
        argumentsText = JSON.stringify(rawArguments) ?? "";
      } catch {
        argumentsText = "";
      }
    }
    return [{
      toolCallId: typeof toolCall.id === "string" ? toolCall.id : `tool-call-${index}`,
      name: typeof toolCall.name === "string" ? toolCall.name : "",
      arguments: argumentsText,
      status: "unknown" as const,
    }];
  });
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
    <article className={styles.liveRun} data-run-id={run.runId ?? run.localId}>
      <div className={styles.historyMessage} data-role="user">
        <div className={styles.historyMessageRole}>{t("sessions.userMessage")}</div>
        <div className={styles.livePrompt}>{run.prompt}</div>
      </div>
      <div className={styles.historyMessage} data-role="assistant">
        <div className={styles.liveRunHeader}>
          <span className={styles.historyMessageRole}>{t("sessions.assistantMessage")}</span>
          <span className={`${styles.statusBadge} ${styles[`status${run.status}`]}`} role="status" aria-live="polite">
            {statusIcon(run.status)}
            {t(runStatusKey(run.status))}
          </span>
        </div>
        {run.tools.length > 0 ? <ToolActivityGroup tools={run.tools} t={t} /> : null}
        {run.assistantContent ? <MarkdownContent content={run.assistantContent} /> : active ? <p className={styles.pendingAnswer}>{t("conversation.assistantPending")}</p> : null}
        {run.error ? <p className={styles.runError}>{run.error}</p> : null}
        {active && run.runId !== null && run.cancellable ? (
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
      </div>
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

function ChatSessionsView({
  authState,
  connectionState,
  projects,
  registeredClient,
  onRestoreConsumed,
  refreshVersion,
  sendServiceCommand,
  subscribeServiceEvents,
  confirmationTriggerRef,
  newChatVersion,
  configurationNeedsSetup,
}: ProjectSessionsViewProps & { newChatVersion: number; configurationNeedsSetup: boolean | null }) {
  const { t } = useTranslation();
  const [setupInput, setSetupInput] = useState("");
  const [workspaceEntry, setWorkspaceEntry] = useState<Awaited<ReturnType<typeof enterChatWorkspace>> | null>(null);
  const [entryVersion, setEntryVersion] = useState(0);
  const [initialSessionId, setInitialSessionId] = useState<string | null>(null);
  const [entryLoadState, setEntryLoadState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [entryError, setEntryError] = useState<string | null>(null);
  const [history, setHistory] = useState<ChatSessionsResponse | null>(null);
  const [historyLoadState, setHistoryLoadState] = useState<SessionLoadState>("idle");
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyBusy, setHistoryBusy] = useState(false);
  const activationSequenceRef = useRef(0);
  const historyRequestRef = useRef(0);

  const refreshHistory = useCallback(async (cursor: string | null = null) => {
    if (authState !== "ready") return;
    const requestNumber = ++historyRequestRef.current;
    setHistoryLoadState("loading");
    setHistoryError(null);
    try {
      const response = await getChatSessions({ cursor: cursor ?? undefined, limit: 100 });
      if (requestNumber !== historyRequestRef.current) return;
      setHistory((current) => cursor === null || current === null
        ? response
        : { ...response, sessions: [...current.sessions, ...response.sessions] });
      setHistoryLoadState("ready");
    } catch (error) {
      if (requestNumber !== historyRequestRef.current) return;
      setHistoryLoadState("error");
      setHistoryError(sessionErrorKey(error));
    }
  }, [authState]);

  const activateWorkspace = useCallback(async (directory?: string, sessionId: string | null = null) => {
    if (authState !== "ready" || configurationNeedsSetup !== false) return;
    const requestNumber = ++activationSequenceRef.current;
    setWorkspaceEntry(null);
    setInitialSessionId(null);
    setEntryLoadState("loading");
    setEntryError(null);
    try {
      const entry = await enterChatWorkspace(directory);
      if (requestNumber !== activationSequenceRef.current) return;
      setWorkspaceEntry(entry);
      setInitialSessionId(sessionId);
      setEntryVersion((version) => version + 1);
      setEntryLoadState("ready");
      void refreshHistory();
    } catch (error) {
      if (requestNumber !== activationSequenceRef.current) return;
      setEntryLoadState("error");
      setEntryError(error instanceof ApiError && error.body?.code === "config_invalid"
        ? "chat.configurationUnavailable"
        : error instanceof ApiError && error.body?.code === "workspace_unavailable"
          ? "chat.workspaceUnavailable"
          : error instanceof ApiError && error.body?.code === "not_found" && directory !== undefined
            ? "chat.workspaceUnavailable"
            : sessionErrorKey(error));
    }
  }, [authState, configurationNeedsSetup, refreshHistory]);

  useEffect(() => {
    if (authState === "ready") void refreshHistory();
  }, [authState, refreshVersion, refreshHistory]);

  useEffect(() => {
    if (authState === "ready" && configurationNeedsSetup === false) void activateWorkspace();
  }, [authState, configurationNeedsSetup, newChatVersion, activateWorkspace]);

  useEffect(() => subscribeServiceEvents((event) => {
    if (event.type.startsWith("session.")
      || ["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
      void refreshHistory();
    }
  }), [refreshHistory, subscribeServiceEvents]);

  async function startNewConversation() {
    await activateWorkspace();
  }

  async function openHistorySession(session: ChatSessionSummary) {
    if (!session.available || historyBusy) return;
    setHistoryBusy(true);
    await activateWorkspace(session.directory, session.id);
    setHistoryBusy(false);
  }

  return (
    <div className={styles.chatWorkspaceLayout}>
      <aside className={styles.chatHistorySidebar} aria-label={t("nav.chatHistory")}>
        <button
          className={styles.primaryButton}
          type="button"
          disabled={authState !== "ready" || entryLoadState === "loading"}
          onClick={() => void startNewConversation()}
        >
          <Plus size={16} aria-hidden="true" />
          {t("controls.newSession")}
        </button>
        <h2 className={styles.chatHistoryHeading}>{t("nav.chatHistory")}</h2>
        {historyLoadState === "loading" && history === null ? (
          <p className={styles.chatHistoryStatus} role="status">{t("chat.historyLoading")}</p>
        ) : null}
        {historyError !== null ? (
          <p className={styles.chatHistoryError} role="alert">{t(historyError)}</p>
        ) : null}
        <nav className={styles.chatHistoryList} aria-label={t("nav.chatHistory")}>
          {(history?.sessions ?? []).map((session) => (
            <button
              className={styles.chatHistoryItem}
              key={`${session.directory}:${session.id}`}
              type="button"
              disabled={!session.available || historyBusy || entryLoadState === "loading"}
              data-active={session.id === initialSessionId && session.directory === workspaceEntry?.directory}
              onClick={() => void openHistorySession(session)}
              title={session.directory}
            >
              <span>{session.title}</span>
              <small>{session.directory}</small>
            </button>
          ))}
        </nav>
        {history?.next_cursor !== null && history !== null ? (
          <button
            className={styles.chatHistoryMore}
            type="button"
            disabled={historyLoadState === "loading"}
            onClick={() => void refreshHistory(history.next_cursor)}
          >
            {t("sessions.loadMore")}
          </button>
        ) : null}
        <button
          className={styles.chatHistoryRefresh}
          type="button"
          disabled={historyLoadState === "loading"}
          onClick={() => void refreshHistory()}
        >
          <RefreshCw size={14} aria-hidden="true" />
          {t("controls.refreshSessions")}
        </button>
      </aside>
      <div className={styles.chatWorkspaceContent}>
        {configurationNeedsSetup === true ? (
          <div className={styles.conversationStage} data-empty="true">
            <div className={styles.conversationViewport}>
              <div className={styles.emptyConversation}>
                <h2 className={styles.emptyBrand}>Omni</h2>
              </div>
            </div>
            <form className={styles.composer} onSubmit={(event) => event.preventDefault()}>
              <label className={styles.srOnly} htmlFor="setup-conversation-input">{t("conversation.inputLabel")}</label>
              <textarea
                id="setup-conversation-input"
                aria-label={t("conversation.inputLabel")}
                className={styles.composerInput}
                rows={3}
                value={setupInput}
                placeholder={t("conversation.inputPlaceholder")}
                onChange={(event) => setSetupInput(event.target.value)}
              />
              <div className={styles.composerFooter}>
                <Link
                  className={styles.secondaryButton}
                  to="/settings#models"
                  state={{ returnTo: { pathname: "/", search: "", hash: "", scrollTop: 0 } }}
                >
                  {t("chat.configureModels")}
                </Link>
                <button className={styles.primaryButton} type="submit" disabled>
                  <Send size={15} aria-hidden="true" />
                  {t("controls.send")}
                </button>
              </div>
            </form>
          </div>
        ) : workspaceEntry === null ? (
          <div className={entryLoadState === "error" ? styles.errorBanner : styles.notice} role={entryLoadState === "error" ? "alert" : "status"}>
            <CircleAlert size={16} aria-hidden="true" />
            <span>{entryError === null ? t("chat.workspaceLoading") : t(entryError)}</span>
          </div>
        ) : (
          <ProjectSessionsContent
            key={`${workspaceEntry.workspace_id}:${entryVersion}`}
            authState={authState}
            connectionState={connectionState}
            projects={projects}
            registeredClient={registeredClient}
            onRestoreConsumed={onRestoreConsumed}
            refreshVersion={refreshVersion}
            sendServiceCommand={sendServiceCommand}
            subscribeServiceEvents={subscribeServiceEvents}
            confirmationTriggerRef={confirmationTriggerRef}
            projectId={null}
            workspaceId={workspaceEntry.workspace_id}
            workspaceDirectory={workspaceEntry.directory}
            initialSessionId={initialSessionId}
            startInDraft={initialSessionId === null}
          />
        )}
      </div>
    </div>
  );
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
  workspaceId: initialWorkspaceId,
  workspaceDirectory,
  initialSessionId,
  startInDraft = false,
}: ProjectSessionsViewProps & {
  projectId: string | null;
  workspaceId?: string;
  workspaceDirectory?: string;
  initialSessionId?: string | null;
  startInDraft?: boolean;
}) {
  const { t, i18n } = useTranslation();
  const isChat = projectId === null;
  const sessionScopeId = projectId ?? initialWorkspaceId ?? "";
  const sessionStorageId = projectId ?? workspaceDirectory ?? sessionScopeId;
  const project = projectId === null ? undefined : projects.find((item) => item.project_id === projectId);
  const [sessions, setSessions] = useState<ProjectSessionsResponse | WorkspaceSessionsResponse | null>(null);
  const [sessionSummaries, setSessionSummaries] = useState<Record<string, SessionSummary>>({});
  const [sessionSearch, setSessionSearch] = useState("");
  const deferredSessionSearch = useDeferredValue(sessionSearch);
  const [sessionNextCursor, setSessionNextCursor] = useState<string | null>(null);
  const [loadState, setLoadState] = useState<SessionLoadState>("idle");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [sessionListOpen, setSessionListOpen] = useState(false);
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
  const [pendingDeletion, setPendingDeletion] = useState<PendingSessionDeletion | null>(() => readPendingDeletion(sessionStorageId));
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
  const sessionSelectionVersionRef = useRef(0);
  const refreshSessionsRef = useRef<((cursor?: string | null, append?: boolean) => Promise<void>) | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const renameTriggerRef = useRef<HTMLButtonElement | null>(null);
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null);
  const deleteRetryTriggerRef = useRef<HTMLButtonElement | null>(null);
  const deleteFocusOriginRef = useRef<"toolbar" | "retry">("toolbar");
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
  const snapshotReadsRef = useRef(new Set<ServiceEvent[]>());
  const snapshotEventsRef = useRef(new WeakMap<SessionSnapshot, ServiceEvent[]>());
  const sessionCursorRef = useRef<Record<string, { streamId: string; seq: number }>>({});
  const attemptedHistorySessionRef = useRef<string | null>(null);
  const defaultDraftWorkspaceRef = useRef<string | null>(null);

  const listSessions = useCallback((options: { title?: string; cursor?: string; limit?: number } = {}) => (
    isChat
      ? getWorkspaceSessions(sessionScopeId, options)
      : getProjectSessions(sessionScopeId, options)
  ), [isChat, sessionScopeId]);

  const claimSession = useCallback((sessionId: string): Promise<Pick<SessionClaimResponse, "claim" | "snapshot">> => (
    isChat
      ? claimWorkspaceSession(sessionScopeId, sessionId)
      : claimProjectSession(sessionScopeId, sessionId)
  ), [isChat, sessionScopeId]);

  const getClaimedSession = useCallback((currentClaim: SessionClaim) => (
    isChat
      ? getWorkspaceSession(
          currentClaim.workspace_id,
          currentClaim.session_id,
          currentClaim.claim_version,
          currentClaim.reconnect_credential,
        )
      : getProjectSession(
          sessionScopeId,
          currentClaim.session_id,
          currentClaim.claim_version,
          currentClaim.reconnect_credential,
        )
  ), [isChat, sessionScopeId]);

  const releaseSessionClaim = useCallback((currentClaim: SessionClaim) => (
    isChat
      ? releaseWorkspaceSession(
          currentClaim.workspace_id,
          currentClaim.session_id,
          currentClaim.claim_version,
          currentClaim.reconnect_credential,
        )
      : releaseProjectSession(
          sessionScopeId,
          currentClaim.session_id,
          currentClaim.claim_version,
          currentClaim.reconnect_credential,
        )
  ), [isChat, sessionScopeId]);

  const getDeletionStatus = useCallback((sessionId: string) => (
    isChat
      ? getWorkspaceSessionDeletionStatus(sessionScopeId, sessionId)
      : getProjectSessionDeletionStatus(sessionScopeId, sessionId)
  ), [isChat, sessionScopeId]);

  const claimDeletion = useCallback((sessionId: string) => (
    isChat
      ? claimWorkspaceSessionDeletion(sessionScopeId, sessionId)
      : claimProjectSessionDeletion(sessionScopeId, sessionId)
  ), [isChat, sessionScopeId]);

  const readWithEvents = useCallback(async <T,>(
    read: () => Promise<T>, extract: (response: T) => SessionSnapshot,
  ): Promise<T> => {
    const events: ServiceEvent[] = [];
    const clientId = pendingClientIdRef.current;
    snapshotReadsRef.current.add(events);
    try {
      const response = await read();
      if (clientId !== pendingClientIdRef.current) throw new ServiceCommandError(null, false);
      snapshotEventsRef.current.set(extract(response), events);
      return response;
    } finally {
      snapshotReadsRef.current.delete(events);
    }
  }, []);

  const readClaimSnapshot = useCallback((sessionId: string) => readWithEvents(
    () => claimSession(sessionId), (response) => response.snapshot,
  ), [claimSession, readWithEvents]);

  const readRunSnapshot = useCallback((currentClaim: SessionClaim) => readWithEvents(
    () => getClaimedSession(currentClaim), (response) => response,
  ), [getClaimedSession, readWithEvents]);

  const clearPendingDeletion = useCallback(() => {
    pendingDeletionRef.current = null;
    setPendingDeletion(null);
    try { sessionStorage.removeItem(`omni.session-delete.${sessionStorageId}`); } catch { /* Storage can be unavailable. */ }
  }, [sessionStorageId]);

  const rememberPendingDeletion = useCallback((operation: PendingSessionDeletion) => {
    pendingDeletionRef.current = operation;
    setPendingDeletion(operation);
    try {
      sessionStorage.setItem(`omni.session-delete.${sessionStorageId}`, JSON.stringify(operation));
    } catch { /* In-memory retries remain available when browser storage is unavailable. */ }
  }, [sessionStorageId]);

  const releaseOrphanClaim = useCallback((orphan: SessionClaim) => {
    void releaseSessionClaim(orphan).catch(() => {});
  }, [releaseSessionClaim]);

  const releaseClaims = useCallback(() => {
    if (!sessionScopeId) return;
    for (const current of Object.values(claimsBySessionRef.current)) {
      if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === current.session_id) continue;
      void releaseSessionClaim(current).catch(() => {});
    }
  }, [releaseSessionClaim, sessionScopeId]);

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
  }, [claim, connectionState, draft, sessionScopeId]);

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

  usePanelKeyboard(sessionListOpen, setSessionListOpen, "project-session-list", "project-session-list-toggle");

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
      claimsBySessionRef.current = {};
      snapshotsBySessionRef.current = {};
      sessionCursorRef.current = {};
      liveRunsRef.current = {};
      setLiveRunsBySession({});
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
    const live = nextSnapshot.live_state;
    if (live !== undefined && live !== null) {
      const previous = snapshotsBySessionRef.current[sessionId]?.live_state;
      if (previous?.stream_id === live.stream_id && previous.seq > live.seq) return false;
      const events = (snapshotEventsRef.current.get(nextSnapshot) ?? []).filter((event) => (
        event.session_id === sessionId && event.stream_id === live.stream_id && event.seq > live.seq
      ));
      const recovered = live.runs.map((run): LiveRun => ({
        ...newLiveRun(run.request_id, run.run_id, run.prompt, run.status),
        assistantContent: run.assistant_content,
        tools: run.tools.map((tool) => ({ toolCallId: tool.tool_call_id, name: tool.name,
          arguments: tool.arguments, status: tool.status })),
        cancelRequested: run.cancel_requested, cancellable: run.cancellable,
      }));
      const requestIds = new Set(live.runs.map((run) => run.request_id));
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (pending) => !requestIds.has(pending.command.request_id),
      );
      updateLiveRuns(sessionId, (runs) => events.reduce(reduceLiveRunEvent, [
        ...recovered, ...runs.filter((run) => run.status === "submitting"
          && !requestIds.has(run.localId)),
      ]));
      const cursor = sessionCursorRef.current[sessionId];
      if (cursor?.streamId !== live.stream_id || cursor.seq < live.seq) {
        sessionCursorRef.current[sessionId] = { streamId: live.stream_id, seq: live.seq };
      }
    }
    const previousCount = snapshotsBySessionRef.current[sessionId]?.messages.length ?? 0;
    const committedPrompts = new Set(nextSnapshot.messages.slice(previousCount)
      .filter((message) => message.role === "user" && typeof message.content === "string")
      .map((message) => message.content as string));
    snapshotsBySessionRef.current[sessionId] = nextSnapshot;
    if (selectedSessionRef.current === sessionId) {
      snapshotRef.current = nextSnapshot;
      setSnapshot(nextSnapshot);
    }
    if ((live === undefined || live === null) && committedPrompts.size > 0) {
      updateLiveRuns(sessionId, (runs) => runs.filter((run) => !committedPrompts.has(run.prompt)));
    }
    return true;
  }, [updateLiveRuns]);

  const rememberSession = useCallback((nextClaim: SessionClaim, nextSnapshot: SessionSnapshot) => {
    const existingClaim = claimsBySessionRef.current[nextClaim.session_id];
    if (existingClaim !== undefined && existingClaim.claim_version > nextClaim.claim_version) return;
    claimsBySessionRef.current[nextClaim.session_id] = nextClaim;
    adoptSnapshot(nextSnapshot);
    claimRef.current = nextClaim;
    setClaim(nextClaim);
    setSnapshot(snapshotsBySessionRef.current[nextClaim.session_id] ?? nextSnapshot);
  }, [adoptSnapshot]);

  const refreshSessions = useCallback(async (cursor: string | null = null, append = false) => {
    if (authState !== "ready" || !sessionScopeId) return;
    const requestNumber = sessionRequestRef.current + 1;
    sessionRequestRef.current = requestNumber;
    setLoadState("loading");
    try {
      const response = await listSessions({
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
        const status = await getDeletionStatus(deletionOperation.claim.session_id);
        if (!mountedRef.current || requestNumber !== sessionRequestRef.current) return;
        if (status.state === "deleted" || status.state === "present") {
          clearPendingDeletion();
          setDeleteOpen(false);
          if (claimRef.current?.session_id === deletionOperation.claim.session_id) clearClaimState();
        } else {
          const recovered = await claimDeletion(deletionOperation.claim.session_id);
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
        connectionState === "online" && registeredClient !== null &&
        pendingClientIdRef.current === registeredClient.client_id &&
        attemptedRestoreRef.current !== registeredClient.web_control_credential &&
        registeredClient.current_workspace_id === response.workspace_id &&
        registeredClient.current_session_id !== null
        && pendingDeletionRef.current?.claim.session_id !== registeredClient.current_session_id
      ) {
        attemptedRestoreRef.current = registeredClient.web_control_credential;
        onRestoreConsumed();
        if (claimRef.current === null) {
          try {
            const selectionVersion = sessionSelectionVersionRef.current;
            const restored = await readClaimSnapshot(registeredClient.current_session_id);
            if (!mountedRef.current) {
              releaseOrphanClaim(restored.claim);
              return;
            }
            if (selectionVersion !== sessionSelectionVersionRef.current) {
              if (claimsBySessionRef.current[restored.claim.session_id] === undefined) {
                releaseOrphanClaim(restored.claim);
              }
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
              const metadata = await listSessions();
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
          const restored = await readClaimSnapshot(currentClaim.session_id);
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
  }, [authState, claimDeletion, clearClaimState, clearPendingDeletion, connectionState, deferredSessionSearch, getDeletionStatus, listSessions, onRestoreConsumed, registeredClient, releaseOrphanClaim, rememberPendingDeletion, rememberSession, readClaimSnapshot, sessionScopeId]);

  useEffect(() => {
    if (connectionState !== "online") needsReclaimRef.current = true;
  }, [connectionState]);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions, refreshVersion]);

  useEffect(() => {
    if (connectionState !== "online" || sessions?.workspace_id === undefined) return;
    void sendServiceCommand({
      request_id: createRequestId(), type: "subscribe", workspace_id: null,
      session_id: null, claim_version: null, payload: { last_seq: null },
    }).catch(() => {});
  }, [connectionState, sessions?.workspace_id, sendServiceCommand]);

  useEffect(() => {
    refreshSessionsRef.current = refreshSessions;
  }, [refreshSessions]);

  const loadMoreSessions = useCallback(() => {
    if (sessionNextCursor === null || loadState === "loading") return;
    void refreshSessions(sessionNextCursor, true);
  }, [loadState, refreshSessions, sessionNextCursor]);

  const refreshRunSnapshot = useCallback(async (sessionId: string, runId: string) => {
    const currentClaim = claimsBySessionRef.current[sessionId];
    if (currentClaim === undefined) {
      void refreshSessionsRef.current?.();
      return;
    }
    try {
      const nextSnapshot = await readRunSnapshot(currentClaim);
      if (!mountedRef.current || claimsBySessionRef.current[sessionId] !== currentClaim) return;
      adoptSnapshot(nextSnapshot);
      updateLiveRuns(sessionId, (runs) => runs.filter((run) => run.runId !== runId));
      void refreshSessionsRef.current?.();
    } catch {
      // Keep the terminal live projection visible when persistence is still settling.
    }
  }, [adoptSnapshot, readRunSnapshot, updateLiveRuns]);

  const handleServiceEvent = useCallback((event: ServiceEvent) => {
    for (const events of snapshotReadsRef.current) events.push(event);
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
          if (entry.workspace_id !== workspaceIdRef.current) continue;
          if (currentClaim !== undefined && (currentClaim.workspace_id !== entry.workspace_id
            || currentClaim.claim_version !== entry.claim_version)) continue;
          adoptSnapshot(nextSnapshot as SessionSnapshot);
          if (nextSnapshot.messages.length === 0
            && (nextSnapshot.live_state?.runs.length ?? 0) > 0) {
            const sessionId = nextSnapshot.session_id;
            setDraftSessionIds((ids) => ids.includes(sessionId) ? ids : [...ids, sessionId]);
          }
          restored.add(nextSnapshot.session_id);
        }
      }
      if (Array.isArray(sessions) && workspaceIdRef.current !== null
        && typeof snapshotPayload === "object" && snapshotPayload !== null
        && "pending_confirmation" in snapshotPayload) {
        for (const sessionId of Object.keys(liveRunsRef.current)) {
          const cursor = sessionCursorRef.current[sessionId];
          if (restored.has(sessionId)
            || (cursor?.streamId === event.stream_id && cursor.seq > event.seq)) continue;
          updateLiveRuns(sessionId, () => []);
          delete snapshotsBySessionRef.current[sessionId];
          delete sessionCursorRef.current[sessionId];
        }
      }
      for (const currentClaim of Object.values(claimsBySessionRef.current)) {
        if (restored.has(currentClaim.session_id)) continue;
        void readRunSnapshot(currentClaim).then((nextSnapshot) => {
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
      delete snapshotsBySessionRef.current[sessionId];
      delete sessionCursorRef.current[sessionId];
      updateLiveRuns(sessionId, () => []);
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
    const cursor = sessionCursorRef.current[sessionId];
    if (cursor?.streamId === event.stream_id && event.seq <= cursor.seq) return;
    sessionCursorRef.current[sessionId] = { streamId: event.stream_id, seq: event.seq };
    if (event.type === "input.accepted" && typeof event.payload.request_id === "string") {
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (pending) => pending.command.request_id !== event.payload.request_id,
      );
    }
    updateLiveRuns(sessionId, (runs) => reduceLiveRunEvent(runs, event));
    if (event.run_id !== null && (event.type === "run.completed" || event.type === "run.failed")) {
      void refreshRunSnapshot(sessionId, event.run_id);
    }
  }, [adoptSnapshot, readRunSnapshot, refreshRunSnapshot, updateLiveRuns]);

  useEffect(() => subscribeServiceEvents(handleServiceEvent), [handleServiceEvent, subscribeServiceEvents]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      releaseClaims();
    };
  }, [sessionScopeId, releaseClaims]);

  const openSession = useCallback(async (sessionId: string, isDraft: boolean, allowBusy = false) => {
    if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === sessionId) return;
    if (busySessionId !== null && !allowBusy) return;
    sessionSelectionVersionRef.current += 1;
    setManagementOpen(false);
    const previousSessionId = claimRef.current?.session_id;
    setBusySessionId(sessionId);
    setActionError(null);
    try {
      const response = await readClaimSnapshot(sessionId);
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
  }, [busySessionId, readClaimSnapshot, refreshSessions, releaseOrphanClaim, rememberSession]);

  useEffect(() => {
    if (initialSessionId == null || loadState !== "ready"
      || !sessions?.sessions.some((item) => item.id === initialSessionId)) return;
    const attemptKey = `${sessionScopeId}:${initialSessionId}`;
    if (attemptedHistorySessionRef.current === attemptKey) return;
    attemptedHistorySessionRef.current = attemptKey;
    void openSession(initialSessionId, false);
  }, [initialSessionId, loadState, openSession, sessionScopeId, sessions]);

  const createDraft = useCallback(async () => {
    if (busySessionId !== null) return;
    sessionSelectionVersionRef.current += 1;
    setBusySessionId("new");
    setActionError(null);
    try {
      const created = isChat
        ? await createWorkspaceSession(sessionScopeId)
        : await createProjectSession(sessionScopeId);
      setDraftSessionIds((ids) => [...ids, created.session_id]);
      setBusySessionId(null);
      await openSession(created.session_id, true, true);
    } catch (error) {
      setActionError(sessionErrorKey(error));
    } finally {
      setBusySessionId(null);
    }
  }, [busySessionId, isChat, openSession, sessionScopeId]);

  useEffect(() => {
    if (!startInDraft || initialSessionId != null || loadState !== "ready"
      || !sessionScopeId || selectedSessionRef.current !== null || claimRef.current !== null
      || defaultDraftWorkspaceRef.current === sessionScopeId) return;
    defaultDraftWorkspaceRef.current = sessionScopeId;
    void createDraft();
  }, [createDraft, initialSessionId, loadState, sessionScopeId, startInDraft]);

  async function releaseCurrent() {
    const current = claimRef.current;
    if (current === null) return;
    if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === current.session_id) return;
    setManagementOpen(false);
    setBusySessionId(current.session_id);
    setActionError(null);
    try {
      await releaseSessionClaim(current);
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
    const localId = createRequestId();
    confirmationTriggerRef.current = inputRef.current;
    const pending: PendingSubmission = {
      localId,
      sessionId,
      command: {
        request_id: localId,
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
      newLiveRun(pending.command.request_id, null, text, "submitting"),
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
      const nextSnapshot = await readRunSnapshot(nextClaim);
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
      const response = isChat
        ? await renameWorkspaceSession(
            currentClaim.workspace_id,
            currentClaim.session_id,
            currentClaim.claim_version,
            currentClaim.reconnect_credential,
            nextTitle,
            currentSummary.metadata_version,
          )
        : await renameProjectSession(
            sessionScopeId,
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
        const metadata = await listSessions();
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
    deleteFocusOriginRef.current = "toolbar";
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
        if (isChat) {
          await deleteWorkspaceSession(
            currentClaim.workspace_id,
            currentClaim.session_id,
            currentClaim.claim_version,
            currentClaim.reconnect_credential,
            operation.requestId,
          );
        } else {
          await deleteProjectSession(
            sessionScopeId,
            currentClaim.session_id,
            currentClaim.claim_version,
            currentClaim.reconnect_credential,
            operation.requestId,
          );
        }
      } catch (error) {
        if (!(error instanceof ApiError) || !["stale_claim", "not_found", "stale_client", "session_busy", "restore_pending", "validation_error"].includes(error.body?.code ?? "")) throw error;
        const status = await getDeletionStatus(currentClaim.session_id);
        if (status.state === "deleting") {
          if (["stale_claim", "not_found", "stale_client"].includes(error.body?.code ?? "")) {
            const recovered = await claimDeletion(currentClaim.session_id);
            rememberPendingDeletion({ ...operation, claim: recovered.claim });
          }
          setDeleteError("sessions.deletionInProgressError");
          return;
        }
        if (status.state === "present") {
          const rejectedOperation = { ...operation, attempted: false };
          pendingDeletionRef.current = rejectedOperation;
          setPendingDeletion(rejectedOperation);
          try { sessionStorage.removeItem(`omni.session-delete.${sessionStorageId}`); } catch { /* Storage can be unavailable. */ }
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
          {!isChat ? (
            <Link className={styles.backLink} to="/projects">
              <ArrowLeft size={15} aria-hidden="true" />
              {t("controls.backToProjects")}
            </Link>
          ) : null}
          <p className={styles.eyebrow}>{t(isChat ? "nav.chatHistory" : "nav.sessions")}</p>
          <h1 id="sessions-heading" tabIndex={-1}>
            {project?.name || t(isChat ? "app.name" : "sessions.title")}
          </h1>
          {project !== undefined
            ? <p className={styles.pageDescription}>{project.path}</p>
            : isChat && workspaceDirectory !== undefined
              ? <p className={styles.pageDescription}>{workspaceDirectory}</p>
              : null}
        </div>
        <div className={styles.pageActions}>
          <button
            className={styles.sessionListToggle}
            id="project-session-list-toggle"
            type="button"
            aria-label={t("sessions.listTitle")}
            aria-controls="project-session-list"
            aria-expanded={sessionListOpen}
            title={t("sessions.listTitle")}
            onClick={() => setSessionListOpen((open) => !open)}
          >
            {sessionListOpen ? <X size={17} aria-hidden="true" /> : <Menu size={17} aria-hidden="true" />}
          </button>
          {pendingDeletion?.attempted ? (
            <button
              className={styles.dangerButton}
              ref={deleteRetryTriggerRef}
              type="button"
              disabled={deleteBusy || connectionState !== "online"}
              onClick={() => {
                deleteFocusOriginRef.current = "retry";
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
          {!isChat ? (
            <>
              <Link className={styles.secondaryButton} to={`/projects/${projectId}/schedule`}>
                <CalendarClock size={15} aria-hidden="true" />
                {t("controls.openSchedule")}
              </Link>
              <button
                className={styles.primaryButton}
                type="button"
                disabled={authUnavailable || (!isChat && project?.available !== true) || busySessionId !== null}
                onClick={() => {
                  setSessionListOpen(false);
                  void createDraft();
                }}
              >
                <Plus size={16} aria-hidden="true" />
                {t("controls.newSession")}
              </button>
            </>
          ) : null}
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
      ) : !isChat && project === undefined ? (
        <div className={styles.emptyState} role="alert">
          <div className={styles.emptyIcon} aria-hidden="true"><CircleAlert size={22} /></div>
          <div><h2>{t("sessions.notFound")}</h2><Link className={styles.secondaryButton} to="/projects">{t("controls.backToProjects")}</Link></div>
        </div>
      ) : !isChat && project !== undefined && project.available !== true ? (
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
        <div className={styles.sessionsLayout} data-list-open={sessionListOpen}>
          {sessionListOpen ? (
            <button
              className={styles.sessionListBackdrop}
              type="button"
              aria-label={t("controls.close")}
              onClick={() => setSessionListOpen(false)}
            />
          ) : null}
          <aside
            className={styles.sessionListPanel}
            id="project-session-list"
            aria-label={t("sessions.listLabel")}
          >
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
                  onClick={() => {
                    setSessionListOpen(false);
                    void openSession(draftId, true);
                  }}
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
                        onClick={() => {
                          setSessionListOpen(false);
                          void openSession(item.id, false);
                        }}
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

          <section
            className={styles.sessionContentPanel}
            aria-label={t("sessions.conversation")}
          >
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
                <div
                  className={styles.conversationStage}
                  data-empty={snapshot.messages.length === 0 && selectedLiveRuns.length === 0}
                >
                  <div className={styles.conversationViewport} role="log" aria-live="off" aria-label={t("sessions.historyLabel")}>
                    {snapshot.messages.length === 0 && selectedLiveRuns.length === 0 ? (
                      <div className={styles.emptyConversation} role="status">
                        <h2 className={styles.emptyBrand}>Omni</h2>
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
                </div>
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
              const trigger = deleteFocusOriginRef.current === "retry"
                ? deleteRetryTriggerRef.current
                : deleteTriggerRef.current;
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
    if (value === "light" || value === "dark" || value === "system") return value;
  } catch {
    // Theme preference is optional when storage is unavailable.
  }
  return "light";
}

function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Theme preference is optional when storage is unavailable.
  }
}
