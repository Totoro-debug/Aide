import * as Dialog from "@radix-ui/react-dialog";
import {
  Activity,
  Check,
  CircleAlert,
  CircleCheck,
  Menu,
  Play,
  Settings2,
  ShieldX,
  Trash2,
  TriangleAlert,
  X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import {
  ApiError,
  ServiceCommandError,
  createRequestId,
  decideServiceConfirmation,
  getConfig,
  getProjectRemoval,
  getProjects,
  pickProjectDirectory,
  registerProject,
  removeProject,
  resumeProjectSchedule
} from "./api";
import styles from "./App.module.css";
import type { Theme } from "./app/theme.ts";
import { applyTheme, readThemePreference } from "./app/theme.ts";
import type { BrowserRecoverySnapshot } from "./features/conversations/browserRecovery";
import {
  browserRecoveryRoute,
  clearBrowserRecoverySnapshot,
  readBrowserRecoverySnapshot,
  reconcileBrowserRecovery,
  writeBrowserRecoverySnapshot,
} from "./features/conversations/browserRecovery";
import { ConversationDraftsProvider } from "./features/conversations/ConversationDraftsProvider";
import { ChatSessionsView, ProjectSessionsView } from "./features/conversations/ConversationsView.tsx";
import type { PendingSessionAction } from "./features/conversations/session.ts";
import { StatusView } from "./features/runtime/RuntimeView.tsx";
import { ScheduleJobHistoryView, ScheduleJobsView } from "./features/schedules/SchedulesView.tsx";
import { SettingsView } from "./features/settings/SettingsView.tsx";
import type { NavigationProjectAction, NavigationSession, NavigationSessionAction } from "./NavigationSidebar";
import NavigationSidebar from "./NavigationSidebar";
import type {
  ClientCommand,
  ConfirmationOrigin,
  ConfirmationRequest,
  RegisteredClient,
  RegisteredProject,
  ServiceCommandResult,
  ServiceEvent,
  ServiceStatus,
  SessionClaim
} from "./protocol";
import { WebServiceClient } from "./serviceClient";
import type { AuthState, ConnectionState, ServiceEventListener } from "./shared/service/types.ts";
import { isNonEmptyString, isRecord } from "./validation.ts";
import type { WorkspaceMemoryTarget } from "./WorkspaceMemoryDialog";
import WorkspaceMemoryDialog from "./WorkspaceMemoryDialog";

type ProjectsLoadState = "idle" | "loading" | "ready" | "error";

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

const initialLaunchTicket = readAndClearTicket();

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
  return <ConversationDraftsProvider><AppShell /></ConversationDraftsProvider>;
}

function AppShell() {
  const { i18n, t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;
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
  const [addProjectRequest, setAddProjectRequest] = useState(0);
  const [addingProject, setAddingProject] = useState(false);
  const [pendingProjectAction, setPendingProjectAction] = useState<NavigationProjectAction | null>(null);
  const consumeProjectAction = useCallback(() => setPendingProjectAction(null), []);
  const [sessionNavigationVersion, setSessionNavigationVersion] = useState(0);
  const [activeNavigationSession, setActiveNavigationSession] = useState<NavigationSession | null>(null);
  const [activeNavigationClaim, setActiveNavigationClaim] = useState<SessionClaim | null>(null);
  const [memoryTarget, setMemoryTarget] = useState<WorkspaceMemoryTarget | null>(null);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const expectedRestartRef = useRef(false);
  const setRestartExpectation = useCallback((expected: boolean) => { expectedRestartRef.current = expected; }, []);
  const [navigationDraftSessions, setNavigationDraftSessions] = useState<NavigationSession[]>([]);
  const [pendingSessionAction, setPendingSessionAction] = useState<PendingSessionAction | null>(null);
  const releasedEmptyDraftSessionsRef = useRef(new Set<string>());
  const [projectSessionRequest, setProjectSessionRequest] = useState<{
    projectId: string;
    requestId: number;
  } | null>(null);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
  const [confirmationNotice, setConfirmationNotice] = useState<string | null>(null);
  const [browserRecovery, setBrowserRecovery] = useState<BrowserRecoverySnapshot | null>(null);
  const [browserRecoveryNotice, setBrowserRecoveryNotice] = useState<string | null>(null);
  const [settingsVisited, setSettingsVisited] = useState(location.pathname === "/settings");
  const mainContentRef = useRef<HTMLElement | null>(null);
  const serviceClientRef = useRef<WebServiceClient | null>(null);
  const eventListenersRef = useRef(new Set<ServiceEventListener>());
  const pendingConfirmationRef = useRef<PendingConfirmation | null>(null);
  const confirmationTriggerRef = useRef<HTMLElement | null>(null);
  const nextProjectSessionRequestRef = useRef(0);
  const resolvingConfirmationTokenRef = useRef<string | null>(null);
  const confirmationNoticeTimerRef = useRef<number | null>(null);
  const browserRecoveryNoticeTimerRef = useRef<number | null>(null);
  const serviceInstanceIdRef = useRef<string | null>(null);
  const consumeRegisteredClient = useCallback(() => setRegisteredClient(null), []);
  const persistBrowserRecovery = useCallback((snapshot: BrowserRecoverySnapshot | null) => {
    if (snapshot === null) {
      clearBrowserRecoverySnapshot();
    } else {
      writeBrowserRecoverySnapshot(snapshot);
    }
    setBrowserRecovery((current) => JSON.stringify(current) === JSON.stringify(snapshot) ? current : snapshot);
  }, []);
  const handleMissingBrowserRecovery = useCallback(() => {
    consumeRegisteredClient();
    clearBrowserRecoverySnapshot();
    setBrowserRecovery(null);
    setBrowserRecoveryNotice("sessions.notFoundError");
    if (browserRecoveryNoticeTimerRef.current !== null) {
      window.clearTimeout(browserRecoveryNoticeTimerRef.current);
    }
    browserRecoveryNoticeTimerRef.current = window.setTimeout(() => {
      setBrowserRecoveryNotice(null);
      browserRecoveryNoticeTimerRef.current = null;
    }, 6000);
    navigateRef.current("/", { replace: true, state: null });
  }, [consumeRegisteredClient]);
  const subscribeServiceEvents = useCallback((listener: ServiceEventListener) => {
    eventListenersRef.current.add(listener);
    return () => eventListenersRef.current.delete(listener);
  }, []);
  const requestAddProject = useCallback(() => {
    setAddProjectRequest((request) => request + 1);
  }, []);
  const requestProjectSession = useCallback((projectId: string) => {
    const requestId = ++nextProjectSessionRequestRef.current;
    setProjectSessionRequest({ projectId, requestId });
    navigate(`/projects/${encodeURIComponent(projectId)}`);
    setSidebarOpen(false);
  }, [navigate]);
  const consumeProjectSessionRequest = useCallback((requestId: number) => {
    setProjectSessionRequest((current) => current?.requestId === requestId ? null : current);
  }, []);
  const consumeAddProjectRequest = useCallback(() => setAddProjectRequest(0), []);
  const requestSessionAction = useCallback((action: NavigationSessionAction) => {
    const requestId = createRequestId();
    setPendingSessionAction({ ...action, requestId });
    setSessionNavigationVersion((version) => version + 1);
    const query = new URLSearchParams({ session: action.sessionId });
    if (action.projectId === null) query.set("directory", action.directory);
    const path = action.projectId === null
      ? `/chat?${query.toString()}`
      : `/projects/${encodeURIComponent(action.projectId)}?${query.toString()}`;
    navigate(path);
    setSidebarOpen(false);
  }, [navigate]);
  const consumeSessionAction = useCallback((requestId: string) => {
    setPendingSessionAction((current) => current?.requestId === requestId ? null : current);
  }, []);
  useEffect(() => {
    const query = new URLSearchParams(location.search);
    setPendingSessionAction((current) => {
      if (current === null) return current;
      const path = current.projectId === null ? "/chat" : `/projects/${encodeURIComponent(current.projectId)}`;
      return location.pathname === path && query.get("session") === current.sessionId
        && (current.projectId !== null || query.get("directory") === current.directory)
        ? current : null;
    });
  }, [location.pathname, location.search]);
  const updateNavigationSession = useCallback((session: NavigationSession | null, claim: SessionClaim | null) => {
    if (session?.draft && releasedEmptyDraftSessionsRef.current.has(session.sessionId ?? "")) return;
    setActiveNavigationSession(session);
    setActiveNavigationClaim(claim);
    if (session?.sessionId === null || session == null) return;
    setNavigationDraftSessions((current) => {
      const otherSessions = current.filter((item) => item.sessionId !== session.sessionId);
      return session.draft ? [...otherSessions, session] : otherSessions;
    });
  }, []);
  const removeNavigationDraft = useCallback((sessionId: string, wasEmptyDraft = false) => {
    if (wasEmptyDraft) releasedEmptyDraftSessionsRef.current.add(sessionId);
    setActiveNavigationSession((current) => current?.sessionId === sessionId ? null : current);
    setNavigationDraftSessions((current) => current.filter((session) => session.sessionId !== sessionId));
  }, []);
  const sendServiceCommand = useCallback(
    (command: ClientCommand): Promise<ServiceCommandResult> => {
      const client = serviceClientRef.current;
      if (client === null) {
        return Promise.reject(new ServiceCommandError(null, false));
      }
      return client.sendCommand(command);
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
    void decideServiceConfirmation(sendServiceCommand, confirmation.token, decision).catch((error: unknown) => {
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
          setConfigurationNeedsSetup(response.configuration.repair_required);
        }
      } catch {
        // SettingsView owns the detailed configuration error state.
      }
    };
    const unsubscribe = subscribeServiceEvents((event) => {
      if (event.type === "config.application") {
        ++sequence;
        setConfigurationNeedsSetup(event.payload.status === "pending-repair");
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
      if (browserRecoveryNoticeTimerRef.current !== null) {
        window.clearTimeout(browserRecoveryNoticeTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const serviceClient = serviceClientRef.current ?? new WebServiceClient({
      onAuthState: setAuthState,
      onConnectionState: setConnectionState,
      onServiceStatus: setServiceStatus,
      onRegisteredClient: setRegisteredClient,
      onServiceInstanceChange: (currentInstanceId, previousInstanceId, initial) => {
        const decision = reconcileBrowserRecovery(readBrowserRecoverySnapshot(), {
          currentInstanceId, previousInstanceId, initial,
          expectedRestart: expectedRestartRef.current,
          location: window.location,
        });
        if (decision.serviceChanged) {
          setMemoryTarget(null);
          expectedRestartRef.current = false;
        }
        serviceInstanceIdRef.current = currentInstanceId;
        if (decision.storage === "write" && decision.snapshot !== null) {
          writeBrowserRecoverySnapshot(decision.snapshot);
        } else if (decision.storage === "clear") {
          clearBrowserRecoverySnapshot();
        }
        setBrowserRecovery(decision.snapshot);
        if (decision.navigation?.kind === "settings-return") {
          const route = new URL(decision.navigation.route, window.location.origin);
          const state = window.history.state?.usr as SettingsNavigationState | null;
          navigateRef.current("/settings", { replace: true, state: {
            ...state,
            returnTo: { ...state?.returnTo, pathname: route.pathname, search: route.search,
              hash: "", scrollTop: state?.returnTo?.scrollTop ?? 0 },
          } satisfies SettingsNavigationState });
        } else if (decision.navigation !== null) {
          navigateRef.current(decision.navigation.route, { replace: true, state: null });
        }
      },
      onConnectionOpen: () => setSessionEventVersion((version) => version + 1),
      onConnectionClose: () => {
        pendingConfirmationRef.current = null;
        resolvingConfirmationTokenRef.current = null;
        setPendingConfirmation(null);
        setSessionEventVersion((version) => version + 1);
      },
      onEvent: (event) => {
        for (const listener of eventListenersRef.current) listener(event);
      },
      onSessionEvent: () => setSessionEventVersion((version) => version + 1),
    }, initialLaunchTicket);
    serviceClientRef.current = serviceClient;
    serviceClient.start();
    return () => {
      serviceClient.close();
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
      ? settingsReturnLocation
      : location;
  const conversationParams = new URLSearchParams(conversationLocation.search);
  const requestNewChat = useCallback(() => {
    persistBrowserRecovery(null);
    const alreadyOnNewChat = conversationLocation.pathname === "/" && conversationLocation.search === "";
    navigate("/", { state: null });
    if (alreadyOnNewChat) setNewChatVersion((version) => version + 1);
  }, [conversationLocation.pathname, conversationLocation.search, navigate, persistBrowserRecovery]);
  const openSettings = (event: React.MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    if (location.pathname === "/settings") return;
    const recovery = readBrowserRecoverySnapshot();
    const viewport = mainContentRef.current?.querySelector<HTMLElement>('[role="log"]');
    if (recovery !== null && recovery.session_id !== null && viewport !== null && viewport !== undefined) {
      persistBrowserRecovery({ ...recovery, scroll_top: viewport.scrollTop });
    }
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
    const recovery = readBrowserRecoverySnapshot();
    const recoveredRoute = recovery !== null && recovery.service_instance_id === serviceInstanceIdRef.current
      ? new URL(browserRecoveryRoute(recovery), window.location.origin) : null;
    const targetSession = new URLSearchParams(target.search).get("session");
    const route = recoveredRoute !== null && recoveredRoute.pathname === target.pathname
      && targetSession !== null && targetSession !== recovery?.session_id
        ? `${recoveredRoute.pathname}${recoveredRoute.search}`
        : `${target.pathname}${target.search}${target.hash}`;
    navigate(route, { replace: true });
    window.requestAnimationFrame(() => {
      if (mainContentRef.current !== null) {
        mainContentRef.current.scrollTop = target.scrollTop;
        const log = mainContentRef.current.querySelector<HTMLElement>('[role="log"]');
        if (log !== null && target.conversationScrollTop !== undefined) log.scrollTop = target.conversationScrollTop;
      }
    });
  };
  return (
    <div className={location.pathname === "/settings" ? `${styles.appShell} ${styles.appShellSettings}` : styles.appShell}>
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
        <NavigationSidebar
          authReady={authState === "ready"}
          projects={projects}
          projectsLoadState={projectsLoadState}
          projectsError={projectsError}
          refreshVersion={sessionEventVersion}
          activeSession={activeNavigationSession}
          draftSessions={navigationDraftSessions}
          onRefreshProjects={() => void refreshProjects()}
          onNewChat={requestNewChat}
          onSessionNavigation={() => {
            setPendingSessionAction(null);
            setSessionNavigationVersion((version) => version + 1);
          }}
          onAddProject={requestAddProject}
          addingProject={addingProject}
          projectSelectionReady={connectionState === "online"}
          onProjectAction={setPendingProjectAction}
          onNewProjectSession={requestProjectSession}
          onSessionAction={requestSessionAction}
          onClose={() => setSidebarOpen(false)}
          onOpenMemory={(projectId, trigger) => {
            if (projectId !== null) {
              const project = projects.find((item) => item.project_id === projectId);
              if (project !== undefined) setMemoryTarget({ projectId, title: project.path, trigger });
            } else if (activeNavigationClaim !== null) {
              setMemoryTarget({ projectId: null, claim: activeNavigationClaim,
                title: activeNavigationSession?.directory ?? t("nav.chatHistory"), trigger });
            }
            setMemoryOpen(true);
          }}
        />
        <div className={styles.sidebarFooter}>
          <NavLink
            className={({ isActive }) => isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink}
            to="/settings"
            onClick={openSettings}
          >
            <Settings2 size={16} aria-hidden="true" />
            <span>{t("nav.settings")}</span>
          </NavLink>
        </div>
      </aside>

      {memoryTarget !== null ? (
        <WorkspaceMemoryDialog
          key={memoryTarget.projectId ?? `${memoryTarget.claim.workspace_id}:${memoryTarget.claim.session_id}:${memoryTarget.claim.claim_version}`}
          target={memoryTarget}
          open={memoryOpen}
          online={connectionState === "online"}
          onClose={() => setMemoryOpen(false)}
        />
      ) : null}

      <div className={styles.mainColumn}>
        <button
          className={`${styles.sidebarToggle} ${styles.floatingNavigationToggle}`}
          id="app-sidebar-toggle"
          type="button"
          aria-label={sidebarOpen ? t("controls.closeNavigation") : t("controls.openNavigation")}
          aria-controls="app-sidebar"
          aria-expanded={sidebarOpen}
          onClick={() => setSidebarOpen((open) => !open)}
        >
          {sidebarOpen ? <X size={18} aria-hidden="true" /> : <Menu size={18} aria-hidden="true" />}
        </button>

        <main
          id="main-content"
          className={isProjectSessionRoute || isChatRoute ? `${styles.mainContent} ${styles.conversationMainContent}` : styles.mainContent}
          ref={mainContentRef}
          tabIndex={-1}
        >
          {authState === "conflict" ? (
            <div className={styles.errorBanner} role="alert">
              <CircleAlert size={16} aria-hidden="true" />
              <span>{t("status.webClientExists")}</span>
            </div>
          ) : null}
          <div style={{ display: location.pathname === "/settings" ? "none" : "contents" }}>
            <Routes location={conversationLocation}>
              <Route
                path="/"
                element={
                  <ChatSessionsView
                        key={serviceStatus?.service_instance_id}
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
                        browserRecovery={browserRecovery}
                        serviceInstanceId={serviceStatus?.service_instance_id ?? null}
                        onBrowserRecoveryChange={persistBrowserRecovery}
                        onBrowserRecoveryUnavailable={handleMissingBrowserRecovery}
                        newChatVersion={newChatVersion}
                        initialSessionId={conversationParams.get("session")}
                        initialDirectory={conversationParams.get("directory")}
                        navigationRequestKey={String(sessionNavigationVersion)}
                        onNavigationSessionChange={updateNavigationSession}
                        onNavigationDraftReleased={removeNavigationDraft}
                        sessionActionRequest={pendingSessionAction}
                        onSessionActionConsumed={consumeSessionAction}
                      />
                }
              />
              <Route
                path="/chat"
                element={
                  <ChatSessionsView
                        key={serviceStatus?.service_instance_id}
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
                        browserRecovery={browserRecovery}
                        serviceInstanceId={serviceStatus?.service_instance_id ?? null}
                        onBrowserRecoveryChange={persistBrowserRecovery}
                        onBrowserRecoveryUnavailable={handleMissingBrowserRecovery}
                        newChatVersion={newChatVersion}
                        initialSessionId={conversationParams.get("session")}
                        initialDirectory={conversationParams.get("directory")}
                        navigationRequestKey={String(sessionNavigationVersion)}
                        onNavigationSessionChange={updateNavigationSession}
                        onNavigationDraftReleased={removeNavigationDraft}
                        sessionActionRequest={pendingSessionAction}
                        onSessionActionConsumed={consumeSessionAction}
                      />
                }
              />
              <Route
                path="/chat/schedule"
                element={
                  <ScheduleJobsView
                    authState={authState}
                    connectionState={connectionState}
                    projects={projects}
                  />
                }
              />
              <Route
                path="/chat/schedule/jobs/:jobId/history"
                element={
                  <ScheduleJobHistoryView
                    authState={authState}
                    connectionState={connectionState}
                    projects={projects}
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
              <Route path="/projects" element={<Navigate replace to="/" />} />
              <Route
                path="/projects/:projectId"
                element={
                  <ProjectSessionsView
                    key={serviceStatus?.service_instance_id}
                    authState={authState}
                    connectionState={connectionState}
                    projects={projects}
                    registeredClient={registeredClient}
                    onRestoreConsumed={consumeRegisteredClient}
                    refreshVersion={sessionEventVersion}
                    sendServiceCommand={sendServiceCommand}
                    subscribeServiceEvents={subscribeServiceEvents}
                    confirmationTriggerRef={confirmationTriggerRef}
                    browserRecovery={browserRecovery}
                    serviceInstanceId={serviceStatus?.service_instance_id ?? null}
                    onBrowserRecoveryChange={persistBrowserRecovery}
                    onBrowserRecoveryUnavailable={handleMissingBrowserRecovery}
                    projectSessionRequest={projectSessionRequest}
                    onProjectSessionRequestConsumed={consumeProjectSessionRequest}
                    navigationRequestKey={String(sessionNavigationVersion)}
                    onNavigationSessionChange={updateNavigationSession}
                    onNavigationDraftReleased={removeNavigationDraft}
                    sessionActionRequest={pendingSessionAction}
                    onSessionActionConsumed={consumeSessionAction}
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
                onRestartExpectation={setRestartExpectation}
                serviceStatus={serviceStatus}
                theme={theme}
                runtimeClaim={activeNavigationClaim}
              />
            </div>
          ) : null}
        </main>
        {browserRecoveryNotice !== null ? (
          <div className={styles.confirmationNotice} role="status" aria-live="polite">
            <CircleAlert size={16} aria-hidden="true" />
            {t(browserRecoveryNotice)}
          </div>
        ) : null}
        {confirmationNotice !== null && pendingConfirmation === null ? (
          <div className={styles.confirmationNotice} role="status" aria-live="polite">
            <CircleCheck size={16} aria-hidden="true" />
            {t(confirmationNotice)}
          </div>
        ) : null}
      </div>
      <ProjectActions
        authState={authState}
        connectionState={connectionState}
        openRegistrationRequest={addProjectRequest}
        onRegistrationRequestConsumed={consumeAddProjectRequest}
        onAddingChange={setAddingProject}
        onRetryRegistration={requestAddProject}
        actionRequest={pendingProjectAction}
        onActionRequestConsumed={consumeProjectAction}
        onRefresh={refreshProjects}
        projects={projects}
        subscribeServiceEvents={subscribeServiceEvents}
      />
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

function ProjectActions({
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

function readString(value: unknown): string | null {
  return isNonEmptyString(value) ? value : null;
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

function readAndClearTicket(): string | null {
  const rawHash = window.location.hash.slice(1);
  const ticket = new URLSearchParams(rawHash).get("ticket");
  if (ticket !== null) {
    window.history.replaceState({}, document.title, `${window.location.pathname}${window.location.search}`);
  }
  return ticket;
}
