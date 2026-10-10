import {
  Activity,
  CircleAlert,
  CircleCheck,
  Menu,
  Settings2,
  X
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { ConfirmationDialog } from "../features/confirmations/ConfirmationDialog.tsx";
import type { PendingConfirmation } from "../features/confirmations/presentation.ts";
import { parseConfirmationEvent } from "../features/confirmations/presentation.ts";
import type { BrowserRecoverySnapshot } from "../features/conversations/browserRecovery";
import {
  browserRecoveryRoute,
  clearBrowserRecoverySnapshot,
  readBrowserRecoverySnapshot,
  reconcileBrowserRecovery,
  writeBrowserRecoverySnapshot,
} from "../features/conversations/browserRecovery";
import { ConversationDraftsProvider } from "../features/conversations/ConversationDraftsProvider";
import { ChatSessionsView, ProjectSessionsView } from "../features/conversations/ConversationsView.tsx";
import type { PendingSessionAction } from "../features/conversations/session.ts";
import { StatusView } from "../features/runtime/RuntimeView.tsx";
import type { WorkspaceMemoryTarget } from "../features/runtime/WorkspaceMemoryDialog";
import WorkspaceMemoryDialog from "../features/runtime/WorkspaceMemoryDialog";
import { ScheduleJobHistoryView, ScheduleJobsView } from "../features/schedules/SchedulesView.tsx";
import type { SettingsNavigationState, SettingsReturnLocation } from "../features/settings/navigation.ts";
import { SettingsView } from "../features/settings/SettingsView.tsx";
import {
  ServiceCommandError,
  createRequestId,
  decideServiceConfirmation,
  getConfig,
  getProjects
} from "../shared/service/api";
import type {
  ClientCommand,
  RegisteredClient,
  RegisteredProject,
  ServiceCommandResult,
  ServiceEvent,
  ServiceStatus,
  SessionClaim
} from "../shared/service/protocol";
import { WebServiceClient } from "../shared/service/serviceClient";
import type { AuthState, ConnectionState, ServiceEventListener } from "../shared/service/types.ts";
import commonStyles from "../shared/styles/controls.module.css";
import moduleStyles from "./App.module.css";
import type { NavigationProjectAction, NavigationSession, NavigationSessionAction } from "./NavigationSidebar";
import NavigationSidebar from "./NavigationSidebar";
import { ProjectActions } from "./ProjectActions.tsx";
import { projectErrorKey } from "./projects.ts";
import type { Theme } from "./theme.ts";
import { applyTheme, readThemePreference } from "./theme.ts";
import { usePanelKeyboard } from "./usePanelKeyboard.ts";

const styles = { ...commonStyles, ...moduleStyles };

type ProjectsLoadState = "idle" | "loading" | "ready" | "error";

const initialLaunchTicket = readAndClearTicket();

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

function readAndClearTicket(): string | null {
  const rawHash = window.location.hash.slice(1);
  const ticket = new URLSearchParams(rawHash).get("ticket");
  if (ticket !== null) {
    window.history.replaceState({}, document.title, `${window.location.pathname}${window.location.search}`);
  }
  return ticket;
}
