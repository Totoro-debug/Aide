import { useCallback, useDeferredValue, useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight, MessageSquare, Plus, RefreshCw, Search } from "lucide-react";

import { getChatSessions, getProjectSessions } from "./api";
import type {
  ChatSessionSummary,
  ChatSessionsResponse,
  ProjectSessionsResponse,
  RegisteredProject,
  SessionSummary,
} from "./protocol";
import styles from "./App.module.css";

const EXPANDED_PROJECTS_KEY = "omni.sidebar.expanded-projects";
const SESSION_PAGE_SIZE = 100;

type ProjectsLoadState = "idle" | "loading" | "ready" | "error";

export interface NavigationSession {
  projectId: string | null;
  directory: string | null;
  sessionId: string | null;
  draft: boolean;
  running?: boolean;
}

interface NavigationSidebarProps {
  authReady: boolean;
  projects: RegisteredProject[];
  projectsLoadState: ProjectsLoadState;
  projectsError: string | null;
  refreshVersion: number;
  activeSession: NavigationSession | null;
  draftSessions: NavigationSession[];
  onRefreshProjects: () => void;
  onNewChat: () => void;
  onSessionNavigation: () => void;
  onAddProject: () => void;
  onNewProjectSession: (projectId: string) => void;
  onClose: () => void;
}

export default function NavigationSidebar({
  authReady,
  projects,
  projectsLoadState,
  projectsError,
  refreshVersion,
  activeSession,
  draftSessions,
  onRefreshProjects,
  onNewChat,
  onSessionNavigation,
  onAddProject,
  onNewProjectSession,
  onClose,
}: NavigationSidebarProps) {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const [expandedProjectIds, setExpandedProjectIds] = useState<string[]>(readExpandedProjectIds);
  const [chatHistory, setChatHistory] = useState<ChatSessionsResponse | null>(null);
  const [chatHistoryLoadState, setChatHistoryLoadState] = useState<ProjectsLoadState>("idle");
  const [chatHistoryError, setChatHistoryError] = useState<string | null>(null);
  const chatHistoryRequestRef = useRef(0);
  const loadedChatCountRef = useRef(0);
  const projectRouteMatch = location.pathname.match(/^\/projects\/([^/]+)/);
  const selectedProjectId = projectRouteMatch === null ? null : decodeURIComponent(projectRouteMatch[1]);
  const searchParams = new URLSearchParams(location.search);
  const selectedSessionId = activeSession?.sessionId ?? null;
  const selectedDirectory = activeSession?.directory ?? searchParams.get("directory");
  const isChatRoute = location.pathname === "/" || location.pathname === "/chat";

  const refreshChatHistory = useCallback(async (cursor: string | null = null) => {
    if (!authReady) return;
    const requestNumber = ++chatHistoryRequestRef.current;
    setChatHistoryLoadState("loading");
    setChatHistoryError(null);
    try {
      let response = await getChatSessions({ cursor: cursor ?? undefined, limit: SESSION_PAGE_SIZE });
      while (cursor === null && response.next_cursor !== null
        && response.sessions.length < loadedChatCountRef.current) {
        if (requestNumber !== chatHistoryRequestRef.current) return;
        const page = await getChatSessions({ cursor: response.next_cursor, limit: SESSION_PAGE_SIZE });
        response = { ...page, sessions: [...response.sessions, ...page.sessions] };
      }
      if (requestNumber !== chatHistoryRequestRef.current) return;
      setChatHistory((current) => {
        const combined = cursor === null || current === null ? response.sessions
          : [...current.sessions, ...response.sessions];
        const sessions = Array.from(new Map(combined.map((session) => [`${session.directory}:${session.id}`, session])).values());
        loadedChatCountRef.current = sessions.length;
        return { ...response, sessions };
      });
      setChatHistoryLoadState("ready");
    } catch {
      if (requestNumber !== chatHistoryRequestRef.current) return;
      setChatHistoryLoadState("error");
      setChatHistoryError("chat.historyError");
    }
  }, [authReady]);

  useEffect(() => {
    if (authReady) void refreshChatHistory();
  }, [authReady, refreshVersion, refreshChatHistory]);

  useEffect(() => {
    try {
      sessionStorage.setItem(EXPANDED_PROJECTS_KEY, JSON.stringify(expandedProjectIds));
    } catch {
      // Navigation remains usable when browser storage is unavailable.
    }
  }, [expandedProjectIds]);

  useEffect(() => {
    if (projectsLoadState !== "ready") return;
    const registeredIds = new Set(projects.map((project) => project.project_id));
    setExpandedProjectIds((current) => current.filter((id) => registeredIds.has(id)));
  }, [projects, projectsLoadState]);

  useEffect(() => {
    if (selectedProjectId === null || !projects.some((project) => project.project_id === selectedProjectId)) return;
    setExpandedProjectIds((current) => current.includes(selectedProjectId) ? current : [...current, selectedProjectId]);
  }, [projects, selectedProjectId]);

  const toggleProject = useCallback((projectId: string) => {
    setExpandedProjectIds((current) => current.includes(projectId)
      ? current.filter((id) => id !== projectId)
      : [...current, projectId]);
  }, []);

  function openChatSession(session: ChatSessionSummary) {
    onSessionNavigation();
    const query = new URLSearchParams({ directory: session.directory, session: session.id });
    navigate(`/chat?${query.toString()}`);
    onClose();
  }

  function openProjectSession(projectId: string, sessionId: string) {
    onSessionNavigation();
    const query = new URLSearchParams({ session: sessionId });
    navigate(`/projects/${encodeURIComponent(projectId)}?${query.toString()}`);
    onClose();
  }

  function openProject(projectId: string) {
    navigate(`/projects/${encodeURIComponent(projectId)}`);
    onClose();
  }

  const activeChatDirectory = isChatRoute ? selectedDirectory : null;

  return (
    <div className={styles.sidebarContents}>
      <nav className={styles.navigation} aria-label={t("app.name")}>
        <Link
          className={styles.navLink}
          to="/"
          onClick={(event) => {
            event.preventDefault();
            onNewChat();
            onClose();
          }}
        >
          <MessageSquare size={16} aria-hidden="true" />
          <span>{t("nav.newChat")}</span>
        </Link>
      </nav>

      <section className={styles.sidebarSection} aria-label={t("nav.projects")}>
        <div className={styles.sidebarSectionHeader}>
          <h2 className={styles.sidebarSectionTitle}>
            <Link to="/projects" onClick={onClose}>{t("nav.projects")}</Link>
          </h2>
          <button
            className={styles.sidebarIconButton}
            type="button"
            aria-label={t("controls.addProject")}
            title={t("controls.addProject")}
            disabled={!authReady}
            onClick={onAddProject}
          >
            <Plus size={16} aria-hidden="true" />
          </button>
        </div>
        {projectsLoadState === "loading" && projects.length === 0 ? (
          <p className={styles.sidebarStatus} role="status">{t("projects.loading")}</p>
        ) : null}
        {projectsLoadState === "error" && projects.length === 0 ? (
          <div className={styles.sidebarStatus} role="alert">
            <span>{t(projectsError ?? "projects.loadError")}</span>
            <button className={styles.sidebarTextButton} type="button" onClick={onRefreshProjects}>
              {t("controls.retry")}
            </button>
          </div>
        ) : null}
        {projectsLoadState === "ready" && projects.length === 0 ? (
          <p className={styles.sidebarStatus}>{t("projects.emptyTitle")}</p>
        ) : null}
        {projects.length > 0 ? (
          <ul className={styles.projectNavigation} aria-label={t("nav.projects")}>
            {projects.map((project) => (
              <ProjectNavigationItem
                key={project.project_id}
                project={project}
                authReady={authReady}
                expanded={expandedProjectIds.includes(project.project_id)}
                activeSessionId={selectedProjectId === project.project_id ? selectedSessionId : null}
                refreshVersion={refreshVersion}
                draftSessions={draftSessions.filter((session) => session.projectId === project.project_id)}
                onToggle={() => toggleProject(project.project_id)}
                onOpenProject={() => openProject(project.project_id)}
                onNewSession={() => onNewProjectSession(project.project_id)}
                onOpenSession={(sessionId) => openProjectSession(project.project_id, sessionId)}
              />
            ))}
          </ul>
        ) : null}
      </section>

      <section className={`${styles.sidebarSection} ${styles.conversationNavigationSection}`}>
        <h2 className={styles.sidebarSectionTitle}>{t("nav.chatHistory")}</h2>
        {chatHistoryLoadState === "loading" && chatHistory === null ? (
          <p className={styles.sidebarStatus} role="status">{t("chat.historyLoading")}</p>
        ) : null}
        {chatHistoryError !== null ? (
          <p className={styles.sidebarError} role="alert">{t(chatHistoryError)}</p>
        ) : null}
        {chatHistoryLoadState === "ready" && chatHistory?.sessions.length === 0 ? (
          <p className={styles.sidebarStatus}>{t("chat.historyEmpty")}</p>
        ) : null}
        <nav className={styles.chatHistoryNavigation} aria-label={t("nav.chatHistory")}>
          <ul className={styles.chatHistoryList} aria-label={t("nav.chatHistory")}>
            {(chatHistory?.sessions ?? []).map((session) => {
              const active = isChatRoute && session.id === selectedSessionId
                && session.directory === activeChatDirectory;
              return (
                <li key={`${session.directory}:${session.id}`}>
                  <button
                    className={styles.chatHistoryItem}
                    type="button"
                    disabled={!session.available}
                    data-active={active}
                    aria-current={active ? "page" : undefined}
                    title={session.directory}
                    onClick={() => openChatSession(session)}
                  >
                    <span>{session.title}</span>
                    <small>{session.directory}</small>
                  </button>
                </li>
              );
            })}
          </ul>
          {chatHistory !== null && chatHistory.unavailable_directories.length > 0 ? (
            <p className={styles.sidebarStatus} role="status" aria-atomic="true">
              {t("chat.historyDirectoriesUnavailable", {
                directories: chatHistory.unavailable_directories.join("; "),
              })}
            </p>
          ) : null}
          {chatHistory?.next_cursor !== null && chatHistory !== null ? (
            <button
              className={styles.chatHistoryMore}
              type="button"
              disabled={chatHistoryLoadState === "loading"}
              onClick={() => void refreshChatHistory(chatHistory.next_cursor)}
            >
              {t("sessions.loadMore")}
            </button>
          ) : null}
          <button
            className={styles.chatHistoryRefresh}
            type="button"
            disabled={!authReady || chatHistoryLoadState === "loading"}
            onClick={() => void refreshChatHistory()}
          >
            <RefreshCw size={14} aria-hidden="true" />
            {t("controls.refreshSessions")}
          </button>
        </nav>
      </section>
    </div>
  );
}

function ProjectNavigationItem({
  project,
  authReady,
  expanded,
  activeSessionId,
  refreshVersion,
  draftSessions,
  onToggle,
  onOpenProject,
  onNewSession,
  onOpenSession,
}: {
  project: RegisteredProject;
  authReady: boolean;
  expanded: boolean;
  activeSessionId: string | null;
  refreshVersion: number;
  draftSessions: NavigationSession[];
  onToggle: () => void;
  onOpenProject: () => void;
  onNewSession: () => void;
  onOpenSession: (sessionId: string) => void;
}) {
  const { t } = useTranslation();
  const [sessionSearch, setSessionSearch] = useState("");
  const deferredSessionSearch = useDeferredValue(sessionSearch);
  const [sessions, setSessions] = useState<ProjectSessionsResponse | null>(null);
  const [loadState, setLoadState] = useState<ProjectsLoadState>("idle");
  const [loadError, setLoadError] = useState(false);
  const requestRef = useRef(0);
  const loadedSessionsRef = useRef({ title: "", count: 0 });
  const sessionListId = `project-sessions-${encodeURIComponent(project.project_id)}`;

  const refreshSessions = useCallback(async (cursor: string | null = null, append = false) => {
    if (!project.available) return;
    const requestNumber = ++requestRef.current;
    setLoadState("loading");
    setLoadError(false);
    const title = deferredSessionSearch.trim();
    try {
      let response = await getProjectSessions(project.project_id, {
        title: title || undefined,
        cursor: cursor ?? undefined,
        limit: SESSION_PAGE_SIZE,
      });
      while (!append && response.next_cursor !== null
        && loadedSessionsRef.current.title === title
        && response.sessions.length < loadedSessionsRef.current.count) {
        if (requestNumber !== requestRef.current) return;
        const page = await getProjectSessions(project.project_id, {
          title: title || undefined, cursor: response.next_cursor, limit: SESSION_PAGE_SIZE,
        });
        response = { ...page, sessions: [...response.sessions, ...page.sessions] };
      }
      if (requestNumber !== requestRef.current) return;
      setSessions((current) => {
        const combined = append && current !== null ? [...current.sessions, ...response.sessions] : response.sessions;
        const sessions = Array.from(new Map(combined.map((session) => [session.id, session])).values());
        loadedSessionsRef.current = { title, count: sessions.length };
        return { ...response, sessions };
      });
      setLoadState("ready");
    } catch {
      if (requestNumber !== requestRef.current) return;
      setLoadState("error");
      setLoadError(true);
    }
  }, [deferredSessionSearch, project.available, project.project_id]);

  useEffect(() => {
    if (expanded && project.available) void refreshSessions();
  }, [expanded, project.available, refreshSessions, refreshVersion]);

  const projectName = project.name || project.path;
  const toggleLabel = t(expanded ? "nav.collapseProjectSessions" : "nav.expandProjectSessions", {
    project: projectName,
  });

  return (
    <li className={styles.projectNavigationItem}>
      <div className={styles.projectNavigationRow}>
        <button
          className={styles.sidebarIconButton}
          type="button"
          aria-label={toggleLabel}
          title={toggleLabel}
          aria-expanded={expanded}
          aria-controls={sessionListId}
          onClick={onToggle}
        >
          {expanded ? <ChevronDown size={15} aria-hidden="true" /> : <ChevronRight size={15} aria-hidden="true" />}
        </button>
        <button
          className={styles.projectNavigationLink}
          type="button"
          aria-current={activeSessionId !== null ? "page" : undefined}
          onClick={onOpenProject}
        >
          <span className={styles.projectNavigationDot} data-available={project.available} />
          <span>{projectName}</span>
        </button>
        <button
          className={styles.sidebarIconButton}
          type="button"
          aria-label={t("nav.newProjectSession", { project: projectName })}
          title={t("nav.newProjectSession", { project: projectName })}
          disabled={!authReady || !project.available}
          onClick={onNewSession}
        >
          <Plus size={15} aria-hidden="true" />
        </button>
      </div>
      {expanded ? (
        <div className={styles.projectSessionTree} id={sessionListId}>
          {project.available ? (
            <label className={styles.sidebarSearch}>
              <Search size={13} aria-hidden="true" />
              <span className={styles.srOnly}>{t("sessions.searchLabel")}</span>
              <input
                type="search"
                aria-label={t("sessions.searchLabel")}
                placeholder={t("sessions.searchPlaceholder")}
                value={sessionSearch}
                onChange={(event) => setSessionSearch(event.target.value)}
              />
            </label>
          ) : (
            <p className={styles.sidebarStatus} role="status">{t("sessions.projectUnavailable")}</p>
          )}
          {loadState === "loading" && sessions === null ? (
            <p className={styles.sidebarStatus} role="status">{t("sessions.loading")}</p>
          ) : null}
          {loadError ? (
            <div className={styles.sidebarError} role="alert">
              <span>{t("sessions.loadError")}</span>
              <button className={styles.sidebarTextButton} type="button" onClick={() => void refreshSessions()}>
                {t("controls.retry")}
              </button>
            </div>
          ) : null}
          {loadState === "ready" && (sessions?.sessions.length ?? 0) === 0 && draftSessions.length === 0 ? (
            <p className={styles.sidebarStatus}>
              {sessionSearch.trim() ? t("sessions.noSearchResults") : t("sessions.empty")}
            </p>
          ) : null}
          {sessions !== null || draftSessions.length > 0 ? (
            <ul className={styles.projectSessionList} aria-label={`${projectName} ${t("nav.sessions")}`}>
              {draftSessions.filter((draft) => !sessions?.sessions.some((session) => session.id === draft.sessionId))
                .map((draft) => (
                  <li key={draft.sessionId}>
                    <button
                      className={styles.projectSessionItem}
                      type="button"
                      aria-current={activeSessionId === draft.sessionId ? "page" : undefined}
                      data-active={activeSessionId === draft.sessionId}
                      onClick={() => { if (draft.sessionId !== null) onOpenSession(draft.sessionId); }}
                    >
                      <MessageSquare size={13} aria-hidden="true" />
                      <span>{t(draft.running ? "sessions.draftTitle" : "sessions.draft")}</span>
                      {draft.running ? <small className={styles.sessionRunBadge}>{t("conversation.running")}</small> : null}
                    </button>
                  </li>
                ))}
              {(sessions?.sessions ?? []).map((session: SessionSummary) => (
                <li key={session.id}>
                  <button
                    className={styles.projectSessionItem}
                    type="button"
                    aria-current={activeSessionId === session.id ? "page" : undefined}
                    data-active={activeSessionId === session.id}
                    onClick={() => onOpenSession(session.id)}
                  >
                    <MessageSquare size={13} aria-hidden="true" />
                    <span>{session.title}</span>
                    {session.occupied ? (
                      <small className={styles.occupiedBadge}>
                        {t(session.occupied_by === "client" ? "sessions.occupied" : "sessions.occupiedHere")}
                      </small>
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
          {sessions?.next_cursor !== null && sessions !== null ? (
            <button
              className={styles.chatHistoryMore}
              type="button"
              disabled={loadState === "loading"}
              onClick={() => void refreshSessions(sessions.next_cursor, true)}
            >
              {t("sessions.loadMore")}
            </button>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

function readExpandedProjectIds(): string[] {
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem(EXPANDED_PROJECTS_KEY) ?? "[]");
    return Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}
