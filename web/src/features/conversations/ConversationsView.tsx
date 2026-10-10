import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowUp,
  Check,
  CircleAlert,
  CircleCheck,
  FolderOpen,
  Info,
  MessageSquare,
  Plus,
  RotateCcw,
  Send,
  Square,
  Trash2,
  TriangleAlert,
  X
} from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import type { NavigationSession } from "../../app/NavigationSidebar";
import {
  ApiError,
  ServiceCommandError,
  acknowledgeRestore,
  cancelConversationRun,
  cancelRestore,
  claimProjectSessionDeletion,
  claimWorkspaceSessionDeletion,
  configureConversationModel,
  createRequestId,
  deleteProjectSession,
  deleteWorkspaceSession,
  executeRestore,
  getAvailableModels,
  getProjectSession,
  getProjectSessionDeletionStatus,
  getProjectSessions,
  getRestoreResult,
  getRuntimeStatus,
  getWorkspaceSession,
  getWorkspaceSessionDeletionStatus,
  getWorkspaceSessions,
  inspectRestore,
  openConversation,
  recallQueuedInputs,
  releaseProjectSession,
  releaseWorkspaceSession,
  renameProjectSession,
  renameWorkspaceSession,
  submitUserInput,
  subscribeState,
  updateRuntimePermission
} from "../../shared/service/api";
import { PERMISSION_LEVELS } from "../../shared/service/permissions.ts";
import type {
  AvailableModelsResponse,
  ClientCommand,
  ConversationOpenResponse,
  ProjectSessionsResponse,
  ReasoningEffort,
  RegisteredClient,
  RegisteredProject,
  RestoreMode,
  RestorePlan,
  RestoreResult,
  ServiceCommandResult,
  ServiceEvent,
  SessionClaim,
  SessionModelConfiguration,
  SessionSnapshot,
  SessionSummary,
  ToolPermissionLevel,
  WorkspaceSessionsResponse
} from "../../shared/service/protocol";
import type { AuthState, ConnectionState, ServiceEventListener } from "../../shared/service/types.ts";
import commonStyles from "../../shared/styles/controls.module.css";
import { ConversationRuntimeStatus, RuntimeManagementDialog } from "../runtime/RuntimeView";
import ComposerControls from "./ComposerControls";
import conversationsStyles from "./Conversations.module.css";
import SubAgentPanel from "./SubAgentPanel";
import type { BrowserRecoverySnapshot, SessionBrowserRecoverySnapshot } from "./browserRecovery";
import {
  readBrowserRecoverySnapshot,
  writeBrowserRecoverySnapshot
} from "./browserRecovery";
import { useConversationDrafts } from "./drafts";
import { ConversationHistoryView, LiveRunView } from "./history.tsx";
import { REASONING_EFFORTS } from "./reasoningEffort.ts";
import type { LiveRun, RunStatus } from "./run.ts";
import { isLiveRunActive, newLiveRun, reduceLiveRunEvent } from "./run.ts";
import type { PendingSessionAction, PendingSessionDeletion, PendingSubmission, SessionLoadState } from "./session.ts";
import { formatSessionTime, mergeSessionSummaries, readPendingDeletion, restoreSessionActionFocus, sessionDeleteErrorKey, sessionErrorKey } from "./session.ts";

const styles = { ...commonStyles, ...conversationsStyles };

interface ProjectSessionsViewProps {
  authState: AuthState;
  connectionState: ConnectionState;
  projects: RegisteredProject[];
  registeredClient: RegisteredClient | null;
  browserRecovery: BrowserRecoverySnapshot | null;
  serviceInstanceId: string | null;
  onRestoreConsumed: () => void;
  onBrowserRecoveryChange: (snapshot: BrowserRecoverySnapshot | null) => void;
  onBrowserRecoveryUnavailable: () => void;
  refreshVersion: number;
  sendServiceCommand: (command: ClientCommand) => Promise<ServiceCommandResult>;
  subscribeServiceEvents: (listener: ServiceEventListener) => () => void;
  confirmationTriggerRef: { current: HTMLElement | null };
  projectSessionRequest?: { projectId: string; requestId: number } | null;
  onProjectSessionRequestConsumed?: (requestId: number) => void;
  sessionActionRequest?: PendingSessionAction | null;
  onSessionActionConsumed?: (requestId: string) => void;
  onNavigationSessionChange?: (session: NavigationSession | null, claim: SessionClaim | null) => void;
  onNavigationDraftReleased?: (sessionId: string, wasEmptyDraft?: boolean) => void;
  navigationRequestKey?: string;
}

export function ChatSessionsView({
  authState,
  connectionState,
  projects,
  registeredClient,
  browserRecovery,
  serviceInstanceId,
  onRestoreConsumed,
  onBrowserRecoveryChange,
  onBrowserRecoveryUnavailable,
  refreshVersion,
  sendServiceCommand,
  subscribeServiceEvents,
  confirmationTriggerRef,
  newChatVersion,
  configurationNeedsSetup,
  initialSessionId: requestedSessionId,
  initialDirectory: requestedDirectory,
  navigationRequestKey,
  onNavigationSessionChange,
  onNavigationDraftReleased,
  sessionActionRequest,
  onSessionActionConsumed,
}: ProjectSessionsViewProps & {
  newChatVersion: number;
  configurationNeedsSetup: boolean | null;
  initialSessionId: string | null;
  initialDirectory: string | null;
  navigationRequestKey: string;
}) {
  const { t } = useTranslation();
  const [setupInput, setSetupInput] = useState("");
  const [workspaceEntry, setWorkspaceEntry] = useState<ConversationOpenResponse | null>(null);
  const [initialSessionId, setInitialSessionId] = useState<string | null>(null);
  const [entryLoadState, setEntryLoadState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [entryError, setEntryError] = useState<string | null>(null);
  const activationSequenceRef = useRef(0);

  useEffect(() => {
    setSetupInput("");
  }, [newChatVersion, serviceInstanceId]);

  useEffect(() => {
    if (authState !== "ready" || configurationNeedsSetup !== true || serviceInstanceId === null
      || requestedSessionId !== null || requestedDirectory !== null) return;
    const recovery = browserRecovery ?? readBrowserRecoverySnapshot();
    if (recovery?.target.kind !== "new-chat" || recovery.service_instance_id !== serviceInstanceId) return;
    setSetupInput(recovery.input_text);
    onRestoreConsumed();
  }, [authState, browserRecovery, configurationNeedsSetup, onRestoreConsumed,
    requestedDirectory, requestedSessionId, serviceInstanceId]);

  const activateWorkspace = useCallback(async (directory?: string, sessionId: string | null = null) => {
    if (authState !== "ready" || configurationNeedsSetup !== false) return;
    const requestNumber = ++activationSequenceRef.current;
    setEntryLoadState("loading");
    setEntryError(null);
    try {
      const entry = await openConversation({
        ...(directory === undefined ? {} : { directory }),
        ...(sessionId === null ? { create_new: true } : { session_id: sessionId }),
      });
      if (requestNumber !== activationSequenceRef.current) return;
      setWorkspaceEntry(entry);
      setInitialSessionId(entry.session_id);
      setEntryLoadState("ready");
    } catch (error) {
      if (requestNumber !== activationSequenceRef.current) return;
      if (sessionId !== null && error instanceof ApiError && error.body?.code === "not_found"
        && error.body.field_errors.session_id !== undefined) {
        const recovery = readBrowserRecoverySnapshot();
        if (recovery?.session_id === sessionId && recovery.draft && recovery.target.kind === "chat"
          && recovery.service_instance_id === serviceInstanceId) {
          try {
            const entry = await openConversation({ directory: recovery.target.directory, create_new: true });
            if (requestNumber !== activationSequenceRef.current) return;
            const restored = { ...recovery, session_id: entry.session_id };
            writeBrowserRecoverySnapshot(restored);
            onBrowserRecoveryChange(restored);
            setWorkspaceEntry(entry);
            setInitialSessionId(entry.session_id);
            setEntryLoadState("ready");
          } catch (failure) {
            if (requestNumber !== activationSequenceRef.current) return;
            setEntryLoadState("error");
            setEntryError(sessionErrorKey(failure));
          }
          return;
        }
        onBrowserRecoveryUnavailable();
        return;
      }
      const current = error instanceof ApiError ? error.body?.conversation?.current_conversation : undefined;
      if (current !== undefined && current.project_id === null) {
        setWorkspaceEntry(current);
        setInitialSessionId(current.session_id);
      }
      setEntryLoadState("error");
      setEntryError(error instanceof ApiError && error.body?.code === "config_invalid"
        ? "chat.configurationUnavailable"
        : error instanceof ApiError && error.body?.code === "workspace_unavailable"
          ? "chat.workspaceUnavailable"
          : error instanceof ApiError && error.body?.code === "not_found" && directory !== undefined
            ? "chat.workspaceUnavailable"
            : sessionErrorKey(error));
    }
  }, [authState, configurationNeedsSetup, onBrowserRecoveryChange, onBrowserRecoveryUnavailable, serviceInstanceId]);

  useEffect(() => {
    if (authState !== "ready" || configurationNeedsSetup !== false) return;
    if (requestedSessionId !== null && requestedDirectory !== null) {
      void activateWorkspace(requestedDirectory, requestedSessionId);
    } else {
      void activateWorkspace();
    }
    return () => {
      activationSequenceRef.current += 1;
    };
  }, [
    activateWorkspace,
    authState,
    configurationNeedsSetup,
    newChatVersion,
    navigationRequestKey,
    requestedDirectory,
    requestedSessionId,
  ]);

  return (
    <div className={styles.chatWorkspaceLayout}>
      <div className={styles.chatWorkspaceContent}>
        {workspaceEntry !== null && entryError !== null && (
          <div className={styles.errorBanner} role="alert">{t(entryError)}</div>
        )}
        {configurationNeedsSetup === true ? (
          <div className={styles.conversationStage} data-empty="true">
            <div className={styles.conversationViewport}>
              <div className={styles.emptyConversation}>
                <h2 className={styles.emptyBrand}>Aide</h2>
              </div>
            </div>
            <form className={styles.composer} onSubmit={(event) => event.preventDefault()}>
              <div className={styles.composerBox}>
              <label className={styles.srOnly} htmlFor="setup-conversation-input">{t("conversation.inputLabel")}</label>
              <textarea
                id="setup-conversation-input"
                aria-label={t("conversation.inputLabel")}
                className={styles.composerInput}
                rows={3}
                value={setupInput}
                placeholder={t("conversation.inputPlaceholder")}
                onChange={(event) => {
                  const input = event.target.value;
                  setSetupInput(input);
                  if (authState !== "ready" || serviceInstanceId === null) return;
                  onBrowserRecoveryChange({
                    version: 1,
                    service_instance_id: serviceInstanceId,
                    target: { kind: "new-chat" },
                    session_id: null,
                    draft: true,
                    input_text: input,
                    model_configuration: null,
                    scroll_top: 0,
                  });
                }}
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
              </div>
            </form>
          </div>
        ) : authState === "required" ? (
          <div className={styles.notice} role="status">
            <CircleAlert size={16} aria-hidden="true" />
            <span>{t("status.authenticationRequired")}</span>
          </div>
        ) : workspaceEntry === null ? (
          <div className={entryLoadState === "error" ? styles.errorBanner : styles.notice} role={entryLoadState === "error" ? "alert" : "status"}>
            <CircleAlert size={16} aria-hidden="true" />
            <span>{entryError === null ? t("chat.workspaceLoading") : t(entryError)}</span>
          </div>
        ) : (
          <ProjectSessionsContent
            key={`${workspaceEntry.workspace_id}:${workspaceEntry.session_id}`}
            authState={authState}
            connectionState={connectionState}
            projects={projects}
            registeredClient={registeredClient}
            browserRecovery={browserRecovery}
            serviceInstanceId={serviceInstanceId}
            onRestoreConsumed={onRestoreConsumed}
            onBrowserRecoveryChange={onBrowserRecoveryChange}
            onBrowserRecoveryUnavailable={onBrowserRecoveryUnavailable}
            refreshVersion={refreshVersion}
            sendServiceCommand={sendServiceCommand}
            subscribeServiceEvents={subscribeServiceEvents}
            confirmationTriggerRef={confirmationTriggerRef}
            target={{ kind: "chat", workspaceId: workspaceEntry.workspace_id, directory: workspaceEntry.directory }}
            initialSessionId={initialSessionId}
            initialConversation={workspaceEntry}
            onNavigationSessionChange={onNavigationSessionChange}
            onNavigationDraftReleased={onNavigationDraftReleased}
            sessionActionRequest={sessionActionRequest}
            onSessionActionConsumed={onSessionActionConsumed}
          />
        )}
      </div>
    </div>
  );
}

export function ProjectSessionsView({
  authState,
  connectionState,
  projects,
  registeredClient,
  browserRecovery,
  serviceInstanceId,
  onRestoreConsumed,
  onBrowserRecoveryChange,
  onBrowserRecoveryUnavailable,
  refreshVersion,
  sendServiceCommand,
  subscribeServiceEvents,
  confirmationTriggerRef,
  projectSessionRequest,
  navigationRequestKey,
  onProjectSessionRequestConsumed,
  onNavigationSessionChange,
  onNavigationDraftReleased,
  sessionActionRequest,
  onSessionActionConsumed,
}: ProjectSessionsViewProps) {
  const { projectId = "" } = useParams();
  const [searchParams] = useSearchParams();
  const requestedSessionId = searchParams.get("session");
  const [displayedProjectId, setDisplayedProjectId] = useState(projectId);
  const [openedConversation, setOpenedConversation] = useState<ConversationOpenResponse>();
  const [navigationError, setNavigationError] = useState<string | null>(null);
  const pendingNavigationRef = useRef(false);
  const displayedClaimRef = useRef<SessionClaim | null>(null);
  const { t } = useTranslation();
  const projectSessionRequestId = projectSessionRequest?.projectId === projectId
    ? projectSessionRequest.requestId : null;
  const trackNavigationSession = useCallback((session: NavigationSession | null, claim: SessionClaim | null) => {
    if (claim !== null) displayedClaimRef.current = claim;
    onNavigationSessionChange?.(session, claim);
  }, [onNavigationSessionChange]);
  useEffect(() => {
    const returning = pendingNavigationRef.current && projectId === displayedProjectId;
    if ((projectId === displayedProjectId && !returning)
      || authState !== "ready" || connectionState !== "online") return;
    let active = true;
    if (!returning && requestedSessionId === null && projectSessionRequestId === null) {
      setDisplayedProjectId(projectId);
      setOpenedConversation(undefined);
      setNavigationError(null);
      return;
    }
    pendingNavigationRef.current = true;
    const targetSessionId = requestedSessionId ?? (returning ? displayedClaimRef.current?.session_id : null);
    void openConversation({
      project_id: projectId,
      ...(targetSessionId == null ? { create_new: true } : { session_id: targetSessionId }),
    }).then((opened) => {
      if (!active) return;
      pendingNavigationRef.current = false;
      setOpenedConversation(opened);
      setDisplayedProjectId(projectId);
      setNavigationError(null);
      if (projectSessionRequestId !== null) onProjectSessionRequestConsumed?.(projectSessionRequestId);
    }).catch((error) => {
      if (!active) return;
      pendingNavigationRef.current = false;
      const current = error instanceof ApiError ? error.body?.conversation?.current_conversation : undefined;
      if (current !== undefined && current.project_id !== null) {
        setOpenedConversation(current);
        setDisplayedProjectId(current.project_id);
      }
      setNavigationError(sessionErrorKey(error));
    });
    return () => { active = false; };
  }, [authState, connectionState, displayedProjectId, onProjectSessionRequestConsumed,
    projectId, projectSessionRequestId, requestedSessionId]);
  const initialSessionId = displayedProjectId === projectId ? requestedSessionId : openedConversation?.session_id;
  return (
    <>
    {navigationError !== null && <div className={styles.errorBanner} role="alert">{t(navigationError)}</div>}
    <ProjectSessionsContent
      key={displayedProjectId}
      authState={authState}
      connectionState={connectionState}
      projects={projects}
      registeredClient={registeredClient}
      browserRecovery={browserRecovery}
      serviceInstanceId={serviceInstanceId}
      onRestoreConsumed={onRestoreConsumed}
      onBrowserRecoveryChange={onBrowserRecoveryChange}
      onBrowserRecoveryUnavailable={onBrowserRecoveryUnavailable}
      refreshVersion={refreshVersion}
      sendServiceCommand={sendServiceCommand}
      subscribeServiceEvents={subscribeServiceEvents}
      confirmationTriggerRef={confirmationTriggerRef}
      target={{ kind: "project", projectId: displayedProjectId }}
      initialSessionId={initialSessionId}
      initialConversation={openedConversation}
      initialSessionRequestKey={navigationRequestKey}
      projectSessionRequestId={displayedProjectId === projectId ? projectSessionRequestId : null}
      onProjectSessionRequestConsumed={onProjectSessionRequestConsumed}
      onNavigationSessionChange={trackNavigationSession}
      onNavigationDraftReleased={onNavigationDraftReleased}
      sessionActionRequest={sessionActionRequest}
      onSessionActionConsumed={onSessionActionConsumed}
    />
    </>
  );
}

type ConversationTarget =
  | { kind: "chat"; workspaceId: string; directory: string }
  | { kind: "project"; projectId: string };

function ProjectSessionsContent({
  authState,
  connectionState,
  projects,
  registeredClient,
  browserRecovery,
  serviceInstanceId,
  onRestoreConsumed,
  onBrowserRecoveryChange,
  onBrowserRecoveryUnavailable,
  refreshVersion,
  sendServiceCommand,
  subscribeServiceEvents,
  confirmationTriggerRef,
  target,
  initialSessionId,
  initialConversation,
  initialSessionRequestKey,
  projectSessionRequestId,
  onProjectSessionRequestConsumed,
  onNavigationSessionChange,
  onNavigationDraftReleased,
  sessionActionRequest,
  onSessionActionConsumed,
  startInDraft = false,
}: ProjectSessionsViewProps & {
  target: ConversationTarget;
  initialSessionId?: string | null;
  initialConversation?: ConversationOpenResponse;
  initialSessionRequestKey?: string;
  projectSessionRequestId?: number | null;
  onProjectSessionRequestConsumed?: (requestId: number) => void;
  sessionActionRequest?: PendingSessionAction | null;
  onSessionActionConsumed?: (requestId: string) => void;
  startInDraft?: boolean;
}) {
  const { t, i18n } = useTranslation();
  const { drafts: conversationDraftsRef, version: conversationDraftVersion, changed: onConversationDraftChanged } = useConversationDrafts();
  const isChat = target.kind === "chat";
  const projectId = target.kind === "project" ? target.projectId : null;
  const initialWorkspaceId = target.kind === "chat" ? target.workspaceId : undefined;
  const workspaceDirectory = target.kind === "chat" ? target.directory : undefined;
  const navigate = useNavigate();
  const sessionScopeId = projectId ?? initialWorkspaceId ?? "";
  const sessionStorageId = projectId ?? workspaceDirectory ?? sessionScopeId;
  const currentServiceInstanceIdRef = useRef(serviceInstanceId);
  currentServiceInstanceIdRef.current = serviceInstanceId;
  const recoveryCandidate = useMemo(() => browserRecovery
    ?? (serviceInstanceId === null ? null : readBrowserRecoverySnapshot()), [browserRecovery, serviceInstanceId]);
  const sessionRecoveryCandidate = recoveryCandidate?.session_id != null ? recoveryCandidate : null;
  const browserRecoveryMatchesScope = sessionRecoveryCandidate !== null
    && sessionRecoveryCandidate.service_instance_id === serviceInstanceId
    && (isChat
      ? sessionRecoveryCandidate.target.kind === "chat" && sessionRecoveryCandidate.target.directory === workspaceDirectory
      : sessionRecoveryCandidate.target.kind === "project" && sessionRecoveryCandidate.target.project_id === projectId);
  const matchingBrowserRecovery = browserRecoveryMatchesScope
    && sessionRecoveryCandidate?.session_id === initialSessionId ? sessionRecoveryCandidate : null;
  const readBrowserRecoveryForSession = useCallback((sessionId: string) => {
    const recovery = readBrowserRecoverySnapshot();
    const currentServiceInstanceId = currentServiceInstanceIdRef.current;
    if (recovery === null || recovery.session_id === null || currentServiceInstanceId === null
      || recovery.service_instance_id !== currentServiceInstanceId || recovery.session_id !== sessionId) return null;
    const targetMatches = isChat
      ? recovery.target.kind === "chat" && recovery.target.directory === workspaceDirectory
      : recovery.target.kind === "project" && recovery.target.project_id === projectId;
    return targetMatches ? recovery : null;
  }, [isChat, projectId, workspaceDirectory]);
  const project = projectId === null ? undefined : projects.find((item) => item.project_id === projectId);
  const [sessions, setSessions] = useState<ProjectSessionsResponse | WorkspaceSessionsResponse | null>(null);
  const [sessionSummaries, setSessionSummaries] = useState<Record<string, SessionSummary>>({});
  const [sessionNextCursor, setSessionNextCursor] = useState<string | null>(null);
  const [loadState, setLoadState] = useState<SessionLoadState>("idle");
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [claim, setClaim] = useState<SessionClaim | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModelsResponse | null>(null);
  const [availableModelsState, setAvailableModelsState] = useState<"loading" | "ready" | "error">("loading");
  const availableModelsRef = useRef<AvailableModelsResponse | null>(null);
  const availableModelsServiceRef = useRef<string | null>(null);
  const availableModelsRequestRef = useRef(0);
  const refreshAvailableModelsRef = useRef<() => void>(() => {});
  const [sessionModelSaving, setSessionModelSaving] = useState(false);
  const [clientPermission, setClientPermission] = useState<ToolPermissionLevel>("workspace-write");
  const [clientPermissionSaving, setClientPermissionSaving] = useState(false);
  const [fullAccessWarningClaim, setFullAccessWarningClaim] = useState<SessionClaim | null>(null);
  const fullAccessCancelRef = useRef<HTMLButtonElement | null>(null);
  const [draft, setDraft] = useState(false);
  const [occupiedSessionId, setOccupiedSessionId] = useState<string | null>(null);
  const [recreateBrowserDraft, setRecreateBrowserDraft] = useState(false);
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
  const [subAgentRefreshVersion, setSubAgentRefreshVersion] = useState(0);
  const [pendingRestoreFailure, setPendingRestoreFailure] = useState<RestoreResult | null>(null);
  const [managementOpen, setManagementOpen] = useState(false);
  const [managementPanel, setManagementPanel] = useState<"runtime" | "memory">("runtime");
  const [pendingDeletion, setPendingDeletion] = useState<PendingSessionDeletion | null>(() => readPendingDeletion(sessionStorageId));
  const pendingDeletionRef = useRef(pendingDeletion);
  const deleteBusyRef = useRef(false);
  const [liveRunsBySession, setLiveRunsBySession] = useState<Record<string, LiveRun[]>>({});
  const [inputText, setInputText] = useState("");
  const inputTextRef = useRef(inputText);
  const setComposerInputText = useCallback((value: string) => {
    inputTextRef.current = value;
    setInputText(value);
  }, []);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [composerNotice, setComposerNotice] = useState<string | null>(null);
  const claimRef = useRef<SessionClaim | null>(null);
  const snapshotRef = useRef<SessionSnapshot | null>(null);
  const sessionModelSavingRef = useRef(false);
  const saveSessionModelConfigurationRef = useRef<(configuration: SessionModelConfiguration) => Promise<void>>(
    async () => undefined,
  );
  const clientPermissionSavingRef = useRef(false);
  const claimsBySessionRef = useRef<Record<string, SessionClaim>>({});
  const snapshotsBySessionRef = useRef<Record<string, SessionSnapshot>>({});
  const liveRunsRef = useRef<Record<string, LiveRun[]>>({});
  const selectedSessionRef = useRef<string | null>(null);
  const workspaceIdRef = useRef<string | null>(null);
  const sessionRequestRef = useRef(0);
  const sessionSelectionVersionRef = useRef(0);
  const refreshSessionsRef = useRef<((cursor?: string | null, append?: boolean) => Promise<void>) | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const renameTriggerRef = useRef<HTMLElement | null>(null);
  const deleteTriggerRef = useRef<HTMLElement | null>(null);
  const deleteRetryTriggerRef = useRef<HTMLButtonElement | null>(null);
  const deleteFocusOriginRef = useRef<"toolbar" | "retry">("toolbar");
  const restoreTriggerRef = useRef<HTMLElement | null>(null);
  const managementTriggerRef = useRef<HTMLElement | null>(null);
  const restoreBusyRef = useRef(false);
  const restoreFocusPendingRef = useRef(false);
  const restorePlanClaimRef = useRef<SessionClaim | null>(null);
  const restoreCompletedClaimRef = useRef<SessionClaim | null>(null);
  const consumedProjectSessionRequestRef = useRef<number | null>(null);
  const recoveredDraftRef = useRef<SessionBrowserRecoverySnapshot | null>(
    matchingBrowserRecovery?.draft === true ? matchingBrowserRecovery : null,
  );
  const draftsBySessionRef = useRef<Record<string, string>>({});
  const draftsScope = `${serviceInstanceId ?? ""}:${sessionScopeId}`;
  draftsBySessionRef.current = conversationDraftsRef.current[draftsScope] ??= {};
  const conversationViewportRef = useRef<HTMLDivElement | null>(null);
  const recoveryScrollTimerRef = useRef<number | null>(null);
  const restoredRecoveryScrollRef = useRef<string | null>(null);
  const pendingSubmissionsRef = useRef<PendingSubmission[]>([]);
  const pendingClientIdRef = useRef<string | null>(null);
  const needsReclaimRef = useRef(false);
  const attemptedRestoreRef = useRef<string | null>(null);
  const mountedRef = useRef(true);
  const adoptedConversationRef = useRef<ConversationOpenResponse | undefined>(undefined);
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

  const openConversationForSession = useCallback((sessionId: string | null, createNew = false) => (
    openConversation({
      ...(isChat ? { workspace_id: sessionScopeId } : { project_id: sessionScopeId }),
      ...(sessionId === null ? {} : { session_id: sessionId }),
      ...(createNew ? { create_new: true } : {}),
    })
  ), [isChat, sessionScopeId]);

  const claimSession = useCallback((sessionId: string) => (
    openConversationForSession(sessionId)
  ), [openConversationForSession]);

  const persistSelectedBrowserRecovery = useCallback((options?: { inputText?: string; scrollTop?: number }) => {
    const currentClaim = claimRef.current;
    const currentSnapshot = snapshotRef.current;
    const currentSessionId = selectedSessionRef.current;
    if (conversationViewportRef.current?.checkVisibility() !== true) return;
    if (serviceInstanceId === null || currentClaim === null || currentSnapshot === null
      || currentSessionId === null) return;
    let target: BrowserRecoverySnapshot["target"];
    if (isChat) {
      if (workspaceDirectory === undefined || workspaceDirectory.length === 0) return;
      target = { kind: "chat", directory: workspaceDirectory };
    } else {
      if (projectId === null) return;
      target = { kind: "project", project_id: projectId };
    }
    const configuration = draft && currentSnapshot.messages.length === 0
      ? currentSnapshot.model_configuration
      : null;
    const recoveredInputText = options?.inputText === undefined && inputTextRef.current === ""
      ? readBrowserRecoveryForSession(currentSessionId)?.input_text
      : undefined;
    const inputTextToPersist = options?.inputText ?? recoveredInputText ?? inputTextRef.current;
    if (options?.inputText === undefined && inputTextToPersist !== inputTextRef.current) {
      setComposerInputText(inputTextToPersist);
    }
    onBrowserRecoveryChange({
      version: 1,
      service_instance_id: serviceInstanceId,
      target,
      session_id: currentSessionId,
      draft,
      input_text: inputTextToPersist,
      model_configuration: configuration === null ? null : {
        provider_id: configuration.provider_id,
        model: configuration.model,
        reasoning_effort: configuration.reasoning_effort,
      },
      scroll_top: options?.scrollTop ?? conversationViewportRef.current?.scrollTop ?? 0,
    });
  }, [draft, isChat, onBrowserRecoveryChange, projectId, readBrowserRecoveryForSession,
    serviceInstanceId, setComposerInputText, workspaceDirectory]);

  useLayoutEffect(() => {
    const sessionId = selectedSessionRef.current;
    if (sessionId === null) return;
    const recoveredInput = draftsBySessionRef.current[sessionId];
    if (recoveredInput === undefined || recoveredInput === inputTextRef.current) return;
    setComposerInputText(recoveredInput);
    persistSelectedBrowserRecovery({ inputText: recoveredInput });
  }, [conversationDraftVersion, persistSelectedBrowserRecovery, setComposerInputText]);

  useEffect(() => {
    const saveBeforeClose = () => persistSelectedBrowserRecovery();
    window.addEventListener("pagehide", saveBeforeClose);
    return () => window.removeEventListener("pagehide", saveBeforeClose);
  }, [persistSelectedBrowserRecovery]);

  useLayoutEffect(() => {
    if (claim === null || snapshot === null || restoredRecoveryScrollRef.current === claim.session_id) return;
    const recovery = readBrowserRecoveryForSession(claim.session_id);
    if (recovery === null) return;
    const viewport = conversationViewportRef.current;
    if (viewport === null) return;
    viewport.scrollTop = recovery.scroll_top;
    restoredRecoveryScrollRef.current = claim.session_id;
  }, [claim, readBrowserRecoveryForSession, snapshot]);

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
    try { sessionStorage.removeItem(`aide.session-delete.${sessionStorageId}`); } catch { /* Storage can be unavailable. */ }
  }, [sessionStorageId]);

  const rememberPendingDeletion = useCallback((operation: PendingSessionDeletion) => {
    pendingDeletionRef.current = operation;
    setPendingDeletion(operation);
    try {
      sessionStorage.setItem(`aide.session-delete.${sessionStorageId}`, JSON.stringify(operation));
    } catch { /* In-memory retries remain available when browser storage is unavailable. */ }
  }, [sessionStorageId]);

  const releaseOrphanClaim = useCallback((orphan: SessionClaim) => {
    void releaseSessionClaim(orphan).catch(() => {});
  }, [releaseSessionClaim]);

  const releaseClaims = useCallback(() => {
    if (!sessionScopeId) return;
    for (const current of Object.values(claimsBySessionRef.current)) {
      if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === current.session_id) continue;
      void releaseSessionClaim(current).then(() => onNavigationDraftReleased?.(current.session_id)).catch(() => {});
    }
  }, [onNavigationDraftReleased, releaseSessionClaim, sessionScopeId]);

  useEffect(() => {
    claimRef.current = claim;
    if (claim !== null) claimsBySessionRef.current[claim.session_id] = claim;
  }, [claim]);

  useEffect(() => {
    snapshotRef.current = snapshot;
    if (snapshot !== null) snapshotsBySessionRef.current[snapshot.session_id] = snapshot;
  }, [snapshot]);

  useEffect(() => {
    if (authState !== "ready" || connectionState !== "online" || serviceInstanceId === null) return;
    let active = true;
    if (availableModelsServiceRef.current !== serviceInstanceId) {
      availableModelsServiceRef.current = serviceInstanceId;
      availableModelsRef.current = null;
      setAvailableModels(null);
    }
    if (availableModelsRef.current === null) setAvailableModelsState("loading");
    function refresh() {
      const requestNumber = ++availableModelsRequestRef.current;
      const isCurrent = () => active && requestNumber === availableModelsRequestRef.current
        && currentServiceInstanceIdRef.current === serviceInstanceId;
      void getAvailableModels().then((result) => {
        if (!isCurrent()) return;
        availableModelsRef.current = result;
        setAvailableModels(result);
        setAvailableModelsState("ready");
      }).catch(() => {
        if (isCurrent() && availableModelsRef.current === null) setAvailableModelsState("error");
      });
    }
    refreshAvailableModelsRef.current = refresh;
    refresh();
    const timer = window.setInterval(refresh, 5000);
    const unsubscribe = subscribeServiceEvents((event) => {
      if (event.service_instance_id === serviceInstanceId
        && (event.type === "config.application" || event.type === "snapshot.required")) refresh();
    });
    return () => {
      active = false;
      availableModelsRequestRef.current += 1;
      refreshAvailableModelsRef.current = () => {};
      window.clearInterval(timer);
      unsubscribe();
    };
  }, [authState, connectionState, serviceInstanceId, subscribeServiceEvents]);

  useEffect(() => {
    if (claim === null || connectionState !== "online") return;
    let active = true;
    void getRuntimeStatus(
      claim.workspace_id,
      claim.session_id,
      claim.claim_version,
      claim.reconnect_credential,
    ).then((result) => {
      if (active && result.status !== undefined
        && PERMISSION_LEVELS.includes(result.status.current_permission_level)) {
        setClientPermission(result.status.current_permission_level);
      }
    }).catch(() => {});
    return () => { active = false; };
  }, [claim, connectionState]);

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
    if (trigger?.isConnected && (!(trigger instanceof HTMLButtonElement) || !trigger.disabled)) trigger.focus();
    else document.getElementById("sessions-heading")?.focus();
  }, [restoreOpen]);

  useEffect(() => {
    selectedSessionRef.current = selectedSessionId;
  }, [selectedSessionId]);

  useEffect(() => {
    if (claim === null || snapshot === null || selectedSessionId === null) return;
    persistSelectedBrowserRecovery();
  }, [claim, draft, inputText, persistSelectedBrowserRecovery, selectedSessionId, snapshot]);

  useEffect(() => {
    onNavigationSessionChange?.({
      projectId,
      directory: workspaceDirectory ?? project?.path ?? null,
      sessionId: selectedSessionId,
      draft,
      running: (liveRunsBySession[selectedSessionId ?? ""] ?? []).some(isLiveRunActive),
    }, claim);
  }, [claim, draft, liveRunsBySession, onNavigationSessionChange, project?.path, projectId, selectedSessionId, workspaceDirectory]);

  useEffect(() => () => onNavigationSessionChange?.(null, null), [onNavigationSessionChange]);

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
      const result = await submitUserInput(
        sendServiceCommand,
        pending.claim,
        pending.text,
        pending.localId,
      );
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (item) => item.localId !== pending.localId,
      );
      if (result.kind === "management") {
        updateLiveRuns(pending.sessionId, (runs) => runs.filter((run) => run.localId !== pending.localId));
        const management = result.management_result;
        if (typeof management !== "object" || management === null || Array.isArray(management)) {
          throw new ServiceCommandError(null, false);
        }
        const fields = management;
        const output = fields.output;
        const effort = fields.effort_selection;
        const permission = fields.permission_selection;
        if (claimRef.current !== pending.claim) return;
        if (fields.management_error !== undefined) {
          setComposerError(fields.management_error.message);
          setComposerNotice(null);
          return;
        }
        if (effort != null || permission != null) {
          managementTriggerRef.current = inputRef.current;
          setManagementPanel("runtime");
          setManagementOpen(true);
        }
        if (fields.restore_listing !== undefined) {
          const firstAnchor = fields.restore_listing.anchors[0];
          if (firstAnchor === undefined) {
            await cancelRestore(pending.claim);
          } else {
            restoreTriggerRef.current = inputRef.current;
            setRestoreAnchorId(firstAnchor.anchor_id);
            setRestorePlan(null);
            setRestoreMode("conversation-only");
            setRestoreError(null);
            setRestoreOpen(true);
          }
        }
        setComposerError(null);
        setComposerNotice(typeof output === "string" ? output
          : typeof effort === "string" ? effort
            : typeof permission === "string" ? t(`settings.permissionLevels.${permission}`)
              : t("conversation.managementCompleted"));
        return;
      }
      if (result.kind !== "conversation_input" || typeof result.run_id !== "string" || !result.run_id) {
        throw new ServiceCommandError(null, false);
      }
      const runId = result.run_id;
      const controlsByRunId = new Map<string, { status: RunStatus; cancellable: boolean; cancelRequested: boolean }>();
      const liveState = result.live_state;
      if (typeof liveState === "object" && liveState !== null && !Array.isArray(liveState)) {
        const liveStateValue = liveState;
        const liveStateSeq = liveStateValue.seq;
        const liveStateStreamId = liveStateValue.stream_id;
        const currentCursor = sessionCursorRef.current[pending.sessionId];
        const staleLiveState = typeof liveStateStreamId === "string"
          && typeof liveStateSeq === "number"
          && Number.isInteger(liveStateSeq)
          && currentCursor?.streamId === liveStateStreamId
          && currentCursor.seq > liveStateSeq;
        const wireRuns = liveStateValue.runs;
        if (!staleLiveState && Array.isArray(wireRuns)) {
          for (const wireRun of wireRuns) {
            if (typeof wireRun !== "object" || wireRun === null || Array.isArray(wireRun)) continue;
            const run = wireRun;
            if (typeof run.run_id !== "string"
              || (run.status !== "accepted" && run.status !== "running")
              || typeof run.cancellable !== "boolean") continue;
            controlsByRunId.set(run.run_id, {
              status: run.status,
              cancellable: run.cancellable,
              cancelRequested: run.cancel_requested === true,
            });
          }
        }
      }
      updateLiveRuns(pending.sessionId, (runs) => runs.map((run) => {
        const accepted = run.localId === pending.localId
          ? { ...run, runId, status: run.status === "submitting" ? "accepted" as const : run.status }
          : run;
        const authoritative = accepted.runId === null ? undefined : controlsByRunId.get(accepted.runId);
        return authoritative === undefined ? accepted : {
          ...accepted,
          status: authoritative.status,
          cancellable: authoritative.cancellable,
          cancelRequested: authoritative.cancelRequested,
        };
      }));
      setComposerError(null);
      setComposerNotice(null);
    } catch (error) {
      if (error instanceof ServiceCommandError && error.resultUnknown) {
        pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
          (item) => item.localId !== pending.localId,
        );
        updateLiveRuns(pending.sessionId, (runs) => runs.filter((run) => (
          run.localId !== pending.localId || run.status !== "submitting"
        )));
        if (selectedSessionRef.current === pending.sessionId) setComposerError("conversation.submitUnknown");
        return;
      }
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
  }, [sendServiceCommand, t, updateLiveRuns]);

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
    // Reconcile unknown submissions through service snapshots; never resend inputs.
  }, [connectionState, registeredClient, updateLiveRuns]);

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
    setComposerNotice(null);
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
        responseSegments: run.response_segments?.slice(0, -1) ?? [],
        assistantContent: run.response_segments?.at(-1) ?? run.assistant_content,
        tools: run.tools.map((tool) => ({ toolCallId: tool.tool_call_id, name: tool.name,
          arguments: tool.arguments, status: tool.status, result: tool.result })),
        cancelRequested: run.cancel_requested, cancellable: run.cancellable,
      }));
      const requestIds = new Set(live.runs.map((run) => run.request_id));
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (pending) => !requestIds.has(pending.localId),
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
      const response = await listSessions({ cursor: cursor ?? undefined, limit: 100 });
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
        initialSessionId == null &&
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
            if (!restoredDraft && !response.sessions.some((item) => item.id === restored.claim.session_id)) {
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
      if (response.sessions.some((item) => item.id === selectedSessionRef.current)) setDraft(false);
      setLoadState("ready");
      setActionError(restoreError);
    } catch (error) {
      if (!mountedRef.current || requestNumber !== sessionRequestRef.current) return;
      if (error instanceof ApiError && error.body?.code === "not_found"
        && initialSessionId != null && readBrowserRecoveryForSession(initialSessionId) !== null) {
        onBrowserRecoveryUnavailable();
        return;
      }
      setLoadState("error");
      setActionError(sessionErrorKey(error));
    }
  }, [authState, claimDeletion, clearClaimState, clearPendingDeletion, connectionState, getDeletionStatus, initialSessionId, listSessions, onBrowserRecoveryUnavailable, onRestoreConsumed, readBrowserRecoveryForSession, registeredClient, releaseOrphanClaim, rememberPendingDeletion, rememberSession, readClaimSnapshot, sessionScopeId]);

  useEffect(() => {
    if (connectionState !== "online") needsReclaimRef.current = true;
  }, [connectionState]);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions, refreshVersion]);

  useEffect(() => {
    if (connectionState !== "online" || sessions?.workspace_id === undefined) return;
    void subscribeState(sendServiceCommand).catch(() => {});
  }, [connectionState, sessions?.workspace_id, sendServiceCommand]);

  useEffect(() => {
    refreshSessionsRef.current = refreshSessions;
  }, [refreshSessions]);

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
      if (claimsBySessionRef.current[sessionId] !== undefined) {
        onNavigationDraftReleased?.(sessionId, snapshotsBySessionRef.current[sessionId]?.messages.length === 0);
      }
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
    if (event.type === "session.model_configuration") {
      const providerId = event.payload.provider_id;
      const model = event.payload.model;
      const effort = event.payload.reasoning_effort;
      const version = event.payload.model_configuration_version;
      const currentSnapshot = snapshotsBySessionRef.current[sessionId];
      if (
        typeof providerId === "string" && providerId.length > 0
        && typeof model === "string" && model.length > 0
        && typeof effort === "string" && REASONING_EFFORTS.includes(effort as ReasoningEffort)
        && typeof version === "number" && Number.isInteger(version) && version > 0
        && currentSnapshot !== undefined
        && version >= currentSnapshot.model_configuration_version
      ) {
        adoptSnapshot({
          ...currentSnapshot,
          model_configuration: {
            provider_id: providerId,
            model,
            reasoning_effort: effort as ReasoningEffort,
          },
          model_configuration_version: version,
          model_configuration_available: true,
        });
      }
      return;
    }
    if (event.type === "input.accepted" && typeof event.payload.request_id === "string") {
      pendingSubmissionsRef.current = pendingSubmissionsRef.current.filter(
        (pending) => pending.localId !== event.payload.request_id,
      );
    }
    updateLiveRuns(sessionId, (runs) => reduceLiveRunEvent(runs, event));
    if (event.run_id !== null && (event.type === "run.completed" || event.type === "run.failed")) {
      void refreshRunSnapshot(sessionId, event.run_id);
    }
  }, [adoptSnapshot, onNavigationDraftReleased, readRunSnapshot, refreshRunSnapshot, updateLiveRuns]);

  useEffect(() => subscribeServiceEvents(handleServiceEvent), [handleServiceEvent, subscribeServiceEvents]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (recoveryScrollTimerRef.current !== null) {
        window.clearTimeout(recoveryScrollTimerRef.current);
      }
      const nextPath = window.location.pathname;
      const opensConversation = nextPath === "/" || nextPath === "/chat"
        || (/^\/projects\/[^/]+$/.test(nextPath) && new URLSearchParams(window.location.search).has("session"));
      if (!opensConversation) {
        releaseClaims();
      }
    };
  }, [sessionScopeId, releaseClaims]);

  useEffect(() => {
    if (matchingBrowserRecovery === null) return;
    const restoredInput = draftsBySessionRef.current[matchingBrowserRecovery.session_id]
      ?? matchingBrowserRecovery.input_text;
    draftsBySessionRef.current[matchingBrowserRecovery.session_id] = restoredInput;
    if (selectedSessionRef.current === matchingBrowserRecovery.session_id) {
      setComposerInputText(restoredInput);
    }
    if (matchingBrowserRecovery.draft) recoveredDraftRef.current = matchingBrowserRecovery;
  }, [matchingBrowserRecovery, setComposerInputText]);

  const openSession = useCallback(async (sessionId: string | null, isDraft: boolean, allowBusy = false) => {
    if (sessionId !== null && pendingDeletionRef.current?.attempted
      && pendingDeletionRef.current.claim.session_id === sessionId) return;
    if (busySessionId !== null && !allowBusy) return;
    sessionSelectionVersionRef.current += 1;
    setManagementOpen(false);
    setBusySessionId(sessionId ?? "new");
    setOccupiedSessionId(null);
    setActionError(null);
    try {
      const response = await readWithEvents(
        () => openConversationForSession(sessionId, sessionId === null),
        (opened) => opened.snapshot,
      );
      if (!mountedRef.current) {
        releaseOrphanClaim(response.claim);
        return;
      }
      const openedSessionId = response.claim.session_id;
      rememberSession(response.claim, response.snapshot);
      setSelectedSessionId(openedSessionId);
      selectedSessionRef.current = openedSessionId;
      const recovery = readBrowserRecoveryForSession(openedSessionId);
      const restoredInput = draftsBySessionRef.current[openedSessionId] ?? recovery?.input_text ?? "";
      draftsBySessionRef.current[openedSessionId] = restoredInput;
      setComposerInputText(restoredInput);
      setComposerError(null);
      setDraft(isDraft || response.snapshot.messages.length === 0);
      await refreshSessions();
    } catch (error) {
      const recovery = sessionId === null ? null : readBrowserRecoveryForSession(sessionId);
      if (error instanceof ApiError && error.body?.code === "session_claimed") {
        setOccupiedSessionId(sessionId);
        setActionError("sessions.claimedError");
      } else if (error instanceof ApiError && error.body?.code === "not_found" && recovery !== null) {
        if (recovery.draft) {
          recoveredDraftRef.current = recovery;
          setRecreateBrowserDraft(true);
          setActionError(null);
        } else {
          onBrowserRecoveryUnavailable();
        }
      } else {
        setActionError(sessionErrorKey(error));
      }
    } finally {
      setBusySessionId(null);
    }
  }, [busySessionId, onBrowserRecoveryUnavailable, openConversationForSession, readBrowserRecoveryForSession,
    readWithEvents, refreshSessions, releaseOrphanClaim, rememberSession, setComposerInputText]);

  useEffect(() => {
    if (initialConversation === undefined || adoptedConversationRef.current === initialConversation) return;
    adoptedConversationRef.current = initialConversation;
    const sessionId = initialConversation.session_id;
    rememberSession(initialConversation.claim, initialConversation.snapshot);
    setSelectedSessionId(sessionId);
    selectedSessionRef.current = sessionId;
    setDraft(initialConversation.snapshot.messages.length === 0);
    const recovery = readBrowserRecoveryForSession(sessionId);
    setComposerInputText(draftsBySessionRef.current[sessionId] ?? recovery?.input_text ?? "");
  }, [initialConversation, readBrowserRecoveryForSession, rememberSession, setComposerInputText]);

  useEffect(() => {
    if (initialSessionId == null) {
      attemptedHistorySessionRef.current = null;
      return;
    }
    if (authState !== "ready" || connectionState !== "online"
      || loadState !== "ready" || busySessionId !== null) return;
    if (initialConversation?.session_id === initialSessionId
      && selectedSessionRef.current === initialSessionId) return;
    const attemptKey = `${sessionScopeId}:${initialSessionId}:${initialSessionRequestKey ?? ""}`;
    if (attemptedHistorySessionRef.current === attemptKey) return;
    if (!sessions?.sessions.some((item) => item.id === initialSessionId) && sessionNextCursor !== null) {
      void refreshSessions(sessionNextCursor, true);
      return;
    }
    attemptedHistorySessionRef.current = attemptKey;
    if (registeredClient !== null) {
      attemptedRestoreRef.current = registeredClient.web_control_credential;
      onRestoreConsumed();
    }
    void openSession(initialSessionId, false);
  }, [authState, busySessionId, connectionState, initialConversation, initialSessionId, initialSessionRequestKey, loadState, onRestoreConsumed, openSession, refreshSessions, registeredClient, sessionNextCursor, sessionScopeId, sessions]);

  const createDraft = useCallback(async (preserveSettings = false) => {
    if (busySessionId !== null) return;
    sessionSelectionVersionRef.current += 1;
    setBusySessionId("new");
    setOccupiedSessionId(null);
    setActionError(null);
    if (!isChat && !(preserveSettings && window.location.pathname === "/settings")) {
      navigate(`/projects/${encodeURIComponent(sessionScopeId)}`, { replace: true });
    }
    try {
      await openSession(null, true, true);
    } catch (error) {
      setActionError(sessionErrorKey(error));
    } finally {
      setBusySessionId(null);
    }
  }, [busySessionId, isChat, navigate, openSession, sessionScopeId]);

  useEffect(() => {
    if (projectSessionRequestId == null || loadState !== "ready" || busySessionId !== null
      || consumedProjectSessionRequestRef.current === projectSessionRequestId) return;
    consumedProjectSessionRequestRef.current = projectSessionRequestId;
    onProjectSessionRequestConsumed?.(projectSessionRequestId);
    void createDraft();
  }, [
    busySessionId,
    createDraft,
    loadState,
    onProjectSessionRequestConsumed,
    projectSessionRequestId,
  ]);

  useEffect(() => {
    if (!startInDraft || initialSessionId != null || loadState !== "ready"
      || !sessionScopeId || selectedSessionRef.current !== null || claimRef.current !== null
      || defaultDraftWorkspaceRef.current === sessionScopeId) return;
    defaultDraftWorkspaceRef.current = sessionScopeId;
    void createDraft();
  }, [createDraft, initialSessionId, loadState, sessionScopeId, startInDraft]);

  useEffect(() => {
    if (!recreateBrowserDraft || loadState !== "ready" || busySessionId !== null) return;
    const recovery = recoveredDraftRef.current;
    setRecreateBrowserDraft(false);
    if (recovery === null) return;
    void (async () => {
      await createDraft(true);
      const currentClaim = claimRef.current;
      const newSessionId = selectedSessionRef.current;
      if (currentClaim === null || newSessionId === null) return;
      draftsBySessionRef.current[newSessionId] = recovery.input_text;
      setComposerInputText(recovery.input_text);
      if (recovery.model_configuration !== null) {
        await saveSessionModelConfigurationRef.current(recovery.model_configuration);
      }
      if (isChat && workspaceDirectory !== undefined) {
        const params = new URLSearchParams({
          directory: workspaceDirectory,
          session: newSessionId,
        });
        navigate(`/chat?${params}`, { replace: true, state: null });
      }
      recoveredDraftRef.current = null;
    })();
  }, [busySessionId, createDraft, isChat, loadState, navigate, recreateBrowserDraft, setComposerInputText, workspaceDirectory]);

  const saveSessionModelConfiguration = useCallback(async (nextConfiguration: SessionModelConfiguration) => {
    const currentClaim = claimRef.current;
    const currentSnapshot = snapshotRef.current;
    if (currentClaim === null || currentSnapshot === null || sessionModelSavingRef.current) return;
    sessionModelSavingRef.current = true;
    setSessionModelSaving(true);
    setComposerError(null);
    try {
      const result = await configureConversationModel(
        sendServiceCommand,
        currentClaim,
        currentSnapshot.model_configuration_version,
        nextConfiguration,
      );
      if (claimRef.current !== currentClaim) return;
      const version = result.model_configuration_version;
      if (typeof version !== "number" || !Number.isInteger(version)) {
        throw new ServiceCommandError(null, false);
      }
      const latestSnapshot = snapshotRef.current;
      if (latestSnapshot === null || latestSnapshot.session_id !== currentClaim.session_id) return;
      if (version >= latestSnapshot.model_configuration_version) {
        adoptSnapshot({
          ...latestSnapshot,
          model_configuration: nextConfiguration,
          model_configuration_version: version,
          model_configuration_available: true,
        });
      }
    } catch (error) {
      if (error instanceof ServiceCommandError && error.body?.code === "model_configuration_conflict") {
        setComposerError("conversation.modelConfigurationConflict");
        try {
          const latest = await readRunSnapshot(currentClaim);
          if (claimRef.current === currentClaim) adoptSnapshot(latest);
        } catch {
          // Claim recovery will refresh the saved selection after reconnection.
        }
      } else {
        setComposerError("conversation.modelConfigurationFailed");
      }
    } finally {
      sessionModelSavingRef.current = false;
      if (mountedRef.current) setSessionModelSaving(false);
    }
  }, [adoptSnapshot, readRunSnapshot, sendServiceCommand]);
  saveSessionModelConfigurationRef.current = saveSessionModelConfiguration;

  useEffect(() => {
    const recovery = recoveredDraftRef.current;
    if (connectionState !== "online" || claim === null || snapshot === null
      || recovery?.session_id !== claim.session_id || recovery.model_configuration === null
      || snapshot.messages.length !== 0 || snapshot.model_configuration !== null) return;
    recoveredDraftRef.current = null;
    void saveSessionModelConfiguration(recovery.model_configuration);
  }, [claim, connectionState, saveSessionModelConfiguration, snapshot]);

  async function saveClientPermission(value: string, confirmed = false) {
    if (!PERMISSION_LEVELS.includes(value as ToolPermissionLevel)
      || clientPermissionSavingRef.current || connectionState !== "online") return;
    const currentClaim = claimRef.current;
    if (currentClaim === null) return;
    const nextPermission = value as ToolPermissionLevel;
    if (nextPermission === "full-access" && clientPermission !== "full-access" && !confirmed) {
      setFullAccessWarningClaim(currentClaim);
      return;
    }
    clientPermissionSavingRef.current = true;
    setClientPermissionSaving(true);
    setComposerError(null);
    try {
      const result = await updateRuntimePermission(
        currentClaim.workspace_id,
        currentClaim.session_id,
        currentClaim.claim_version,
        currentClaim.reconnect_credential,
        nextPermission,
      );
      if (claimRef.current !== currentClaim) return;
      const published = result.published_permission_level;
      if (published == null || !PERMISSION_LEVELS.includes(published)) {
        throw new Error("The service returned an invalid permission level.");
      }
      setClientPermission(published);
    } catch {
      setComposerError("conversation.permissionSaveFailed");
    } finally {
      clientPermissionSavingRef.current = false;
      if (mountedRef.current) setClientPermissionSaving(false);
    }
  }

  async function submitInput(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const currentClaim = claimRef.current;
    const sessionId = selectedSessionRef.current;
    const text = inputText;
    if (currentClaim === null || sessionId === null || !text.trim()) return;
    if (pendingDeletionRef.current?.attempted && pendingDeletionRef.current.claim.session_id === sessionId) return;
    if (connectionState !== "online") return;
    const localId = createRequestId();
    confirmationTriggerRef.current = inputRef.current;
    const pending: PendingSubmission = {
      localId,
      sessionId,
      claim: currentClaim,
      text,
    };
    pendingSubmissionsRef.current.push(pending);
    updateLiveRuns(sessionId, (runs) => [
      ...runs,
      newLiveRun(pending.localId, null, text, "submitting"),
    ]);
    setComposerInputText("");
    persistSelectedBrowserRecovery({ inputText: "" });
    delete draftsBySessionRef.current[sessionId];
    setComposerError(null);
    setComposerNotice(null);
    inputRef.current?.focus();
    await sendPendingSubmission(pending);
  }

  async function recallQueuedConversationInputs() {
    const currentClaim = claimRef.current;
    if (currentClaim === null || connectionState !== "online") return;
    const sessionDrafts = draftsBySessionRef.current;
    setComposerError(null);
    try {
      const result = await recallQueuedInputs(sendServiceCommand, currentClaim);
      const recalledIds = new Set(result.recalled_inputs.map((item) => item.run_id));
      const cursor = sessionCursorRef.current[currentClaim.session_id];
      const staleState = result.live_state !== null
        && cursor?.streamId === result.live_state.stream_id
        && cursor.seq > result.live_state.seq;
      const serverRuns = new Map((staleState ? [] : result.live_state?.runs ?? [])
        .map((run) => [run.run_id, run]));
      updateLiveRuns(currentClaim.session_id, (runs) => runs
        .filter((run) => run.runId === null || !recalledIds.has(run.runId))
        .map((run) => {
          const serverRun = run.runId === null ? undefined : serverRuns.get(run.runId);
          return serverRun === undefined ? run : {
            ...run,
            status: serverRun.status,
            cancellable: serverRun.cancellable,
            cancelRequested: serverRun.cancel_requested,
          };
        }));
      const recalledText = result.recalled_inputs.map((item) => item.text).join("\n");
      if (recalledText) {
        const stillSelected = mountedRef.current && claimRef.current === currentClaim;
        const currentDraft = stillSelected ? inputTextRef.current
          : sessionDrafts[currentClaim.session_id] ?? "";
        const nextDraft = [recalledText, currentDraft].filter(Boolean).join("\n");
        sessionDrafts[currentClaim.session_id] = nextDraft;
        onConversationDraftChanged();
        if (stillSelected) {
          setComposerInputText(nextDraft);
          persistSelectedBrowserRecovery({ inputText: nextDraft });
        }
      }
      if (mountedRef.current && claimRef.current === currentClaim) setComposerNotice(null);
    } catch (error) {
      if (!mountedRef.current || claimRef.current !== currentClaim) return;
      if (error instanceof ServiceCommandError && error.resultUnknown) {
        setComposerError("conversation.recallUnknown");
      } else {
        setComposerError("conversation.recallFailed");
      }
    }
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
      await cancelConversationRun(sendServiceCommand, currentClaim, run.runId);
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
  const queuedInputCount = selectedLiveRuns.filter((run) => run.status === "accepted" && !run.cancellable).length;
  const savedSessionModel = snapshot?.model_configuration ?? null;
  const chatDefaultModel = availableModels?.default_combination ?? null;
  const displayedModel = savedSessionModel ?? chatDefaultModel;
  const displayedEffort = savedSessionModel?.reasoning_effort
    ?? chatDefaultModel?.reasoning_effort
    ?? "mid";
  const modelSelectionNeedsAttention = snapshot?.model_configuration_available === false
    || (savedSessionModel === null && (availableModelsState !== "ready" || chatDefaultModel === null));
  const selectedRestoreAnchor = snapshot?.restore_anchors?.find(
    (anchor) => anchor.anchor_id === restoreAnchorId,
  );

  function beginRestore(
    anchorId?: number,
    mode: RestoreMode = "files",
    trigger: HTMLElement | null = null,
  ) {
    if (draft || claim === null || snapshot === null || activeRun !== null) return;
    const anchors = snapshot.restore_anchors ?? [];
    if (anchors.length === 0) return;
    const target = anchorId === undefined
      ? anchors[0]
      : anchors.find((anchor) => anchor.anchor_id === anchorId);
    if (target === undefined) return;
    if (trigger !== null) restoreTriggerRef.current = trigger;
    setRestoreAnchorId(target.anchor_id);
    setRestorePlan(null);
    setRestoreMode(mode);
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
      setRestoreMode((current) => plan.available_modes.includes(current) ? current : "conversation-only");
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
    const anchorContent = snapshotRef.current?.restore_anchors?.find(
      (anchor) => anchor.anchor_id === currentPlan.anchor_id,
    )?.content;
    if (anchorContent === undefined) return;
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
      const nextClaim = executed.claim;
      if (!mountedRef.current || claimRef.current !== currentClaim) return;
      claimsBySessionRef.current[nextClaim.session_id] = nextClaim;
      claimRef.current = nextClaim;
      restoreCompletedClaimRef.current = nextClaim;
      setClaim(nextClaim);
      restorePlanClaimRef.current = null;
      setRestoreNotice(executed.result);
      setSubAgentRefreshVersion((version) => version + 1);
      setPendingRestoreFailure(executed.result.file_results.some((item) => item.status === "failed")
        && !executed.result.failure_notification_acknowledged ? executed.result : null);
      rememberSession(nextClaim, executed.snapshot);
      draftsBySessionRef.current[nextClaim.session_id] = anchorContent;
      setComposerInputText(anchorContent);
      persistSelectedBrowserRecovery({ inputText: anchorContent });
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

  const beginRename = useCallback(() => {
    if (draft || claim === null || selectedSummary === undefined) return;
    setRenameTitle(selectedSummary.title);
    setRenameError(null);
    setRenameOpen(true);
  }, [claim, draft, selectedSummary]);

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

  const beginDelete = useCallback((trigger: HTMLElement | null = null) => {
    if (draft || claim === null || selectedSummary === undefined) return;
    if (pendingDeletionRef.current?.attempted) return;
    deleteTriggerRef.current = trigger;
    deleteFocusOriginRef.current = "toolbar";
    if (pendingDeletionRef.current === null) {
      const operation = { claim: { ...claim }, requestId: createRequestId(), attempted: false, title: selectedSummary.title };
      pendingDeletionRef.current = operation;
      setPendingDeletion(operation);
    }
    setDeleteError(null);
    setDeleteOpen(true);
  }, [claim, draft, selectedSummary]);

  useEffect(() => {
    if (sessionActionRequest == null || claim === null || selectedSummary === undefined || draft) return;
    if (sessionActionRequest.sessionId !== claim.session_id || sessionActionRequest.projectId !== projectId) return;
    const currentDirectory = isChat ? workspaceDirectory : project?.path;
    if (currentDirectory !== sessionActionRequest.directory) return;
    if (sessionActionRequest.action === "rename") {
      renameTriggerRef.current = sessionActionRequest.trigger;
      beginRename();
    } else {
      beginDelete(sessionActionRequest.trigger);
    }
    onSessionActionConsumed?.(sessionActionRequest.requestId);
  }, [
    claim,
    draft,
    isChat,
    onSessionActionConsumed,
    project?.path,
    projectId,
    selectedSummary,
    sessionActionRequest,
    beginDelete,
    beginRename,
    workspaceDirectory,
  ]);

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
          try { sessionStorage.removeItem(`aide.session-delete.${sessionStorageId}`); } catch { /* Storage can be unavailable. */ }
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
        setComposerInputText("");
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
      <h1 id="sessions-heading" className={styles.srOnly} tabIndex={-1}>
        {project?.name || t(isChat ? "app.name" : "sessions.title")}
      </h1>
      {pendingDeletion?.attempted ? (
        <button className={styles.dangerButton} ref={deleteRetryTriggerRef} type="button"
          disabled={deleteBusy || connectionState !== "online"}
          onClick={() => { deleteFocusOriginRef.current = "retry"; setDeleteOpen(true); }}>
          <Trash2 size={15} aria-hidden="true" />{t("controls.retry")}
        </button>
      ) : null}

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
      {pendingRestoreFailure !== null && pendingRestoreFailure.session_id === claim?.session_id && restoreNotice === null ? (
        <button className={styles.secondaryButton} type="button" onClick={() => setRestoreNotice(pendingRestoreFailure)}>
          <Info size={15} aria-hidden="true" />{t("controls.reviewRestoreFailure")}
        </button>
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
          <div><h2>{t("sessions.notFound")}</h2><Link className={styles.secondaryButton} to="/">{t("controls.backToChat")}</Link></div>
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
        <div className={styles.sessionsLayout}>
          <section
            className={styles.sessionContentPanel}
            aria-label={t("sessions.conversation")}
          >
            {claim !== null && snapshot !== null ? (
              <>
                <h2 className={styles.srOnly}>
                  {draft ? t("sessions.draftTitle") : selectedSummary?.title ?? t("sessions.title")}
                </h2>
                <div
                  className={styles.conversationStage}
                  data-empty={snapshot.messages.length === 0 && selectedLiveRuns.length === 0}
                >
                  <div className={styles.conversationToolbar}>
                    <SubAgentPanel
                      key={`${claim.workspace_id}:${claim.session_id}`}
                      claim={claim}
                      connectionState={connectionState}
                      refreshVersion={subAgentRefreshVersion}
                      subscribeServiceEvents={subscribeServiceEvents}
                      renderConversation={(messages) => (
                        <div className={styles.messageHistory}>
                          <ConversationHistoryView messages={messages} t={t} />
                        </div>
                      )}
                    />
                  </div>
                  <div
                    className={styles.conversationViewport}
                    ref={conversationViewportRef}
                    role="log"
                    aria-live="off"
                    aria-label={t("sessions.historyLabel")}
                    onScroll={(event) => {
                      const viewport = event.currentTarget;
                      if (recoveryScrollTimerRef.current !== null) {
                        window.clearTimeout(recoveryScrollTimerRef.current);
                      }
                      recoveryScrollTimerRef.current = window.setTimeout(() => {
                        persistSelectedBrowserRecovery({ scrollTop: viewport.scrollTop });
                        recoveryScrollTimerRef.current = null;
                      }, 120);
                    }}
                  >
                    {snapshot.messages.length === 0 && selectedLiveRuns.length === 0 ? (
                      <div className={styles.emptyConversation} role="status">
                        <h2 className={styles.emptyBrand}>Aide</h2>
                      </div>
                    ) : (
                      <div className={styles.messageHistory}>
                        <ConversationHistoryView
                          messages={snapshot.messages}
                          t={t}
                          restoreAnchors={activeRun === null && busySessionId === null && connectionState === "online"
                            ? snapshot.restore_anchors
                            : []}
                          onRestoreAnchor={(anchorId, mode, trigger) => beginRestore(anchorId, mode, trigger)}
                        />
                        {selectedLiveRuns.map((run) => (
                          <LiveRunView key={run.localId} run={run} t={t} />
                        ))}
                      </div>
                    )}
                  </div>
                  <form className={styles.composer} onSubmit={(event) => void submitInput(event)}>
                  <div className={styles.composerBox}>
                  <label className={styles.srOnly} htmlFor="conversation-input">{t("conversation.inputLabel")}</label>
                  <textarea
                    ref={inputRef}
                    id="conversation-input"
                    aria-label={t("conversation.inputLabel")}
                    className={styles.composerInput}
                    rows={3}
                    value={inputText}
                    disabled={connectionState !== "online"}
                    placeholder={t("conversation.inputPlaceholder")}
                    onChange={(event) => {
                      const nextInput = event.target.value;
                      draftsBySessionRef.current[claim.session_id] = nextInput;
                      setComposerInputText(nextInput);
                      setComposerNotice(null);
                      persistSelectedBrowserRecovery({ inputText: nextInput });
                    }}
                    onKeyDown={handleInputKeyDown}
                  />
                  <div className={styles.composerToolbar}>
                  <ComposerControls
                    key={claim.session_id}
                    permission={clientPermission}
                    permissionDisabled={clientPermissionSaving}
                    onPermissionChange={(level) => void saveClientPermission(level)}
                    modelSummary={displayedModel?.model ?? t("conversation.modelsUnavailable")}
                    effortSummary={displayedEffort}
                    disabled={connectionState !== "online"}
                  >
                    <label className={styles.composerSetting}>
                      <span>{t("conversation.modelMenuLabel")}</span>
                      <select
                        className={styles.composerSelect}
                        aria-label={t("conversation.sessionModel")}
                        value={savedSessionModel === null
                          ? ""
                          : JSON.stringify([savedSessionModel.provider_id, savedSessionModel.model])}
                        disabled={sessionModelSaving || connectionState !== "online" || availableModelsState !== "ready"
                          || (availableModels?.models.length ?? 0) === 0}
                        onChange={(event) => {
                          const selected = availableModels?.models.find((model) => (
                            JSON.stringify([model.provider_id, model.model]) === event.target.value
                          ));
                          if (selected === undefined) return;
                          void saveSessionModelConfiguration({
                            provider_id: selected.provider_id,
                            model: selected.model,
                            reasoning_effort: displayedEffort,
                          });
                        }}
                      >
                        {savedSessionModel === null ? (
                          <option value="" disabled>
                            {chatDefaultModel === null
                              ? t("conversation.modelsUnavailable")
                              : `${t("conversation.defaultModel")} · ${chatDefaultModel.provider_id}/${chatDefaultModel.model}`}
                          </option>
                        ) : null}
                        {savedSessionModel !== null && snapshot?.model_configuration_available === false ? (
                          <option value={JSON.stringify([savedSessionModel.provider_id, savedSessionModel.model])}>
                            {`${savedSessionModel.provider_id}/${savedSessionModel.model} · ${t("conversation.unavailableModel")}`}
                          </option>
                        ) : null}
                        {(availableModels?.models ?? []).map((model) => (
                          <option
                            key={JSON.stringify([model.provider_id, model.model])}
                            value={JSON.stringify([model.provider_id, model.model])}
                          >
                            {`${model.provider_id} · ${model.model} (${model.context_window.toLocaleString(i18n.language)})`}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className={styles.composerSetting}>
                      <span>{t("conversation.effortMenuLabel")}</span>
                      <select
                        className={styles.composerSelect}
                        aria-label={t("conversation.sessionEffort")}
                        lang="en"
                        value={displayedEffort}
                        disabled={sessionModelSaving || connectionState !== "online" || displayedModel === null}
                        onChange={(event) => {
                          const effort = event.target.value as ReasoningEffort;
                          const selected = savedSessionModel ?? chatDefaultModel;
                          if (!REASONING_EFFORTS.includes(effort) || selected === null) return;
                          void saveSessionModelConfiguration({
                            provider_id: selected.provider_id,
                            model: selected.model,
                            reasoning_effort: effort,
                          });
                        }}
                      >
                        {REASONING_EFFORTS.map((effort) => (
                          <option key={effort} value={effort}>
                            {effort}
                          </option>
                        ))}
                      </select>
                    </label>
                  </ComposerControls>
                    <button
                      className={activeRun === null ? styles.composerSend : `${styles.composerSend} ${styles.composerStop}`}
                      type={activeRun === null ? "submit" : "button"}
                      aria-label={activeRun === null ? t("controls.send") : activeRun.cancelRequested
                        ? t("controls.cancelingRun") : t("controls.cancelRun")}
                      title={activeRun === null ? t("controls.send") : activeRun.cancelRequested
                        ? t("controls.cancelingRun") : t("controls.cancelRun")}
                      disabled={connectionState !== "online" || (activeRun === null
                        ? !inputText.trim()
                        : activeRun.runId === null || !activeRun.cancellable || activeRun.cancelRequested)}
                      onClick={activeRun === null ? undefined : () => void cancelRun(activeRun)}
                    >
                      {activeRun === null ? <ArrowUp size={20} aria-hidden="true" />
                        : <Square size={14} fill="currentColor" aria-hidden="true" />}
                    </button>
                  </div>
                  </div>
                  <div className={styles.composerFooter}>
                    {availableModelsState !== "loading" && modelSelectionNeedsAttention ? (
                      <Link className={styles.secondaryButton} to="/settings">
                        {t("conversation.configureModels")}
                      </Link>
                    ) : null}
                    {queuedInputCount > 0 ? (
                      <button
                        className={styles.secondaryButton}
                        type="button"
                        disabled={connectionState !== "online"}
                        onClick={() => void recallQueuedConversationInputs()}
                      >
                        <RotateCcw size={14} aria-hidden="true" />
                        {t("conversation.recallQueued", { count: queuedInputCount })}
                      </button>
                    ) : null}
                    <p className={composerError !== null ? styles.composerError
                      : composerNotice !== null ? styles.composerNotice : styles.composerHint}
                    role={composerError !== null ? "alert" : "status"}>
                      {composerError !== null
                        ? t(composerError)
                        : composerNotice !== null
                          ? composerNotice
                        : availableModelsState === "error"
                          ? t("conversation.modelsUnavailable")
                          : modelSelectionNeedsAttention
                            ? t("conversation.modelUnavailable")
                        : activeRun !== null
                          ? t("conversation.queueBehindActive")
                          : t("conversation.enterHint")}
                    </p>
                  </div>
                    <ConversationRuntimeStatus
                      key={`${claim.workspace_id}:${claim.session_id}:${claim.claim_version}`}
                      claim={claim}
                      connectionState={connectionState}
                      refreshVersion={refreshVersion}
                      activeRuns={selectedLiveRuns}
                    />
                  </form>
                </div>
              </>
            ) : occupiedSessionId !== null ? (
              <div className={styles.sessionPrompt} role="status">
                <Info size={22} aria-hidden="true" />
                <h2>{t("sessions.occupied")}</h2>
                <p>{t("sessions.claimedError")}</p>
                <button
                  className={styles.primaryButton}
                  type="button"
                  disabled={busySessionId !== null || connectionState !== "online"}
                  onClick={() => void createDraft()}
                >
                  <Plus size={16} aria-hidden="true" />
                  {t("controls.newSession")}
                </button>
              </div>
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
          onOpenChange={(open) => {
            setManagementOpen(open);
            if (!open) refreshAvailableModelsRef.current();
          }}
          panel={managementPanel}
          onPermissionChanged={setClientPermission}
          claim={claim}
          sessionTitle={draft ? t("sessions.draftTitle") : selectedSummary?.title ?? t("sessions.title")}
          sessionModelConfiguration={savedSessionModel}
          activeRuns={selectedLiveRuns}
          connectionState={connectionState}
          triggerRef={managementTriggerRef}
        />
      ) : null}
      <Dialog.Root
        open={fullAccessWarningClaim !== null}
        onOpenChange={(open) => { if (!open) setFullAccessWarningClaim(null); }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content
            className={styles.dialogContent}
            onOpenAutoFocus={(event) => {
              event.preventDefault();
              fullAccessCancelRef.current?.focus();
            }}
          >
            <Dialog.Title className={styles.dialogTitle}>{t("conversation.fullAccessTitle")}</Dialog.Title>
            <Dialog.Description className={styles.dialogDescription}>
              {t("conversation.fullAccessWarning")}
            </Dialog.Description>
            <div className={styles.dialogActions}>
              <Dialog.Close asChild>
                <button ref={fullAccessCancelRef} className={styles.secondaryButton} type="button">
                  {t("controls.cancel")}
                </button>
              </Dialog.Close>
              <button
                className={styles.primaryButton}
                type="button"
                disabled={claim !== fullAccessWarningClaim || connectionState !== "online"}
                onClick={() => {
                  if (claimRef.current !== fullAccessWarningClaim) return;
                  setFullAccessWarningClaim(null);
                  void saveClientPermission("full-access", true);
                }}
              >
                {t("conversation.enableFullAccess")}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
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
              if (!restoreSessionActionFocus(trigger)) document.getElementById("sessions-heading")?.focus();
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
              if (trigger?.isConnected && (!(trigger instanceof HTMLButtonElement) || !trigger.disabled)) trigger.focus();
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
              if (!restoreSessionActionFocus(renameTriggerRef.current)) document.getElementById("sessions-heading")?.focus();
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
