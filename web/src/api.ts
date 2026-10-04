import type {
  BrowserSession,
  BrowserTicketExchange,
  ClientCommand,
  ProjectListResponse,
  ProjectRemoval,
  ProjectRemovalStatus,
  ProjectRegistration,
  ProjectScheduleResume,
  ScheduleJobInput,
  ScheduleJobHistoryResponse,
  ScheduleJobResponse,
  ScheduleJobsResponse,
  RegisteredClient,
  ProjectSessionsResponse,
  WorkspaceSessionsResponse,
  ChatSessionsResponse,
  ChatWorkspaceEntry,
  SessionClaimResponse,
  SessionCreation,
  SessionDeletion,
  SessionDeletionStatus,
  SessionDeletionClaim,
  SessionRenameResponse,
  SessionRelease,
  SessionSnapshot,
  SessionClaim,
  ManagementResult,
  ManagementResponse,
  ToolPermissionLevel,
  ReasoningEffort,
  RestoreMode,
  RestorePlan,
  RestoreResult,
  ServiceCommandResult,
  ServiceErrorBody,
  ServiceEvent,
  ServiceStatus,
  ConfigPatchResponse,
  ConfigResponse,
  ConfigPatchFields,
  ConfigSecrets,
  AvailableModelsResponse,
} from "./protocol";

const API_PREFIX = "/api/v1";
let csrfToken: string | null = null;
let webControlCredential: string | null = null;

export class ApiError extends Error {
  readonly status: number;
  readonly body: ServiceErrorBody | null;

  constructor(status: number, body: ServiceErrorBody | null) {
    super(body?.message ?? "The local service request failed.");
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

export class ServiceCommandError extends Error {
  readonly body: ServiceErrorBody | null;
  readonly resultUnknown: boolean;

  constructor(body: ServiceErrorBody | null, resultUnknown: boolean) {
    super(body?.message ?? "The local service connection closed before the command was confirmed.");
    this.name = "ServiceCommandError";
    this.body = body;
    this.resultUnknown = resultUnknown;
  }
}

export interface EventStreamConnection {
  close: () => void;
  sendCommand: (command: ClientCommand) => Promise<ServiceCommandResult>;
}

type WorkspaceSessionCreation = Omit<SessionCreation, "project_id"> & { project_id: null };
type WorkspaceSessionClaimResponse = Omit<SessionClaimResponse, "project_id"> & { project_id: null };
type WorkspaceSessionDeletion = Omit<SessionDeletion, "project_id">;
type WorkspaceSessionDeletionClaim = Omit<SessionDeletionClaim, "project_id"> & { project_id: null };
type WorkspaceSessionDeletionStatus = Omit<SessionDeletionStatus, "project_id"> & { project_id: null };
type WorkspaceSessionRenameResponse = Omit<SessionRenameResponse, "project_id">;

export function enterChatWorkspace(directory?: string): Promise<ChatWorkspaceEntry> {
  return request<ChatWorkspaceEntry>("/chat/workspaces/enter", {
    method: "POST",
    mutation: true,
    body: { request_id: createRequestId(), ...(directory === undefined ? {} : { directory }) },
  });
}

export function getChatSessions(
  options: { title?: string; cursor?: string; limit?: number } = {},
): Promise<ChatSessionsResponse> {
  const query = new URLSearchParams();
  if (options.title !== undefined) query.set("title", options.title);
  if (options.cursor !== undefined) query.set("cursor", options.cursor);
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  const queryString = query.toString();
  return request<ChatSessionsResponse>(`/chat/sessions${queryString ? `?${queryString}` : ""}`);
}

export async function getWorkspaceSessions(
  workspaceId: string,
  options: { title?: string; cursor?: string; limit?: number } = {},
): Promise<WorkspaceSessionsResponse> {
  const query = new URLSearchParams();
  if (options.title !== undefined) query.set("title", options.title);
  if (options.cursor !== undefined) query.set("cursor", options.cursor);
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  const queryString = query.toString();
  const response = await request<Omit<WorkspaceSessionsResponse, "workspace_id">>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions${queryString ? `?${queryString}` : ""}`,
  );
  return { ...response, workspace_id: workspaceId };
}

export function createWorkspaceSession(workspaceId: string): Promise<WorkspaceSessionCreation> {
  return request<WorkspaceSessionCreation>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions`,
    { method: "POST", mutation: true, body: { request_id: createRequestId() } },
  );
}

export function claimWorkspaceSession(
  workspaceId: string,
  sessionId: string,
): Promise<WorkspaceSessionClaimResponse> {
  return request<WorkspaceSessionClaimResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}/claim`,
    { method: "POST", mutation: true, body: { request_id: createRequestId() } },
  );
}

export function getWorkspaceSession(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<SessionSnapshot> {
  return request<{ snapshot: SessionSnapshot }>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}?claim_version=${claimVersion}`,
    { extraHeaders: { "X-Omni-Claim": claimCredential } },
  ).then((response) => response.snapshot);
}

export function releaseWorkspaceSession(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<SessionRelease> {
  return request<SessionRelease>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}/release`,
    {
      method: "POST",
      mutation: true,
      body: { request_id: createRequestId(), claim_version: claimVersion },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
}

export function renameWorkspaceSession(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  title: string,
  metadataVersion: number,
): Promise<WorkspaceSessionRenameResponse> {
  return request<WorkspaceSessionRenameResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      mutation: true,
      body: {
        request_id: createRequestId(),
        claim_version: claimVersion,
        metadata_version: metadataVersion,
        title,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
}

export function deleteWorkspaceSession(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  requestId: string,
): Promise<WorkspaceSessionDeletion> {
  return request<WorkspaceSessionDeletion>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "DELETE",
      mutation: true,
      body: { request_id: requestId, claim_version: claimVersion, confirm: true },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
}

export function getWorkspaceSessionDeletionStatus(
  workspaceId: string,
  sessionId: string,
): Promise<WorkspaceSessionDeletionStatus> {
  return request<WorkspaceSessionDeletionStatus>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}/deletion-status`,
  );
}

export function claimWorkspaceSessionDeletion(
  workspaceId: string,
  sessionId: string,
): Promise<WorkspaceSessionDeletionClaim> {
  return request<WorkspaceSessionDeletionClaim>(
    `/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}/deletion-claim`,
    { method: "POST", mutation: true, body: { request_id: createRequestId() } },
  );
}

export function setCsrfToken(value: string): void {
  csrfToken = value;
}

export function clearCsrfToken(): void {
  csrfToken = null;
}

async function request<T>(
  path: string,
  options: {
    method?: "GET" | "POST" | "PATCH" | "DELETE";
    body?: Record<string, unknown>;
    mutation?: boolean;
    extraHeaders?: Record<string, string>;
  } = {},
): Promise<T> {
  const headers = new Headers({ Accept: "application/json" });
  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
  }
  if (options.mutation) {
    if (csrfToken === null) {
      throw new ApiError(403, null);
    }
    headers.set("X-Omni-CSRF", csrfToken);
  }
  if (options.extraHeaders !== undefined) {
    for (const [name, value] of Object.entries(options.extraHeaders)) {
      headers.set(name, value);
    }
  }
  if (webControlCredential !== null) {
    headers.set("X-Omni-Control", webControlCredential);
  }
  const response = await fetch(`${API_PREFIX}${path}`, {
    method: options.method ?? "GET",
    headers,
    credentials: "include",
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  const value: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const body = isServiceError(value) ? value : null;
    throw new ApiError(response.status, body);
  }
  return value as T;
}

export async function exchangeTicket(ticket: string): Promise<BrowserTicketExchange> {
  const response = await request<BrowserTicketExchange>("/web/ticket", {
    method: "POST",
    body: { ticket },
  });
  setCsrfToken(response.csrf_token);
  return response;
}

export async function restoreBrowserSession(): Promise<BrowserSession | null> {
  try {
    const response = await request<BrowserSession>("/web/session");
    setCsrfToken(response.csrf_token);
    return response;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      clearCsrfToken();
      return null;
    }
    throw error;
  }
}

export async function registerWebClient(): Promise<RegisteredClient> {
  const client = await request<RegisteredClient>("/clients", {
    method: "POST",
    mutation: true,
    body: { request_id: createRequestId(), kind: "web" },
  });
  webControlCredential = client.web_control_credential;
  return client;
}

export function getServiceStatus(): Promise<ServiceStatus> {
  return request<ServiceStatus>("/service");
}

export function getConfig(): Promise<ConfigResponse> {
  return request<ConfigResponse>("/config");
}

export function getAvailableModels(): Promise<AvailableModelsResponse> {
  return request<AvailableModelsResponse>("/models/available");
}

export function patchConfig(
  revision: string,
  fields: ConfigPatchFields,
  secrets: ConfigSecrets,
  requestId: string = createRequestId(),
): Promise<ConfigPatchResponse> {
  return request<ConfigPatchResponse>("/config", {
    method: "PATCH",
    mutation: true,
    body: { request_id: requestId, revision, fields, secrets },
  });
}

export function repairConfig(
  revision: string,
  fields: ConfigPatchFields,
  secrets: ConfigSecrets,
  requestId: string = createRequestId(),
): Promise<ConfigPatchResponse> {
  return request<ConfigPatchResponse>("/config/repair", {
    method: "POST",
    mutation: true,
    body: { request_id: requestId, revision, fields, secrets },
  });
}


export async function getRuntimeStatus(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<ManagementResult> {
  const response = await request<ManagementResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/management/status`,
    {
      method: "POST",
      mutation: true,
      body: {
        request_id: createRequestId(),
        current_session_id: sessionId,
        claim_version: claimVersion,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
  return response.result;
}

export async function updateRuntimePermission(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  permissionLevel: ToolPermissionLevel,
): Promise<ManagementResult> {
  const response = await request<ManagementResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/management/permission`,
    {
      method: "POST",
      mutation: true,
      body: {
        request_id: createRequestId(),
        current_session_id: sessionId,
        claim_version: claimVersion,
        permission_level: permissionLevel,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
  return response.result;
}

export async function updateRuntimeEffort(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  effort: ReasoningEffort,
): Promise<ManagementResult> {
  const response = await request<ManagementResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/management/effort`,
    {
      method: "POST",
      mutation: true,
      body: {
        request_id: createRequestId(),
        current_session_id: sessionId,
        claim_version: claimVersion,
        effort,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
  return response.result;
}

async function postRuntimeManagement(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  action: "memory" | "dream" | "skills/reload",
): Promise<ManagementResult> {
  const response = await request<ManagementResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/management/${action}`,
    {
      method: "POST",
      mutation: true,
      body: {
        request_id: createRequestId(),
        current_session_id: sessionId,
        claim_version: claimVersion,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
  return response.result;
}

export function getRuntimeMemory(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<ManagementResult> {
  return postRuntimeManagement(workspaceId, sessionId, claimVersion, claimCredential, "memory");
}

export function triggerRuntimeDream(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<ManagementResult> {
  return postRuntimeManagement(workspaceId, sessionId, claimVersion, claimCredential, "dream");
}

export function reloadRuntimeSkills(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<ManagementResult> {
  return postRuntimeManagement(workspaceId, sessionId, claimVersion, claimCredential, "skills/reload");
}

export function getProjects(): Promise<ProjectListResponse> {
  return request<ProjectListResponse>("/projects");
}

export function registerProject(path: string): Promise<ProjectRegistration> {
  return request<ProjectRegistration>("/projects", {
    method: "POST",
    mutation: true,
    body: { request_id: createRequestId(), path },
  });
}

export function removeProject(projectId: string): Promise<ProjectRemoval> {
  return request<ProjectRemoval>(`/projects/${encodeURIComponent(projectId)}`, {
    method: "DELETE",
    mutation: true,
    body: { request_id: createRequestId() },
  });
}

export function getProjectRemoval(projectId: string, operationId: string): Promise<ProjectRemovalStatus> {
  return request<ProjectRemovalStatus>(
    `/projects/${encodeURIComponent(projectId)}/removal/${encodeURIComponent(operationId)}`,
  );
}

export function resumeProjectSchedule(
  projectId: string,
  jobIds: string[],
): Promise<ProjectScheduleResume> {
  return request<ProjectScheduleResume>(`/projects/${encodeURIComponent(projectId)}/schedule-resume`, {
    method: "POST",
    mutation: true,
    body: { request_id: createRequestId(), job_ids: jobIds },
  });
}

export function getScheduleJobs(workspaceId: string): Promise<ScheduleJobsResponse> {
  return request<ScheduleJobsResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/schedule/jobs`,
  );
}

export function createScheduleJob(
  workspaceId: string,
  input: ScheduleJobInput,
  requestId: string,
): Promise<ScheduleJobResponse> {
  return request<ScheduleJobResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/schedule/jobs`,
    {
      method: "POST",
      mutation: true,
      body: { request_id: requestId, ...input },
    },
  );
}

export function getScheduleJob(
  workspaceId: string,
  jobId: string,
): Promise<ScheduleJobResponse> {
  return request<ScheduleJobResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/schedule/jobs/${encodeURIComponent(jobId)}`,
  );
}

export function getScheduleJobHistory(
  workspaceId: string,
  jobId: string,
  options: { cursor?: string; limit?: number } = {},
): Promise<ScheduleJobHistoryResponse> {
  const query = new URLSearchParams();
  if (options.cursor !== undefined) query.set("cursor", options.cursor);
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  const queryString = query.toString();
  return request<ScheduleJobHistoryResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/schedule/jobs/${encodeURIComponent(jobId)}/history${queryString ? `?${queryString}` : ""}`,
  );
}

export function deleteScheduleJob(
  workspaceId: string,
  jobId: string,
  requestId: string,
): Promise<ScheduleJobResponse> {
  return request<ScheduleJobResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/schedule/jobs/${encodeURIComponent(jobId)}`,
    {
      method: "DELETE",
      mutation: true,
      body: { request_id: requestId },
    },
  );
}

export function getProjectSessions(
  projectId: string,
  options: { title?: string; cursor?: string; limit?: number } = {},
): Promise<ProjectSessionsResponse> {
  const query = new URLSearchParams();
  if (options.title !== undefined) query.set("title", options.title);
  if (options.cursor !== undefined) query.set("cursor", options.cursor);
  if (options.limit !== undefined) query.set("limit", String(options.limit));
  const queryString = query.toString();
  return request<ProjectSessionsResponse>(
    `/projects/${encodeURIComponent(projectId)}/sessions${queryString ? `?${queryString}` : ""}`,
  );
}

export function renameProjectSession(
  projectId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  title: string,
  metadataVersion: number,
): Promise<SessionRenameResponse> {
  return request<SessionRenameResponse>(
    `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "PATCH",
      mutation: true,
      body: {
        request_id: createRequestId(),
        claim_version: claimVersion,
        metadata_version: metadataVersion,
        title,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
}

export function getProjectSession(
  projectId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<SessionSnapshot> {
  return request<{ snapshot: SessionSnapshot }>(
    `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}?claim_version=${claimVersion}`,
    { extraHeaders: { "X-Omni-Claim": claimCredential } },
  ).then((response) => response.snapshot);
}

function postRestoreManagement(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  action: string,
  payload: Record<string, unknown> = {},
): Promise<ManagementResult> {
  return request<{ request_id: string; result: ManagementResult }>(
    `/workspaces/${encodeURIComponent(workspaceId)}/management/${action}`,
    {
      method: "POST",
      mutation: true,
      body: {
        request_id: createRequestId(),
        current_session_id: sessionId,
        claim_version: claimVersion,
        ...payload,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  ).then((response) => response.result);
}

export async function inspectRestore(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  anchorId: number,
): Promise<RestorePlan> {
  const result = await postRestoreManagement(
    workspaceId,
    sessionId,
    claimVersion,
    claimCredential,
    "restore/inspect",
    { anchor_id: anchorId },
  );
  if (result.restore_plan === undefined) {
    throw new ApiError(409, null);
  }
  return result.restore_plan;
}

export async function executeRestore(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  plan: RestorePlan,
  mode: RestoreMode,
): Promise<{ result: RestoreResult; claimVersion: number; claimCredential: string }> {
  const response = await request<{
    request_id: string;
    result: ManagementResult & { claim_version?: number; claim_credential?: string };
  }>(`/workspaces/${encodeURIComponent(workspaceId)}/management/restore/execute`, {
    method: "POST",
    mutation: true,
    body: {
      request_id: createRequestId(),
      current_session_id: sessionId,
      claim_version: claimVersion,
      plan: { anchor_id: plan.anchor_id },
      mode,
    },
    extraHeaders: { "X-Omni-Claim": claimCredential },
  });
  const nextResult = response.result.restore_result;
  const nextClaimVersion = response.result.claim_version;
  const nextClaimCredential = response.result.claim_credential;
  if (
    nextResult === undefined ||
    nextResult === null ||
    typeof nextClaimVersion !== "number" ||
    typeof nextClaimCredential !== "string"
  ) {
    throw new ApiError(409, null);
  }
  return {
    result: nextResult,
    claimVersion: nextClaimVersion,
    claimCredential: nextClaimCredential,
  };
}

export function getRestoreResult(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<RestoreResult | null> {
  return postRestoreManagement(
    workspaceId,
    sessionId,
    claimVersion,
    claimCredential,
    "restore/result",
  ).then((result) => result.restore_result ?? null);
}

export async function cancelRestore(claim: SessionClaim): Promise<void> {
  await postRestoreManagement(
    claim.workspace_id, claim.session_id, claim.claim_version,
    claim.reconnect_credential, "restore/cancel",
  );
}

export function acknowledgeRestore(
  workspaceId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<RestoreResult | null> {
  return postRestoreManagement(
    workspaceId,
    sessionId,
    claimVersion,
    claimCredential,
    "restore/acknowledge",
  ).then((result) => result.restore_result ?? null);
}

export function createProjectSession(projectId: string): Promise<SessionCreation> {
  return request<SessionCreation>(
    `/projects/${encodeURIComponent(projectId)}/sessions`,
    {
      method: "POST",
      mutation: true,
      body: { request_id: createRequestId() },
    },
  );
}

export function claimProjectSession(
  projectId: string,
  sessionId: string,
): Promise<SessionClaimResponse> {
  return request<SessionClaimResponse>(
    `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/claim`,
    {
      method: "POST",
      mutation: true,
      body: { request_id: createRequestId() },
    },
  );
}

export function releaseProjectSession(
  projectId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
): Promise<SessionRelease> {
  return request<SessionRelease>(
    `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/release`,
    {
      method: "POST",
      mutation: true,
      body: { request_id: createRequestId(), claim_version: claimVersion },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
}

export function deleteProjectSession(
  projectId: string,
  sessionId: string,
  claimVersion: number,
  claimCredential: string,
  requestId: string,
): Promise<SessionDeletion> {
  return request<SessionDeletion>(
    `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
    {
      method: "DELETE",
      mutation: true,
      body: {
        request_id: requestId,
        claim_version: claimVersion,
        confirm: true,
      },
      extraHeaders: { "X-Omni-Claim": claimCredential },
    },
  );
}

export function getProjectSessionDeletionStatus(projectId: string, sessionId: string): Promise<SessionDeletionStatus> {
  return request<SessionDeletionStatus>(
    `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/deletion-status`,
  );
}

export function claimProjectSessionDeletion(projectId: string, sessionId: string): Promise<SessionDeletionClaim> {
  return request<SessionDeletionClaim>(
    `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/deletion-claim`,
    { method: "POST", mutation: true, body: { request_id: createRequestId() } },
  );
}

export function openEventStream(
  onOpen: () => void,
  onClose: () => void,
  onMessage: (value: ServiceEvent) => void,
): EventStreamConnection {
  if (webControlCredential === null) throw new ApiError(403, null);
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${window.location.host}${API_PREFIX}/events`,
    ["omni-v1", webControlCredential],
  );
  const pending = new Map<
    string,
    { resolve: (result: ServiceCommandResult) => void; reject: (error: ServiceCommandError) => void }
  >();
  socket.addEventListener("open", onOpen);
  socket.addEventListener("close", () => {
    for (const { reject } of pending.values()) reject(new ServiceCommandError(null, true));
    pending.clear();
    onClose();
  });
  socket.addEventListener("error", onClose, { once: true });
  socket.addEventListener("message", (event) => {
    try {
      const value: unknown = JSON.parse(event.data as string);
      if (isServiceCommandResponse(value)) {
        const command = pending.get(value.request_id);
        if (command === undefined) return;
        pending.delete(value.request_id);
        if (value.accepted === true && isServiceCommandResult(value.result)) {
          command.resolve(value.result);
        } else {
          command.reject(new ServiceCommandError(isServiceError(value) ? value : null, false));
        }
        return;
      }
      if (isServiceEvent(value)) onMessage(value);
    } catch {
      // Invalid events do not affect the authenticated connection state.
    }
  });
  return {
    close: () => socket.close(),
    sendCommand: (command) => {
      if (socket.readyState !== WebSocket.OPEN) {
        return Promise.reject(new ServiceCommandError(null, false));
      }
      return new Promise<ServiceCommandResult>((resolve, reject) => {
        pending.set(command.request_id, { resolve, reject });
        try {
          socket.send(JSON.stringify(command));
        } catch {
          pending.delete(command.request_id);
          reject(new ServiceCommandError(null, true));
        }
      });
    },
  };
}

export function createRequestId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `web-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function isServiceError(value: unknown): value is ServiceErrorBody {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ServiceErrorBody>;
  return typeof candidate.code === "string" && typeof candidate.message === "string";
}

function isServiceCommandResponse(
  value: unknown,
): value is { request_id: string; accepted?: boolean; result?: unknown } & Partial<ServiceErrorBody> {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as { request_id?: unknown; accepted?: unknown; code?: unknown };
  return (
    typeof candidate.request_id === "string"
    && (candidate.accepted === true || typeof candidate.code === "string")
  );
}

function isServiceCommandResult(value: unknown): value is ServiceCommandResult {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isServiceEvent(value: unknown): value is ServiceEvent {
  if (typeof value !== "object" || value === null) return false;
  const event = value as Partial<ServiceEvent>;
  return (
    event.protocol_version === 1 &&
    typeof event.service_instance_id === "string" &&
    typeof event.stream_id === "string" &&
    typeof event.seq === "number" &&
    typeof event.type === "string" &&
    (event.workspace_id === null || typeof event.workspace_id === "string") &&
    (event.project_id === null || typeof event.project_id === "string") &&
    (event.session_id === null || typeof event.session_id === "string") &&
    (event.run_id === null || typeof event.run_id === "string") &&
    typeof event.payload === "object" &&
    event.payload !== null
  );
}
