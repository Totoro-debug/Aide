import type {
  BrowserSession,
  BrowserTicketExchange,
  ProjectListResponse,
  ProjectRegistration,
  ProjectScheduleResume,
  RegisteredClient,
  ProjectSessionsResponse,
  SessionClaimResponse,
  SessionCreation,
  SessionRelease,
  ServiceErrorBody,
  ServiceEvent,
  ServiceStatus,
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

export function setCsrfToken(value: string): void {
  csrfToken = value;
}

export function clearCsrfToken(): void {
  csrfToken = null;
}

async function request<T>(
  path: string,
  options: {
    method?: "GET" | "POST";
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
    headers.set("X-MyClaw-CSRF", csrfToken);
  }
  if (options.extraHeaders !== undefined) {
    for (const [name, value] of Object.entries(options.extraHeaders)) {
      headers.set(name, value);
    }
  }
  if (webControlCredential !== null) {
    headers.set("X-MyClaw-Control", webControlCredential);
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

export function getProjectSessions(projectId: string): Promise<ProjectSessionsResponse> {
  return request<ProjectSessionsResponse>(
    `/projects/${encodeURIComponent(projectId)}/sessions`,
  );
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
      extraHeaders: { "X-MyClaw-Claim": claimCredential },
    },
  );
}

export function openEventStream(
  onOpen: () => void,
  onClose: () => void,
  onMessage: (value: ServiceEvent) => void,
): WebSocket {
  if (webControlCredential === null) throw new ApiError(403, null);
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${window.location.host}${API_PREFIX}/events`,
    ["myclaw-v1", webControlCredential],
  );
  socket.addEventListener("open", onOpen);
  socket.addEventListener("close", onClose);
  socket.addEventListener("error", onClose, { once: true });
  socket.addEventListener("message", (event) => {
    try {
      const value: unknown = JSON.parse(event.data as string);
      if (isServiceEvent(value)) onMessage(value);
    } catch {
      // Invalid events do not affect the authenticated connection state.
    }
  });
  return socket;
}

function createRequestId(): string {
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

function isServiceEvent(value: unknown): value is ServiceEvent {
  if (typeof value !== "object" || value === null) return false;
  const event = value as Partial<ServiceEvent>;
  return (
    event.protocol_version === 1 &&
    typeof event.service_instance_id === "string" &&
    typeof event.stream_id === "string" &&
    typeof event.seq === "number" &&
    typeof event.type === "string" &&
    typeof event.workspace_id === "string" &&
    (event.project_id === null || typeof event.project_id === "string") &&
    (event.session_id === null || typeof event.session_id === "string") &&
    (event.run_id === null || typeof event.run_id === "string") &&
    typeof event.payload === "object" &&
    event.payload !== null
  );
}
