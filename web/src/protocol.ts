export type ServiceState = "starting" | "ready" | "reconnecting" | "draining" | "stopped";

export interface ServiceEvent {
  protocol_version: 1;
  service_instance_id: string;
  stream_id: string;
  seq: number;
  type: string;
  workspace_id: string | null;
  project_id: string | null;
  session_id: string | null;
  run_id: string | null;
  payload: Record<string, unknown>;
}

export type ConfirmationOrigin = "foreground" | "background";

export interface ConfirmationRequest {
  confirmation_id: string;
  tool_call_id: string;
  tool_name: string;
  reason: string;
  summary: string;
  details: Record<string, unknown>;
  warnings: string[];
}

export interface ConfirmationEventOwner {
  kind: ConfirmationOrigin;
  generation_id: string;
  run_id?: string;
  job_id?: string;
  occurrence_id?: string;
}

export interface ConfirmationRequestedPayload {
  token: string;
  origin: ConfirmationOrigin;
  request: ConfirmationRequest;
  job_id?: string;
  title?: string;
  owner?: ConfirmationEventOwner;
}

export type ClientCommandType =
  | "claim"
  | "release"
  | "input"
  | "cancel"
  | "confirmation_decide"
  | "subscribe";

export interface ClientCommand {
  request_id: string;
  type: ClientCommandType;
  workspace_id: string | null;
  session_id: string | null;
  claim_version: number | null;
  payload: Record<string, unknown>;
}

export type ServiceCommandResult = Record<string, unknown>;

export interface ServiceStatus {
  service_instance_id: string;
  protocol_version: number;
  state: ServiceState;
  active_workspace_count: number;
}

export type ProjectScheduleState =
  | "available"
  | "unavailable"
  | "awaiting_resume"
  | "removing"
  | "failed";

export type ProjectScheduleKind = "at" | "every" | "cron";

export interface ProjectSchedule {
  kind: ProjectScheduleKind;
  at_time: string | null;
  every_seconds: number | null;
  cron_expr: string | null;
  timezone: string | null;
}

export interface ProjectJob {
  job_id: string;
  title: string;
  schedule: ProjectSchedule;
  due_at: string | null;
  review_status: "overdue" | "upcoming" | "next_on_resume" | "completed";
}

export interface ProjectScheduleStatus {
  admitted: boolean;
  status: "available" | "faulted";
  active_job_count: number;
}

export interface RegisteredProject {
  project_id: string;
  path: string;
  name: string;
  schedule_state: ProjectScheduleState;
  available: boolean;
  saved_jobs: ProjectJob[];
  schedule_status: ProjectScheduleStatus | null;
  removal_operation_id?: string;
  removal_error?: string;
}

export interface ProjectListResponse {
  projects: RegisteredProject[];
}

export interface ProjectRegistration {
  request_id: string;
  project_id: string;
  workspace_id: string;
  schedule_state: ProjectScheduleState;
  saved_jobs: ProjectJob[];
}

export interface ProjectScheduleResume {
  request_id: string;
  schedule_state: ProjectScheduleState;
}

export interface ProjectRemoval {
  request_id: string;
  project_id: string;
  operation_id: string;
  status: "removing" | "completed" | "failed";
}

export type ProjectRemovalStatus = Omit<ProjectRemoval, "request_id">;

export interface SessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  occupied: boolean;
  occupied_by: string | null;
  metadata_version: number;
}

export interface SessionSnapshot {
  session_id: string;
  messages: Record<string, unknown>[];
}

export interface SessionClaim {
  workspace_id: string;
  session_id: string;
  claim_version: number;
  reconnect_credential: string;
}

export interface ProjectSessionsResponse {
  project_id: string;
  workspace_id: string;
  sessions: SessionSummary[];
  next_cursor: string | null;
}

export interface SessionRenameResponse {
  request_id: string;
  project_id: string;
  session: SessionSummary;
}

export interface SessionCreation {
  request_id: string;
  project_id: string;
  workspace_id: string;
  session_id: string;
}

export interface SessionClaimResponse {
  request_id: string;
  project_id: string;
  workspace_id: string;
  session_id: string;
  claim: SessionClaim;
  snapshot: SessionSnapshot;
}

export interface SessionRelease {
  request_id: string;
  released: true;
}

export interface BrowserSession {
  authenticated: true;
  csrf_token: string;
  client_id: string | null;
}

export interface BrowserTicketExchange {
  authenticated: true;
  csrf_token: string;
}

export interface RegisteredClient {
  request_id: string;
  client_id: string;
  reconnect_credential: string;
  web_control_credential: string;
  permission_level: string;
  current_workspace_id: string | null;
  current_session_id: string | null;
}

export interface WebLaunchTicket {
  request_id: string;
  ticket: string;
  expires_in: number;
}

export interface ServiceErrorBody {
  code: string;
  message: string;
  field_errors: Record<string, string>;
  retryable: boolean;
  request_id: string;
}
