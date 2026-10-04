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

export interface PendingConfirmationSnapshot {
  workspace_id: string;
  project_id: string | null;
  session_id: string | null;
  run_id: string | null;
  payload: ConfirmationRequestedPayload;
}

export interface RecoverySnapshot {
  sessions: SessionRecoverySnapshot[];
  pending_confirmation: PendingConfirmationSnapshot | null;
}

export interface SessionRecoverySnapshot {
  workspace_id: string;
  claim_version: number;
  snapshot: SessionSnapshot;
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

export interface ConfigFields {
  runtime: {
    max_tool_result_chars: number;
    max_iterations: number;
    enable_skill_always_load: boolean;
    compact_ratio: number;
    permission_level: ToolPermissionLevel;
    exec_shell: "auto" | "powershell" | "pwsh";
  };
  memory: {
    batch_size: number;
    schedule: string;
  };
  web: {
    default_chat_workspace: string;
  };
  models: ConfigModelsFields;
  mcp: Record<string, ConfigMcpFields>;
}

export interface ConfigRedactedSecret {
  configured: boolean;
}

export interface ConfigProviderFields {
  protocol: string;
  base_url: string;
  models: string[];
  api_key: ConfigRedactedSecret;
}

export interface ConfigRouteFields {
  provider_id: string;
  model: string;
  context_window: number;
  max_output: number;
  temperature: number;
  reasoning_effort: ReasoningEffort;
  timeout: number;
}

export interface ConfigMcpFields {
  enabled: boolean;
  transport: "stdio" | "streamable-http";
  command: string | null;
  args: string[];
  cwd: string | null;
  url: string | null;
  headers: Record<string, ConfigRedactedSecret>;
  connect_timeout: number;
  call_timeout: number;
  tool_keywords: Record<string, string[]>;
}

export interface ConfigModelsFields {
  providers: Record<string, ConfigProviderFields>;
  routes: Record<string, ConfigRouteFields>;
}

export type ConfigPatchFields = {
  runtime?: Partial<ConfigFields["runtime"]>;
  memory?: Partial<ConfigFields["memory"]>;
  web?: Partial<ConfigFields["web"]>;
  models?: {
    providers?: Record<string, Partial<Omit<ConfigProviderFields, "api_key">>>;
    routes?: Record<string, Partial<ConfigRouteFields>>;
  };
  mcp?: Record<string, Partial<ConfigMcpFields>>;
};

export type ConfigSecretChange =
  | { action: "keep" }
  | { action: "clear" }
  | { action: "replace"; value: string };

export type ConfigSecrets = Record<string, ConfigSecretChange>;

export type ConfigApplicationStatus = "active" | "restart-required" | "pending-repair";

export type ConfigProjectionState = "active" | "missing" | "invalid" | "malformed";

export interface ConfigProjection {
  state: ConfigProjectionState;
  repair_required: boolean;
  backup_required: boolean;
  requires_secret_reentry: boolean;
  error: { code: string; message: string } | null;
}

export interface ConfigApplication {
  status: ConfigApplicationStatus;
  saved_revision: string;
  active_revision: string | null;
  restart_required: boolean;
}

export interface ConfigResponse {
  revision: string;
  fields: ConfigFields;
  configuration: ConfigProjection;
  application: ConfigApplication;
}

export interface ConfigPatchResponse extends ConfigResponse {
  request_id: string;
  backup_id?: string | null;
}

export type ToolPermissionLevel = "read-only" | "workspace-write" | "full-access";
export type ReasoningEffort = "low" | "medium" | "high" | "xhigh" | "max";

export interface ManagementError {
  code: string;
  message: string;
  retryable: boolean;
  retry_after_seconds: number | null;
}

export interface DreamResult {
  status: string;
  processed_count: number;
  memory_updated: boolean;
  cursor: number;
  error: ManagementError | null;
}

export interface SkillMetadata {
  name: string;
  description: string;
  path: string;
}

export interface RuntimeStatus {
  version: string;
  chat_model: string;
  chat_reasoning_effort: ReasoningEffort;
  uptime_seconds: number;
  context_window: number;
  max_output: number;
  available_context: number;
  compact_ratio: number;
  compact_context_window: number;
  projected_next_request_tokens: number;
  projection_source: "estimated" | "reported_delta";
  input_budget_used_percent: number;
  session_message_count: number;
  last_compacted: number;
  cumulative_usage: Record<string, number>;
  configured_permission_level: ToolPermissionLevel;
  current_permission_level: ToolPermissionLevel;
  schedule?: {
    status?: string;
    active_job_count?: number;
  };
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

export interface ScheduleJobState {
  last_finished_at_ms: number | null;
  last_status: "ok" | "error" | null;
  last_error: string | null;
}

export type ScheduleJobStatus = "scheduled" | "running" | "ok" | "error" | "deleted";

export interface ScheduleJob {
  job_id: string;
  source: "user" | "system";
  message: string;
  title: string;
  schedule: ProjectSchedule;
  state: ScheduleJobState;
  created_at_ms: number;
  updated_at_ms: number;
  session_id: string;
  active: boolean;
  status: ScheduleJobStatus;
}

export interface ScheduleStatus {
  admitted: boolean;
  status: "available" | "faulted";
  active_job_count: number;
}

export interface ScheduleJobsResponse {
  workspace_id: string;
  jobs: ScheduleJob[];
  status: ScheduleStatus;
}

export interface ScheduleJobResponse {
  request_id?: string;
  workspace_id: string;
  job: ScheduleJob;
  status: ScheduleStatus;
  deleted?: boolean;
  canceled?: boolean;
}

export type ScheduleHistoryResultState = "success" | "failure" | "canceled" | "unknown";

export interface ScheduleHistoryGroup {
  started_at: string | null;
  finished_at: string | null;
  result_state: ScheduleHistoryResultState;
  complete: boolean;
  messages: Record<string, unknown>[];
}

export interface ScheduleJobHistoryResponse {
  workspace_id: string;
  job_id: string;
  session_id: string;
  job: ScheduleJob;
  status: ScheduleStatus;
  groups: ScheduleHistoryGroup[];
  next_cursor: string | null;
}

export interface ScheduleJobInput {
  message: string;
  title?: string;
  kind?: ProjectScheduleKind;
  at_time?: string;
  every_seconds?: number;
  cron_expr?: string;
  timezone?: string;
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
  restore_anchors: RestoreAnchor[];
  live_state: SessionLiveState | null;
}

export interface SessionLiveState {
  stream_id: string;
  seq: number;
  runs: ActiveRunSnapshot[];
}

export interface ActiveRunSnapshot {
  run_id: string;
  request_id: string;
  prompt: string;
  status: "accepted" | "running";
  assistant_content: string;
  tools: ActiveToolSnapshot[];
  cancel_requested: boolean;
  cancellable: boolean;
}

export interface ActiveToolSnapshot {
  tool_call_id: string;
  name: string;
  arguments: string;
  status: "running" | "completed" | "failed" | "rejected" | "canceled" | "unknown";
}

export interface RestoreAnchor {
  anchor_id: number;
  run_token: string;
  content: string;
  timestamp: string;
}

export type RestoreMode = "conversation-only" | "files";

export interface RestoreTarget {
  canonical_target: string;
  operation_id: number;
  requested_targets: string[];
  before_exists: boolean;
  before_sha256: string | null;
  latest_after_exists: boolean | null;
  latest_after_sha256: string | null;
  external: boolean;
  session_owned: boolean;
  backup_error: string | null;
}

export interface RestorePlan {
  session_id: string;
  anchor_id: number;
  session_digest: string;
  journal_revision: number;
  removed_users: number;
  removed_messages: number;
  targets: RestoreTarget[];
  external_target_count: number;
  backup_gaps: Record<string, unknown>[];
  integrity_issues: Record<string, unknown>[];
  conflict_targets: string[];
  discarded_run_tokens: string[];
  available_modes: RestoreMode[];
}

export interface RestoreFileResult {
  target: string;
  operation_id: number;
  status: "restored" | "unchanged" | "failed";
  conflict: boolean;
  error: string | null;
}

export interface RestoreResult {
  session_id: string;
  anchor_id: number;
  mode: RestoreMode;
  removed_users: number;
  removed_messages: number;
  file_results: RestoreFileResult[];
  session_result: Record<string, unknown>;
  failure_notification_acknowledged: boolean;
}

export interface ManagementResult {
  handled: boolean;
  output: string | null;
  memory_content?: string | null;
  dream_result?: DreamResult;
  management_error?: ManagementError;
  status_view?: RuntimeStatus;
  effort_selection?: ReasoningEffort | null;
  permission_selection?: ToolPermissionLevel | null;
  published_effort?: ReasoningEffort | null;
  published_permission_level?: ToolPermissionLevel | null;
  skill_metadata?: SkillMetadata[];
  restore_plan?: RestorePlan;
  restore_result?: RestoreResult | null;
}

export interface ManagementResponse {
  request_id: string;
  result: ManagementResult;
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

export interface WorkspaceSessionsResponse {
  workspace_id: string;
  sessions: SessionSummary[];
  next_cursor: string | null;
}

export interface ChatSessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  directory: string;
  available: boolean;
}

export interface ChatSessionsResponse {
  sessions: ChatSessionSummary[];
  next_cursor: string | null;
}

export interface ChatWorkspaceEntry {
  request_id: string;
  workspace_id: string;
  directory: string;
  project_id: null;
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

export interface SessionDeletion {
  request_id: string;
  project_id: string;
  workspace_id: string;
  session_id: string;
  deleted: true;
}

export interface SessionDeletionStatus {
  project_id: string;
  workspace_id: string;
  session_id: string;
  state: "deleted" | "deleting" | "present";
}

export interface SessionDeletionClaim {
  request_id: string;
  project_id: string;
  workspace_id: string;
  session_id: string;
  claim: SessionClaim;
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
  permission_level: ToolPermissionLevel;
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
