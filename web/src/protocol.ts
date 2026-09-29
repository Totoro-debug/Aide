export type ServiceState = "starting" | "ready" | "reconnecting" | "draining" | "stopped";

export interface ServiceEvent {
  protocol_version: 1;
  service_instance_id: string;
  stream_id: string;
  seq: number;
  type: string;
  workspace_id: string;
  project_id: string | null;
  session_id: string | null;
  run_id: string | null;
  payload: Record<string, unknown>;
}

export interface ServiceStatus {
  service_instance_id: string;
  protocol_version: number;
  state: ServiceState;
  active_workspace_count: number;
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
