import type { EventStreamConnection } from "./api";
import {
  ApiError,
  exchangeTicket,
  getServiceStatus,
  openEventStream,
  registerWebClient,
  restoreBrowserSession,
  ServiceCommandError,
  subscribeState,
} from "./api";
import type {
  ClientCommand,
  RegisteredClient,
  ServiceCommandResult,
  ServiceEvent,
  ServiceStatus,
} from "./protocol";

export type WebServiceAuthState = "checking" | "ready" | "required" | "error" | "conflict";
export type WebServiceConnectionState = "checking" | "online" | "offline" | "recovering";

export interface WebServiceClientHandlers {
  onAuthState: (state: WebServiceAuthState) => void;
  onConnectionState: (state: WebServiceConnectionState) => void;
  onServiceStatus: (status: ServiceStatus) => void;
  onRegisteredClient: (client: RegisteredClient) => void;
  onServiceInstanceChange: (
    currentInstanceId: string,
    previousInstanceId: string | null,
    initial: boolean,
  ) => void;
  onConnectionOpen: () => void;
  onConnectionClose: () => void;
  onEvent: (event: ServiceEvent) => void;
  onSessionEvent: () => void;
}

interface EventCursor {
  serviceInstanceId: string;
  clientId: string;
  streamId: string;
  seq: number;
}

export class WebServiceClient {
  private active = false;
  private retryTimer: number | null = null;
  private statusTimer: number | null = null;
  private connection: EventStreamConnection | null = null;
  private serviceInstanceId: string | null = null;
  private clientId: string | null = null;
  private eventCursor: EventCursor | null = null;
  private authentication: Promise<void> | null = null;
  private generation = 0;

  constructor(
    private readonly handlers: WebServiceClientHandlers,
    private readonly launchTicket: string | null = null,
  ) {}

  start(): void {
    if (this.active) return;
    this.active = true;
    this.generation += 1;
    void this.connect(true, this.generation);
  }

  sendCommand(command: ClientCommand): Promise<ServiceCommandResult> {
    if (this.connection === null) {
      return Promise.reject(new ServiceCommandError(null, false));
    }
    return this.connection.sendCommand(command);
  }

  close(): void {
    this.active = false;
    this.generation += 1;
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    if (this.statusTimer !== null) window.clearInterval(this.statusTimer);
    this.retryTimer = null;
    this.statusTimer = null;
    const connection = this.connection;
    this.connection = null;
    connection?.close();
  }

  private scheduleReconnect(recovering: boolean): void {
    if (!this.active || this.retryTimer !== null) return;
    if (recovering) this.handlers.onConnectionState("recovering");
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      void this.connect(false, this.generation);
    }, recovering ? 1000 : 3000);
  }

  private async refreshStatus(): Promise<void> {
    const generation = this.generation;
    try {
      const status = await getServiceStatus();
      if (this.active && generation === this.generation
        && status.service_instance_id === this.serviceInstanceId) this.handlers.onServiceStatus(status);
    } catch {
      // The event connection reports transport failures separately.
    }
  }

  private async authenticate(): Promise<void> {
    if (this.launchTicket !== null) {
      await exchangeTicket(this.launchTicket);
    } else if ((await restoreBrowserSession()) === null) {
      throw new ApiError(401, null);
    }
  }

  private async connect(initial: boolean, generation: number): Promise<void> {
    try {
      if (initial) await (this.authentication ??= this.authenticate());
      if (!this.active || generation !== this.generation) return;
      const client = await registerWebClient();
      if (!this.active || generation !== this.generation) return;
      const status = await getServiceStatus();
      if (!this.active || generation !== this.generation) return;

      const previousInstanceId = this.serviceInstanceId;
      const previousClientId = this.clientId;
      this.serviceInstanceId = status.service_instance_id;
      this.clientId = client.client_id;
      if (previousInstanceId !== status.service_instance_id || previousClientId !== client.client_id) {
        this.eventCursor = null;
      }
      this.handlers.onServiceInstanceChange(status.service_instance_id, previousInstanceId, initial);
      this.handlers.onServiceStatus(status);
      this.handlers.onRegisteredClient(client);
      this.handlers.onAuthState("ready");
      this.handlers.onSessionEvent();
      this.statusTimer ??= window.setInterval(() => void this.refreshStatus(), 5000);

      const connection = openEventStream(
        () => {
          if (!this.active || this.connection !== connection) return;
          this.handlers.onConnectionState("online");
          this.handlers.onConnectionOpen();
          void this.subscribe(
            this.eventCursor?.seq ?? null,
            this.eventCursor?.streamId ?? null,
            connection,
          );
        },
        () => {
          if (!this.active || this.connection !== connection) return;
          this.connection = null;
          this.handlers.onConnectionClose();
          this.scheduleReconnect(true);
        },
        (event) => this.handleEvent(event, client, connection),
      );
      this.connection = connection;
    } catch (error) {
      if (!this.active || generation !== this.generation) return;
      if (error instanceof ApiError && error.body?.code === "web_client_exists") {
        this.handlers.onAuthState("conflict");
        this.handlers.onConnectionState("offline");
        return;
      }
      if (initial || (error instanceof ApiError && error.status === 401)) {
        this.handlers.onAuthState(error instanceof ApiError && error.status === 401 ? "required" : "error");
      }
      this.handlers.onConnectionState("offline");
      if (!initial && !(error instanceof ApiError && error.status === 401)) {
        this.scheduleReconnect(false);
      }
    }
  }

  private async subscribe(
    lastSeq: number | null,
    streamId: string | null,
    connection: EventStreamConnection,
  ): Promise<void> {
    try {
      await subscribeState(connection.sendCommand, lastSeq, streamId);
    } catch {
      connection.close();
    }
  }

  private handleEvent(
    event: ServiceEvent,
    client: RegisteredClient,
    connection: EventStreamConnection,
  ): void {
    if (!this.active || this.connection !== connection
      || event.service_instance_id !== this.serviceInstanceId) return;
    const cursor = this.eventCursor;
    if (cursor !== null && cursor.streamId !== event.stream_id) return;
    const sameStream = cursor !== null
      && cursor.serviceInstanceId === event.service_instance_id
      && cursor.clientId === client.client_id
      && cursor.streamId === event.stream_id;
    if (sameStream && event.seq <= cursor.seq) return;
    if (sameStream && event.seq > cursor.seq + 1 && event.type !== "snapshot.required") {
      void this.subscribe(null, event.stream_id, connection);
      return;
    }

    this.eventCursor = {
      serviceInstanceId: event.service_instance_id,
      clientId: client.client_id,
      streamId: event.stream_id,
      seq: event.seq,
    };
    this.handlers.onEvent(event);
    if (event.type.startsWith("session.")
      || ["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
      this.handlers.onSessionEvent();
    }
    void this.refreshStatus();
  }
}
