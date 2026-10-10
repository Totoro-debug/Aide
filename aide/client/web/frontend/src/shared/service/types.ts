import type {
  ServiceEvent
} from "./protocol";

export type AuthState = "checking" | "ready" | "required" | "error" | "conflict";

export type ConnectionState = "checking" | "online" | "offline" | "recovering";

export type ServiceEventListener = (event: ServiceEvent) => void;
