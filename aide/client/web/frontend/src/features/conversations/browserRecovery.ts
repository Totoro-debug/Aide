import type { ReasoningEffort, SessionModelConfiguration } from "../../shared/service/protocol";
import { isNonEmptyString, isRecord } from "../../shared/validation.ts";
import { REASONING_EFFORTS } from "./reasoningEffort.ts";

const BROWSER_RECOVERY_KEY = "aide.browser-recovery";

export type BrowserRecoveryTarget =
  | { kind: "project"; project_id: string }
  | { kind: "chat"; directory: string }
  | { kind: "new-chat" };

export interface SessionBrowserRecoverySnapshot {
  version: 1;
  service_instance_id: string;
  target: Exclude<BrowserRecoveryTarget, { kind: "new-chat" }>;
  session_id: string;
  draft: boolean;
  input_text: string;
  model_configuration: SessionModelConfiguration | null;
  scroll_top: number;
}

export interface NewChatBrowserRecoverySnapshot {
  version: 1;
  service_instance_id: string;
  target: { kind: "new-chat" };
  session_id: null;
  draft: true;
  input_text: string;
  model_configuration: null;
  scroll_top: 0;
}

export type BrowserRecoverySnapshot = SessionBrowserRecoverySnapshot | NewChatBrowserRecoverySnapshot;

export interface BrowserRecoveryConnection {
  currentInstanceId: string;
  previousInstanceId: string | null;
  initial: boolean;
  expectedRestart: boolean;
  location: { pathname: string; search: string };
}

export interface BrowserRecoveryDecision {
  snapshot: BrowserRecoverySnapshot | null;
  storage: "keep" | "write" | "clear";
  serviceChanged: boolean;
  navigation: { kind: "conversation" | "settings-return"; route: string } | null;
}

export function reconcileBrowserRecovery(
  snapshot: BrowserRecoverySnapshot | null,
  connection: BrowserRecoveryConnection,
): BrowserRecoveryDecision {
  const { currentInstanceId, previousInstanceId, initial, location } = connection;
  const serviceChanged = previousInstanceId !== null && previousInstanceId !== currentInstanceId;
  const expectedRestart = serviceChanged && connection.expectedRestart;
  let recovery = snapshot;
  let storage: BrowserRecoveryDecision["storage"] = "keep";
  let navigation: BrowserRecoveryDecision["navigation"] = null;

  if (expectedRestart && recovery !== null) {
    recovery = { ...recovery, service_instance_id: currentInstanceId };
    storage = "write";
    navigation = {
      kind: location.pathname === "/settings" ? "settings-return" : "conversation",
      route: browserRecoveryRoute(recovery),
    };
  }
  if ((serviceChanged && !expectedRestart)
    || (recovery !== null && recovery.service_instance_id !== currentInstanceId)) {
    recovery = null;
    storage = "clear";
    if ((initial || serviceChanged) && isConversationPath(location.pathname)) {
      navigation = { kind: "conversation", route: "/" };
    }
  } else if (recovery !== null && initial
    && shouldRestoreBrowserRecovery(location.pathname, location.search, recovery)) {
    navigation = { kind: "conversation", route: browserRecoveryRoute(recovery) };
  }
  return { snapshot: recovery, storage, serviceChanged, navigation };
}

export function readBrowserRecoverySnapshot(): BrowserRecoverySnapshot | null {
  try {
    const raw = window.localStorage.getItem(BROWSER_RECOVERY_KEY);
    if (raw === null) return null;
    const value: unknown = JSON.parse(raw);
    return isBrowserRecoverySnapshot(value) ? value : null;
  } catch {
    return null;
  }
}

export function writeBrowserRecoverySnapshot(snapshot: BrowserRecoverySnapshot): void {
  try {
    const serialized = JSON.stringify(snapshot);
    if (window.localStorage.getItem(BROWSER_RECOVERY_KEY) !== serialized) {
      window.localStorage.setItem(BROWSER_RECOVERY_KEY, serialized);
    }
  } catch {
    // Recovery is optional when browser storage is unavailable.
  }
}

export function clearBrowserRecoverySnapshot(): void {
  try {
    window.localStorage.removeItem(BROWSER_RECOVERY_KEY);
  } catch {
    // Recovery is optional when browser storage is unavailable.
  }
}

export function browserRecoveryRoute(snapshot: BrowserRecoverySnapshot): string {
  if (snapshot.session_id === null) return "/";
  const params = new URLSearchParams({ session: snapshot.session_id });
  if (snapshot.target.kind === "project") {
    return `/projects/${encodeURIComponent(snapshot.target.project_id)}?${params}`;
  }
  params.set("directory", snapshot.target.directory);
  return `/chat?${params}`;
}

export function isBrowserRecoverySnapshot(value: unknown): value is BrowserRecoverySnapshot {
  if (!isRecord(value)
    || !hasOnlyKeys(value, [
      "version", "service_instance_id", "target", "session_id", "draft", "input_text",
      "model_configuration", "scroll_top",
    ])
    || value.version !== 1
    || !isNonEmptyString(value.service_instance_id)
    || typeof value.draft !== "boolean"
    || typeof value.input_text !== "string"
    || typeof value.scroll_top !== "number"
    || !Number.isFinite(value.scroll_top)
    || value.scroll_top < 0
    || !(value.model_configuration === null || isSessionModelConfiguration(value.model_configuration))) {
    return false;
  }
  if (!isRecord(value.target)) return false;
  if (value.target.kind === "new-chat") {
    return hasOnlyKeys(value.target, ["kind"])
      && value.session_id === null
      && value.draft === true
      && value.model_configuration === null
      && value.scroll_top === 0;
  }
  if (!isNonEmptyString(value.session_id)) return false;
  return value.target.kind === "project"
    ? hasOnlyKeys(value.target, ["kind", "project_id"]) && isNonEmptyString(value.target.project_id)
    : value.target.kind === "chat"
      && hasOnlyKeys(value.target, ["kind", "directory"])
      && isNonEmptyString(value.target.directory);
}

function isSessionModelConfiguration(value: unknown): value is SessionModelConfiguration {
  return isRecord(value)
    && hasOnlyKeys(value, ["provider_id", "model", "reasoning_effort"])
    && isNonEmptyString(value.provider_id)
    && isNonEmptyString(value.model)
    && typeof value.reasoning_effort === "string"
    && REASONING_EFFORTS.includes(value.reasoning_effort as ReasoningEffort);
}

function hasOnlyKeys(value: Record<string, unknown>, keys: string[]): boolean {
  return Object.keys(value).length === keys.length && Object.keys(value).every((key) => keys.includes(key));
}

function isConversationPath(pathname: string): boolean {
  return pathname === "/" || pathname === "/chat" || /^\/projects\/[^/]+$/.test(pathname);
}

function shouldRestoreBrowserRecovery(
  pathname: string,
  search: string,
  snapshot: BrowserRecoverySnapshot,
): boolean {
  if (!isConversationPath(pathname)) return false;
  const params = new URLSearchParams(search);
  if (params.has("session")) return false;
  if (pathname.startsWith("/projects/")) {
    let projectId: string;
    try {
      projectId = decodeURIComponent(pathname.slice("/projects/".length));
    } catch {
      return false;
    }
    return snapshot.target.kind === "project" && snapshot.target.project_id === projectId;
  }
  const requestedDirectory = params.get("directory");
  return requestedDirectory === null
    || (snapshot.target.kind === "chat" && snapshot.target.directory === requestedDirectory);
}
