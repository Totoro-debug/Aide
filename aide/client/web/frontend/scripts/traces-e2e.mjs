import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { normalize } from "node:path";
import { readFile } from "node:fs/promises";
import { URL } from "node:url";
import { chromium, expect } from "@playwright/test";
import ts from "typescript";
import setup from "./e2e-setup.mjs";
import { registerProjectFromSidebar, showProjectNavigation } from "./project-ui.mjs";

const routingSource = await readFile(new URL("../src/features/traces/routing.ts", import.meta.url), "utf8");
const routingCode = ts.transpileModule(routingSource, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { traceWorkspaceKeyForPath } = await import(`data:text/javascript;base64,${Buffer.from(routingCode).toString("base64")}`);
const identityPaths = process.platform === "win32"
  ? ["C:\\", "C:/Trace/../Workspace/", "\\\\server\\share\\", "\\\\server\\share\\Trace\\..\\", "D:\\测试\\Workspace"]
  : ["/", "/trace/../workspace/", "/测试/workspace"];
const backendKeys = spawnSync("python", ["-c",
  "import json,sys; from pathlib import Path; from aide.trace.context import workspace_key_for; print(json.dumps([workspace_key_for(Path(p)) for p in json.loads(sys.argv[1])]))",
  JSON.stringify(identityPaths),
], { cwd: new URL("../../../../../", import.meta.url), encoding: "utf8" });
assert.equal(backendKeys.status, 0, backendKeys.stderr);
assert.deepEqual(await Promise.all(identityPaths.map(traceWorkspaceKeyForPath)), JSON.parse(backendKeys.stdout),
  "Conversation Trace links must use the backend Workspace identity");

const control = await setup();
let browser;
let page;
const requests = [];
const unauthenticatedTraceRequests = [];
let webControlCredential = null;
const sessionPaths = new Map();
const htmlResponse = '<html><body>Trace response<script>window.__traceExecuted = true</script></body></html>';
let traceMode = "populated";
let lastSessionId = null;

function workspaceKeyFor(path) {
  const normalized = normalize(path);
  const identity = process.platform === "win32" ? normalized.toLowerCase() : normalized;
  return `sha256:${createHash("sha256").update(identity, "utf8").digest("hex")}`;
}

function record(sessionId, workspaceKey, workspacePath, overrides = {}) {
  return {
    record_id: "record-main",
    root_id: "root-main",
    workspace_key: workspaceKey,
    workspace_path: workspacePath,
    owner_session_id: sessionId,
    subject_session_id: sessionId,
    origin: "foreground",
    run_id: "run-main",
    schedule_run_id: null,
    restore_run_token: null,
    agent_id: null,
    parent_root_id: null,
    parent_record_id: null,
    tool_call_id: null,
    kind: "model",
    purpose: "agent_run",
    tool_name: null,
    model: "model-main",
    reasoning_effort: "high",
    endpoint: "https://provider.example/v1/responses",
    protocol: "openai-compatible",
    output_complete: true,
    status: "success",
    error: null,
    retry_count: 2,
    started_at: "2026-10-10T10:00:00.000Z",
    finished_at: "2026-10-10T10:00:01.000Z",
    duration_ms: 1000,
    final_attempt_duration_ms: 800,
    usage: {
      state: "reported",
      input_tokens: 120,
      output_tokens: 60,
      total_tokens: 180,
      cached_input_tokens: 20,
      cache_creation_input_tokens: 5,
    },
    capture_state: "complete",
    restored: false,
    ...overrides,
  };
}

function recordsFor(sessionId, workspaceKey, workspacePath) {
  return [
    record(sessionId, workspaceKey, workspacePath),
    record(sessionId, workspaceKey, workspacePath, {
      record_id: "record-subagent",
      root_id: "root-subagent",
      subject_session_id: "subagent-session",
      origin: "subagent",
      run_id: "run-subagent",
      agent_id: "agent-child",
      parent_root_id: "root-main",
      kind: "model",
      purpose: "subagent_run",
      model: "model-child",
      usage: { state: "unknown", input_tokens: null, output_tokens: null, total_tokens: null, cached_input_tokens: null, cache_creation_input_tokens: null },
    }),
    record(sessionId, workspaceKey, workspacePath, {
      record_id: "record-cancelled",
      root_id: "root-main",
      kind: "tool",
      purpose: "agent_run",
      tool_name: "exec",
      parent_record_id: "record-main",
      tool_call_id: "call-exec",
      model: null,
      reasoning_effort: null,
      status: "failed",
      error: { category: "cancelled", code: null, message: "Process cancelled by user." },
      usage: {
        state: "not_applicable", input_tokens: null, output_tokens: null, total_tokens: null,
        cached_input_tokens: null, cache_creation_input_tokens: null,
      },
    }),
    record(sessionId, workspaceKey, workspacePath, {
      record_id: "record-rejected",
      root_id: "root-main",
      kind: "tool",
      tool_name: "write_file",
      model: null,
      reasoning_effort: null,
      status: "failed",
      error: { category: "permission_denied", code: "tool_refused", message: "The Tool was denied." },
      usage: {
        state: "not_applicable", input_tokens: null, output_tokens: null, total_tokens: null,
        cached_input_tokens: null, cache_creation_input_tokens: null,
      },
    }),
    record(sessionId, workspaceKey, workspacePath, {
      record_id: "record-restored",
      root_id: "root-main",
      restored: true,
      capture_state: "partial",
      output_complete: false,
    }),
    record(sessionId, workspaceKey, workspacePath, {
      record_id: "record-archive",
      root_id: "root-main",
      kind: "tool",
      purpose: "mcp_keywords",
      tool_name: "search",
      model: null,
      reasoning_effort: null,
      usage: {
        state: "not_applicable", input_tokens: null, output_tokens: null, total_tokens: null,
        cached_input_tokens: null, cache_creation_input_tokens: null,
      },
    }),
  ];
}

function detailFor(selectedRecord) {
  const main = selectedRecord.record_id === "record-main";
  return {
    record: selectedRecord,
    content: selectedRecord.kind === "model" ? {
      output: { text: main ? htmlResponse : "Trace fixture result.", reasoning: "Semantic reasoning.",
        tool_calls: main ? [{ id: "call-exec", name: "exec" }, { id: "missing", name: "missing-tool" }] : [] },
    } : { arguments: { path: "fixture.txt" }, result: { content: "Trace fixture result.", artifact: null } },
    related_tools: main ? [{ record_id: "record-cancelled", tool_call_id: "call-exec", tool_name: "exec", status: "failed" }] : [],
    capture_issues: selectedRecord.record_id === "record-restored"
      ? [{ state: "partial", message: "An output fragment could not be saved.", source: "model_router" }] : [],
  };
}

try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith("/api/v1/")) {
      webControlCredential = request.headers()["x-aide-control"] ?? webControlCredential;
      requests.push({ method: request.method(), path: url.pathname, query: url.searchParams });
    }
  });
  await page.addInitScript(() => {
    window.__traceClaimCommands = [];
    window.__traceClaimCount = Number(window.sessionStorage.getItem("aide.trace-claim-count") ?? "0");
    window.localStorage.setItem("aide.language", "en");
    const Original = window.WebSocket;
    window.WebSocket = class extends Original {
      send(data) {
        try {
          const command = JSON.parse(data);
          if (command.type === "claim") {
            window.__traceClaimCommands.push(command);
            window.__traceClaimCount += 1;
            window.sessionStorage.setItem("aide.trace-claim-count", String(window.__traceClaimCount));
          }
        } catch { /* Non-JSON WebSocket frames are outside this assertion. */ }
        super.send(data);
      }
    };
  });
  await page.route((url) => url.pathname.startsWith("/api/v1/traces"), async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.headers()["x-aide-control"] === undefined) {
      unauthenticatedTraceRequests.push(url.pathname);
      await route.fulfill({ status: 401, json: { code: "unauthorized", message: "Authenticate first" } });
      return;
    }
    if (request.method() !== "GET") {
      await route.fulfill({ status: 405, json: { code: "method_not_allowed", message: "GET only" } });
      return;
    }
    if (traceMode === "unavailable" && url.pathname === "/api/v1/traces") {
      await route.fulfill({ status: 503, json: { code: "trace_unavailable", message: "Unavailable" } });
      return;
    }
    const sessionId = url.searchParams.get("owner_session_id") ?? lastSessionId;
    if (url.pathname === "/api/v1/traces" && sessionId !== null) lastSessionId = sessionId;
    const workspaceKey = url.searchParams.get("workspace_key")
      ?? (sessionId === null ? null : workspaceKeyFor(sessionPaths.get(sessionId)
        ?? (sessionId === control.details.occupied_session_id ? control.details.cli_workspace : control.details.home_root)));
    const workspacePath = sessionId === null
      ? control.details.home_root
      : sessionPaths.get(sessionId) ?? (sessionId === control.details.occupied_session_id
        ? control.details.first_project : control.details.home_root);
    const expectedWorkspaceKey = workspaceKeyFor(workspacePath);
    const workspaceRecords = recordsFor(sessionId ?? "session-main", expectedWorkspaceKey, workspacePath);
    if (sessionId !== null) sessionPaths.set(sessionId, workspacePath);
    if (url.pathname === "/api/v1/traces") {
      const matching = traceMode === "empty" || workspaceKey !== expectedWorkspaceKey ? [] : workspaceRecords.filter((item) => (
        (url.searchParams.get("kind") === null || item.kind === url.searchParams.get("kind"))
        && (url.searchParams.get("model") === null || item.model === url.searchParams.get("model"))
        && (url.searchParams.get("tool_name") === null || item.tool_name === url.searchParams.get("tool_name"))
        && (url.searchParams.get("status") === null || item.status === url.searchParams.get("status"))
        && (url.searchParams.get("purpose") === null || item.purpose === url.searchParams.get("purpose"))
      ));
      const cursor = url.searchParams.get("cursor");
      const hasFilters = ["kind", "model", "tool_name", "status", "purpose"].some((name) => url.searchParams.has(name));
      const items = cursor !== null
        ? matching.filter((item) => item.record_id === "record-archive")
        : matching.filter((item) => hasFilters || item.record_id !== "record-archive");
      await route.fulfill({
        status: 200,
        json: {
          items: items.map((item) => ({ ...item, workspace_key: workspaceKey ?? "", workspace_path: workspacePath })),
          next_cursor: cursor === null && !hasFilters && items.length > 0 ? "trace-page-2" : null,
          snapshot_sequence: 10,
        },
      });
      return;
    }
    const recordPath = url.pathname.match(/^\/api\/v1\/traces\/([^/]+)$/);
    if (recordPath !== null) {
      const recordId = decodeURIComponent(recordPath[1]);
      const found = traceMode === "empty" ? undefined : workspaceRecords.find((item) => item.record_id === recordId);
      if (found === undefined) {
        await route.fulfill({ status: 404, json: { code: "not_found", message: "Missing" } });
      } else {
        await route.fulfill({ status: 200, json: detailFor({ ...found, workspace_key: workspaceKey ?? "", workspace_path: workspacePath }) });
      }
      return;
    }
    await route.fallback();
  });

  const openedResponse = page.waitForResponse((response) => (
    response.request().method() === "POST"
    && new URL(response.url()).pathname === "/api/v1/conversations/open"
  ));
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  const opened = await openedResponse;
  assert.equal(opened.status(), 200);
  const conversation = await opened.json();
  sessionPaths.set(conversation.session_id, conversation.directory);
  await expect(page.locator("#conversation-input")).toBeEnabled();
  await expect(page.getByRole("button", { name: "Execution trace", exact: true })).toBeVisible();
  const openCountBeforeTrace = requests.filter((item) => item.path === "/api/v1/conversations/open").length;
  const claimCountBeforeTrace = await page.evaluate(() => window.__traceClaimCount);

  await page.getByRole("button", { name: "Execution trace", exact: true }).click();
  const routeKey = workspaceKeyFor(conversation.directory);
  await expect(page).toHaveURL(new RegExp(`/traces/sessions/${encodeURIComponent(routeKey)}/${conversation.session_id}`));
  await expect(page.getByRole("heading", { name: "Execution Trace", exact: true })).toBeVisible();
  assert.equal(requests.filter((item) => item.path === "/api/v1/conversations/open").length, openCountBeforeTrace);
  assert.equal(await page.evaluate(() => window.__traceClaimCount), claimCountBeforeTrace);
  const listRequest = requests.find((item) => item.path === "/api/v1/traces");
  assert.equal(listRequest.method, "GET");
  assert.equal(listRequest.query.get("workspace_key"), routeKey);
  assert.equal(listRequest.query.get("owner_session_id"), conversation.session_id);
  await expect(page.getByRole("heading", { name: /Run run-main/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: /SubAgent agent-child/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open record-main", exact: true })).toBeVisible();
  // A purpose/tool present only after the first page must still be queryable.
  await page.getByLabel("Purpose", { exact: true }).selectOption("mcp_keywords");
  await expect(page.getByRole("button", { name: "Open record-archive", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters", exact: true }).click();
  await page.getByLabel("Tool", { exact: true }).fill("search");
  await expect(page.getByRole("button", { name: "Open record-archive", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Clear filters", exact: true }).click();
  await page.getByRole("button", { name: "Load more calls", exact: true }).click();
  await expect(page.getByRole("button", { name: "Open record-archive", exact: true })).toBeVisible();

  await expect(page.getByRole("heading", { name: "model-main", exact: true })).toBeVisible();
  await expect(page.getByText("https://provider.example/v1/responses", { exact: true })).not.toBeVisible();
  await expect(page.getByText("Input tokens", { exact: true }).locator("..")).toContainText("120");
  await expect(page.getByText("Output tokens", { exact: true }).locator("..")).toContainText("60");
  await page.getByText("More information", { exact: true }).click();
  await expect(page.getByText("https://provider.example/v1/responses", { exact: true })).toBeVisible();
  await expect(page.getByText("Cached input tokens", { exact: true }).locator("..")).toContainText("20");
  await expect(page.getByText("Cache creation input tokens", { exact: true }).locator("..")).toContainText("5");
  await expect(page.getByText("Retry count", { exact: true }).locator("..")).toContainText("2");
  await expect(page.locator("pre").filter({ hasText: "window.__traceExecuted = true" })).toHaveText(htmlResponse);
  assert.equal(await page.evaluate(() => window.__traceExecuted === true), false, "Model output must render as text");
  await expect(page.getByText("missing-tool · Trace unavailable", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "exec · Failed", exact: true }).click();
  await expect(page).toHaveURL(/traces\/records\/record-cancelled/);
  await expect(page.getByRole("heading", { name: "exec", exact: true })).toBeVisible();
  await page.getByRole("link", { name: "View initiating model call", exact: true }).click();
  await expect(page).toHaveURL(/traces\/records\/record-main/);
  await expect(page.getByRole("heading", { name: "model-main", exact: true })).toBeVisible();
  await page.goto(`${control.details.url}/traces/sessions/${encodeURIComponent(routeKey)}/${conversation.session_id}?recordId=record-main`);

  await page.getByRole("button", { name: "Open record-subagent", exact: true }).click();
  await expect(page).toHaveURL(/recordId=record-subagent/);
  await expect(page.getByRole("heading", { name: "model-child", exact: true })).toBeVisible();
  await expect(page.getByText("Input tokens", { exact: true }).locator("..")).toContainText("Unknown");
  await page.getByRole("button", { name: "Open record-cancelled", exact: true }).click();
  await expect(page.getByText("Process cancelled by user.", { exact: true })).toBeVisible();
  for (const label of ["Input tokens", "Output tokens", "Total tokens", "Cached input tokens", "Cache creation input tokens"]) {
    await expect(page.getByText(label, { exact: true }).locator("..")).toHaveCount(0);
  }
  await page.getByRole("button", { name: "Open record-rejected", exact: true }).click();
  await expect(page.getByText("The Tool was denied.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Open record-restored", exact: true }).click();
  await expect(page.getByText("Restored", { exact: true })).toBeVisible();
  await expect(page.getByText("An output fragment could not be saved.", { exact: true })).toBeVisible();
  await expect(page.getByText("Output is incomplete", { exact: true })).toBeVisible();

  for (const [label, value, queryName] of [
    ["Kind", "tool", "kind"],
    ["Model", "model-child", "model"],
    ["Tool", "exec", "tool_name"],
    ["Status", "failed", "status"],
    ["Purpose", "subagent_run", "purpose"],
  ]) {
    await page.getByRole("button", { name: "Clear filters", exact: true }).click();
    if (label === "Model" || label === "Tool") await page.getByLabel(label, { exact: true }).fill(value);
    else await page.getByLabel(label, { exact: true }).selectOption(value);
    await expect.poll(() => requests.filter((item) => item.path === "/api/v1/traces").at(-1)?.query.get(queryName))
      .toBe(value);
  }

  const stableRequestCount = requests.length;
  await page.waitForTimeout(500);
  assert.equal(requests.length, stableRequestCount, "Trace data must not poll");
  await page.getByRole("button", { name: "Clear filters", exact: true }).click();
  await expect(page.getByRole("button", { name: "Open record-restored", exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  const mobileLayout = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
  }));
  assert.ok(mobileLayout.documentWidth <= mobileLayout.viewportWidth, JSON.stringify(mobileLayout));
  await page.getByRole("tab", { name: "Calls", exact: true }).click();
  await expect(page.getByRole("button", { name: "Open record-restored", exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Details", exact: true }).click();
  await expect(page.getByText("Restored", { exact: true })).toBeVisible();

  const listOnlyUrl = new URL(page.url());
  listOnlyUrl.searchParams.delete("recordId");
  await page.goto(listOnlyUrl.href);
  await expect(page.getByRole("button", { name: "Open record-main", exact: true })).toBeVisible();
  traceMode = "empty";
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByText("No saved trace calls in this session.", { exact: true })).toBeVisible();
  traceMode = "unavailable";
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByText("Trace storage is unavailable.", { exact: true })).toBeVisible();

  traceMode = "populated";
  const recordUrl = new URL(page.url());
  recordUrl.searchParams.set("recordId", "record-main");
  await page.goto(recordUrl.href);
  await expect(page.getByRole("heading", { name: "model-main", exact: true })).toBeVisible();
  const detailsBeforeRefresh = requests.filter((item) => item.path.endsWith("/record-main")).length;
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect.poll(() => requests.filter((item) => item.path.endsWith("/record-main")).length).toBeGreaterThan(detailsBeforeRefresh);
  assert.equal(requests.filter((item) => item.path.endsWith("/raw")).length, 0);
  traceMode = "empty";
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByText("This call is no longer available. It may have expired or its Session may have been deleted.", { exact: true })).toBeVisible();

  traceMode = "populated";
  const expiredRecordUrl = new URL(page.url());
  expiredRecordUrl.searchParams.set("recordId", "expired-record");
  await page.goto(expiredRecordUrl.href);
  await expect(page.getByText("This call is no longer available. It may have expired or its Session may have been deleted.", { exact: true })).toBeVisible();

  const cliSessionId = control.details.occupied_session_id;
  const cliWorkspaceKey = workspaceKeyFor(control.details.first_project);
  sessionPaths.set(cliSessionId, control.details.first_project);
  lastSessionId = cliSessionId;
  await page.setViewportSize({ width: 1440, height: 960 });
  // Remember this directory through the real Web API before reopening its CLI-owned history.
  await control.command(`config-patch ${JSON.stringify({ web: { default_chat_workspace: control.details.first_project } })}`);
  const browserSession = await (await page.request.get(`${control.details.url}/api/v1/web/session`)).json();
  const rememberDirectory = await page.request.post(`${control.details.url}/api/v1/conversations/open`, {
    headers: { "X-Aide-CSRF": browserSession.csrf_token, "X-Aide-Control": webControlCredential },
    data: { request_id: "trace-remember-directory", create_new: true },
  });
  assert.equal(rememberDirectory.status(), 200);
  assert.equal((await rememberDirectory.json()).directory, control.details.first_project);
  const occupiedChatUrl = new URL(control.details.url);
  occupiedChatUrl.searchParams.set("session", cliSessionId);
  occupiedChatUrl.searchParams.set("directory", control.details.first_project);
  await page.goto(occupiedChatUrl.href);
  const occupiedChatEntry = page.getByTitle(cliSessionId, { exact: true });
  await expect(occupiedChatEntry).toBeVisible();
  const claimCountBeforeChatTrace = await page.evaluate(() => window.__traceClaimCount);
  const openCountBeforeChatTrace = requests.filter((item) => item.path === "/api/v1/conversations/open").length;
  await occupiedChatEntry.click();
  await expect(page).toHaveURL(new RegExp(`/traces/sessions/${encodeURIComponent(cliWorkspaceKey)}/${cliSessionId}`));
  await expect(page.getByRole("heading", { name: "model-main", exact: true })).toBeVisible();
  assert.equal(await page.evaluate(() => window.__traceClaimCount), claimCountBeforeChatTrace);
  assert.equal(requests.filter((item) => item.path === "/api/v1/conversations/open").length, openCountBeforeChatTrace);
  await registerProjectFromSidebar(page, control.details.first_project);
  await showProjectNavigation(page);
  await page.locator("#app-sidebar").getByRole("button", { name: "project-one", exact: true }).click();
  await page.locator("#app-sidebar").getByRole("button", { name: /Web available history/ }).click();
  await expect(page.locator("#conversation-input")).toBeEnabled();
  await page.locator("#app-sidebar").getByRole("button", { name: /CLI occupied history/ }).click();
  const occupiedTraceEntry = page.getByRole("alert").getByRole("button", { name: "Execution trace", exact: true });
  await expect(occupiedTraceEntry).toBeVisible();
  const claimCountBeforeCliTrace = await page.evaluate(() => window.__traceClaimCount);
  const conversationOpenCountBeforeCliTrace = requests.filter((item) => item.path === "/api/v1/conversations/open").length;
  await occupiedTraceEntry.click();
  await expect(page).toHaveURL(new RegExp(`/traces/sessions/${encodeURIComponent(cliWorkspaceKey)}/${cliSessionId}`));
  await expect(page.getByRole("heading", { name: "model-main", exact: true })).toBeVisible();
  assert.equal(await page.evaluate(() => window.__traceClaimCount), claimCountBeforeCliTrace);
  assert.equal(requests.filter((item) => item.path === "/api/v1/conversations/open").length, conversationOpenCountBeforeCliTrace);
  await page.goto(`${control.details.url}/traces/sessions/${encodeURIComponent(cliWorkspaceKey)}/${encodeURIComponent(cliSessionId)}?recordId=record-main`);
  await expect(page.getByRole("heading", { name: "Execution Trace", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "model-main", exact: true })).toBeVisible();
  assert.equal(await page.evaluate(() => window.__traceClaimCount), claimCountBeforeCliTrace);
  assert.equal(requests.filter((item) => item.path === "/api/v1/conversations/open").length, conversationOpenCountBeforeCliTrace);
  assert.deepEqual(unauthenticatedTraceRequests, [], "Trace deep links must wait for browser authentication");
  assert.deepEqual(pageErrors, []);
  console.log("Trace session E2E passed.");
} catch (error) {
  if (page !== undefined) console.error("Trace E2E state:", page.url(), await page.getByRole("alert").allTextContents());
  throw error;
} finally {
  if (browser !== undefined) await browser.close();
  await control.shutdown();
}
