import assert from "node:assert/strict";
import { mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { URL } from "node:url";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import browserRecoveryAcceptance from "./browser-recovery-e2e.mjs";

const expect = playwrightExpect.configure({ timeout: 30000 });
const control = await setup({ shutdownTimeoutMs: 60000 });
const browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
const diagnostics = [];
let diagnosticPage;
function redact(value) {
  if (Array.isArray(value)) return value.map(redact);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [
      key, /credential|csrf|token|api_key|authorization|secret|ticket/i.test(key) ? "<REDACTED>" : redact(item),
    ]));
  }
  return value;
}
function recordDiagnostic(value) {
  diagnostics.push(redact(value));
  if (diagnostics.length > 500) diagnostics.shift();
}
try {
  const context = await browser.newContext({ locale: "en" });
  context.on("page", target => {
    diagnosticPage = target;
    target.on("pageerror", error => recordDiagnostic({ type: "pageerror", message: error.message }));
    target.on("requestfailed", request => recordDiagnostic({
      type: "requestfailed", path: new URL(request.url()).pathname, error: request.failure()?.errorText,
    }));
    target.on("response", async response => {
      const path = new URL(response.url()).pathname;
      if (!path.startsWith("/api/v1/") || path.includes("/config") || path.includes("/web/")) return;
      const request = response.request();
      recordDiagnostic({
        method: request.method(), path, status: response.status(),
        request: request.postDataJSON(), response: await response.json().catch(() => null),
      });
    });
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const Original = window.WebSocket;
    window.WebSocket = class extends Original {
      constructor(...args) {
        super(...args);
        window.chatControl = args[1][1];
        window.chatSocket = this;
      }
    };
  });
  async function api(path, method = "GET", body) {
    return page.evaluate(async ({ path, method, body }) => {
      const session = await (await window.fetch("/api/v1/web/session")).json();
      const response = await window.fetch(`/api/v1${path}`, {
        method,
        headers: { "Content-Type": "application/json", "X-Omni-Control": window.chatControl, "X-Omni-CSRF": session.csrf_token },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      return { status: response.status, body: await response.json() };
    }, { path, method, body });
  }
  async function newChat() {
    const entry = page.waitForResponse(response => response.url().endsWith("/chat/workspaces/enter") && response.request().method() === "POST");
    const claimed = page.waitForResponse(response => response.request().method() === "POST"
      && /\/sessions\/[^/]+\/claim$/.test(new URL(response.url()).pathname));
    void claimed.catch(() => {});
    await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
    const response = await entry;
    if (response.ok()) {
      assert.equal((await claimed).ok(), true);
      await expect(page.getByRole("textbox", { name: "Message input", exact: true })).toBeEnabled();
    }
    return { status: response.status(), body: await response.json() };
  }
  async function openSessionMore(action) {
    const trigger = page.locator('summary[aria-label^="More options for "]');
    const label = await trigger.getAttribute("aria-label");
    assert.ok(label);
    await trigger.click();
    await page.getByRole("group", { name: label, exact: true }).getByRole("button", { name: action, exact: true }).click();
  }
  async function changeDirectory(directory) {
    await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
    await page.getByRole("navigation", { name: "Settings sections", exact: true }).getByRole("button", { name: "General & appearance", exact: true }).click();
    const saved = page.waitForResponse(response => response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH");
    await page.getByLabel("Default conversation workspace", { exact: true }).fill(directory);
    await page.getByLabel("Default conversation workspace", { exact: true }).blur();
    assert.equal((await saved).status(), 200);
  }
  async function createScheduleJob(title, message) {
    await page.locator("#schedule-field-message").fill(message);
    await page.locator("#schedule-field-title").fill(title);
    await page.getByRole("group", { name: "Schedule type", exact: true }).getByRole("button", { name: "Every", exact: true }).click();
    await page.locator("#schedule-field-every_seconds").fill("3600");
    const created = page.waitForResponse(response => response.request().method() === "POST"
      && /\/schedule\/jobs$/.test(new URL(response.url()).pathname));
    await page.getByRole("button", { name: "Create Job", exact: true }).click();
    assert.equal((await created).ok(), true);
    await expect(page.getByRole("listitem").filter({ hasText: title })).toBeVisible();
  }
  async function deleteScheduleJob(title) {
    const job = page.getByRole("listitem").filter({ hasText: title });
    await job.getByRole("button", { name: "Delete", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Delete this Schedule Job?", exact: true });
    const deleted = page.waitForResponse(response => response.request().method() === "DELETE"
      && /\/schedule\/jobs\//.test(new URL(response.url()).pathname));
    await dialog.getByRole("button", { name: "Delete Job", exact: true }).click();
    assert.equal((await deleted).ok(), true);
    await expect(page.getByRole("listitem").filter({ hasText: title })).toHaveCount(0);
  }
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Omni", exact: true }).last()).toBeVisible();
  try {
    await expect(page.getByRole("textbox", { name: "Message input", exact: true })).toBeEnabled();
  } catch (error) {
    console.log((await page.locator("#main-content").innerText()).slice(0, 1600));
    throw error;
  }
  const firstDirectory = join(control.details.home_root, ".omni", "chat");
  assert.deepEqual(await readdir(join(firstDirectory, ".omni", "sessions")), [], "An empty draft was persisted");
  const sessionModel = page.getByRole("combobox", { name: "Session model", exact: true });
  const sessionEffort = page.getByRole("combobox", { name: "Reasoning effort", exact: true });
  const permission = page.getByRole("combobox", { name: "Client permission", exact: true });
  await expect(permission).toHaveValue("full-access");
  await permission.selectOption("workspace-write");
  await expect(permission).toHaveValue("workspace-write");
  await permission.selectOption("full-access");
  const warning = page.getByRole("dialog", { name: "Enable full access?", exact: true });
  await expect(warning.getByRole("button", { name: "Cancel", exact: true })).toBeFocused();
  await warning.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(permission).toHaveValue("workspace-write");
  await permission.selectOption("full-access");
  await warning.getByRole("button", { name: "Enable full access", exact: true }).click();
  await expect(permission).toHaveValue("full-access");
  await sessionEffort.selectOption("high");
  await expect(sessionEffort).toHaveValue("high");
  await expect(sessionModel).toHaveValue(JSON.stringify(["primary", "small-model"]));
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("chat acceptance message");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  await expect.poll(async () => (await api("/chat/sessions")).body.sessions.length).toBe(1);
  const first = (await api("/chat/sessions")).body.sessions[0];
  const header = JSON.parse((await readFile(join(first.directory, ".omni", "sessions", `${first.id}.jsonl`), "utf8")).split("\n")[0]);
  assert.equal(header.metadata.creation_scope, "chat");
  assert.deepEqual(header.metadata.model_configuration, {
    provider_id: "primary",
    model: "small-model",
    reasoning_effort: "high",
  });
  assert.equal(header.metadata.model_configuration_version, 1);
  const firstTurnObservation = (await readFile(control.details.provider_observation_path, "utf8"))
    .trim().split("\n").map(line => JSON.parse(line))
    .find(observation => observation.prompt.includes("chat acceptance message")
      && observation.tools.length > 0);
  assert.ok(firstTurnObservation, "The first chat turn must reach the configured model");
  assert.equal(firstTurnObservation.model, "small-model");
  assert.equal(firstTurnObservation.reasoning_effort, "high");
  assert.equal(firstTurnObservation.max_output, 1024);
  let releaseModels;
  let modelsRequested;
  const modelsGate = new Promise(resolve => { releaseModels = resolve; });
  const modelsStarted = new Promise(resolve => { modelsRequested = resolve; });
  await page.route("**/api/v1/models/available", async route => {
    modelsRequested();
    await modelsGate;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      models: [], default_combination: null,
    }) });
  });
  await page.reload();
  await modelsStarted;
  await expect(page.getByRole("textbox", { name: "Message input", exact: true })).toBeEnabled();
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("unsent model check");
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
  releaseModels();
  await expect(page.getByRole("link", { name: "Configure models", exact: true })).toHaveAttribute("href", "/settings");
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeDisabled();
  await page.unroute("**/api/v1/models/available");
  await page.reload();
  await expect(sessionModel).toHaveValue(JSON.stringify(["primary", "small-model"]));
  await expect(sessionEffort).toHaveValue("high");
  assert.equal((await api("/projects")).body.projects.length, 0);
  const forbidden = await api("/workspaces/attach", "POST", { request_id: "chat-attach-denied", path: control.details.cli_workspace });
  assert.equal(forbidden.status, 403, "Web gained arbitrary CLI directory attachment");
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("streaming");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.getByRole("button", { name: "Cancel run", exact: true }).click();
  await expect(page.getByRole("textbox", { name: "Message input", exact: true })).toBeEnabled();
  const nextDirectory = join(control.details.home_root, "chat-next");
  await changeDirectory(nextDirectory);
  const entered = await newChat();
  assert.equal(entered.status, 200);
  assert.equal(entered.body.directory.toLowerCase(), nextDirectory.toLowerCase());
  await expect(page.getByRole("heading", { name: "Omni", exact: true }).last()).toBeVisible();
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("second directory acceptance message");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  await expect.poll(async () => (await api("/chat/sessions")).body.sessions.length).toBe(2);
  const bothDirectories = (await api("/chat/sessions")).body.sessions;
  const firstConversation = bothDirectories.find(session => session.directory.toLowerCase() === firstDirectory.toLowerCase());
  const secondConversation = bothDirectories.find(session => session.directory.toLowerCase() === nextDirectory.toLowerCase());
  assert.ok(firstConversation && secondConversation, "Both chat directories should appear in history");
  const historyNavigation = page.getByRole("navigation", { name: "Conversations", exact: true });
  await historyNavigation.getByTitle(firstConversation.directory).click();
  await expect(page.getByRole("log").getByText("chat acceptance message", { exact: true })).toBeVisible();
  await historyNavigation.getByTitle(secondConversation.directory).click();
  await expect(page.getByRole("log").getByText("second directory acceptance message", { exact: true })).toBeVisible();
  const fileDirectory = join(control.details.home_root, "chat-file");
  await writeFile(fileDirectory, "not a directory");
  await changeDirectory(fileDirectory);
  const invalidEntry = await newChat();
  assert.equal(invalidEntry.status, 422, JSON.stringify(invalidEntry.body));
  await expect(page.getByRole("alert").filter({ hasText: "workspace is unavailable" })).toBeVisible();
  await changeDirectory(nextDirectory);
  assert.equal((await newChat()).status, 200);
  const restarted = await control.restart();
  await page.goto("about:blank");
  await page.goto(`${restarted.url}/#ticket=${restarted.ticket}`);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Omni", exact: true }).last()).toBeVisible();
  const restartedSessions = (await api("/chat/sessions")).body.sessions;
  assert.ok(restartedSessions.some(session => session.id === first.id));
  assert.ok(restartedSessions.some(session => session.id === secondConversation.id));
  const restartedNavigation = page.getByRole("navigation", { name: "Conversations", exact: true });
  await restartedNavigation.getByTitle(firstConversation.directory).filter({ hasText: firstConversation.title }).click();
  await expect(page.getByRole("log").getByText("chat acceptance message", { exact: true })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Session model", exact: true }))
    .toHaveValue(JSON.stringify(["primary", "small-model"]));
  await expect(page.getByRole("combobox", { name: "Reasoning effort", exact: true }))
    .toHaveValue("high");
  await restartedNavigation.getByTitle(secondConversation.directory).click();
  await expect(page.getByRole("log").getByText("second directory acceptance message", { exact: true })).toBeVisible();

  // Persisted fixtures exercise the real service list, pagination and Claim paths.
  const seeded = await control.command("chat-history-seed");
  const projectsNavigation = page.locator("#app-sidebar").getByRole("region", { name: "Projects", exact: true });
  await projectsNavigation.getByRole("button", { name: "Add project", exact: true }).click();
  const projectDialog = page.getByRole("dialog");
  await projectDialog.getByLabel("Absolute local path", { exact: true }).fill(nextDirectory);
  await projectDialog.getByRole("button", { name: "Register project", exact: true }).click();
  await expect(projectDialog).toBeHidden();
  const projectDisclosure = projectsNavigation.getByRole("button", { name: "Expand sessions for chat-next", exact: true });
  await projectDisclosure.click();
  const projectSessions = projectsNavigation.getByRole("list", { name: "chat-next Sessions", exact: true });
  await expect(projectSessions.getByRole("button", { name: /Legacy project/ })).toBeVisible();
  await expect(projectSessions.getByRole("button", { name: /Explicit project/ })).toBeVisible();
  await expect(projectSessions.getByRole("button")).toHaveCount(2);

  const projectCreation = page.waitForResponse(response => response.request().method() === "POST"
    && /\/projects\/[^/]+\/sessions$/.test(new URL(response.url()).pathname));
  await projectsNavigation.getByRole("button", { name: "New session in chat-next", exact: true }).click();
  const createdProjectSession = await (await projectCreation).json();
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("project row creates in its workspace");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  await expect.poll(async () => (await api(`/projects/${createdProjectSession.project_id}/sessions`)).body.sessions
    .some(session => session.id === createdProjectSession.session_id)).toBe(true);
  await expect(page.getByRole("complementary", { name: "Conversation Sessions", exact: true })).toHaveCount(0);

  await projectSessions.getByRole("button", { name: /Legacy project/ }).click();
  await expect(page.getByRole("log").getByText("Legacy project body", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("log").getByText("Legacy project body", { exact: true })).toBeVisible();
  await expect(projectsNavigation.getByRole("button", { name: "Collapse sessions for chat-next", exact: true }))
    .toHaveAttribute("aria-expanded", "true");

  // A slow Claim must not consume the newer navigation request.
  console.log("Chat E2E: checking rapid session navigation");
  let releaseClaim;
  let claimStarted;
  const delayedClaim = new Promise(resolve => { releaseClaim = resolve; });
  const claimRequestStarted = new Promise(resolve => { claimStarted = resolve; });
  const explicitProject = seeded.history.find(session => session.title === "Explicit project");
  const claimPattern = `**/projects/*/sessions/${explicitProject.id}/claim`;
  await page.route(claimPattern, async route => {
    const response = await route.fetch();
    claimStarted();
    await delayedClaim;
    await route.fulfill({ response });
  });
  await projectSessions.locator('summary[aria-label="Session actions for Explicit project"]').click();
  await projectSessions.getByRole("group", { name: "Session actions for Explicit project", exact: true })
    .getByRole("button", { name: "Rename session", exact: true }).click();
  await Promise.race([claimRequestStarted, new Promise((_, reject) => {
    const timer = setTimeout(() => reject(new Error("Delayed Claim was not requested")), 30000);
    timer.unref();
  })]);
  const legacyProject = seeded.history.find(session => session.title === "Legacy project");
  const newestClaim = page.waitForResponse(response => response.request().method() === "POST"
    && response.url().endsWith(`/sessions/${legacyProject.id}/claim`));
  await projectSessions.getByRole("button", { name: /Legacy project/ }).click();
  releaseClaim();
  await newestClaim;
  await expect(page.getByRole("log").getByText("Legacy project body", { exact: true })).toBeVisible();
  await expect(projectSessions.getByRole("button", { name: /Legacy project/ })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.unroute(claimPattern);

  // A rejected Claim can be retried by clicking the same row again.
  await page.route(claimPattern, route => route.fulfill({
    status: 409,
    contentType: "application/json",
    body: JSON.stringify({ code: "session_claimed", message: "Session is occupied", retryable: true, field_errors: {}, request_id: "navigation-occupied" }),
  }));
  await projectSessions.getByRole("button", { name: /Explicit project/ }).click();
  await expect(page.getByRole("alert").filter({ hasText: "occupied" })).toBeVisible();
  await page.unroute(claimPattern);
  await projectSessions.getByRole("button", { name: /Explicit project/ }).click();
  await expect(page.getByRole("log").getByText("Explicit project body", { exact: true })).toBeVisible();
  await expect(page.getByRole("dialog")).toHaveCount(0);

  // Returning to project management must not replay a consumed Add request.
  await projectsNavigation.getByRole("link", { name: "Projects", exact: true }).click();
  await expect(page.locator("#main-content").getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await projectsNavigation.getByRole("button", { name: "Add project", exact: true }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.getByRole("dialog").getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(page.getByRole("dialog")).toHaveCount(0);
  }

  assert.equal((await newChat()).status, 200);
  const conversationList = restartedNavigation.getByRole("list", { name: "Conversations", exact: true });
  await expect(conversationList.getByRole("button")).toHaveCount(100);
  await expect(restartedNavigation.getByText("Legacy chat", { exact: true })).toBeVisible();
  await expect(restartedNavigation.getByText("Legacy project", { exact: true })).toHaveCount(0);
  await expect(restartedNavigation.getByText("Explicit project", { exact: true })).toHaveCount(0);
  await expect(restartedNavigation.getByTitle(secondConversation.directory)).toBeVisible();
  await restartedNavigation.getByRole("button", { name: "Load more", exact: true }).click();
  await expect(conversationList.getByRole("button")).toHaveCount(104);
  const oldest = seeded.history.find(session => session.title === "Paginated chat 000");
  assert.ok(oldest);
  await restartedNavigation.getByRole("button", { name: /Paginated chat 000/ }).click();
  await expect(page.getByRole("log").getByText("Paginated chat 000 body", { exact: true })).toBeVisible();
  await expect(restartedNavigation.getByRole("button", { name: /Paginated chat 000/ })).toHaveAttribute("aria-current", "page");
  await expect(page.getByText(firstConversation.directory, { exact: true }).last()).toBeVisible();
  await restartedNavigation.getByRole("button", { name: /Legacy chat/ }).click();
  await expect(page.getByRole("log").getByText("Legacy chat body", { exact: true })).toBeVisible();
  await restartedNavigation.getByTitle(secondConversation.directory).click();
  await expect(page.getByRole("log").getByText("second directory acceptance message", { exact: true })).toBeVisible();
  const secondConversationRow = restartedNavigation.getByTitle(secondConversation.directory).locator("xpath=../..");
  const sessionActionTrigger = secondConversationRow.locator("summary");
  const initialSessionActionLabel = await sessionActionTrigger.getAttribute("aria-label");
  assert.ok(initialSessionActionLabel?.startsWith("Session actions for "));
  const sessionActions = secondConversationRow.getByRole("group", { name: initialSessionActionLabel, exact: true });
  await sessionActionTrigger.click();
  await sessionActions.getByRole("button", { name: "View workspace directory", exact: true }).click();
  const directoryDialog = page.getByRole("dialog", { name: "Workspace directory", exact: true });
  await expect(directoryDialog.getByText(secondConversation.directory, { exact: true })).toBeVisible();
  await directoryDialog.getByRole("button", { name: "Close", exact: true }).click();
  await expect(sessionActionTrigger).toBeFocused();
  await sessionActionTrigger.click();
  await sessionActions.getByRole("button", { name: "Rename session", exact: true }).click();
  const renameDialog = page.getByRole("dialog");
  await renameDialog.getByLabel("Session title", { exact: true }).fill("Renamed shared chat");
  await renameDialog.getByRole("button", { name: "Save", exact: true }).click();
  await expect(restartedNavigation.getByText("Renamed shared chat", { exact: true })).toBeVisible();
  const renamed = (await api("/chat/sessions")).body.sessions.find(session => session.id === secondConversation.id);
  assert.equal(renamed.directory.toLowerCase(), nextDirectory.toLowerCase());
  const sharedProjectId = (await api("/projects")).body.projects.find(project => project.path.toLowerCase() === nextDirectory.toLowerCase()).project_id;

  console.log("Chat E2E: checking Workspace management menus");
  await openSessionMore("Runtime status and controls");
  const runtimeDialog = page.getByRole("dialog", { name: "Runtime status", exact: true });
  await expect(runtimeDialog.getByText("Client permission", { exact: true })).toBeVisible();
  await expect(runtimeDialog.getByText("Session model", { exact: true })).toBeVisible();
  await expect(runtimeDialog.getByText("Chat model", { exact: true })).toBeVisible();
  await runtimeDialog.getByRole("button", { name: "Close", exact: true }).click();

  const firstMemoryPath = join(firstDirectory, ".omni", "memory", "memory.md");
  const nextMemoryPath = join(nextDirectory, ".omni", "memory", "memory.md");
  await writeFile(firstMemoryPath, "# First workspace memory marker\n", "utf8");
  await writeFile(nextMemoryPath, "# Next workspace memory marker\n", "utf8");
  await openSessionMore("Workspace Memory and Dream");
  const memoryDialog = page.getByRole("dialog", { name: "Workspace Memory and Dream", exact: true });
  await memoryDialog.getByRole("button", { name: "View Memory", exact: true }).click();
  await expect(memoryDialog.getByRole("region", { name: "Long-term Memory", exact: true }).last()).toBeVisible();
  await expect(memoryDialog.locator("pre")).toContainText("Next workspace memory marker");
  await expect(memoryDialog.locator("pre")).not.toContainText("First workspace memory marker");
  await memoryDialog.getByRole("button", { name: "Run Dream", exact: true }).click();
  await expect(memoryDialog.getByRole("status").filter({ hasText: /Dream completed\.|No pending summaries\./ }).last()).toBeVisible();
  await memoryDialog.getByRole("button", { name: "Close", exact: true }).click();

  await restartedNavigation.getByTitle(firstConversation.directory).filter({ hasText: firstConversation.title }).click();
  await expect(page.getByRole("log").getByText("chat acceptance message", { exact: true })).toBeVisible();
  await openSessionMore("Workspace Memory and Dream");
  await memoryDialog.getByRole("button", { name: "View Memory", exact: true }).click();
  await expect(memoryDialog.locator("pre")).toContainText("First workspace memory marker");
  await expect(memoryDialog.locator("pre")).not.toContainText("Next workspace memory marker");
  await memoryDialog.getByRole("button", { name: "Run Dream", exact: true }).click();
  await expect(memoryDialog.getByRole("status").filter({ hasText: /Dream completed\.|No pending summaries\./ }).last()).toBeVisible();
  await memoryDialog.getByRole("button", { name: "Close", exact: true }).click();
  const openChatSchedule = page.waitForURL(url => url.pathname === "/chat/schedule");
  await openSessionMore("Schedule tasks");
  await openChatSchedule;
  await expect(page.getByRole("heading", { name: "Schedule Jobs", exact: true })).toBeVisible();
  const chatScheduleUrl = new URL(page.url());
  assert.equal(chatScheduleUrl.searchParams.get("directory")?.toLowerCase(), firstDirectory.toLowerCase());
  assert.equal(chatScheduleUrl.searchParams.get("session"), firstConversation.id);
  const chatScheduleJobTitle = "Chat-only schedule job";
  await createScheduleJob(chatScheduleJobTitle, "chat Workspace schedule acceptance task");

  const legacyProjectActions = projectSessions.getByRole("group", { name: "Session actions for Legacy project", exact: true });
  await projectSessions.locator('summary[aria-label="Session actions for Legacy project"]').click();
  await legacyProjectActions.getByRole("button", { name: "View workspace directory", exact: true }).click();
  const projectDirectoryDialog = page.getByRole("dialog", { name: "Workspace directory", exact: true });
  await expect(projectDirectoryDialog.getByText(nextDirectory, { exact: true })).toBeVisible();
  await projectDirectoryDialog.getByRole("button", { name: "Close", exact: true }).click();
  await expect(projectSessions.locator('summary[aria-label="Session actions for Legacy project"]')).toBeFocused();

  await projectSessions.getByRole("button", { name: /Legacy project/ }).click();
  await expect(page.getByRole("log").getByText("Legacy project body", { exact: true })).toBeVisible();
  const openProjectSchedule = page.waitForURL(url => url.pathname === `/projects/${sharedProjectId}/schedule`);
  await openSessionMore("Schedule tasks");
  await openProjectSchedule;
  await expect(page.getByRole("listitem").filter({ hasText: chatScheduleJobTitle })).toHaveCount(0);
  const projectScheduleJobTitle = "Project-only schedule job";
  await createScheduleJob(projectScheduleJobTitle, "Project Workspace schedule acceptance task");

  await restartedNavigation.getByTitle(secondConversation.directory).click();
  await expect(page.getByRole("log").getByText("second directory acceptance message", { exact: true })).toBeVisible();
  const sharedChatSchedule = page.waitForURL(url => url.pathname === "/chat/schedule");
  await openSessionMore("Schedule tasks");
  await sharedChatSchedule;
  await expect(page.getByRole("listitem").filter({ hasText: projectScheduleJobTitle })).toBeVisible();
  await expect(page.getByRole("listitem").filter({ hasText: chatScheduleJobTitle })).toHaveCount(0);
  await restartedNavigation.getByTitle(firstConversation.directory).filter({ hasText: firstConversation.title }).click();
  await expect(page.getByRole("log").getByText("chat acceptance message", { exact: true })).toBeVisible();
  const returnToChatSchedule = page.waitForURL(url => url.pathname === "/chat/schedule");
  await openSessionMore("Schedule tasks");
  await returnToChatSchedule;
  await expect(page.getByRole("listitem").filter({ hasText: chatScheduleJobTitle })).toBeVisible();
  await expect(page.getByRole("listitem").filter({ hasText: projectScheduleJobTitle })).toHaveCount(0);
  await deleteScheduleJob(chatScheduleJobTitle);

  await projectSessions.getByRole("button", { name: /Legacy project/ }).click();
  const deleteProjectSchedule = page.waitForURL(url => url.pathname === `/projects/${sharedProjectId}/schedule`);
  await openSessionMore("Schedule tasks");
  await deleteProjectSchedule;
  await expect(page.getByRole("listitem").filter({ hasText: chatScheduleJobTitle })).toHaveCount(0);
  await deleteScheduleJob(projectScheduleJobTitle);

  await restartedNavigation.getByTitle(secondConversation.directory).click();
  await expect(page.getByRole("log").getByText("second directory acceptance message", { exact: true })).toBeVisible();
  const modelConfigurationBeforeRestore = JSON.parse((await readFile(
    join(nextDirectory, ".omni", "sessions", `${secondConversation.id}.jsonl`), "utf8",
  )).split("\n")[0]).metadata.model_configuration;
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("shared chat branch to restore");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toHaveCount(2);
  const restoreEntry = page.getByRole("article").filter({ hasText: "shared chat branch to restore" });
  await restoreEntry.locator("summary").click();
  await restoreEntry.getByRole("group", { name: "Restore options for this message", exact: true })
    .getByRole("button", { name: "Restore conversation to before this message", exact: true }).click();
  await expect(page.locator("#restore-anchor-select")).toHaveValue("2");
  await page.getByRole("button", { name: "Inspect restore", exact: true }).click();
  await expect(page.getByText("Restore preview", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Restore session", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "Restore completed" })).toBeVisible();
  await expect(page.getByRole("log").getByText("shared chat branch to restore", { exact: true })).toHaveCount(0);
  const restored = (await api("/chat/sessions")).body.sessions.find(session => session.id === secondConversation.id);
  assert.equal(restored.directory.toLowerCase(), nextDirectory.toLowerCase());
  assert.equal(restored.title, "Renamed shared chat");
  const restoredHeader = JSON.parse((await readFile(join(nextDirectory, ".omni", "sessions", `${secondConversation.id}.jsonl`), "utf8")).split("\n")[0]);
  assert.equal(restoredHeader.metadata.creation_scope, "chat");
  assert.deepEqual(restoredHeader.metadata.model_configuration, modelConfigurationBeforeRestore);
  await page.getByRole("status").filter({ hasText: "Restore completed" }).getByRole("button", { name: "Close", exact: true }).click();
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("file restore menu branch");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toHaveCount(2);
  const chatTitleAfterAutoTitle = (await api("/chat/sessions")).body.sessions.find(session => session.id === secondConversation.id);
  assert.equal(chatTitleAfterAutoTitle.title, "Renamed shared chat");
  const fileRestoreEntry = page.getByRole("article").filter({ hasText: "file restore menu branch" });
  await fileRestoreEntry.locator("summary").click();
  await fileRestoreEntry.getByRole("group", { name: "Restore options for this message", exact: true })
    .getByRole("button", { name: "Restore conversation and files to before this message", exact: true }).click();
  await expect(page.locator("#restore-anchor-select")).toHaveValue("3");
  await page.getByRole("button", { name: "Inspect restore", exact: true }).click();
  await expect(page.locator('input[name="restore-mode"][value="files"]')).toBeDisabled();
  await expect(page.locator('input[name="restore-mode"][value="conversation-only"]')).toBeChecked();
  await page.getByRole("dialog").getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);

  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  await page.getByRole("navigation", { name: "Settings sections", exact: true })
    .getByRole("button", { name: "Runtime", exact: true }).click();
  await page.getByRole("button", { name: "Reload Skills", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: /Skills reloaded:/ })).toBeVisible();
  await page.getByRole("button", { name: "Back to conversation", exact: true }).click();
  await expect(page.getByRole("log").getByText("file restore menu branch", { exact: true })).toBeVisible();

  const deleteSessionActions = restartedNavigation.getByRole("group", {
    name: "Session actions for Renamed shared chat", exact: true,
  });
  await restartedNavigation.locator('summary[aria-label="Session actions for Renamed shared chat"]').click();
  await deleteSessionActions.getByRole("button", { name: "Delete session", exact: true }).click();
  const deleteSessionDialog = page.getByRole("dialog", { name: "Delete this Session permanently?", exact: true });
  await deleteSessionDialog.getByRole("button", { name: "Delete session", exact: true }).click();
  await expect.poll(async () => (await api("/chat/sessions")).body.sessions.some(session => session.id === secondConversation.id)).toBe(false);
  const remainingProjectSessions = (await api(`/projects/${sharedProjectId}/sessions`)).body.sessions;
  assert.deepEqual(new Set(remainingProjectSessions.map(session => session.title)), new Set([
    "Legacy project",
    "Explicit project",
    "Fixture response.",
  ]));

  const legacyProjectMenu = projectSessions.getByRole("group", { name: "Session actions for Legacy project", exact: true });
  await projectSessions.locator('summary[aria-label="Session actions for Legacy project"]').click();
  await legacyProjectMenu.getByRole("button", { name: "View workspace directory", exact: true }).click();
  const legacyDirectoryDialog = page.getByRole("dialog", { name: "Workspace directory", exact: true });
  await expect(legacyDirectoryDialog.getByText(nextDirectory, { exact: true })).toBeVisible();
  await legacyDirectoryDialog.getByRole("button", { name: "Close", exact: true }).click();
  await expect(projectSessions.locator('summary[aria-label="Session actions for Legacy project"]')).toBeFocused();

  const explicitProjectMenu = projectSessions.getByRole("group", { name: "Session actions for Explicit project", exact: true });
  await projectSessions.locator('summary[aria-label="Session actions for Explicit project"]').click();
  await explicitProjectMenu.getByRole("button", { name: "Rename session", exact: true }).click();
  const projectRenameDialog = page.getByRole("dialog");
  await projectRenameDialog.getByLabel("Session title", { exact: true }).fill("Manual project title");
  await projectRenameDialog.getByRole("button", { name: "Save", exact: true }).click();
  await expect(projectSessions.getByText("Manual project title", { exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("keep the manual project title");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  const projectTitleAfterAutoTitle = (await api(`/projects/${sharedProjectId}/sessions`)).body.sessions
    .find(session => session.id === seeded.history.find(session => session.title === "Explicit project").id);
  assert.equal(projectTitleAfterAutoTitle.title, "Manual project title");

  const projectSessionToDelete = (await api(`/projects/${sharedProjectId}/sessions`)).body.sessions
    .find(session => session.id === createdProjectSession.session_id);
  const createdProjectMenu = projectSessions.getByRole("group", {
    name: `Session actions for ${projectSessionToDelete.title}`, exact: true,
  });
  await projectSessions.locator(`summary[aria-label="Session actions for ${projectSessionToDelete.title}"]`).click();
  await createdProjectMenu.getByRole("button", { name: "Delete session", exact: true }).click();
  const projectDeleteDialog = page.getByRole("dialog", { name: "Delete this Session permanently?", exact: true });
  await projectDeleteDialog.getByRole("button", { name: "Delete session", exact: true }).click();
  await expect.poll(async () => (await api(`/projects/${sharedProjectId}/sessions`)).body.sessions
    .some(session => session.id === createdProjectSession.session_id)).toBe(false);

  const unavailableDirectory = join(control.details.home_root, "unavailable-chat");
  await changeDirectory(unavailableDirectory);
  assert.equal((await newChat()).status, 200);
  await changeDirectory(nextDirectory);
  assert.equal((await newChat()).status, 200);
  await rename(unavailableDirectory, join(control.details.home_root, "unavailable-chat-moved"));
  const refreshHistoryButton = restartedNavigation.getByRole("button", { name: "Refresh sessions", exact: true });
  await expect(refreshHistoryButton).toBeEnabled();
  await refreshHistoryButton.click();
  await expect(page.getByRole("status").filter({ hasText: unavailableDirectory })).toBeVisible();
  await assert.rejects(stat(unavailableDirectory), { code: "ENOENT" });
  await rename(join(control.details.home_root, "unavailable-chat-moved"), unavailableDirectory);

  // The first Run has no durable list entry yet; its draft remains reachable.
  console.log("Chat E2E: checking running draft navigation");
  await projectsNavigation.getByRole("button", { name: "New session in chat-next", exact: true }).click();
  const draftButton = projectSessions.getByRole("button", { name: /Empty draft|New Session draft/ });
  await expect(draftButton).toHaveAttribute("aria-current", "page");
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("recovery streaming markdown");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  const draftActivity = page.locator("article[data-run-id]").filter({ hasText: "recovery streaming markdown" })
    .getByRole("group", { name: "Run activity", exact: true });
  await expect(draftActivity).not.toHaveAttribute("open");
  await draftActivity.locator("summary").click();
  await expect(page.getByRole("log").getByText("Streamed answer", { exact: true })).toBeVisible();
  await projectSessions.getByRole("button", { name: /Legacy project/ }).click();
  await expect(page.getByRole("log").getByText("Legacy project body", { exact: true })).toBeVisible();
  await draftButton.click();
  await expect(page.getByRole("button", { name: "Cancel run", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Cancel run", exact: true }).click();
  await control.command("settings-release");
  await expect(page.getByRole("textbox", { name: "Message input", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Release session", exact: true }).click();
  await expect(draftButton).toHaveCount(0);
  await expect(projectSessions.locator('[aria-current="page"]')).toHaveCount(0);

  // Project pagination keeps later pages visible through Claim refreshes.
  console.log("Chat E2E: checking project pagination");
  const expectedProjectSessions = (await api(`/projects/${sharedProjectId}/sessions`)).body.sessions.length + 101;
  await control.command("project-history-seed");
  await projectsNavigation.getByRole("button", { name: "Collapse sessions for chat-next", exact: true }).click();
  await projectsNavigation.getByRole("button", { name: "Expand sessions for chat-next", exact: true }).click();
  await expect(projectSessions.getByRole("button")).toHaveCount(100);
  await projectsNavigation.getByRole("button", { name: "Load more", exact: true }).click();
  await expect(projectSessions.getByRole("button")).toHaveCount(expectedProjectSessions);
  await projectSessions.getByRole("button", { name: /Paginated project 000/ }).click();
  await expect(page.getByRole("log").getByText("Paginated project 000 body", { exact: true })).toBeVisible();
  await expect(projectSessions.getByRole("button", { name: /Paginated project 000/ })).toHaveAttribute("aria-current", "page");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation", exact: true }).click();
  await expect(page.locator("#app-sidebar")).toHaveAttribute("data-open", "true");
  await expect(page.getByRole("region", { name: "Projects", exact: true })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Conversations", exact: true })).toBeVisible();
  await expect(page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true })).toBeVisible();
  assert.deepEqual(errors, []);
  console.log("Chat E2E: cross-directory/restart history, pagination, legacy and shared-directory classification, Claim, rename/restore/delete ownership, unavailable history, mobile navigation, draft/send/cancel/settings and CLI boundary passed");
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.keyboard.press("Escape");
  await browserRecoveryAcceptance({ page, control });
} catch (error) {
  await mkdir("test-results/chat-e2e", { recursive: true });
  const pageState = diagnosticPage?.isClosed() === false
    ? await diagnosticPage.locator("#main-content").innerText().catch(() => "") : "";
  const controls = diagnosticPage?.isClosed() === false ? await diagnosticPage.evaluate(() => ({
    route: window.location.pathname + window.location.search,
    controls: [...document.querySelectorAll("main select, main textarea")].map(item => ({
      label: item.getAttribute("aria-label"), disabled: item.disabled, value: item.value,
    })),
  })).catch(() => null) : null;
  await writeFile("test-results/chat-e2e/diagnostics.json", JSON.stringify({
    error: error.message, page: pageState, controls, network: diagnostics, service: control.serviceLog(),
  }, null, 2));
  console.error(error);
  throw error;
} finally {
  await browser.close();
  await control.shutdown();
}
