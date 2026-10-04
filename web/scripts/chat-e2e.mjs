import assert from "node:assert/strict";
import { readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";

const expect = playwrightExpect.configure({ timeout: 30000 });
const control = await setup({ shutdownTimeoutMs: 60000 });
const browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
try {
  const page = await browser.newPage({ locale: "en" });
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
    await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
    const response = await entry;
    return { status: response.status(), body: await response.json() };
  }
  async function changeDirectory(directory) {
    await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
    await page.getByRole("navigation", { name: "Settings sections", exact: true }).getByRole("button", { name: "General & appearance", exact: true }).click();
    const saved = page.waitForResponse(response => response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH");
    await page.getByLabel("Default conversation workspace", { exact: true }).fill(directory);
    await page.getByLabel("Default conversation workspace", { exact: true }).blur();
    assert.equal((await saved).status(), 200);
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
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("chat acceptance message");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  await expect.poll(async () => (await api("/chat/sessions")).body.sessions.length).toBe(1);
  const first = (await api("/chat/sessions")).body.sessions[0];
  const header = JSON.parse((await readFile(join(first.directory, ".omni", "sessions", `${first.id}.jsonl`), "utf8")).split("\n")[0]);
  assert.equal(header.metadata.creation_scope, "chat");
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
  assert.equal((await newChat()).status, 422);
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
  await restartedNavigation.getByTitle(firstConversation.directory).click();
  await expect(page.getByRole("log").getByText("chat acceptance message", { exact: true })).toBeVisible();
  await restartedNavigation.getByTitle(secondConversation.directory).click();
  await expect(page.getByRole("log").getByText("second directory acceptance message", { exact: true })).toBeVisible();

  // Persisted fixtures exercise the real service list, pagination and Claim paths.
  const seeded = await control.command("chat-history-seed");
  await page.locator("#app-sidebar").getByRole("link", { name: "Projects", exact: true }).click();
  await page.getByRole("button", { name: "Add project", exact: true }).first().click();
  const projectDialog = page.getByRole("dialog");
  await projectDialog.getByLabel("Absolute local path", { exact: true }).fill(nextDirectory);
  await projectDialog.getByRole("button", { name: "Register project", exact: true }).click();
  await expect(projectDialog).toBeHidden();
  await page.locator("#app-sidebar").getByRole("link", { name: "chat-next", exact: true }).click();
  const projectSessions = page.getByRole("list", { name: "Conversation Sessions", exact: true });
  await expect(projectSessions.getByRole("button", { name: /Legacy project/ })).toBeVisible();
  await expect(projectSessions.getByRole("button", { name: /Explicit project/ })).toBeVisible();
  await expect(projectSessions.getByRole("button")).toHaveCount(2);
  await projectSessions.getByRole("button", { name: /Legacy project/ }).click();
  await expect(page.getByRole("log").getByText("Legacy project body", { exact: true })).toBeVisible();
  assert.equal((await newChat()).status, 200);
  await expect(restartedNavigation.getByRole("button")).toHaveCount(100);
  await expect(restartedNavigation.getByText("Legacy chat", { exact: true })).toBeVisible();
  await expect(restartedNavigation.getByText("Legacy project", { exact: true })).toHaveCount(0);
  await expect(restartedNavigation.getByText("Explicit project", { exact: true })).toHaveCount(0);
  await expect(restartedNavigation.getByTitle(secondConversation.directory)).toBeVisible();
  const historySidebar = page.getByRole("complementary", { name: "Conversations", exact: true });
  await historySidebar.getByRole("button", { name: "Load more", exact: true }).click();
  await expect(restartedNavigation.getByRole("button")).toHaveCount(104);
  const oldest = seeded.history.find(session => session.title === "Paginated chat 000");
  assert.ok(oldest);
  await restartedNavigation.getByRole("button", { name: /Paginated chat 000/ }).click();
  await expect(page.getByRole("log").getByText("Paginated chat 000 body", { exact: true })).toBeVisible();
  await expect(page.getByText(firstConversation.directory, { exact: true }).last()).toBeVisible();
  await restartedNavigation.getByRole("button", { name: /Legacy chat/ }).click();
  await expect(page.getByRole("log").getByText("Legacy chat body", { exact: true })).toBeVisible();
  await restartedNavigation.getByTitle(secondConversation.directory).click();
  await expect(page.getByRole("log").getByText("second directory acceptance message", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Rename session", exact: true }).click();
  const renameDialog = page.getByRole("dialog");
  await renameDialog.getByLabel("Session title", { exact: true }).fill("Renamed shared chat");
  await renameDialog.getByRole("button", { name: "Save", exact: true }).click();
  await expect(restartedNavigation.getByText("Renamed shared chat", { exact: true })).toBeVisible();
  const renamed = (await api("/chat/sessions")).body.sessions.find(session => session.id === secondConversation.id);
  assert.equal(renamed.directory.toLowerCase(), nextDirectory.toLowerCase());
  await page.getByRole("textbox", { name: "Message input", exact: true }).fill("shared chat branch to restore");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toHaveCount(2);
  await page.getByRole("button", { name: "Restore", exact: true }).click();
  await page.locator("#restore-anchor-select").selectOption("2");
  await page.getByRole("button", { name: "Inspect restore", exact: true }).click();
  await expect(page.getByText("Restore preview", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Restore session", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "Restore completed" })).toBeVisible();
  await expect(page.getByRole("log").getByText("shared chat branch to restore", { exact: true })).toHaveCount(0);
  const restored = (await api("/chat/sessions")).body.sessions.find(session => session.id === secondConversation.id);
  assert.equal(restored.directory.toLowerCase(), nextDirectory.toLowerCase());
  const restoredHeader = JSON.parse((await readFile(join(nextDirectory, ".omni", "sessions", `${secondConversation.id}.jsonl`), "utf8")).split("\n")[0]);
  assert.equal(restoredHeader.metadata.creation_scope, "chat");
  await page.getByRole("status").filter({ hasText: "Restore completed" }).getByRole("button", { name: "Close", exact: true }).click();
  await page.getByRole("button", { name: "Delete session", exact: true }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete session", exact: true }).click();
  await expect.poll(async () => (await api("/chat/sessions")).body.sessions.some(session => session.id === secondConversation.id)).toBe(false);
  const sharedProjectId = (await api("/projects")).body.projects.find(project => project.path.toLowerCase() === nextDirectory.toLowerCase()).project_id;
  const remainingProjectSessions = (await api(`/projects/${sharedProjectId}/sessions`)).body.sessions;
  assert.deepEqual(new Set(remainingProjectSessions.map(session => session.title)), new Set(["Legacy project", "Explicit project"]));

  const unavailableDirectory = join(control.details.home_root, "unavailable-chat");
  await changeDirectory(unavailableDirectory);
  assert.equal((await newChat()).status, 200);
  await changeDirectory(nextDirectory);
  assert.equal((await newChat()).status, 200);
  await rename(unavailableDirectory, join(control.details.home_root, "unavailable-chat-moved"));
  const refreshHistoryButton = page.getByRole("complementary", { name: "Conversations", exact: true })
    .getByRole("button", { name: "Refresh sessions", exact: true });
  await expect(refreshHistoryButton).toBeEnabled();
  await refreshHistoryButton.click();
  await expect(page.getByRole("status").filter({ hasText: unavailableDirectory })).toBeVisible();
  await assert.rejects(stat(unavailableDirectory), { code: "ENOENT" });
  await rename(join(control.details.home_root, "unavailable-chat-moved"), unavailableDirectory);
  assert.deepEqual(errors, []);
  console.log("Chat E2E: cross-directory/restart history, pagination, legacy and shared-directory classification, Claim, rename/restore/delete ownership, unavailable history, draft/send/cancel/settings and CLI boundary passed");
} catch (error) {
  console.error(error);
  throw error;
} finally {
  await browser.close();
  await control.shutdown();
}
