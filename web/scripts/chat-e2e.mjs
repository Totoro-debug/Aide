import assert from "node:assert/strict";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";

const control = await setup();
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
  const oldButton = page.getByRole("navigation", { name: "Conversations", exact: true }).getByRole("button").first();
  await oldButton.click();
  await expect(page.getByRole("log").getByText("chat acceptance message", { exact: true })).toBeVisible();
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
  assert.ok((await api("/chat/sessions")).body.sessions.some(session => session.id === first.id));
  await page.getByRole("navigation", { name: "Conversations", exact: true }).getByRole("button").first().click();
  await expect(page.getByRole("log").getByText("chat acceptance message", { exact: true })).toBeVisible();
  assert.deepEqual(errors, []);
  console.log("Chat E2E: missing directory, draft persistence, send/cancel/Claim, immediate settings, old directory/restart history, directory error recovery and CLI boundary passed");
} finally {
  await browser.close();
  await control.shutdown();
}
