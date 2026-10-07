import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium, expect as playwrightExpect } from "@playwright/test";

import setup from "./e2e-setup.mjs";
import { newProjectConversation, projectItemByPath, projectMenuAction, registerProjectFromSidebar } from "./project-ui.mjs";
import { setInterfaceLanguage, setInterfaceTheme } from "./settings-e2e.mjs";

const expect = playwrightExpect.configure({ timeout: 30000 });
const control = await setup({ shutdownTimeoutMs: 60000 });
const browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
const network = [];
page.on("pageerror", error => errors.push(error.message));
page.on("response", async response => {
  const path = new globalThis.URL(response.url()).pathname;
  if (!path.startsWith("/api/v1/") || path.includes("/config") || path.includes("/web/session")) return;
  const body = await response.json().catch(() => ({}));
  network.push({ path, status: response.status(), code: body.code, state: body.state,
    session: body.session_id, instance: body.service_instance_id });
});
page.on("websocket", socket => {
  socket.on("close", () => network.push({ event: "socket closed" }));
  socket.on("framereceived", ({ payload }) => {
    const event = JSON.parse(String(payload));
    network.push({ event: event.type, code: event.code, accepted: event.accepted });
  });
});
const output = resolve("test-results", "workspace-layout");
await mkdir(output, { recursive: true });
try {
  await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
  await setInterfaceLanguage(page, "en");
  await expect(page.getByLabel("Message input", { exact: true })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Runtime status and controls", exact: true })).toHaveCount(0);
  await expect(page.locator("header")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "EN", exact: true })).toHaveCount(0);
  assert.equal(await page.getByRole("heading", { name: "New Session draft", exact: true }).evaluate(
    heading => heading.getBoundingClientRect().height), 1);
  await expect(page.getByLabel("Search by title", { exact: true })).toHaveCount(0);
  const usageSummary = page.getByLabel("Context and Token usage", { exact: true });
  await expect(usageSummary).toContainText("Historical input 0");
  await expect(usageSummary).toContainText("Input —");
  await expect(usageSummary).toContainText("Cache hit — Tokens");
  let cachedFixture = 12000;
  await page.route("**/runtime/status", async route => {
    const response = await route.fetch();
    const body = await response.json();
    body.status.projected_next_request_tokens = 14559;
    body.status.context_window = 1000000;
    body.status.cumulative_usage.input_tokens = 80000;
    body.status.last_request_usage = { input_tokens: 14000, cached_input_tokens: cachedFixture };
    await route.fulfill({ response, json: body });
  });
  await expect(usageSummary).toContainText("Cache hit 12,000 Tokens");
  for (const language of ["en", "zh-CN"]) {
    await setInterfaceLanguage(page, language);
    for (const theme of ["light", "dark"]) {
      await setInterfaceTheme(page, theme);
      for (const viewport of [{ width: 1920, height: 1080 }, { width: 1440, height: 900 },
        { width: 768, height: 900 }, { width: 375, height: 800 }, { width: 1280, height: 440 }]) {
        await page.setViewportSize(viewport);
        await expect.poll(() => page.locator("textarea").evaluate(input => input.getBoundingClientRect().width)).toBeGreaterThan(0);
        const tokenUsage = page.getByLabel(language === "en" ? "Context and Token usage" : "上下文与 Token 用量", { exact: true });
        await expect(tokenUsage).toBeVisible();
        await expect(tokenUsage.locator(":scope > span")).toHaveCount(5);
        await expect(tokenUsage.getByRole("button")).toHaveCount(0);
        for (const value of ["14,559", "1,000,000", "80,000", "14,000", "12,000"]) {
          await expect(tokenUsage).toContainText(value);
        }
        const layout = await page.locator("textarea").evaluate(input => {
          const box = input.parentElement.getBoundingClientRect();
          const main = document.getElementById("main-content").getBoundingClientRect();
          const send = input.parentElement.querySelector('button[type="submit"]').getBoundingClientRect();
          return { center: (box.top + box.height / 2 - main.top) / main.height,
            top: box.top, bottom: send.bottom, width: document.documentElement.scrollWidth,
            parent: input.parentElement.className, input: input.getBoundingClientRect().toJSON(),
            main: main.toJSON() };
        });
        assert.ok(layout.width <= viewport.width + 1, `Horizontal overflow: ${JSON.stringify(layout)}`);
        assert.ok(layout.top >= 0 && layout.bottom <= viewport.height + 1,
          `Input or send button is clipped: ${JSON.stringify(layout)}`);
        const usageBounds = await tokenUsage.boundingBox();
        assert.ok(usageBounds.y >= layout.input.bottom && usageBounds.x >= 0
          && usageBounds.x + usageBounds.width <= viewport.width + 1
          && usageBounds.y + usageBounds.height <= viewport.height + 1,
        `Token usage is clipped or above the input: ${JSON.stringify(usageBounds)}`);
        if (viewport.width >= 1024 && viewport.height >= 500) {
          assert.ok(layout.center >= 0.55 && layout.center <= 0.65,
            `Input is not below the middle: ${JSON.stringify(layout)}`);
        }
        await page.screenshot({ path: resolve(output, `${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await setInterfaceLanguage(page, "en");
  await setInterfaceTheme(page, "light");
  cachedFixture = 0;
  await expect(usageSummary).toContainText("Cache hit 0 Tokens");
  cachedFixture = null;
  await expect(usageSummary).toContainText("Cache hit — Tokens");
  await page.unroute("**/runtime/status");
  const project = await registerProjectFromSidebar(page, control.details.first_project);
  const item = projectItemByPath(page, control.details.first_project);
  await page.goto(new globalThis.URL(`/projects/${project.project_id}`, page.url()).href);
  await expect(page.locator("#sessions-heading")).toHaveText("project-one");
  await expect(page.getByRole("button", { name: "Review restore failure", exact: true })).toHaveCount(0);
  const originalUrl = page.url();
  const memoryResponse = page.waitForResponse(response => response.url().endsWith(`/projects/${project.project_id}/memory/read`));
  await projectMenuAction(item, "Workspace Memory and Dream");
  const memory = page.getByRole("dialog", { name: "Workspace Memory and Dream", exact: true });
  await memory.getByRole("button", { name: "View Memory", exact: true }).click();
  assert.equal((await memoryResponse).status(), 200);
  await expect(memory.locator("pre")).toBeVisible();
  await memory.getByRole("button", { name: "Run Dream", exact: true }).click();
  await expect(memory.getByRole("status").filter({ hasText: /Dream completed|No pending summaries/ })).toBeVisible();
  let releaseRead;
  let readArrived;
  const readGate = new Promise(done => { releaseRead = done; });
  const readArrival = new Promise(done => { readArrived = done; });
  await page.route(`**/projects/${project.project_id}/memory/read`, async route => {
    const response = await route.fetch();
    const body = await response.json();
    readArrived();
    await readGate;
    await route.fulfill({ json: { ...body, content: "STALE_MEMORY_RESPONSE" } });
  }, { times: 1 });
  await memory.getByRole("button", { name: "View Memory", exact: true }).click();
  await readArrival;
  await memory.press("Escape");
  await projectMenuAction(item, "Workspace Memory and Dream");
  const completedRead = page.waitForResponse(response => response.url().endsWith(`/projects/${project.project_id}/memory/read`));
  releaseRead();
  await completedRead;
  await expect(memory.locator("pre")).not.toContainText("STALE_MEMORY_RESPONSE");
  await page.setViewportSize({ width: 375, height: 800 });
  await memory.press("Escape");
  await expect(page.locator("#app-sidebar-toggle")).toBeFocused();
  await page.setViewportSize({ width: 1440, height: 900 });
  assert.equal(page.url(), originalUrl, "Project Memory created or switched a conversation");
  await projectMenuAction(item, "Schedule tasks");
  await expect(page.getByRole("heading", { name: "Schedule Jobs", exact: true })).toBeVisible();
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  await page.getByLabel("Message input").fill("workspace layout acceptance message");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  const runtime = page.getByRole("button", { name: "Runtime status and controls", exact: true });
  await expect(runtime).toHaveCount(0);
  const tokenUsage = page.getByLabel("Context and Token usage", { exact: true });
  await expect(tokenUsage).toBeVisible();
  const positions = await tokenUsage.evaluate(element => ({
    status: element.getBoundingClientRect().top,
    box: document.querySelector("textarea").parentElement.getBoundingClientRect().bottom,
  }));
  assert.ok(positions.status >= positions.box, "Token usage is above the input");
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  await control.command("settings-arm");
  await page.getByLabel("Message input").fill("recovery streaming markdown runtime activity acceptance");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await control.command("settings-wait");
  await expect(tokenUsage).toBeVisible();
  await expect(runtime).toHaveCount(0);
  await control.command("settings-release");
  await expect(page.getByRole("button", { name: "Cancel run", exact: true })).toHaveCount(0);
  await page.getByLabel("Message input").fill("unsent survives restart");
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  const restart = page.getByRole("button", { name: "Restart Aide", exact: true });
  await expect(restart).toBeEnabled();
  await page.locator("#settings-web-default_chat_workspace").fill(control.details.second_project);
  await page.locator("#settings-web-default_chat_workspace").blur();
  await expect(restart).toBeEnabled();
  const before = await page.evaluate(async () => (await globalThis.fetch("/api/v1/service")).json());
  await restart.click();
  const confirmation = page.getByRole("dialog", { name: "Restart Aide", exact: true });
  await expect(confirmation).toContainText("connected CLI clients");
  await confirmation.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(restart).toBeFocused();
  await restart.click();
  await confirmation.getByRole("button", { name: "Restart Aide", exact: true }).click();
  await expect(restart).toBeEnabled({ timeout: 60000 });
  const after = await page.evaluate(async () => (await globalThis.fetch("/api/v1/service")).json());
  assert.notEqual(before.service_instance_id, after.service_instance_id);
  await expect(page.locator('[data-state="active"]')).toBeVisible();
  await page.getByRole("button", { name: "Back to app", exact: true }).click();
  await expect(page.getByLabel("Message input")).toBeEnabled();
  await expect(page.getByLabel("Message input")).toHaveValue("unsent survives restart");
  await page.getByLabel("Message input").fill("conversation after settings restart");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true }).last()).toBeVisible();
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  await expect(page.getByLabel("Message input")).toBeEnabled();
  await page.locator("#composer-model-trigger").click();
  await page.getByLabel("Reasoning effort", { exact: true }).selectOption("xhigh");
  await expect(page.locator("#composer-model-trigger")).toContainText("xhigh");
  await page.getByLabel("Message input").fill("empty draft survives restart");
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  await restart.click();
  await confirmation.getByRole("button", { name: "Restart Aide", exact: true }).click();
  await expect(restart).toBeEnabled({ timeout: 60000 });
  await page.getByRole("button", { name: "Back to app", exact: true }).click();
  await expect(page.getByLabel("Message input")).toHaveValue("empty draft survives restart");
  await expect(page.locator("#composer-model-trigger")).toContainText("xhigh");
  await expect(runtime).toHaveCount(0);
  await page.goto(new globalThis.URL(`/projects/${project.project_id}?session=${control.details.available_session_id}`, page.url()).href);
  await expect(page.getByLabel("Message input")).toBeEnabled();
  await newProjectConversation(page);
  await expect(page.getByLabel("Message input")).toBeEnabled();
  await page.getByLabel("Message input").fill("project draft survives restart");
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  await restart.click();
  await confirmation.getByRole("button", { name: "Restart Aide", exact: true }).click();
  await expect(restart).toBeEnabled({ timeout: 60000 });
  await page.getByRole("button", { name: "Back to app", exact: true }).click();
  await expect(page.getByLabel("Message input")).toHaveValue("project draft survives restart");
  await expect(page.getByLabel("Message input")).toBeEnabled();
  await expect(runtime).toHaveCount(0);
  await control.command("settings-arm");
  await page.getByLabel("Message input").fill("recovery streaming markdown first accepted run");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await control.command("settings-wait");
  await expect(tokenUsage).toBeVisible();
  await expect(runtime).toHaveCount(0);
  await control.command("settings-release");
  await expect(page.getByRole("button", { name: "Cancel run", exact: true })).toHaveCount(0);
  assert.deepEqual(errors, []);
  console.log("Workspace layout, Project Memory, formal-session status and settings restart: passed");
} catch (error) {
  await page.screenshot({ path: resolve(output, "failure.png") });
  await writeFile(resolve(output, "failure.json"), JSON.stringify({
    error: error.message, route: new globalThis.URL(page.url()).pathname,
    page: await page.locator("#main-content").innerText(), errors,
    service: control.serviceLog(), network: network.slice(-100),
  }, null, 2));
  throw error;
} finally {
  await browser.close();
  await control.shutdown();
}
