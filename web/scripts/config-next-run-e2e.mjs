import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { newProjectConversation } from "./project-ui.mjs";
import settingsAcceptance, { setInterfaceLanguage } from "./settings-e2e.mjs";

const output = resolve("test-results");
await mkdir(output, { recursive: true });
const control = await setup();
let browser;
let page;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const context = await browser.newContext();
  page = await context.newPage();
  await page.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    window.__aideTestMessages = [];
    window.WebSocket = class extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        window.__aideTestControlCredential = Array.isArray(args[1]) ? args[1][1] : null;
        window.__aideTestSocket = this;
        this.addEventListener("message", (event) => {
          try {
            window.__aideTestMessages.push(JSON.parse(event.data));
          } catch {
            // Ignore non-JSON messages.
          }
        });
      }
    };
  });
  await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
  await expect(page.locator("#app-sidebar").getByRole("link", { name: /^(Settings|设置)$/ })).toBeVisible({ timeout: 15000 });
  await expect.poll(() => page.evaluate(() => window.__aideTestSocket?.readyState)).toBe(1);
  const serviceInstance = async () => page.evaluate(async () => {
    const response = await globalThis.fetch("/api/v1/service", { credentials: "include" });
    if (!response.ok) throw new Error(`Service status failed: ${response.status}`);
    return (await response.json()).service_instance_id;
  });
  const instance = await serviceInstance();
  assert.ok(instance);
  const registration = await page.evaluate(async (path) => {
    const session = await globalThis.fetch("/api/v1/web/session", { credentials: "include" });
    const { csrf_token: csrf } = await session.json();
    const response = await globalThis.fetch("/api/v1/projects", {
      method: "POST", credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "X-Aide-Control": window.__aideTestControlCredential,
        "X-Aide-CSRF": csrf,
      },
      body: JSON.stringify({ request_id: globalThis.crypto.randomUUID(), path }),
    });
    return { status: response.status, body: await response.json() };
  }, control.details.first_project);
  assert.equal(registration.status, 200, JSON.stringify(registration.body));
  await setInterfaceLanguage(page, "en");
  await settingsAcceptance({ page, control, output, viewports: [{ width: 1440, height: 900 }] });

  const initial = await control.command("config-read");
  const providers = initial.fields.models.providers;
  for (const provider of Object.values(providers)) {
    delete provider.api_key;
    for (const name of Object.keys(provider.models)) {
      provider.models[name] = {
        context_window: name === "large-model" ? 32768 : 8192,
        max_output: 1024, temperature: 0, reasoning_effort: "mid", timeout: 30,
      };
    }
  }
  const fixture = initial.fields.mcp.fixture;
  fixture.tool_keywords = { fixture_echo_v1: ["resource"] };
  const configured = await control.command(`config-patch ${JSON.stringify({
    runtime: { permission_level: "full-access" },
    models: { providers, routes: {
      default: { provider_id: "primary", model: "large-model" },
      chat: { provider_id: "primary", model: "large-model" },
    } },
    mcp: { fixture },
  })}`);
  const project = registration.body;
  assert.ok(project?.project_id);
  await page.goto(`${control.details.url}/projects/${project.project_id}`);
  await newProjectConversation(page);
  await expect(page.getByLabel("Message input", { exact: true })).toBeEnabled();
  await control.command("model-mcp-arm");
  await page.getByLabel("Message input", { exact: true }).fill("model MCP generation barrier");
  await page.getByLabel("Message input", { exact: true }).press("Enter");
  await control.command("model-mcp-wait");
  const oldRun = await page.evaluate(() => window.__aideTestMessages.find(event => (
    event.type === "input.accepted" && event.payload?.text === "model MCP generation barrier"
  ))?.run_id);
  assert.ok(oldRun);
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  const section = name => page.getByRole("navigation", { name: "Settings sections", exact: true })
    .getByRole("button", { name, exact: true }).click();
  const saveField = async (field, value, select = false) => {
    const saved = page.waitForResponse(response => response.url().endsWith("/api/v1/config")
      && response.request().method() === "PATCH");
    if (select) await field.selectOption(value);
    else await field.fill(value);
    await field.press("Tab");
    const response = await saved;
    const body = await response.json();
    assert.equal(response.status(), 200, JSON.stringify(body));
    return body;
  };
  await section("Models");
  await saveField(page.locator("#settings-models-routes-default-model"), "small-model", true);
  await saveField(page.locator("#settings-models-routes-chat-model"), "small-model", true);
  await section("MCP");
  await saveField(page.locator("#settings-mcp-fixture-args").getByRole("textbox").last(), control.details.mcp_v2_path);
  const saved = await saveField(page.locator("#settings-mcp-fixture-tool_keywords")
    .getByRole("textbox", { name: "Tool name", exact: true }), "fixture_echo_v2");
  assert.equal(saved.application.status, "next-run-required");
  assert.equal(saved.application.active_revision, configured.revision);
  assert.equal(saved.application.restart_required, false);
  await control.command("model-mcp-release");
  await expect.poll(() => page.evaluate(runId => window.__aideTestMessages.some(event => (
    event.type === "run.completed" && event.run_id === runId
  )), oldRun), { timeout: 30000 }).toBe(true);
  await page.getByRole("button", { name: "Back to app", exact: true }).click();
  await page.getByLabel("Message input", { exact: true }).fill("model MCP resource");
  await page.getByLabel("Message input", { exact: true }).press("Enter");
  await expect(page.getByText("New model and MCP resource completed.", { exact: true }))
    .toBeVisible({ timeout: 30000 });
  const observations = (await readFile(control.details.provider_observation_path, "utf8"))
    .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  for (const [model, tool] of [["large-model", "fixture_echo_v1"], ["small-model", "fixture_echo_v2"]]) {
    assert.ok(observations.some(item => item.model === model && item.tools.some(name => name.endsWith(tool))));
  }
  assert.equal(await serviceInstance(), instance, "Applying configuration replaced the Service");
  console.log("Configuration next Run E2E: saved models, capacities, routes and MCP reach the next Run while the old Run retains its resources; Service identity is unchanged.");
} catch (error) {
  if (page?.isClosed() === false) {
    console.error(await page.locator("body").innerText());
    await page.screenshot({ path: resolve(output, "config-next-run-failure.png") });
  }
  console.error(control.serviceLog());
  throw error;
} finally {
  await browser?.close();
  await control.shutdown();
}
