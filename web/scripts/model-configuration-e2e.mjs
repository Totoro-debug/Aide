import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { showProjectNavigation } from "./project-ui.mjs";
import { setInterfaceLanguage, setInterfaceTheme } from "./settings-e2e.mjs";

const expect = playwrightExpect.configure({ timeout: 15000 });
const control = await setup({ shutdownTimeoutMs: 60000 });
const output = resolve("test-results/model-configuration");
const configPath = resolve(control.details.home_root, ".aide/config.toml");
let browser;
let page;
try {
  await mkdir(output, { recursive: true });
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  page = await browser.newPage({ locale: "en", viewport: { width: 1440, height: 900 } });
  const errors = [];
  let csrf;
  let runtimeStatus;
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    const token = request.headers()["x-aide-csrf"];
    if (token) csrf = token;
  });
  page.on("response", async response => {
    if (response.url().endsWith("/runtime/status") && response.ok()) runtimeStatus = (await response.json()).status;
  });
  await page.addInitScript(() => {
    const Original = window.WebSocket;
    window.WebSocket = class extends Original {
      constructor(...args) {
        super(...args);
        window.modelControl = args[1][1];
      }
    };
  });
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await setInterfaceLanguage(page, "en");
  await showProjectNavigation(page);
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  const section = page.getByRole("navigation", { name: "Settings sections", exact: true });
  await section.getByRole("button", { name: "Models", exact: true }).click();
  console.log("Model configuration: opened legacy settings");
  const providerCard = page.locator('[id="settings-models-providers-primary"]');
  const card = (provider, model) => page.locator(`[id="settings-model-${encodeURIComponent(provider)}-${encodeURIComponent(model).replaceAll(".", "%2E")}"]`);
  const openCard = async target => {
    if (!await target.locator("details").evaluate(element => element.open)) await target.locator("summary").click();
  };
  const fillModel = async (target, { id, context = 8192, output = 512, temperature = 0.7, effort = "mid", timeout = 17 } = {}) => {
    await openCard(target);
    if (id !== undefined) await target.getByLabel("Model", { exact: true }).fill(id);
    await target.getByLabel("Maximum output", { exact: true }).fill(String(output));
    await target.getByLabel("Temperature", { exact: true }).fill(String(temperature));
    await target.getByLabel("Reasoning effort", { exact: true }).selectOption(effort);
    await target.getByLabel("Timeout (seconds)", { exact: true }).fill(String(timeout));
    await target.getByLabel("Context window", { exact: true }).fill(String(context));
    await target.getByLabel("Context window", { exact: true }).press("Tab");
  };
  const savedModel = async (provider, model, expectedOutput) => {
    await expect.poll(async () => (await control.command("config-read")).fields.models.providers[provider]?.models[model]?.max_output).toBe(expectedOutput);
    await expect(page.locator('[role="status"][data-state="saving"]')).toHaveCount(0);
  };
  const initial = await control.command("config-read");
  assert.equal(initial.fields.models.providers.primary.models["small-model"].reasoning_effort, null);
  await card("primary", "small-model").getByRole("button", { name: /^Use chat parameters/ }).click();
  await page.locator('[id="settings-models-providers-retired"]').getByRole("button", { name: "Remove provider", exact: true }).click();
  await fillModel(card("primary", "large-model"), { context: 65536, output: 16384, temperature: 0.1, effort: "high", timeout: 91 });
  await savedModel("primary", "large-model", 16384);
  console.log("Model configuration: legacy parameters migrated");
  const migrated = await control.command("config-read");
  assert.equal(migrated.fields.models.providers.primary.models["small-model"].migration_candidates, undefined);
  for (const route of Object.values(migrated.fields.models.routes)) assert.deepEqual(Object.keys(route).sort(), ["model", "provider_id"]);
  assert.equal(JSON.stringify(migrated).includes("e2e-provider-secret-302"), false);
  assert.match(await readFile(configPath, "utf8"), /e2e-provider-secret-302/);
  await expect(card("primary", "small-model").getByRole("button", { name: "Remove model", exact: true })).toBeDisabled();

  const chatRoute = page.locator('[id="settings-models-routes-chat"]');
  await expect(chatRoute.getByRole("button", { name: "Remove route", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Add route", exact: true }).click();
  const subagentRoute = page.locator('[id="settings-models-routes-subagent"]');
  await expect(subagentRoute).toBeVisible();
  await page.locator('[id="settings-models-routes-subagent-model"]').selectOption("small-model");
  await page.locator('[id="settings-models-routes-subagent-model"]').press("Tab");
  await expect.poll(async () => (await control.command("config-read")).fields.models.routes.subagent?.model)
    .toBe("small-model");
  await expect(page.locator('[id="settings-models-routes-title"]')).toBeVisible();

  const unchanged = await readFile(configPath);
  await openCard(card("primary", "large-model"));
  const maximum = card("primary", "large-model").getByLabel("Maximum output", { exact: true });
  await maximum.fill("65536");
  await maximum.press("Tab");
  await expect(maximum).toHaveAttribute("aria-invalid", "true");
  assert.deepEqual(await readFile(configPath), unchanged);
  await maximum.fill("16384");
  await maximum.press("Tab");
  await expect(maximum).toHaveAttribute("aria-invalid", "false");

  await providerCard.getByRole("button", { name: "Add model", exact: true }).click();
  const newCard = providerCard.locator('div[id^="settings-model-primary-"]').last();
  await fillModel(newCard, { id: "small-model" });
  const identity = newCard.getByLabel("Model", { exact: true });
  await expect(identity).toHaveAttribute("aria-invalid", "true");
  await expect(identity).toBeEditable();
  await page.getByRole("alert").getByRole("link").click();
  await expect(identity).toBeFocused();
  assert.deepEqual(await readFile(configPath), unchanged);
  await identity.fill("vendor/compact.v1");
  await identity.press("Tab");
  await savedModel("primary", "vendor/compact.v1", 512);
  await expect(identity).toHaveAttribute("readonly", "");
  console.log("Model configuration: duplicate/range validation and model creation passed");

  // A second provider may use the same model ID with different defaults.
  const models = (await control.command("config-read")).fields.models;
  for (const provider of Object.values(models.providers)) delete provider.api_key;
  models.providers.secondary = {
    protocol: "openai-compatible", base_url: models.providers.primary.base_url,
    models: { "vendor/compact.v1": { context_window: 16384, max_output: 1024, temperature: 0.4, reasoning_effort: "low", timeout: 19 } },
  };
  const revision = (await control.command("config-read")).revision;
  assert.ok(csrf);
  assert.equal(await page.evaluate(async ({ payload, csrf }) => {
    const response = await window.fetch("/api/v1/config", {
      method: "PATCH", headers: { "X-Aide-Control": window.modelControl, "X-Aide-CSRF": csrf, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return response.status;
  }, { csrf, payload: { request_id: "add-secondary-provider", revision, fields: { models }, secrets: { "models.providers.secondary.api_key": { action: "replace", value: "secondary-secret" } } } }), 200);
  await page.reload();
  await section.getByRole("button", { name: "Models", exact: true }).click();
  const routeField = (route, field) => page.locator(`[id="settings-models-routes-${route}-${field}"]`);
  const beforePair = await readFile(configPath);
  await routeField("memory", "provider_id").selectOption("secondary");
  await expect(routeField("memory", "model")).toHaveValue("");
  assert.deepEqual(await readFile(configPath), beforePair);
  await routeField("memory", "model").selectOption("vendor/compact.v1");
  await routeField("schedule", "model").selectOption("vendor/compact.v1");
  await routeField("chat", "model").selectOption("large-model");
  await routeField("chat", "model").press("Tab");
  await expect.poll(async () => (await control.command("config-read")).fields.models.routes.chat.model).toBe("large-model");
  const routes = (await control.command("config-read")).fields.models.routes;
  assert.deepEqual(routes.memory, { provider_id: "secondary", model: "vendor/compact.v1" });
  assert.deepEqual(routes.schedule, { provider_id: "primary", model: "vendor/compact.v1" });
  assert.equal(await routeField("memory", "model").locator('option[value="large-model"]').count(), 0);
  assert.equal(await routeField("chat", "model").locator('option[value="missing-model"]').count(), 0);
  console.log("Model configuration: provider-scoped route selection passed");

  for (const language of ["en", "zh-CN"]) {
    await setInterfaceLanguage(page, language);
    for (const theme of ["light", "dark"]) {
      await setInterfaceTheme(page, theme);
      for (const width of [375, 768, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        await routeField("chat", "model").focus();
        await expect(routeField("chat", "model")).toBeFocused();
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
        await page.screenshot({ path: resolve(output, `routes-${language}-${theme}-${width}.png`), fullPage: true });
        const summary = providerCard.locator('div[id^="settings-model-primary-"]').first().locator("summary");
        await summary.focus();
        await expect(summary).toBeFocused();
        await page.screenshot({ path: resolve(output, `models-${language}-${theme}-${width}.png`), fullPage: true });
      }
    }
  }
  await setInterfaceLanguage(page, "en");
  await page.setViewportSize({ width: 1440, height: 900 });
  await subagentRoute.getByRole("button", { name: "Remove route", exact: true }).click();
  await expect.poll(async () => (await control.command("config-read")).fields.models.routes.subagent)
    .toBeUndefined();
  await expect(page.locator('[role="status"][data-state="saving"]')).toHaveCount(0);
  const restarted = await control.restart();
  await page.goto("about:blank");
  await page.goto(`${restarted.url}/#ticket=${restarted.ticket}`);
  await expect(page.locator("#composer-model-trigger")).toBeEnabled();
  await page.locator("#composer-model-trigger").click();
  const sessionModel = page.getByLabel("Session model", { exact: true });
  await expect(sessionModel).toBeEnabled();
  // Its capacity is below the chat model's output limit: it still uses its own budget.
  await sessionModel.selectOption(JSON.stringify(["primary", "vendor/compact.v1"]));
  await page.getByLabel("Message input", { exact: true }).fill("model selection persistence");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  const observations = (await readFile(restarted.provider_observation_path, "utf8")).split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  assert.ok(observations.some(request => request.model === "vendor/compact.v1" && request.max_output === 512 && request.temperature === 0.7));
  await expect.poll(() => runtimeStatus?.context_window).toBe(8192);
  const budget = runtimeStatus;
  assert.equal(budget.context_window, 8192);
  assert.equal(budget.max_output, 512);
  assert.equal(budget.available_context, 7680);
  assert.deepEqual(errors, []);
  console.log("Model configuration E2E passed: explicit migration, model cards, duplicate/range validation, provider-scoped routes, 12 layout variants, restart and real request budget.");
} catch (error) {
  if (page) {
    console.error("Model configuration diagnostics:", JSON.stringify(await page.getByRole("alert").allTextContents()));
    await page.screenshot({ path: resolve(output, "failure.png"), fullPage: true });
  }
  throw error;
} finally {
  try { await browser?.close(); }
  finally { await control.shutdown(); }
}
