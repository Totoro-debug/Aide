import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { showProjectNavigation } from "./project-ui.mjs";
import { setInterfaceLanguage, setInterfaceTheme } from "./settings-e2e.mjs";

const expect = playwrightExpect.configure({ timeout: 15000 });
const output = resolve("test-results/api-key-settings");
const control = await setup();
const configPath = resolve(control.details.home_root, ".aide/config.toml");
let browser;
let page;
const configTraffic = [];
try {
  const initial = await control.command("config-read");
  const providers = initial.fields.models.providers;
  for (const provider of Object.values(providers)) {
    delete provider.api_key;
    for (const name of Object.keys(provider.models)) {
      provider.models[name] = {
        context_window: 8192, max_output: 1024, temperature: 0, reasoning_effort: "mid", timeout: 30,
      };
    }
  }
  await control.command(`config-patch ${JSON.stringify({ models: { providers } })}`);
  await mkdir(output, { recursive: true });
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  page = await browser.newPage({ locale: "en", viewport: { width: 1440, height: 900 } });
  const errors = [];
  const writes = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (request.url().endsWith("/api/v1/config")) configTraffic.push(`request ${request.method()}`);
    if (request.url().endsWith("/api/v1/config") && request.method() === "PATCH") writes.push(request.postDataJSON());
  });
  page.on("response", response => {
    if (response.url().endsWith("/api/v1/config")) configTraffic.push(`response ${response.request().method()} ${response.status()}`);
  });
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await setInterfaceLanguage(page, "en");
  await showProjectNavigation(page);
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  const section = name => page.getByRole("navigation", { name: "Settings sections", exact: true })
    .getByRole("button", { name, exact: true }).click();
  const input = provider => page.locator(`[id="settings-models-providers-${provider}-api_key-value"]`);
  const save = provider => page.locator(`[id="settings-models-providers-${provider}-api_key-save"]`);
  const saved = () => expect(page.locator('[role="status"][data-state="active"], [role="status"][data-state="next-run-required"]')).toBeVisible();
  const persisted = () => readFile(configPath, "utf8");
  const replacementValues = () => writes.flatMap(write => Object.values(write.secrets).filter(change => change.action === "replace").map(change => change.value));
  await section("Models");
  for (const provider of ["primary", "retired"]) {
    await expect(input(provider)).toHaveValue("");
    await expect(input(provider)).toHaveAttribute("type", "password");
    await expect(save(provider)).toHaveCount(0);
  }
  await expect(page.locator('select[id$="api_key-action"]')).toHaveCount(0);
  const original = await persisted();
  await input("primary").fill("pending-primary-key");
  await input("retired").fill("pending-retired-key");
  await input("retired").press("Tab");
  await section("Runtime");
  await page.getByLabel("Maximum iterations", { exact: true }).fill("73");
  await page.getByLabel("Maximum iterations", { exact: true }).press("Tab");
  await saved();
  await expect.poll(async () => (await control.command("config-read")).fields.runtime.max_iterations).toBe(73);
  await section("Models");
  await expect(input("primary")).toHaveValue("pending-primary-key");
  await input("primary").press("Enter");
  await page.waitForResponse(response => response.url().endsWith("/api/v1/config") && response.request().method() === "GET");
  assert.deepEqual(replacementValues(), []);
  assert.equal((await persisted()).includes("pending-primary-key"), false);
  assert.equal((await persisted()).includes("pending-retired-key"), false);
  await input("primary").fill("");
  await expect(save("primary")).toHaveCount(0);
  await input("primary").fill("confirmed-primary-key");
  const beforeClick = writes.length;
  await save("primary").click();
  await saved();
  await expect(input("primary")).toHaveValue("");
  await expect(save("primary")).toHaveCount(0);
  assert.equal(writes.length - beforeClick, 1);
  assert.deepEqual(replacementValues(), ["confirmed-primary-key"]);
  assert.equal((await persisted()).includes("confirmed-primary-key"), true);
  assert.equal((await persisted()).includes("e2e-provider-secret-302"), false);
  assert.equal((await persisted()).includes("e2e-retired-secret-302"), true);
  await expect(input("retired")).toHaveValue("pending-retired-key");
  console.log("API key: input isolation, blur, other settings, polling, Enter and per-provider confirmation passed");

  const rejectSave = route => route.request().method() === "PATCH" ? route.abort("failed") : route.continue();
  await page.route("**/api/v1/config", rejectSave);
  await input("primary").fill("retry-primary-key");
  await save("primary").click();
  await expect(page.locator('[role="status"][data-state="error"]')).toBeVisible();
  await expect(input("primary")).toHaveValue("retry-primary-key");
  assert.equal((await persisted()).includes("retry-primary-key"), false);
  await page.unroute("**/api/v1/config", rejectSave);
  await save("primary").click();
  await saved();
  await expect(input("primary")).toHaveValue("");
  assert.equal((await persisted()).includes("retry-primary-key"), true);

  const loseResponse = async route => {
    if (route.request().method() !== "PATCH") return route.continue();
    await route.fetch();
    await route.abort("failed");
  };
  await page.route("**/api/v1/config", loseResponse);
  await input("primary").fill("lost-response-key");
  await save("primary").click();
  await expect(page.locator('[role="status"][data-state="error"]')).toBeVisible();
  await expect(input("primary")).toHaveValue("lost-response-key");
  assert.equal((await persisted()).includes("lost-response-key"), true);
  const lostRequestId = writes.at(-1).request_id;
  await page.unroute("**/api/v1/config", loseResponse);
  await save("primary").click();
  await saved();
  await expect(input("primary")).toHaveValue("");
  assert.equal(writes.at(-1).request_id, lostRequestId);

  let releaseSave;
  let arriveSave;
  const gate = new Promise(done => { releaseSave = done; });
  const arrival = new Promise(done => { arriveSave = done; });
  const delaySave = async route => {
    if (route.request().method() !== "PATCH") return route.continue();
    arriveSave();
    await gate;
    await route.continue();
  };
  await page.route("**/api/v1/config", delaySave);
  await input("primary").fill("in-flight-primary-key");
  await save("primary").click();
  await arrival;
  await input("primary").fill("unconfirmed-next-key");
  releaseSave();
  await saved();
  await page.unroute("**/api/v1/config", delaySave);
  await expect(input("primary")).toHaveValue("unconfirmed-next-key");
  assert.equal((await persisted()).includes("in-flight-primary-key"), true);
  assert.equal((await persisted()).includes("unconfirmed-next-key"), false);
  console.log("API key: failed save retry and input preservation during a request passed");

  let releaseConflict;
  let arriveConflict;
  const conflictGate = new Promise(done => { releaseConflict = done; });
  const conflictArrival = new Promise(done => { arriveConflict = done; });
  const delayConflict = async route => {
    if (route.request().method() !== "PATCH") return route.continue();
    arriveConflict(route.request().headers());
    await conflictGate;
    await route.continue();
  };
  await page.route("**/api/v1/config", delayConflict);
  await input("primary").fill("conflicting-primary-key");
  const conflictResponse = page.waitForResponse(response => response.url().endsWith("/api/v1/config")
    && response.request().method() === "PATCH");
  await save("primary").click();
  const headers = await conflictArrival;
  const external = await page.request.patch(`${control.details.url}/api/v1/config`, {
    headers,
    data: {
      request_id: "api-key-external-edit", revision: (await control.command("config-read")).revision,
      fields: {}, secrets: { "models.providers.primary.api_key": { action: "replace", value: "external-primary-key" } },
    },
  });
  assert.equal(external.status(), 200);
  releaseConflict();
  assert.equal((await conflictResponse).status(), 409);
  await expect(page.locator('[role="status"][data-state="error"]')).toBeVisible();
  await page.unroute("**/api/v1/config", delayConflict);
  await expect(input("primary")).toHaveValue("conflicting-primary-key");
  await expect(input("primary")).toHaveAttribute("aria-invalid", "true");
  assert.equal((await persisted()).includes("external-primary-key"), true);
  assert.equal((await persisted()).includes("conflicting-primary-key"), false);
  await page.getByRole("button", { name: "Reload saved values", exact: true }).click();
  await expect(input("primary")).toHaveValue("");
  await input("primary").fill("unconfirmed-next-key");
  await input("retired").fill("pending-retired-key");
  console.log("API key: concurrent secret conflict preserves the external key and local input");

  await page.getByRole("button", { name: "Add provider", exact: true }).click();
  await page.locator("#settings-models-providers-new-provider-base_url").fill("http://127.0.0.1:1/provider");
  await page.locator("#settings-models-providers-new-provider-base_url").press("Tab");
  await saved();
  await expect(input("new-provider")).toHaveValue("");
  await input("new-provider").fill("first-provider-key");
  await input("new-provider").press("Tab");
  await expect.poll(async () => (await control.command("config-read")).fields.models.providers["new-provider"].api_key.configured).toBe(false);
  await save("new-provider").click();
  await saved();
  await expect(input("new-provider")).toHaveValue("");
  assert.equal((await control.command("config-read")).fields.models.providers["new-provider"].api_key.configured, true);
  assert.equal((await persisted()).includes("unconfirmed-next-key"), false);
  assert.equal((await persisted()).includes("pending-retired-key"), false);
  assert.equal(original.includes("e2e-provider-secret-302"), true);

  for (const language of ["en", "zh-CN"]) {
    await setInterfaceLanguage(page, language);
    for (const theme of ["light", "dark"]) {
      await setInterfaceTheme(page, theme);
      for (const width of [375, 768, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        await input("primary").scrollIntoViewIfNeeded();
        const button = await save("primary").boundingBox();
        const field = await input("primary").boundingBox();
        assert.ok(button && field);
        assert.ok(button.x + button.width < field.x);
        assert.ok(Math.abs(button.width - button.height) < 1);
        assert.ok(Math.abs(button.y + button.height / 2 - field.y - field.height / 2) < 1);
        assert.equal(await save("primary").evaluate(element => globalThis.getComputedStyle(element).borderRadius), "50%");
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
        await page.screenshot({ path: resolve(output, `api-key-${language}-${theme}-${width}.png`) });
      }
    }
  }
  await page.reload();
  await setInterfaceLanguage(page, "en");
  await section("Models");
  await expect(input("primary")).toHaveValue("");
  await expect(save("primary")).toHaveCount(0);
  assert.equal((await persisted()).includes("unconfirmed-next-key"), false);
  assert.deepEqual(errors, []);
  console.log("API key settings E2E passed: explicit confirmation, retry, in-flight input, first key, reload and 12 layout variants.");
} catch (error) {
  if (page) {
    console.error("API key settings diagnostics:", JSON.stringify(await page.getByRole("alert").allTextContents()));
    console.error("API key settings status:", JSON.stringify(await page.locator('[role="status"][data-state]').evaluateAll(items => items.map(item => ({ state: item.dataset.state, text: item.textContent })))));
    console.error("API key config traffic:", configTraffic.slice(-25));
    await page.screenshot({ path: resolve(output, "failure.png"), fullPage: true });
  }
  throw error;
} finally {
  try { await browser?.close(); }
  finally { await control.shutdown(); }
}
