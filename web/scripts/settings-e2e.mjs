import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { expect } from "@playwright/test";

async function waitForSavedSettings(target) {
  try {
    await expect(target.locator('[role="status"][data-state="active"], [role="status"][data-state="restart-required"]')).toBeVisible({ timeout: 30000 });
  } catch (error) {
    const statuses = await target.getByRole("status").allTextContents();
    const alerts = await target.getByRole("alert").allTextContents();
    const badges = await target.locator('[role="status"][data-state]').evaluateAll((items) => (
      items.map((item) => ({ state: item.getAttribute("data-state"), text: item.textContent }))
    ));
    throw new Error(`${error.message}\nStatus badges: ${JSON.stringify(badges)}\nSettings statuses: ${JSON.stringify(statuses)}\nAlerts: ${JSON.stringify(alerts)}`);
  }
}

async function settingsSection(target, section) {
  await target.bringToFront();
  const language = await target.locator("html").getAttribute("lang");
  const sectionName = language === "zh-CN" ? ({
    "General & appearance": "常规与外观",
    Models: "模型",
    Runtime: "运行时",
    Memory: "记忆",
    MCP: "MCP",
  })[section] ?? section : section;
  await target.getByRole("navigation", {
    name: language === "zh-CN" ? "设置分类" : "Settings sections",
    exact: true,
  }).getByRole("button", { name: sectionName, exact: true }).click();
}

export async function openServiceStatus(target) {
  const settingsLink = target.locator("#app-sidebar").getByRole("link", {
    name: /^(Settings|设置)$/,
  });
  if (!await settingsLink.isVisible()) await target.locator("#app-sidebar-toggle").click();
  await settingsLink.click();
  await settingsSection(target, "Runtime");
  await target.getByRole("main").getByRole("link", { name: /^(Status|状态)$/ }).click();
  await expect(target.locator("#status-heading")).toBeVisible();
}

async function blurSettingsField(target) {
  await target.evaluate(() => {
    const active = document.activeElement;
    if (active?.matches('[role="alert"][tabindex="-1"]')) return;
    if (active instanceof globalThis.HTMLInputElement
      || active instanceof globalThis.HTMLTextAreaElement
      || active instanceof globalThis.HTMLSelectElement) {
      active.blur();
      return;
    }
    const field = [...document.querySelectorAll("form input:not([type=checkbox]):not(:disabled), form textarea:not(:disabled), form select:not(:disabled)")]
      .find((candidate) => candidate.getClientRects().length > 0);
    if (field instanceof globalThis.HTMLElement) {
      field.focus();
      field.blur();
    }
  });
}

export async function settingsConfirmationAcceptance({ page, control }) {
  await page.bringToFront();
  await page.getByRole("button", { name: "EN", exact: true }).click();
  const draftResponses = Promise.all([
    page.waitForResponse((response) => (
      /\/projects\/[^/]+\/sessions$/.test(new globalThis.URL(response.url()).pathname)
      && response.request().method() === "POST"
    )),
    page.waitForResponse((response) => (
      /\/projects\/[^/]+\/sessions\/[^/]+\/claim$/.test(new globalThis.URL(response.url()).pathname)
      && response.request().method() === "POST"
    )),
  ]);
  await page.getByRole("button", { name: "New session", exact: true }).click();
  const [createdResponse, claimedResponse] = await draftResponses;
  assert.equal(createdResponse.status(), 200, "Settings draft creation failed");
  assert.equal(claimedResponse.status(), 200, "Settings draft Claim failed");
  const created = await createdResponse.json();
  const claimed = await claimedResponse.json();
  assert.equal(claimed.claim.session_id, created.session_id);
  await expect(page.getByRole("button", { name: "New session", exact: true })).toBeEnabled();
  await control.command("settings-arm");
  await page.locator("textarea").fill("settings generation barrier confirmation");
  await page.locator("textarea").press("Enter");
  let acceptedRun;
  await expect.poll(async () => {
    acceptedRun = await page.evaluate((sessionId) => [...(window.__omniTestMessages ?? [])]
      .reverse().find((event) => (
        event.type === "input.accepted" && event.session_id === sessionId
        && event.payload?.text === "settings generation barrier confirmation"
      )), created.session_id);
    return acceptedRun?.session_id;
  }, { timeout: 10000, message: "Settings input must be accepted in the newly claimed draft" }).toBe(created.session_id);
  await control.command("settings-wait");
  await page.reload();
  await expect(page.getByRole("log").getByText("settings generation barrier confirmation", { exact: true }))
    .toBeVisible({ timeout: 5000 });
  await expect(page.getByRole("button", { name: "Cancel run", exact: true })).toBeEnabled();
  await expect.poll(async () => page.evaluate((runId) => window.__omniTestMessages
    .filter((event) => event.type === "snapshot.required")
    .some((event) => event.payload?.snapshot?.sessions?.some((entry) => (
      entry.snapshot.live_state?.runs?.some((run) => run.run_id === runId)
    ))), acceptedRun.run_id)).toBe(true);
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  await settingsSection(page, "Runtime");
  const field = page.getByLabel("Maximum iterations", { exact: true });
  await expect(field).toBeEnabled();
  const savedResponse = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  await field.fill("65");
  await field.press("Tab");
  const saved = await savedResponse;
  assert.equal(saved.status(), 200);
  assert.equal((await saved.json()).application.status, "restart-required");
  await control.command("settings-release");
  const dialog = page.getByRole("dialog", { name: "Tool Confirmation", exact: true });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("confirmation-outside.txt");
  const original = await page.evaluate(() => [...window.__omniTestMessages]
    .reverse().find((event) => event.type === "confirmation.requested"));
  await page.reload();
  await expect(dialog).toBeVisible({ timeout: 5000 });
  const restored = await page.evaluate(() => [...window.__omniTestMessages]
    .reverse().find((event) => event.type === "snapshot.required"
      && event.payload?.snapshot?.pending_confirmation)?.payload.snapshot.pending_confirmation);
  assert.equal(restored?.payload.token, original.payload.token);
  await dialog.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(dialog).toBeHidden();
  await waitForSavedSettings(page);
  console.log("Settings pending generation: existing browser Run Tool confirmation remains usable and finishes naturally after save");
  return acceptedRun.run_id;
}

export async function settingsModelMcpAcceptance({ page, secondPage, control, output }) {
  const configPath = resolve(control.details.home_root, ".omni", "config.toml");
  const providerObservationPath = process.env.OMNI_E2E_PROVIDER_OBSERVATION_PATH;
  const mcpV1Path = process.env.OMNI_E2E_MCP_V1_PATH;
  const mcpV2Path = process.env.OMNI_E2E_MCP_V2_PATH;
  assert.ok(providerObservationPath);
  assert.ok(mcpV1Path);
  assert.ok(mcpV2Path);

  const responseBodies = [];
  const captureConfigResponses = (target) => {
    target.on("response", (response) => {
      if (response.url().endsWith("/api/v1/config")) {
        responseBodies.push(response.text().catch(() => ""));
      }
    });
  };
  captureConfigResponses(page);
  captureConfigResponses(secondPage);

  const field = (target, id) => target.locator(`[id="${id}"]`);
  const openRestartedPage = async (target, launchUrl) => {
    const exchanged = target.waitForResponse(response => (
      response.url().endsWith("/api/v1/web/ticket") && response.request().method() === "POST"
    ));
    // A new ticket on the same root URL needs a new document to authenticate.
    await target.goto("about:blank");
    await target.goto(launchUrl);
    assert.equal((await exchanged).status(), 200);
  };
  const openSettings = async (target) => {
    await target.bringToFront();
    await target.getByRole("button", { name: "EN", exact: true }).click();
    const received = target.waitForResponse((response) => (
      response.url().endsWith("/api/v1/config") && response.request().method() === "GET"
    ));
    await target.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
    const response = await received;
    assert.equal(response.status(), 200);
    await settingsSection(target, "Models");
    await expect(field(target, "settings-models-providers-primary-base_url")).toBeEnabled();
    return response.json();
  };
  const openProject = async (target) => {
    const project = await target.evaluate(async () => {
      const credential = window.__omniTestControlCredential;
      const response = await globalThis.fetch("/api/v1/projects", {
        credentials: "include",
        headers: credential == null ? {} : { "X-Omni-Control": credential },
      });
      if (!response.ok) throw new Error(`Project catalog request failed: ${response.status}`);
      const body = await response.json();
      return body.projects.find((item) => item.name === "project-one");
    });
    assert.ok(project?.project_id, "The E2E project catalog did not contain project-one");
    await target.goto(new globalThis.URL(`/projects/${encodeURIComponent(project.project_id)}`, target.url()).href);
    await expect(target.getByRole("heading", { name: "project-one", exact: true })).toBeVisible();
  };
  const createDraft = async (target) => {
    const button = target.getByRole("button", { name: "New session", exact: true });
    for (let attempt = 0; attempt < 30; attempt += 1) {
      await expect(button).toBeEnabled({ timeout: 30000 });
      const result = target.waitForResponse((response) => (
        /\/projects\/[^/]+\/sessions$/.test(new globalThis.URL(response.url()).pathname)
        && response.request().method() === "POST"
      ));
      const claimed = target.waitForResponse(async (response) => {
        if (!/\/projects\/[^/]+\/sessions\/[^/]+\/claim$/.test(new globalThis.URL(response.url()).pathname)
          || response.request().method() !== "POST") return false;
        const created = await result;
        if (!created.ok()) return false;
        return new globalThis.URL(response.url()).pathname.endsWith(`/${(await created.json()).session_id}/claim`);
      }).catch(() => null);
      await button.click();
      const response = await result;
      const body = await response.json();
      if (response.ok()) {
        const claimResponse = await claimed;
        assert.ok(claimResponse);
        assert.equal(claimResponse.status(), 200);
        assert.equal((await claimResponse.json()).claim.session_id, body.session_id);
        await expect(button).toBeEnabled({ timeout: 30000 });
        await expect(target.locator("textarea")).toBeVisible({ timeout: 30000 });
        return;
      }
      assert.equal(body.code, "admission_closed", `Draft failed: ${JSON.stringify(body)}`);
      await new Promise((done) => setTimeout(done, 100));
    }
    throw new Error("New session remained admission-closed after 30 retries.");
  };
  const save = async (target, expectedStatus = 200) => {
    await target.bringToFront();
    await blurSettingsField(target);
    if (expectedStatus !== 200) {
      await expect(target.getByRole("alert").filter({ hasText: "Settings need attention" })).toBeVisible();
      return null;
    }
    try {
      await expect(target.locator('[role="status"][data-state="saving"]')).toBeVisible({ timeout: 1000 });
    } catch {
      // A quick save may already have returned the status badge to its saved state.
    }
    await waitForSavedSettings(target);
    const response = await target.evaluate(async () => {
      const credential = window.__omniTestControlCredential;
      const result = await globalThis.fetch("/api/v1/config", {
        credentials: "include",
        headers: credential == null ? {} : { "X-Omni-Control": credential },
      });
      if (!result.ok) throw new Error(`Configuration read failed: ${result.status}`);
      return result.json();
    });
    return response;
  };
  const readJsonLines = async (path) => {
    const text = await readFile(path, "utf8");
    return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
  };
  const availableModels = async (target) => target.evaluate(async () => {
    const credential = window.__omniTestControlCredential;
    const response = await globalThis.fetch("/api/v1/models/available", {
      credentials: "include",
      headers: credential == null ? {} : { "X-Omni-Control": credential },
    });
    return { status: response.status, body: await response.json() };
  });
  const primaryApiKeyAction = (target) => field(target, "settings-models-providers-primary-api_key-action");
  const primaryApiKeyValue = (target) => field(target, "settings-models-providers-primary-api_key-value");
  const retiredApiKeyAction = (target) => field(target, "settings-models-providers-retired-api_key-action");
  const remoteHeaderAction = (target) => field(target, "settings-mcp-remote-headers-Authorization-action");
  const remoteHeaderValue = (target) => field(target, "settings-mcp-remote-headers-Authorization-value");
  const defaultModel = (target) => field(target, "settings-models-routes-default-model");
  const chatModel = (target) => field(target, "settings-models-routes-chat-model");
  const smallModelContextWindow = (target) => field(target, "settings-models-providers-primary-model_context_windows-small-model");
  const largeModelContextWindow = (target) => field(target, "settings-models-providers-primary-model_context_windows-large-model");

  const initial = await openSettings(page);
  assert.equal(initial.fields.models.providers.primary.models[0], "small-model");
  assert.equal(initial.fields.models.routes.default.model, "small-model");
  const defaultReasoningEffort = initial.fields.models.routes.default.reasoning_effort;
  assert.equal(await smallModelContextWindow(page).inputValue(), "8192");
  assert.equal(await largeModelContextWindow(page).inputValue(), "");
  const activeModels = await availableModels(page);
  assert.equal(activeModels.status, 200);
  assert.deepEqual(activeModels.body.models, [
    { provider_id: "primary", model: "small-model", context_window: 8192 },
  ]);
  assert.deepEqual(activeModels.body.default_combination, {
    provider_id: "primary",
    model: "small-model",
    reasoning_effort: defaultReasoningEffort,
  });
  assert.equal(initial.fields.mcp.fixture.transport, "stdio");
  assert.equal(initial.fields.mcp.remote.headers.Authorization.configured, true);
  for (const secret of [
    "e2e-provider-secret-302",
    "e2e-retired-secret-302",
    "e2e-mcp-secret-302",
  ]) {
    assert.equal(JSON.stringify(initial).includes(secret), false, `Initial config response leaked ${secret}`);
  }

  const modelList = field(page, "settings-models-providers-primary-models");
  await modelList.getByRole("button", { name: "Add item", exact: true }).click();
  await modelList.getByRole("textbox", { name: "Models 3", exact: true }).fill("expanded-model-302");
  await primaryApiKeyAction(page).selectOption("replace");
  await primaryApiKeyValue(page).fill("e2e-provider-secret-replaced-302");
  await save(page);
  await settingsSection(page, "MCP");
  await field(page, "settings-mcp-remote-url").fill("http://127.0.0.1:1/edited-mcp-302");
  await field(page, "settings-mcp-remote-connect_timeout").fill("31");
  await field(page, "settings-mcp-remote-call_timeout").fill("61");

  await remoteHeaderAction(page).selectOption("replace");
  await remoteHeaderValue(page).fill("e2e-mcp-secret-replaced-302");
  await field(page, "settings-mcp-fixture-connect_timeout").fill("31");
  await field(page, "settings-mcp-fixture-call_timeout").fill("61");
  const fixtureKeywords = field(page, "settings-mcp-fixture-tool_keywords");
  await fixtureKeywords.getByRole("button", { name: "Add tool keywords", exact: true }).click();
  await fixtureKeywords.getByRole("textbox", { name: "Tool name", exact: true }).fill("fixture_echo_v1");
  const keywordList = field(page, "settings-mcp-fixture-tool_keywords-0");
  await keywordList.getByRole("button", { name: "Add item", exact: true }).click();
  await keywordList.getByRole("textbox").fill("resource");
  await field(page, "settings-mcp-fixture-cwd").fill("");

  await save(page);
  await waitForSavedSettings(page);
  let savedText = await readFile(configPath, "utf8");
  assert.match(savedText, /expanded-model-302/);
  assert.match(savedText, /e2e-provider-secret-replaced-302/);
  assert.match(savedText, /e2e-mcp-secret-replaced-302/);
  assert.doesNotMatch(savedText, /e2e-provider-secret-302/);

  await settingsSection(page, "Models");
  await primaryApiKeyAction(page).selectOption("keep");
  await retiredApiKeyAction(page).selectOption("clear");
  await save(page);
  await waitForSavedSettings(page);
  savedText = await readFile(configPath, "utf8");
  assert.match(savedText, /e2e-provider-secret-replaced-302/);
  assert.match(savedText, /e2e-prototype-header-canary-302/);
  assert.doesNotMatch(savedText, /e2e-provider-secret-302/);
  assert.doesNotMatch(savedText, /e2e-retired-secret-302/);
  await settingsSection(page, "MCP");
  await remoteHeaderAction(page).selectOption("clear");
  await save(page);
  await waitForSavedSettings(page);
  savedText = await readFile(configPath, "utf8");
  assert.doesNotMatch(savedText, /e2e-mcp-secret-302/);
  assert.doesNotMatch(savedText, /e2e-mcp-secret-replaced-302/);

  await settingsSection(page, "Models");
  await page.getByRole("button", { name: "Add provider", exact: true }).click();
  const providerId = field(page, "settings-models-providers-new-provider-id");
  await providerId.fill("primary");
  const collisionBytes = await readFile(configPath);
  await save(page, 422);
  const collisionSummary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(collisionSummary).toBeFocused();
  await collisionSummary.getByRole("link").filter({ hasText: "models.providers.new-provider.id" }).click();
  await expect(providerId).toBeFocused();
  assert.deepEqual(await readFile(configPath), collisionBytes);
  await settingsSection(page, "Runtime");
  await page.getByLabel("Maximum iterations", { exact: true }).fill("69");
  await page.getByLabel("Maximum iterations", { exact: true }).press("Tab");
  await expect.poll(async () => (await readFile(configPath, "utf8")).includes("max_iterations = 69")).toBe(true);
  // Successful saves in another section must preserve the invalid composite row.
  await collisionSummary.getByRole("link").filter({ hasText: "models.providers.new-provider.id" }).click();
  await expect(providerId).toBeFocused();
  await expect(providerId).toHaveValue("primary");
  await providerId.fill("");
  await providerId.pressSequentially("review-provider-302");
  await expect(providerId).toBeFocused();
  await field(page, "settings-models-providers-review-provider-302-protocol").selectOption("anthropic");
  await field(page, "settings-models-providers-review-provider-302-base_url").fill("http://127.0.0.1:1/models");
  // An unreferenced provider may keep an empty model list and cleared key.
  await save(page);
  await waitForSavedSettings(page);
  const addedProvider = await openSettings(secondPage);
  assert.equal(addedProvider.fields.models.providers["review-provider-302"].protocol, "anthropic");
  assert.deepEqual(addedProvider.fields.models.providers["review-provider-302"].models, []);
  await field(page, "settings-models-providers-review-provider-302").getByRole("button", { name: "Remove provider", exact: true }).click();
  await save(page);
  await waitForSavedSettings(page);

  // Editable new names keep a stable row while typing; list values are lossless.
  await settingsSection(page, "MCP");
  await page.getByRole("button", { name: "Add MCP server", exact: true }).click();
  const newName = page.getByRole("textbox", { name: "Server name", exact: true }).last();
  await newName.focus();
  await newName.pressSequentially("-custom");
  await expect(newName).toBeFocused();
  const customName = await newName.inputValue();
  const exactArgs = ["--flag", "--flag", "a,b", " x ", "", "line\nvalue"];
  const customArgs = field(page, `settings-mcp-${customName}-args`);
  for (const argument of exactArgs) {
    await customArgs.getByRole("button", { name: "Add item", exact: true }).click();
    await customArgs.getByRole("textbox").last().fill(argument);
  }
  await save(page);
  await waitForSavedSettings(page);
  let readback = await openSettings(secondPage);
  assert.deepEqual(readback.fields.mcp[customName].args, exactArgs);
  assert.equal(readback.fields.mcp[customName].cwd, null);
  await field(page, `settings-mcp-${customName}-transport`).selectOption("streamable-http");
  await field(page, `settings-mcp-${customName}-url`).fill("http://127.0.0.1:1/custom");
  const customCard = field(page, `settings-mcp-${customName}`);
  await customCard.getByRole("button", { name: "Add header", exact: true }).click();
  await customCard.getByRole("textbox", { name: "Header name", exact: true }).fill("X-Api-Key");
  await field(page, `settings-mcp-${customName}-headers-Authorization-action`).selectOption("replace");
  await field(page, `settings-mcp-${customName}-headers-Authorization-value`).fill("custom-header-canary-302");
  await save(page);
  await waitForSavedSettings(page);
  readback = await openSettings(secondPage);
  assert.deepEqual(readback.fields.mcp[customName].headers, { "X-Api-Key": { configured: true } });
  for (const [name, row, secret] of [["X.Test", "Authorization", "dot-header-canary-302"], ["X-Test", "X-Header-2", "dash-header-canary-302"]]) {
    await customCard.getByRole("button", { name: "Add header", exact: true }).click();
    await customCard.getByRole("textbox", { name: "Header name", exact: true }).last().fill(name);
    await field(page, `settings-mcp-${customName}-headers-${row}-action`).selectOption("replace");
    await field(page, `settings-mcp-${customName}-headers-${row}-value`).fill(secret);
  }
  await save(page);
  await waitForSavedSettings(page);
  await field(page, `settings-mcp-${customName}-headers-X%2ETest-action`).selectOption("replace");
  await save(page, 422);
  const headerSummary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(headerSummary).toBeFocused();
  await headerSummary.getByRole("link").click();
  await expect(field(page, `settings-mcp-${customName}-headers-X%2ETest-value`)).toBeFocused();
  await field(page, `settings-mcp-${customName}-headers-X%2ETest-value`).fill("dot-header-canary-302");
  await field(page, `settings-mcp-${customName}-transport`).selectOption("stdio");
  await field(page, `settings-mcp-${customName}-command`).fill("python");
  await save(page);
  await waitForSavedSettings(page);
  readback = await openSettings(secondPage);
  assert.deepEqual(readback.fields.mcp[customName].headers, {});
  await customCard.getByRole("button", { name: "Remove MCP server", exact: true }).click();
  await save(page);
  await waitForSavedSettings(page);

  const keywordsBefore = await readFile(configPath);
  const keywordInput = field(page, "settings-mcp-fixture-tool_keywords-0").getByRole("textbox").first();
  await keywordInput.fill("中文");
  await save(page, 422);
  const keywordSummary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(keywordSummary).toBeFocused();
  await keywordSummary.getByRole("link").click();
  await expect(keywordInput).toBeFocused();
  await expect(keywordInput).toHaveAttribute("aria-invalid", "true");
  assert.deepEqual(await readFile(configPath), keywordsBefore);
  await keywordInput.fill("resource");

  await settingsSection(page, "Models");
  const beforeInvalid = await readFile(configPath);
  await defaultModel(page).fill("missing-model-302");
  await save(page, 422);
  await expect(page.getByRole("alert").filter({ hasText: "Settings need attention" })).toBeVisible();
  assert.deepEqual(await readFile(configPath), beforeInvalid, "Invalid model candidate changed config bytes");

  let releaseStaleSave;
  let staleSaveArrived;
  const staleSaveGate = new Promise((done) => { releaseStaleSave = done; });
  const staleSaveArrival = new Promise((done) => { staleSaveArrived = done; });
  const delayedStaleSave = async (route) => {
    if (route.request().method() !== "PATCH"
      || route.request().postDataJSON()?.fields?.models?.routes?.chat?.temperature !== 0.8) return route.continue();
    staleSaveArrived();
    await staleSaveGate;
    await route.continue();
  };
  await page.route("**/api/v1/config", delayedStaleSave);
  const staleResponsePromise = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
    && response.request().postDataJSON()?.fields?.models?.routes?.chat?.temperature === 0.8
  ));
  // Keep the model candidate invalid until all conflicting edits are complete.
  const chatTemperature = field(page, "settings-models-routes-chat-temperature");
  await chatTemperature.fill("0.8");
  await chatModel(page).fill("large-model");
  await defaultModel(page).fill("large-model");
  await defaultModel(page).press("Tab");
  await staleSaveArrival;
  await openSettings(secondPage);
  const competingSave = secondPage.waitForResponse(response => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
    && response.request().postDataJSON()?.fields?.models?.routes?.chat?.temperature === 0.1
  ));
  await field(secondPage, "settings-models-routes-chat-temperature").fill("0.1");
  await save(secondPage);
  const competingResponse = await competingSave;
  assert.equal(competingResponse.status(), 200);
  assert.equal((await competingResponse.json()).fields.models.routes.chat.temperature, 0.1);
  await waitForSavedSettings(secondPage);
  const beforeConflict = await readFile(configPath);
  await page.bringToFront();
  releaseStaleSave();
  const staleResponse = await staleResponsePromise;
  assert.equal(staleResponse.status(), 409);
  await page.unroute("**/api/v1/config", delayedStaleSave);
  await expect(page.getByText("These settings changed elsewhere. Your edits are still here.", { exact: true })).toBeVisible();
  assert.deepEqual(await readFile(configPath), beforeConflict, "Stale model save changed config bytes");
  await page.getByRole("button", { name: "Reload saved values", exact: true }).click();
  await expect(defaultModel(page)).toHaveValue("small-model");
  await expect(chatModel(page)).toHaveValue("small-model");
  const capacitySave = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config")
    && response.request().method() === "PATCH"
    && response.request().postDataJSON()?.fields?.models?.providers?.primary?.model_context_windows?.["large-model"] === 65536
  ));
  await largeModelContextWindow(page).fill("65536");
  await largeModelContextWindow(page).press("Tab");
  const capacityResponse = await capacitySave;
  assert.equal(capacityResponse.status(), 200, "Model capacity must save automatically on blur");
  const capacitySaved = await capacityResponse.json();
  assert.equal(capacitySaved.fields.models.providers.primary.model_context_windows["large-model"], 65536);
  assert.equal(capacitySaved.application.status, "restart-required");
  await waitForSavedSettings(page);
  const beforeCapacityRestart = await availableModels(page);
  assert.deepEqual(beforeCapacityRestart.body.models, [
    { provider_id: "primary", model: "small-model", context_window: 8192 },
  ], "Saved capacity must not replace the active model projection before restart");
  await defaultModel(page).fill("large-model");
  await chatModel(page).fill("large-model");
  const conflictResolved = await save(page);
  await waitForSavedSettings(page);

  const v1Startup = await control.restart();
  await openRestartedPage(page, `${v1Startup.url}/#ticket=${encodeURIComponent(v1Startup.ticket)}`);
  await openRestartedPage(secondPage, `${v1Startup.url}/#ticket=${encodeURIComponent(v1Startup.second_ticket)}`);
  await openSettings(page);
  assert.equal(await largeModelContextWindow(page).inputValue(), "65536");
  const restartedModels = await availableModels(page);
  assert.equal(restartedModels.status, 200);
  assert.deepEqual(restartedModels.body.models, [
    { provider_id: "primary", model: "small-model", context_window: 8192 },
    { provider_id: "primary", model: "large-model", context_window: 65536 },
  ]);
  assert.deepEqual(restartedModels.body.default_combination, {
    provider_id: "primary",
    model: "large-model",
    reasoning_effort: defaultReasoningEffort,
  });
  assert.equal(JSON.stringify(restartedModels).includes("e2e-provider-secret-replaced-302"), false);
  console.log("Settings model/provider/route/MCP E2E: starting old-generation v1 barrier");
  await control.command("model-mcp-arm");
  await openProject(page);
  await createDraft(page);
  await page.locator("textarea").fill("model MCP generation barrier");
  await page.locator("textarea").press("Enter");
  await control.command("model-mcp-wait");
  console.log("Settings model/provider/route/MCP E2E: v1 provider request is holding");
  const oldObservations = await readJsonLines(providerObservationPath);
  console.log(`Settings model/provider/route/MCP E2E: observations=${JSON.stringify(oldObservations.slice(-5))}`);
  assert.ok(oldObservations.some((observation) => (
    observation.model === "large-model"
    && observation.tools.includes("tool_search")
  )), "The active generation did not reach the large-model Tool Search request");

  const runtimeResponse = page.waitForResponse((response) => (
    response.url().endsWith("/management/status") && response.request().method() === "POST"
  ));
  const runtimeMenu = page.locator("details").filter({
    has: page.getByRole("button", { name: "Runtime status and controls", exact: true, includeHidden: true }),
  });
  await runtimeMenu.locator("summary").click();
  await runtimeMenu.getByRole("button", { name: "Runtime status and controls", exact: true }).click();
  const runtime = await runtimeResponse;
  assert.equal(runtime.status(), 200);
  const budget = (await runtime.json()).result.status_view;
  assert.equal(budget.chat_model, "primary/large-model");
  assert.equal(budget.context_window, 65536);
  assert.equal(budget.max_output, conflictResolved.fields.models.routes.chat.max_output);
  assert.equal(budget.available_context, 65536 - budget.max_output);
  assert.equal(budget.compact_context_window, Math.ceil(budget.available_context * budget.compact_ratio));
  assert.equal(budget.compact_ratio, conflictResolved.fields.runtime.compact_ratio);
  await page.keyboard.press("Escape");

  await openSettings(page);
  await defaultModel(page).fill("small-model");
  await chatModel(page).fill("small-model");
  await settingsSection(page, "MCP");
  const args = field(page, "settings-mcp-fixture-args").getByRole("textbox");
  await args.last().fill(mcpV2Path);
  await field(page, "settings-mcp-fixture-tool_keywords").getByRole("textbox", { name: "Tool name", exact: true }).fill("fixture_echo_v2");
  console.log("Settings model/provider/route/MCP E2E: saving new model and v2 MCP while v1 is active");
  const pending = await save(page);
  assert.equal(pending.application.status, "restart-required");
  assert.equal(pending.application.active_revision, conflictResolved.revision);
  assert.notEqual(pending.application.restart_required, false);
  await control.command("model-mcp-release");
  await waitForSavedSettings(page);
  assert.equal((await readJsonLines(providerObservationPath)).some((observation) => (
    observation.tools.some((name) => name.endsWith("fixture_echo_v2"))
  )), false, "Saved MCP settings activated before restart");
  const v2Startup = await control.restart();
  await openRestartedPage(page, `${v2Startup.url}/#ticket=${encodeURIComponent(v2Startup.ticket)}`);
  await openRestartedPage(secondPage, `${v2Startup.url}/#ticket=${encodeURIComponent(v2Startup.second_ticket)}`);
  await openSettings(page);
  console.log("Settings model/provider/route/MCP E2E: explicit restart activated v2 settings");

  await openProject(page);
  await createDraft(page);
  await page.locator("textarea").fill("model MCP resource");
  await page.locator("textarea").press("Enter");
  try {
    await expect(page.getByText("New model and MCP resource completed.", { exact: true })).toBeVisible({ timeout: 30000 });
  } catch (error) {
    const observations = await readJsonLines(providerObservationPath);
    const alerts = await page.getByRole("alert").allTextContents();
    const statuses = await page.getByRole("status").allTextContents();
    console.log(`Settings model/provider/route/MCP E2E: v2 diagnostics=${JSON.stringify({
      observations: observations.slice(-8),
      alerts,
      statuses,
    })}`);
    throw error;
  }
  console.log("Settings model/provider/route/MCP E2E: v2 provider/MCP request completed");

  const observations = await readJsonLines(providerObservationPath);
  assert.ok(observations.some((observation) => (
    observation.model === "large-model"
    && observation.tools.some((name) => name.endsWith("fixture_echo_v1"))
  )), "Provider did not observe the old model and MCP v1 resource");
  assert.ok(observations.some((observation) => (
    observation.model === "small-model"
    && observation.tools.some((name) => name.endsWith("fixture_echo_v2"))
  )), "Provider did not observe the new model and MCP v2 resource");
  const mcpV1Requests = await readJsonLines(mcpV1Path.replace(/\.json$/, ".jsonl"));
  const mcpV2Requests = await readJsonLines(mcpV2Path.replace(/\.json$/, ".jsonl"));
  assert.ok(mcpV1Requests.some((request) => request.method === "tools/list"));
  assert.ok(mcpV1Requests.some((request) => request.method === "tools/call"));
  assert.ok(mcpV2Requests.some((request) => request.method === "tools/list"));
  assert.ok(mcpV2Requests.some((request) => request.method === "tools/call"));

  await page.getByRole("button", { name: "Release session", exact: true }).click();
  await expect(page.getByRole("button", { name: "New session", exact: true })).toBeVisible();
  await openSettings(page);
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const width of [390, 768, 1024, 1440]) {
        await page.setViewportSize({ width, height: 900 });
        for (const [section, id] of Object.entries({ models: "settings-models-providers-primary-base_url", routes: "settings-models-routes-chat-model", mcp: "settings-mcp-fixture-command", headers: "settings-mcp-remote-headers-__proto__-action" })) {
          await settingsSection(page, section === "mcp" || section === "headers" ? "MCP" : "Models");
          const input = field(page, id);
          await input.scrollIntoViewIfNeeded();
          await input.focus();
          await expect(input).toBeFocused();
          await page.screenshot({ path: resolve(output, `settings-302-${section}-${language}-${theme}-${width}.png`) });
        }
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
      }
    }
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.setViewportSize({ width: 1440, height: 900 });
  await openProject(page);
  const bodies = await Promise.all(responseBodies);
  const events = await page.evaluate(() => window.__omniTestMessages ?? []);
  for (const secret of [
    "e2e-provider-secret-302",
    "e2e-retired-secret-302",
    "e2e-provider-secret-replaced-302",
    "e2e-mcp-secret-302",
    "e2e-mcp-secret-replaced-302",
    "e2e-prototype-header-canary-302",
    "custom-header-canary-302",
    "dot-header-canary-302",
    "dash-header-canary-302",
  ]) {
    assert.equal(bodies.some((body) => body.includes(secret)), false, `Config response leaked ${secret}`);
    assert.equal(JSON.stringify(events).includes(secret), false, `Browser event leaked ${secret}`);
  }
  console.log("Settings model/provider/route/MCP E2E: structured safe readback, secret replace/keep/clear, invalid bytes, stale CAS, saved settings with an active Run, and real old/new model plus stdio MCP resources passed");
}

export default async function settingsAcceptance({ page, secondPage, control, output, viewports }) {
  const settings = async (target) => {
    await target.bringToFront();
    await target.getByRole("button", { name: "EN", exact: true }).click();
    await openServiceStatus(target);
    await target.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
    await target.getByRole("navigation", { name: "Settings sections", exact: true }).waitFor();
  };
  const runtimeSettings = async (target) => {
    await settingsSection(target, "Runtime");
    await expect(target.getByLabel("Maximum iterations", { exact: true })).toBeEnabled();
  };
  const memorySettings = (target) => settingsSection(target, "Memory");
  const save = async (target, expectedStatus = 200) => {
    await target.bringToFront();
    await blurSettingsField(target);
    if (expectedStatus === 200) {
      try {
        await expect(target.locator('[role="status"][data-state="saving"]')).toBeVisible({ timeout: 1000 });
      } catch {
        // A quick save may already have returned the status badge to its saved state.
      }
      await waitForSavedSettings(target);
      return target.evaluate(async () => {
        const credential = window.__omniTestControlCredential;
        const response = await globalThis.fetch("/api/v1/config", {
          credentials: "include",
          headers: credential == null ? {} : { "X-Omni-Control": credential },
        });
        if (!response.ok) throw new Error(`Configuration read failed: ${response.status}`);
        return response.json();
      });
    }
    await expect(target.getByRole("alert").filter({ hasText: "Settings need attention" })).toBeVisible();
    return null;
  };
  const configPath = resolve(control.details.home_root, ".omni", "config.toml");
  await settings(page);
  await settingsSection(page, "General & appearance");
  await page.locator("#settings-theme").selectOption("light");
  await page.locator("#settings-language").selectOption("zh-CN");
  await expect(page.getByRole("heading", { name: "常规与外观", exact: true })).toBeVisible();
  await page.locator("#settings-language").selectOption("en");
  const sectionNavigation = page.getByRole("navigation", { name: "Settings sections", exact: true });
  const sections = ["General & appearance", "Models", "Runtime", "Memory", "MCP"];
  for (const section of sections) {
    await sectionNavigation.getByRole("button", { name: section, exact: true }).click();
    await expect(page.getByRole("heading", { name: section, exact: true, level: 2 })).toBeVisible();
  }
  await expect(page.getByRole("button", { name: "Save settings", exact: true })).toHaveCount(0);
  await sectionNavigation.getByRole("button", { name: "Runtime", exact: true }).click();
  await expect(page.getByRole("region", { name: "Settings", exact: true }).getByText("Service status", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Back to conversation", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Service status", exact: true })).toBeVisible();
  await settings(page);
  await runtimeSettings(page);
  const original = await readFile(configPath);
  const invalidField = page.getByLabel("Maximum iterations", { exact: true });
  await invalidField.fill("1");
  await invalidField.press("Tab");
  const summary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(summary).toContainText("Review the highlighted settings.");
  await expect(invalidField).toHaveAttribute("aria-invalid", "true");
  assert.deepEqual(await readFile(configPath), original, "Invalid browser edits changed config bytes");

  const iterations = page.getByLabel("Maximum iterations", { exact: true });
  const automaticSave = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  await iterations.fill("61");
  await iterations.press("Tab");
  assert.equal((await automaticSave).status(), 200, "A valid field must save on blur");
  await waitForSavedSettings(page);
  await settings(secondPage);
  await runtimeSettings(secondPage);
  await memorySettings(secondPage);
  await secondPage.getByLabel("Memory batch size", { exact: true }).fill("12");
  const competing = await save(secondPage);
  await page.bringToFront();
  await expect(page.getByRole("definition").filter({ hasText: competing.revision })).toHaveCount(1);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("61");
  await runtimeSettings(page);
  await page.getByLabel("Maximum iterations", { exact: true }).fill("62");
  await page.getByLabel("Maximum iterations", { exact: true }).press("Tab");
  const saved = await save(page);
  await waitForSavedSettings(page);
  assert.match(await readFile(configPath, "utf8"), /max_iterations = 62/);
  assert.match(await readFile(configPath, "utf8"), /large-model/);

  const hold = await control.command("settings-hold");
  await page.evaluate(() => { window.__settingsSocketBefore = window.__omniTestSocket; });
  await memorySettings(page);
  await page.getByLabel("Memory batch size", { exact: true }).fill("13");
  const pending = await save(page);
  assert.equal(pending.application.status, "restart-required");
  assert.equal(pending.application.active_revision, saved.application.active_revision);
  await expect(page.getByText("Saved; restart Omni to use these settings.", { exact: true }).first()).toBeVisible();
  await openServiceStatus(page);
  await expect(page.getByRole("heading", { name: "Service status", exact: true })).toBeVisible();
  await settings(page);
  await runtimeSettings(page);
  const released = await control.command("settings-release");
  assert.equal(released.pid, hold.pid, "Configuration application restarted the service");
  await waitForSavedSettings(page);
  assert.equal(await page.evaluate(() => window.__settingsSocketBefore === window.__omniTestSocket), true,
    "Configuration application replaced the browser connection");

  const memoryPath = resolve(control.details.cli_workspace, ".omni", "memory", "memory.md");
  const memory = await readFile(memoryPath);
  try {
    await writeFile(memoryPath, Buffer.from([0xff, 0xfe]));
    await runtimeSettings(page);
    await page.getByLabel("Maximum iterations", { exact: true }).fill("63");
    const failedSave = await save(page);
    assert.equal(failedSave.application.status, "restart-required");
    assert.equal(failedSave.application.active_revision, pending.application.active_revision);
    await expect(page.getByRole("button", { name: "Retry application", exact: true })).toHaveCount(0);
  } finally {
    await writeFile(memoryPath, memory);
  }

  let releasePoll;
  let pollArrived;
  const gate = new Promise((done) => { releasePoll = done; });
  const arrival = new Promise((done) => { pollArrived = done; });
  const delayedPoll = async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    const response = await route.fetch();
    pollArrived();
    await gate;
    await route.fulfill({ response });
  };
  await page.route("**/api/v1/config", delayedPoll);
  await arrival;
  const saveAgainstOldPoll = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  await page.getByLabel("Maximum iterations", { exact: true }).fill("64");
  await page.getByLabel("Maximum iterations", { exact: true }).press("Tab");
  const freshRevision = (await (await saveAgainstOldPoll).json()).revision;
  await waitForSavedSettings(page);
  const deliveredPoll = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "GET"
  ));
  releasePoll();
  await (await deliveredPoll).finished();
  await page.evaluate(() => new Promise((done) => window.requestAnimationFrame(done)));
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("64");
  await expect(page.getByRole("definition").filter({ hasText: freshRevision })).toHaveCount(1);
  await page.unroute("**/api/v1/config", delayedPoll);
  await save(page);

  let releaseSave;
  let saveArrived;
  const saveGate = new Promise((done) => { releaseSave = done; });
  const saveArrival = new Promise((done) => { saveArrived = done; });
  const delayedSave = async (route) => {
    if (route.request().method() !== "PATCH") return route.continue();
    const response = await route.fetch();
    assert.equal(response.status(), 200);
    saveArrived();
    await saveGate;
    await route.fulfill({ response });
  };
  await page.route("**/api/v1/config", delayedSave);
  await page.getByLabel("Maximum iterations", { exact: true }).fill("66");
  await page.getByLabel("Maximum iterations", { exact: true }).press("Tab");
  await saveArrival;
  await openServiceStatus(page);
  await settings(page);
  await runtimeSettings(page);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("66");
  await page.getByLabel("Maximum iterations", { exact: true }).fill("67");
  await page.getByLabel("Maximum iterations", { exact: true }).press("Tab");
  await page.getByRole("button", { name: "Back to conversation", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Service status", exact: true })).toBeVisible();
  await settings(page);
  await runtimeSettings(page);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  const deliveredSave = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  releaseSave();
  await (await deliveredSave).finished();
  await page.evaluate(() => new Promise((done) => window.requestAnimationFrame(done)));
  await page.unroute("**/api/v1/config", delayedSave);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  const iterationsField = page.getByLabel("Maximum iterations", { exact: true });
  await expect(iterationsField).toBeEnabled();
  await page.route("**/api/v1/clients", (route) => route.abort());
  await page.evaluate(() => window.__omniTestSocket.close());
  await expect(iterationsField).toBeDisabled({ timeout: 10000 });
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  await page.unroute("**/api/v1/clients");
  await expect(iterationsField).toBeEnabled({ timeout: 10000 });
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  await save(page);
  await waitForSavedSettings(page);

  const lostRequests = [];
  let loseResponse = true;
  const lostAcknowledgement = async (route) => {
    if (route.request().method() !== "PATCH") return route.continue();
    lostRequests.push(route.request().postDataJSON());
    const response = await route.fetch();
    assert.equal(response.status(), 200);
    if (loseResponse) {
      loseResponse = false;
      await route.abort();
    } else await route.fulfill({ response });
  };
  await page.route("**/api/v1/config", lostAcknowledgement);
  await iterationsField.fill("68");
  await iterationsField.press("Tab");
  await expect(page.getByRole("button", { name: "Retry save", exact: true })).toBeVisible();
  assert.match(await readFile(configPath, "utf8"), /max_iterations = 68/);
  await page.getByRole("button", { name: "Retry save", exact: true }).click();
  await waitForSavedSettings(page);
  assert.equal(lostRequests.length, 2);
  assert.deepEqual(lostRequests[1], lostRequests[0], "Unknown-result retry must use the identical request ID and candidate");
  loseResponse = true;
  await iterationsField.fill("69");
  await iterationsField.press("Tab");
  await expect(page.getByRole("button", { name: "Retry save", exact: true })).toBeVisible();
  await iterationsField.fill("70");
  await iterationsField.press("Tab");
  await waitForSavedSettings(page);
  assert.equal(lostRequests.length, 5);
  assert.deepEqual(lostRequests[3], lostRequests[2]);
  assert.notEqual(lostRequests[4].request_id, lostRequests[3].request_id);
  assert.match(await readFile(configPath, "utf8"), /max_iterations = 70/);
  await page.unroute("**/api/v1/config", lostAcknowledgement);

  await mkdir(output, { recursive: true });
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of [...viewports, { width: 390, height: 844 }]) {
        await page.setViewportSize(viewport);
        const label = language === "en" ? "Maximum iterations" : "最大迭代次数";
        await page.getByLabel(label, { exact: true }).focus();
        await expect(page.getByLabel(label, { exact: true })).toBeFocused();
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
          `Settings overflow at ${language}/${theme}/${viewport.width}`);
        await page.screenshot({ path: resolve(output, `settings-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.setViewportSize(viewports[0]);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  console.log("Settings production CSP E2E: global without Claim, invalid bytes, cross-client stale CAS and explicit reload, dirty late poll, real active Run save/restart with same PID/WS, save remains independent of resource preparation, versions, keyboard and 4 locale/theme x 4 viewports passed");
}
