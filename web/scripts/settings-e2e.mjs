import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { expect } from "@playwright/test";

async function waitForActiveGeneration(target) {
  try {
    await expect(target.locator('[role="status"][data-state="active"]')).toBeVisible({ timeout: 30000 });
  } catch (error) {
    const statuses = await target.getByRole("status").allTextContents();
    const alerts = await target.getByRole("alert").allTextContents();
    const badges = await target.locator('[role="status"][data-state]').evaluateAll((items) => (
      items.map((item) => ({ state: item.getAttribute("data-state"), text: item.textContent }))
    ));
    throw new Error(`${error.message}\nStatus badges: ${JSON.stringify(badges)}\nSettings statuses: ${JSON.stringify(statuses)}\nAlerts: ${JSON.stringify(alerts)}`);
  }
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
    acceptedRun = await page.evaluate((sessionId) => [...(window.__myclawTestMessages ?? [])]
      .reverse().find((event) => (
        event.type === "input.accepted" && event.session_id === sessionId
        && event.payload?.text === "settings generation barrier confirmation"
      )), created.session_id);
    return acceptedRun?.session_id;
  }, { timeout: 10000, message: "Settings input must be accepted in the newly claimed draft" }).toBe(created.session_id);
  await control.command("settings-wait");
  await page.getByRole("navigation").getByRole("link", { name: "Settings", exact: true }).click();
  const field = page.getByLabel("Maximum iterations", { exact: true });
  await expect(field).toBeEnabled();
  await field.fill("65");
  const savedResponse = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  const saved = await savedResponse;
  assert.equal(saved.status(), 200);
  assert.equal((await saved.json()).application.status, "pending");
  await control.command("settings-release");
  const dialog = page.getByRole("dialog", { name: "Tool Confirmation", exact: true });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("confirmation-outside.txt");
  await dialog.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(dialog).toBeHidden();
  await waitForActiveGeneration(page);
  console.log("Settings pending generation: existing browser Run Tool confirmation remains usable and finishes naturally after save");
  return acceptedRun.run_id;
}

export async function settingsModelMcpAcceptance({ page, secondPage, control, output }) {
  const configPath = resolve(control.details.home_root, ".myclaw", "config.toml");
  const providerObservationPath = process.env.MYCLAW_E2E_PROVIDER_OBSERVATION_PATH;
  const mcpV1Path = process.env.MYCLAW_E2E_MCP_V1_PATH;
  const mcpV2Path = process.env.MYCLAW_E2E_MCP_V2_PATH;
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
  const openSettings = async (target) => {
    await target.bringToFront();
    await target.getByRole("button", { name: "EN", exact: true }).click();
    const received = target.waitForResponse((response) => (
      response.url().endsWith("/api/v1/config") && response.request().method() === "GET"
    ));
    await target.getByRole("navigation").getByRole("link", { name: "Settings", exact: true }).click();
    const response = await received;
    assert.equal(response.status(), 200);
    await expect(field(target, "settings-models-providers-primary-base_url")).toBeEnabled();
    return response.json();
  };
  const openProject = async (target) => {
    const project = await target.evaluate(async () => {
      const credential = window.__myclawTestControlCredential;
      const response = await globalThis.fetch("/api/v1/projects", {
        credentials: "include",
        headers: credential == null ? {} : { "X-MyClaw-Control": credential },
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
      await button.click();
      const response = await result;
      const body = await response.json();
      if (response.ok()) {
        await expect(target.locator("textarea")).toBeVisible({ timeout: 30000 });
        return;
      }
      assert.equal(body.code, "admission_closed", `Draft failed: ${JSON.stringify(body)}`);
      await new Promise((done) => setTimeout(done, 100));
    }
    throw new Error("New session remained admission-closed after 30 retries.");
  };
  const save = async (target, expectedStatus = 200) => {
    const received = target.waitForResponse((response) => (
      response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
    ));
    await target.getByRole("button", { name: "Save settings", exact: true }).click();
    const response = await received;
    assert.equal(response.status(), expectedStatus);
    return response.json();
  };
  const readJsonLines = async (path) => {
    const text = await readFile(path, "utf8");
    return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
  };
  const primaryApiKeyAction = (target) => field(target, "settings-models-providers-primary-api_key-action");
  const primaryApiKeyValue = (target) => field(target, "settings-models-providers-primary-api_key-value");
  const retiredApiKeyAction = (target) => field(target, "settings-models-providers-retired-api_key-action");
  const remoteHeaderAction = (target) => field(target, "settings-mcp-remote-headers-Authorization-action");
  const remoteHeaderValue = (target) => field(target, "settings-mcp-remote-headers-Authorization-value");
  const defaultModel = (target) => field(target, "settings-models-routes-default-model");
  const chatModel = (target) => field(target, "settings-models-routes-chat-model");

  const initial = await openSettings(page);
  assert.equal(initial.fields.models.providers.primary.models[0], "small-model");
  assert.equal(initial.fields.models.routes.default.model, "small-model");
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
  await waitForActiveGeneration(page);
  let savedText = await readFile(configPath, "utf8");
  assert.match(savedText, /expanded-model-302/);
  assert.match(savedText, /e2e-provider-secret-replaced-302/);
  assert.match(savedText, /e2e-mcp-secret-replaced-302/);
  assert.doesNotMatch(savedText, /e2e-provider-secret-302/);

  await primaryApiKeyAction(page).selectOption("keep");
  await retiredApiKeyAction(page).selectOption("clear");
  await remoteHeaderAction(page).selectOption("clear");
  await save(page);
  await waitForActiveGeneration(page);
  savedText = await readFile(configPath, "utf8");
  assert.match(savedText, /e2e-provider-secret-replaced-302/);
  assert.match(savedText, /e2e-prototype-header-canary-302/);
  assert.doesNotMatch(savedText, /e2e-provider-secret-302/);
  assert.doesNotMatch(savedText, /e2e-retired-secret-302/);
  assert.doesNotMatch(savedText, /e2e-mcp-secret-302/);
  assert.doesNotMatch(savedText, /e2e-mcp-secret-replaced-302/);

  await page.getByRole("button", { name: "Add provider", exact: true }).click();
  const providerId = field(page, "settings-models-providers-new-provider-id");
  await providerId.fill("primary");
  const collisionBytes = await readFile(configPath);
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  const collisionSummary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(collisionSummary).toBeFocused();
  await collisionSummary.getByRole("link").filter({ hasText: "models.providers.new-provider.id" }).click();
  await expect(providerId).toBeFocused();
  assert.deepEqual(await readFile(configPath), collisionBytes);
  await providerId.fill("");
  await providerId.pressSequentially("review-provider-302");
  await expect(providerId).toBeFocused();
  await field(page, "settings-models-providers-review-provider-302-protocol").selectOption("anthropic");
  await field(page, "settings-models-providers-review-provider-302-base_url").fill("http://127.0.0.1:1/models");
  // An unreferenced provider may keep an empty model list and cleared key.
  await save(page);
  await waitForActiveGeneration(page);
  const addedProvider = await openSettings(secondPage);
  assert.equal(addedProvider.fields.models.providers["review-provider-302"].protocol, "anthropic");
  assert.deepEqual(addedProvider.fields.models.providers["review-provider-302"].models, []);
  await field(page, "settings-models-providers-review-provider-302").getByRole("button", { name: "Remove provider", exact: true }).click();
  await save(page);
  await waitForActiveGeneration(page);

  // Editable new names keep a stable row while typing; list values are lossless.
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
  await waitForActiveGeneration(page);
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
  await waitForActiveGeneration(page);
  readback = await openSettings(secondPage);
  assert.deepEqual(readback.fields.mcp[customName].headers, { "X-Api-Key": { configured: true } });
  for (const [name, row, secret] of [["X.Test", "Authorization", "dot-header-canary-302"], ["X-Test", "X-Header-2", "dash-header-canary-302"]]) {
    await customCard.getByRole("button", { name: "Add header", exact: true }).click();
    await customCard.getByRole("textbox", { name: "Header name", exact: true }).last().fill(name);
    await field(page, `settings-mcp-${customName}-headers-${row}-action`).selectOption("replace");
    await field(page, `settings-mcp-${customName}-headers-${row}-value`).fill(secret);
  }
  await save(page);
  await waitForActiveGeneration(page);
  await field(page, `settings-mcp-${customName}-headers-X%2ETest-action`).selectOption("replace");
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  const headerSummary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(headerSummary).toBeFocused();
  await headerSummary.getByRole("link").click();
  await expect(field(page, `settings-mcp-${customName}-headers-X%2ETest-value`)).toBeFocused();
  await field(page, `settings-mcp-${customName}-headers-X%2ETest-value`).fill("dot-header-canary-302");
  await field(page, `settings-mcp-${customName}-transport`).selectOption("stdio");
  await field(page, `settings-mcp-${customName}-command`).fill("python");
  await save(page);
  await waitForActiveGeneration(page);
  readback = await openSettings(secondPage);
  assert.deepEqual(readback.fields.mcp[customName].headers, {});
  await customCard.getByRole("button", { name: "Remove MCP server", exact: true }).click();
  await save(page);
  await waitForActiveGeneration(page);

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

  const beforeInvalid = await readFile(configPath);
  await defaultModel(page).fill("missing-model-302");
  await save(page, 422);
  await expect(page.getByRole("alert").filter({ hasText: "Settings need attention" })).toBeVisible();
  assert.deepEqual(await readFile(configPath), beforeInvalid, "Invalid model candidate changed config bytes");
  await defaultModel(page).fill("large-model");

  await defaultModel(page).fill("large-model");
  await chatModel(page).fill("large-model");
  await openSettings(secondPage);
  await field(secondPage, "settings-models-routes-chat-temperature").fill("0.1");
  await save(secondPage);
  await waitForActiveGeneration(secondPage);
  const beforeConflict = await readFile(configPath);
  await save(page, 409);
  await expect(page.getByText("These settings changed elsewhere. Your edits are still here.", { exact: true })).toBeVisible();
  assert.deepEqual(await readFile(configPath), beforeConflict, "Stale model save changed config bytes");
  await page.getByRole("button", { name: "Reload saved values", exact: true }).click();
  await expect(defaultModel(page)).toHaveValue("small-model");
  await expect(chatModel(page)).toHaveValue("small-model");
  await defaultModel(page).fill("large-model");
  await chatModel(page).fill("large-model");
  const conflictResolved = await save(page);
  await waitForActiveGeneration(page);

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

  await openSettings(page);
  await defaultModel(page).fill("small-model");
  await chatModel(page).fill("small-model");
  const args = field(page, "settings-mcp-fixture-args").getByRole("textbox");
  await args.last().fill(mcpV2Path);
  await field(page, "settings-mcp-fixture-tool_keywords").getByRole("textbox", { name: "Tool name", exact: true }).fill("fixture_echo_v2");
  console.log("Settings model/provider/route/MCP E2E: saving new model and v2 MCP while v1 is active");
  const pending = await save(page);
  assert.equal(pending.application.status, "pending");
  assert.equal(pending.application.active_revision, conflictResolved.revision);
  assert.notEqual(pending.application.pending_revision, null);
  await control.command("model-mcp-release");
  await waitForActiveGeneration(page);
  console.log("Settings model/provider/route/MCP E2E: old generation released and new generation active");

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
  const events = await page.evaluate(() => window.__myclawTestMessages ?? []);
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
  console.log("Settings model/provider/route/MCP E2E: structured safe readback, secret replace/keep/clear, invalid bytes, stale CAS, pending active Run, and real old/new model plus stdio MCP resources passed");
}

export default async function settingsAcceptance({ page, secondPage, control, output, viewports }) {
  const settings = async (target) => {
    await target.bringToFront();
    await target.getByRole("button", { name: "EN", exact: true }).click();
    await target.getByRole("navigation").getByRole("link", { name: "Settings", exact: true }).click();
    await expect(target.getByLabel("Maximum iterations", { exact: true })).toBeEnabled();
  };
  const save = async (target, expectedStatus = 200) => {
    const received = target.waitForResponse((response) => (
      response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
    ));
    await target.getByRole("button", { name: "Save settings", exact: true }).click();
    const response = await received;
    assert.equal(response.status(), expectedStatus);
    return response.json();
  };
  const configPath = resolve(control.details.home_root, ".myclaw", "config.toml");
  await settings(page);
  const original = await readFile(configPath);
  await page.getByLabel("Maximum iterations", { exact: true }).fill("1");
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  const summary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(summary).toContainText("Review the highlighted settings.");
  await expect(summary).toBeFocused();
  await summary.getByRole("link").click();
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toBeFocused();
  assert.deepEqual(await readFile(configPath), original, "Invalid browser edits changed config bytes");

  await page.getByLabel("Maximum iterations", { exact: true }).fill("61");
  await settings(secondPage);
  await secondPage.getByLabel("Memory batch size", { exact: true }).fill("12");
  const competing = await save(secondPage);
  await page.bringToFront();
  await expect(page.getByRole("definition").filter({ hasText: competing.revision })).toHaveCount(2);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("61");
  const beforeConflict = await readFile(configPath);
  await save(page, 409);
  await expect(page.getByText("These settings changed elsewhere. Your edits are still here.", { exact: true })).toBeVisible();
  assert.deepEqual(await readFile(configPath), beforeConflict, "Stale browser save changed config bytes");
  await page.getByRole("button", { name: "Reload saved values", exact: true }).click();
  await expect(page.getByLabel("Memory batch size", { exact: true })).toHaveValue("12");
  await page.getByLabel("Maximum iterations", { exact: true }).fill("62");
  const saved = await save(page);
  await waitForActiveGeneration(page);
  assert.match(await readFile(configPath, "utf8"), /max_iterations = 62/);
  assert.match(await readFile(configPath, "utf8"), /large-model/);

  const hold = await control.command("settings-hold");
  await page.evaluate(() => { window.__settingsSocketBefore = window.__myclawTestSocket; });
  await page.getByLabel("Memory batch size", { exact: true }).fill("13");
  const pending = await save(page);
  assert.equal(pending.application.status, "pending");
  assert.equal(pending.application.active_revision, saved.revision);
  await expect(page.getByText("Waiting for", { exact: true })).toBeVisible();
  await page.getByRole("navigation").getByRole("link", { name: "Status", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Service status", exact: true })).toBeVisible();
  await settings(page);
  const released = await control.command("settings-release");
  assert.equal(released.pid, hold.pid, "Configuration application restarted the service");
  await waitForActiveGeneration(page);
  assert.equal(await page.evaluate(() => window.__settingsSocketBefore === window.__myclawTestSocket), true,
    "Configuration application replaced the browser connection");

  const memoryPath = resolve(control.details.cli_workspace, ".myclaw", "memory", "memory.md");
  const memory = await readFile(memoryPath);
  try {
    await writeFile(memoryPath, Buffer.from([0xff, 0xfe]));
    await page.getByLabel("Maximum iterations", { exact: true }).fill("63");
    const failedSave = await save(page);
    await expect(page.getByRole("button", { name: "Retry application", exact: true })).toBeVisible({ timeout: 15000 });
    const versions = page.getByRole("definition");
    await expect(versions.filter({ hasText: failedSave.revision })).toHaveCount(2);
    await expect(versions.filter({ hasText: pending.revision })).toHaveCount(1);
    await writeFile(memoryPath, memory);
    const retryResponse = page.waitForResponse((response) => response.url().endsWith("/api/v1/config/retry"));
    await page.getByRole("button", { name: "Retry application", exact: true }).click();
    assert.equal((await retryResponse).status(), 200);
    await waitForActiveGeneration(page);
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
  await page.getByLabel("Maximum iterations", { exact: true }).fill("64");
  const deliveredPoll = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "GET"
  ));
  releasePoll();
  await (await deliveredPoll).finished();
  await page.evaluate(() => new Promise((done) => window.requestAnimationFrame(done)));
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("64");
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
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  await saveArrival;
  await page.getByRole("navigation").getByRole("link", { name: "Status", exact: true }).click();
  await settings(page);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("66");
  await page.getByLabel("Maximum iterations", { exact: true }).fill("67");
  const deliveredSave = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  releaseSave();
  await (await deliveredSave).finished();
  await page.evaluate(() => new Promise((done) => window.requestAnimationFrame(done)));
  await page.unroute("**/api/v1/config", delayedSave);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  const saveButton = page.getByRole("button", { name: "Save settings", exact: true });
  await expect(saveButton).toBeEnabled();
  await page.route("**/api/v1/clients", (route) => route.abort());
  await page.evaluate(() => window.__myclawTestSocket.close());
  await expect(saveButton).toBeDisabled({ timeout: 10000 });
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  await page.unroute("**/api/v1/clients");
  await expect(saveButton).toBeEnabled({ timeout: 10000 });
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  await save(page);
  await waitForActiveGeneration(page);

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
  console.log("Settings production CSP E2E: global without Claim, invalid bytes, cross-client stale CAS and explicit reload, dirty late poll, real active Run pending/activation with same PID/WS, real candidate resource failure/retry, versions, keyboard and 4 locale/theme x 4 viewports passed");
}
