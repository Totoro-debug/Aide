import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { URL } from "node:url";

import { chromium, expect } from "@playwright/test";

if (process.platform !== "win32") {
  console.error("Omni requires Windows.");
  process.exit(1);
}

const repoRoot = resolve(process.cwd(), "..");
const output = resolve(process.cwd(), "test-results", "config-startup-e2e");
const narrowViewport = { width: 390, height: 844 };
const viewports = [narrowViewport, { width: 768, height: 1024 }, { width: 1024, height: 768 }, { width: 1440, height: 900 }];
const states = ["missing", "invalid", "malformed"];

function servicePort(homeRoot) {
  let hash = 2166136261;
  for (const character of resolve(homeRoot)) {
    hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
  }
  return 20000 + (hash >>> 0) % 40000;
}

function startupCli(homeRoot, command) {
  const argv = JSON.stringify(["omni", ...command]);
  const source = [
    "import sys, webbrowser",
    "from omni.terminal.process_entry import run",
    "webbrowser.open_new_tab = lambda _url: False",
    `sys.argv = ${argv}`,
    "run()",
  ].join("; ");
  return spawnSync("python", ["-c", source], {
    cwd: repoRoot,
    env: {
      ...process.env,
      USERPROFILE: homeRoot,
      HOME: homeRoot,
      OMNI_SERVICE_PORT: String(servicePort(homeRoot)),
    },
    encoding: "utf8",
    timeout: 30000,
  });
}

async function startHarness(state, root) {
  const child = spawn(
    "python",
    ["-u", "-m", "web.scripts.config_startup_e2e_service", "--state", state, "--root", root],
    { cwd: repoRoot, stdio: ["pipe", "pipe", "pipe"] },
  );
  let stdout = "";
  let stderr = "";
  let closed = false;
  const waiters = [];
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
    let newline;
    while ((newline = stdout.indexOf("\n")) >= 0) {
      const line = stdout.slice(0, newline).trim();
      stdout = stdout.slice(newline + 1);
      waiters.shift()?.resolve(line);
    }
  });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  child.once("exit", (code) => {
    closed = true;
    const error = new Error(`Startup E2E service exited (${code}): ${stderr}`);
    for (const waiter of waiters.splice(0)) waiter.reject(error);
  });
  child.once("error", (error) => {
    closed = true;
    for (const waiter of waiters.splice(0)) waiter.reject(error);
  });

  const readLine = () => {
    if (closed) return Promise.reject(new Error(`Startup E2E service closed: ${stderr}`));
    return new Promise((resolveLine, rejectLine) => {
      const timeout = setTimeout(() => {
        const index = waiters.findIndex((waiter) => waiter.resolve === resolveLine);
        if (index >= 0) waiters.splice(index, 1);
        rejectLine(new Error(`Startup E2E service timed out: ${stderr}`));
      }, 30000);
      waiters.push({
        resolve: (line) => { clearTimeout(timeout); resolveLine(line); },
        reject: (error) => { clearTimeout(timeout); rejectLine(error); },
      });
    });
  };

  try {
    const line = await readLine();
    return {
      details: JSON.parse(line),
      async restart() {
        child.stdin.write("restart\n");
        return JSON.parse(await readLine());
      },
      async shutdown() {
        if (child.exitCode !== null) return;
        child.stdin.end("stop\n");
        await new Promise((resolveExit, rejectExit) => {
          const timeout = setTimeout(() => {
            child.kill();
            rejectExit(new Error(`Startup E2E service shutdown timed out: ${stderr}`));
          }, 15000);
          child.once("exit", (code) => {
            clearTimeout(timeout);
            if (code === 0) resolveExit();
            else rejectExit(new Error(`Startup E2E service shutdown failed (${code}): ${stderr}`));
          });
        });
      },
    };
  } catch (error) {
    child.kill();
    throw error;
  }
}

async function fetchJson(page, path, method = "GET", body = undefined) {
  return page.evaluate(async ({ path: requestPath, method: requestMethod, body: requestBody }) => {
    const headers = {};
    const control = window.__startupControlCredential;
    if (typeof control === "string") headers["X-Omni-Control"] = control;
    if (requestMethod !== "GET") {
      const session = await window.fetch("/api/v1/web/session", { credentials: "include" });
      const sessionBody = await session.json();
      headers["Content-Type"] = "application/json";
      headers["X-Omni-CSRF"] = sessionBody.csrf_token;
    }
    const response = await window.fetch(`/api/v1${requestPath}`, {
      method: requestMethod,
      credentials: "include",
      headers,
      body: requestBody === undefined ? undefined : JSON.stringify(requestBody),
    });
    const text = await response.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
    return { status: response.status, body: parsed };
  }, { path, method, body });
}

async function readConfig(page) {
  return fetchJson(page, "/config");
}

async function setupDraftAcceptance(context, initialPage, details, state, root) {
  let page = initialPage;
  const text = "无模型草稿\n保留 emoji 🐾 和末尾空格  ";
  const input = () => page.getByRole("textbox", { name: /Message input|消息输入/, exact: true });
  const sessionFiles = async () => (await readdir(root, { recursive: true }))
    .filter(path => /[\\/]sessions[\\/].*\.jsonl$/.test(path)).sort();
  const before = await sessionFiles();
  await expect.poll(() => page.evaluate(() => window.__startupSocket?.readyState)).toBe(1);
  await input().fill(text);
  const saved = await page.evaluate(() => JSON.parse(window.localStorage.getItem("omni.browser-recovery")));
  assert.deepEqual(saved.target, { kind: "new-chat" });
  assert.equal(saved.session_id, null);
  assert.equal(saved.model_configuration, null);
  assert.equal(saved.scroll_top, 0);
  assert.equal(saved.draft, true);
  assert.equal(saved.input_text, text);
  await page.reload();
  await expect(input()).toHaveValue(text);
  if (state === "missing") {
    for (const waitMs of [1000, 31000]) {
      await page.close();
      await delay(waitMs);
      page = await context.newPage();
      await page.goto(details.url);
      await expect(input()).toHaveValue(text);
    }
  }
  await page.getByRole("link", { name: /Configure models|配置模型/, exact: true }).click();
  await page.getByRole("button", { name: /Back to conversation|返回对话/, exact: true }).click();
  await expect(input()).toHaveValue(text);
  await page.locator("#app-sidebar").getByRole("link", { name: /New conversation|新建对话/, exact: true }).click();
  await expect(input()).toHaveValue("");
  await page.reload();
  await expect(input()).toHaveValue("");
  assert.deepEqual(await sessionFiles(), before, "Setup drafts created persistent Sessions");
  await input().fill(text);
  return page;
}

async function waitForActiveConfig(page) {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    const response = await readConfig(page);
    if (
      response.status === 200
      && response.body.configuration?.state === "active"
      && response.body.application?.status === "active"
      && response.body.application?.active_revision === response.body.revision
    ) {
      return response.body;
    }
    await delay(50);
  }
  throw new Error("Configuration did not reach an active generation.");
}

async function assertAdmissionClosed(page, projectId) {
  const service = await fetchJson(page, "/service");
  assert.equal(service.status, 200);
  assert.equal(service.body.active_workspace_count, 0, "Initial repair admitted a Workspace");
  const projects = await fetchJson(page, "/projects");
  assert.equal(projects.status, 200);
  const project = projects.body.projects.find((item) => item.project_id === projectId);
  assert.ok(project, "Persistent Project disappeared before repair");
  assert.equal(project.available, false, "Project was available before repair");
  assert.equal(project.schedule_state, "awaiting_resume");
  assert.equal(project.saved_jobs.length, 1, "Saved Schedule Jobs were not preserved");
  const session = await fetchJson(
    page,
    `/projects/${encodeURIComponent(projectId)}/sessions`,
    "POST",
    { request_id: `startup-admission-${Date.now()}` },
  );
  assert.equal(session.status, 422);
  assert.equal(session.body.code, "config_invalid");
  return project;
}

async function screenshotStates(page, state, phase) {
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", {
        name: theme === "light" ? /Light|浅色/ : /Dark|深色/,
      }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        assert.ok(
          await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
          `Settings overflow at ${state}/${phase}/${language}/${theme}/${viewport.width}`,
        );
        await page.screenshot({
          path: resolve(output, `${state}-${phase}-${language}-${theme}-${viewport.width}x${viewport.height}.png`),
        });
      }
    }
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("button", { name: /Light/ }).click();
  await page.setViewportSize(narrowViewport);
}

async function keyboardActivate(locator) {
  await keyboardReach(locator);
  await locator.press("Enter");
}

async function keyboardReach(locator) {
  await expect(locator).toBeVisible();
  for (let attempt = 0; attempt < 300; attempt += 1) {
    if (await locator.evaluate((element) => element === document.activeElement)) return;
    await locator.page().keyboard.press("Tab");
  }
  const diagnostics = await locator.page().evaluate(() => ({
    active: document.activeElement?.outerHTML,
    fields: [...document.querySelectorAll("input, select")].filter((field) => field.getClientRects().length > 0)
      .map((field) => ({ id: field.id, disabled: field.disabled })),
    alerts: [...document.querySelectorAll('[role="alert"]')].map((alert) => alert.textContent),
  }));
  throw new Error(`Required control cannot be reached using Tab: ${JSON.stringify(diagnostics)}`);
}

async function keyboardFill(page, locator, value) {
  await keyboardReach(locator);
  await locator.press("ControlOrMeta+A");
  await page.keyboard.insertText(value);
}

async function fillRepairForm(page, state, providerBaseUrl) {
  const sections = page.getByRole("navigation", { name: "Settings sections", exact: true });
  await keyboardActivate(sections.getByRole("button", { name: "Runtime", exact: true }));
  // Keep the full repair candidate invalid while the composite model fields are completed.
  await keyboardFill(page, page.getByLabel("Maximum iterations", { exact: true }), "1");
  await keyboardActivate(sections.getByRole("button", { name: "Models", exact: true }));
  const provider = page.locator("#settings-models-providers-openai-local");
  await expect(provider).toBeVisible();
  await keyboardFill(page, page.locator("#settings-models-providers-openai-local-base_url"), providerBaseUrl);
  const modelList = page.locator("#settings-models-providers-openai-local-models");
  if (await modelList.locator("textarea").count() === 0) {
    await keyboardActivate(modelList.getByRole("button", { name: "Add item", exact: true }));
  }
  await keyboardFill(page, modelList.locator("textarea").first(), "small-model");
  const routeDefault = page.locator("#settings-models-routes-default");
  if (state === "invalid" && await routeDefault.count() === 0) {
    await keyboardActivate(page.getByRole("button", { name: "Add route", exact: true }));
  }
  for (const input of await page.locator('input[id^="settings-models-routes-"][id$="-model"]').all()) {
    await keyboardFill(page, input, "small-model");
  }
  const action = page.locator("#settings-models-providers-openai-local-api_key-action");
  await keyboardReach(action);
  await action.press("Home");
  await action.press("ArrowDown");
  await keyboardFill(page, page.locator("#settings-models-providers-openai-local-api_key-value"),
    `startup-secret-${state}-303`,
  );
  await keyboardActivate(sections.getByRole("button", { name: "Runtime", exact: true }));
  await keyboardFill(page, page.getByLabel("Maximum iterations", { exact: true }), "64");
}

async function saveRepair(page, expectedStatus) {
  const responsePromise = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config/repair") && response.request().method() === "POST"
  ));
  const retry = page.getByRole("button", { name: "Retry save", exact: true });
  const lastField = page.getByLabel("Maximum iterations", { exact: true });
  if (await lastField.isVisible()) await lastField.press("Tab");
  else await keyboardActivate(retry);
  const response = await responsePromise;
  assert.equal(response.status(), expectedStatus);
  return response.json();
}

async function conversationAfterRepair(page, details, beforeSocket) {
  await page.setViewportSize({ width: 1440, height: 900 });
  assert.equal(
    await page.evaluate(() => window.__startupSocketBefore === window.__startupSocket),
    true,
    "Configuration repair replaced the browser WebSocket",
  );
  const serviceJson = JSON.parse(await readFile(join(details.home_root, "service.json"), "utf8"));
  assert.equal(serviceJson.pid, details.pid, "Configuration repair restarted the service PID");
  assert.equal(serviceJson.port, details.port, "Configuration repair changed the service port");
  assert.equal(beforeSocket, true);
  assert.deepEqual(await page.evaluate(() => ({
    created: window.__startupSocketCount,
    closed: window.__startupSocketCloseCount,
    ready: window.__startupSocket.readyState,
  })), { created: 1, closed: 0, ready: 1 });
  const firstActiveService = await fetchJson(page, "/service");
  assert.equal(firstActiveService.body.active_workspace_count, 2, "The available Project and default Chat workspace were not activated before Session open");

  await keyboardActivate(page.getByRole("navigation", { name: "Settings sections", exact: true })
    .getByRole("button", { name: "Runtime", exact: true }));
  await keyboardActivate(page.getByRole("main").getByRole("link", { name: "Status", exact: true }));
  await expect(page.locator("#status-heading")).toBeVisible();
  await expect(page).toHaveURL(/\/status$/);
  await keyboardActivate(page.locator("#app-sidebar").getByRole("link", { name: "Projects", exact: true }));
  await page.getByRole("main").getByRole("heading", { name: "Projects", exact: true }).waitFor();
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  const projectResponse = await fetchJson(page, "/projects");
  const project = projectResponse.body.projects.find((item) => item.project_id === details.project_id);
  assert.ok(project);
  assert.equal(project.available, true);
  assert.equal(project.schedule_state, "awaiting_resume");
  assert.equal(project.saved_jobs.length, 1);
  assert.equal(project.schedule_status?.admitted ?? false, false);
  await expect(page.getByText("Schedule paused for review", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Resume schedule", exact: true })).toBeEnabled();

  const item = page.locator(`#project-${details.project_id}`);
  await keyboardActivate(item.getByRole("link", { name: "Open sessions", exact: true }));
  await page.getByRole("heading", { name: "persistent-project", exact: true }).waitFor();
  await keyboardActivate(page.getByRole("button", { name: "New session", exact: true }));
  await keyboardFill(page, page.getByLabel("Message input", { exact: true }),
    `startup repair conversation ${details.state}`,
  );
  const prompt = `startup repair conversation ${details.state}`;
  await page.getByLabel("Message input", { exact: true }).press("Enter");
  try {
    await expect(page.getByRole("heading", { name: "Fixture response.", exact: true })).toBeVisible({ timeout: 30000 });
  } catch (error) {
    console.error(`Conversation diagnostics: ${JSON.stringify({
      state: details.state,
      text: await page.locator("body").innerText(),
      events: await page.evaluate(() => window.__startupMessages.slice(-20)),
      config: (await readConfig(page)).body,
    })}`);
    throw error;
  }
  await expect.poll(async () => page.evaluate((text) => {
    const messages = window.__startupMessages ?? [];
    const accepted = messages.find((event) => event.type === "input.accepted" && event.payload?.text === text);
    return accepted !== undefined && messages.some(
      (event) => event.type === "run.completed" && event.run_id === accepted.run_id,
    );
  }, prompt), { timeout: 30000 }).toBe(true);
  const afterService = await fetchJson(page, "/service");
  assert.equal(afterService.body.active_workspace_count, 3);
}

async function cleanupState(context, harness, root, failure) {
  const cleanup = await Promise.allSettled([context?.close()]);
  try {
    await harness?.shutdown();
  } catch (error) {
    cleanup.push({ status: "rejected", reason: error });
  }
  try {
    await rm(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  } catch (error) {
    cleanup.push({ status: "rejected", reason: error });
  }
  const errors = cleanup.filter((result) => result.status === "rejected").map((result) => result.reason);
  if (errors.length > 0) throw new AggregateError(failure ? [failure, ...errors] : errors, "Startup acceptance or cleanup failed");
}

async function runState(browser, state) {
  const root = await mkdtemp(join(tmpdir(), `omni-startup-${state}-`));
  let harness;
  let context;
  let failure;
  try {
    harness = await startHarness(state, root);
    let details = harness.details;
    assert.equal(details.state, state);
    assert.equal(details.initial_service.active_workspace_count, 0);
    assert.ok(details.cold_launch_url.startsWith(`${details.url}/#ticket=`), "Cold production Web startup was bypassed");
    console.log(`${state}: ${details.url} pid=${details.pid} port=${details.port}`);

    const bareRoot = await mkdtemp(join(tmpdir(), `omni-bare-${state}-`));
    try {
      const source = join(details.home_root, "config.toml");
      if (state !== "missing") {
        await mkdir(join(bareRoot, ".omni"));
        await writeFile(join(bareRoot, ".omni", "config.toml"), await readFile(source));
      }
      const bare = startupCli(bareRoot, []);
      assert.equal(bare.status, 2, `Bare CLI unexpectedly started for ${state}: ${bare.stdout}`);
      const bareOutput = `${bare.stdout ?? ""}\n${bare.stderr ?? ""}`;
      const expectedCode = state === "missing" ? "config_missing" : state === "invalid" ? "config_invalid" : "config_parse_error";
      assert.match(bareOutput, new RegExp(expectedCode));
      assert.equal(bareOutput.includes(details.malformed_secret), false);
      const stopped = startupCli(bareRoot, ["service", "stop"]);
      assert.equal(stopped.status, 0, `Isolated bare CLI service did not stop: ${stopped.stdout}`);
    } finally {
      await rm(bareRoot, { recursive: true, force: true });
    }

    const web = startupCli(details.user_home_root, ["web"]);
    assert.equal(web.status, 0, `omni web failed for ${state}: ${web.stderr}`);
    const webOutput = `${web.stdout ?? ""}\n${web.stderr ?? ""}`;
    const launchUrl = webOutput.match(/http:\/\/127\.0\.0\.1:\d+\/#ticket=[\w-]+/)?.[0];
    assert.ok(launchUrl?.startsWith(`${details.url}/#ticket=`), "omni web did not reuse the isolated service");
    const sharedBare = startupCli(details.user_home_root, []);
    assert.equal(sharedBare.status, 2, "Existing Web service swallowed bare CLI startup errors");
    const sharedBareOutput = `${sharedBare.stdout ?? ""}\n${sharedBare.stderr ?? ""}`;
    assert.match(sharedBareOutput, new RegExp(state === "missing" ? "config_missing" : state === "invalid" ? "config_invalid" : "config_parse_error"));
    assert.equal(sharedBareOutput.includes(details.malformed_secret), false);
    if (state === "missing") await rm(join(details.home_root, "config.toml"));

    context = await browser.newContext({ locale: "zh-CN", reducedMotion: "reduce" });
    await context.addInitScript(() => {
      const OriginalWebSocket = window.WebSocket;
      window.__startupMessages = [];
      window.__startupSocket = null;
      window.__startupSocketCount = 0;
      window.__startupSocketCloseCount = 0;
      window.__startupControlCredential = null;
      window.WebSocket = class extends OriginalWebSocket {
        constructor(...args) {
          super(...args);
          window.__startupControlCredential = Array.isArray(args[1]) ? args[1][1] : null;
          window.__startupSocket = this;
          window.__startupSocketCount += 1;
          this.addEventListener("close", () => { window.__startupSocketCloseCount += 1; });
          this.addEventListener("message", (event) => {
            try {
              window.__startupMessages.push(JSON.parse(event.data));
            } catch {
              // Browser transport may carry non-JSON frames.
            }
          });
        }
      };
    });
    let page = await context.newPage();
    const configBodies = [];
    page.on("response", (response) => {
      if (response.url().includes("/api/v1/config")) {
        configBodies.push(response.text().catch(() => ""));
      }
    });
    const documentResponse = await page.goto(launchUrl);
    assert.equal(documentResponse.status(), 200);
    assert.match(documentResponse.headers()["content-security-policy"] ?? "", /default-src 'self'/);
    await expect(page.getByRole("heading", { name: "Omni", exact: true }).last()).toBeVisible();
    await expect(page.getByRole("button", { name: /Send|发送/, exact: true })).toBeDisabled();
    page = await setupDraftAcceptance(context, page, details, state, root);
    if (state === "missing") {
      const oldService = (await fetchJson(page, "/service")).body.service_instance_id;
      await page.close();
      details = await harness.restart();
      page = await context.newPage();
      await page.goto(details.cold_launch_url);
      await expect(page.getByRole("textbox", { name: /Message input|消息输入/, exact: true })).toHaveValue("");
      assert.equal(new URL(page.url()).pathname, "/");
      assert.notEqual((await fetchJson(page, "/service")).body.service_instance_id, oldService);
      assert.equal(await page.evaluate(() => window.localStorage.getItem("omni.browser-recovery")), null);
    }
    await expect(page.getByRole("button", { name: /Send|发送/, exact: true })).toBeDisabled();
    await page.getByRole("link", { name: /Configure models|配置模型/, exact: true }).click();
    try {
      await page.getByRole("heading", { name: /Settings|设置/, exact: true }).waitFor();
    } catch (error) {
      await page.screenshot({ path: resolve(output, `${state}-startup-timeout.png`) });
      console.error(`Startup page timeout for ${state}: url=${page.url()} text=${(await page.locator("body").innerText()).slice(0, 1200)}`);
      throw error;
    }
    await expect.poll(async () => page.evaluate(() => window.__startupControlCredential), {
      timeout: 15000,
    }).not.toBeNull();
    const initial = await readConfig(page);
    assert.equal(initial.status, 200);
    assert.equal(initial.body.configuration.state, state);
    assert.equal(initial.body.application.status, "pending-repair");
    assert.equal(initial.body.configuration.repair_required, true);
    assert.equal(JSON.stringify(initial.body).includes(details.malformed_secret), false);
    assert.equal((await page.locator("body").innerText()).includes(details.malformed_secret), false);
    await assertAdmissionClosed(page, details.project_id);
    for (const path of ["/status", "/settings"]) {
      await page.goto(`${details.url}${path}`);
      await expect(page).toHaveURL(/\/settings$/);
      await expect(page.getByRole("heading", { name: /Settings|设置/, exact: true })).toBeVisible();
    }
    await expect.poll(() => page.evaluate(() => window.__startupSocket?.readyState)).toBe(1);
    await screenshotStates(page, state, "repair-required");
    await page.evaluate(() => { window.__startupSocketBefore = window.__startupSocket; });

    await fillRepairForm(page, state, details.provider_base_url);
    const configPath = join(details.home_root, "config.toml");
    const originalBytes = await readFile(configPath).catch(() => new Uint8Array());
    if (state === "malformed") {
      await mkdir(details.backup_blocker);
      const failed = await saveRepair(page, 500);
      assert.equal(failed.code, "persistence_error");
      assert.deepEqual(await readFile(configPath), originalBytes);
      await keyboardActivate(page.getByRole("navigation", { name: "Settings sections", exact: true })
        .getByRole("button", { name: "Models", exact: true }));
      await expect(page.locator("#settings-models-providers-openai-local-api_key-value")).toHaveValue("startup-secret-malformed-303");
      await expect(page.locator("#settings-models-providers-openai-local-base_url")).toHaveValue(details.provider_base_url);
      assert.equal((await Promise.all(configBodies)).some((body) => body.includes("startup-secret-malformed-303")), false);
      // Keep persistence blocked while Tab traverses fields that can auto-save.
      await keyboardReach(page.getByRole("button", { name: "Retry save", exact: true }));
      await rm(details.backup_blocker, { recursive: true, force: true });
    }
    const repaired = await saveRepair(page, 200);
    if (state === "malformed") assert.match(repaired.backup_id, /^sha256:/);
    assert.equal(repaired.application.status, "restart-required");
    assert.equal(repaired.application.active_revision, null);
    await assertAdmissionClosed(page, details.project_id);
    await expect(page.getByText("Saved; restart Omni to use these settings.", { exact: true }).first()).toBeVisible();
    assert.equal(await page.evaluate(() => window.__startupSocketBefore === window.__startupSocket), true);
    const oldPid = details.pid;
    await context.close();
    details = await harness.restart();
    assert.notEqual(details.pid, oldPid);
    context = await browser.newContext({ locale: "en", reducedMotion: "reduce" });
    await context.addInitScript(() => {
      window.__startupMessages = [];
      window.__startupSocketCount = 0;
      window.__startupSocketCloseCount = 0;
      const OriginalWebSocket = window.WebSocket;
      window.WebSocket = class extends OriginalWebSocket {
        constructor(...args) {
          super(...args);
          window.__startupControlCredential = Array.isArray(args[1]) ? args[1][1] : null;
          window.__startupSocket = this;
          window.__startupSocketCount += 1;
          this.addEventListener("close", () => { window.__startupSocketCloseCount += 1; });
          this.addEventListener("message", (event) => { window.__startupMessages.push(JSON.parse(event.data)); });
        }
      };
    });
    const restartedPage = await context.newPage();
    await restartedPage.goto(details.cold_launch_url);
    await expect(restartedPage.getByRole("heading", { name: "Omni", exact: true }).last()).toBeVisible();
    await expect(restartedPage.getByRole("textbox", { name: /Message input|消息输入/, exact: true })).toBeEnabled();
    await restartedPage.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
    await keyboardActivate(restartedPage.getByRole("navigation", { name: "Settings sections", exact: true })
      .getByRole("button", { name: "Runtime", exact: true }));
    await expect(restartedPage.getByLabel("Maximum iterations", { exact: true })).toBeEnabled();
    page = restartedPage;
    await page.evaluate(() => { window.__startupSocketBefore = window.__startupSocket; });
    const active = await waitForActiveConfig(page);
    assert.equal(active.configuration.state, "active");
    assert.equal(JSON.stringify(active).includes(details.malformed_secret), false);
    await expect(page.getByText("Active generation", { exact: true })).toBeVisible({ timeout: 15000 });
    const savedBytes = await readFile(configPath);
    assert.equal(savedBytes.toString("utf8").includes(details.malformed_secret), false);
    if (state === "malformed") {
      const entries = await readdir(details.home_root, { withFileTypes: true });
      const backups = entries.filter((entry) => entry.name.startsWith("config.toml.backup.") && entry.isFile());
      assert.ok(backups.length > 0, "Malformed configuration backup was not published");
      assert.ok(
        await Promise.any(backups.map(async (entry) => {
          const bytes = await readFile(join(details.home_root, entry.name));
          if (!bytes.equals(originalBytes)) throw new Error("different backup");
          return true;
        })),
        "Malformed backup did not preserve exact original bytes",
      );
      const backupFile = backups[0].name;
      const backupResponse = await page.request.get(`${details.url}/${backupFile}`);
      assert.equal(backupResponse.status(), 404, "Private backup was exposed by static serving");
    }
    await screenshotStates(page, state, "active");
    await conversationAfterRepair(page, details, true);
    const observed = [
      ...(await Promise.all(configBodies)),
      await page.evaluate(() => JSON.stringify({
        events: window.__startupMessages,
        local: { ...window.localStorage },
        session: { ...window.sessionStorage },
      })),
      await page.locator("body").innerText(),
    ].join("\n");
    assert.equal(observed.includes(`startup-secret-${state}-303`), false);
    assert.equal(observed.includes(details.malformed_secret), false);
    await writeFile(configPath, "[broken-after-active");
    await page.goto(`${details.url}/status`);
    await expect(page.locator("#status-heading")).toBeVisible();
    await expect(page).toHaveURL(/\/status$/);
    await expect.poll(() => page.evaluate(() => window.__startupSocket?.readyState)).toBe(1);
    const oldActive = await readConfig(page);
    assert.equal(oldActive.status, 200, `Authenticated saved configuration read: ${JSON.stringify(oldActive.body)}`);
    assert.equal(oldActive.body.configuration.state, "malformed");
    assert.equal(oldActive.body.application.active_revision, active.revision);
    assert.equal(oldActive.body.application.status, "pending-repair");
    await writeFile(configPath, savedBytes);
    await context.close();
    context = null;
    await harness.shutdown();
    harness = null;
    await rm(root, { recursive: true, force: true });
    return details.url;
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    await cleanupState(context, harness, root, failure);
  }
}

let browser;
const urls = [];
try {
  await mkdir(output, { recursive: true });
  browser = await chromium.launch({
    channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge",
  });
  for (const state of states) urls.push(await runState(browser, state));
  console.log(
    `Config startup production E2E: missing, semantic-invalid, malformed TOML, backup failure/exact bytes, `
    + `bare CLI errors, active workspace admission, awaiting_resume Schedule state, repair preserves PID/WebSocket and explicit restart activates settings, `
    + `keyboard fixture conversation, cold production Web startup, CSP, and en/zh-CN light/dark 390/768/1024/1440 screenshots passed; URLs=${urls.join(",")}`,
  );
} finally {
  await browser?.close();
}
