import assert from "node:assert/strict";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { URL } from "node:url";
import { chromium, expect } from "@playwright/test";

if (process.platform !== "win32") {
  console.error("MyClaw requires Windows.");
  process.exit(1);
}

const baseUrl = process.env.MYCLAW_E2E_URL;
const ticket = process.env.MYCLAW_E2E_TICKET;
const secondTicket = process.env.MYCLAW_E2E_SECOND_TICKET;
const workspace = process.env.MYCLAW_E2E_WORKSPACE;
const output = process.env.MYCLAW_E2E_OUTPUT;
const expectedInstance = process.env.MYCLAW_E2E_INSTANCE;
const observationPath = process.env.MYCLAW_PROVIDER_OBSERVATION_PATH;
const cliReadyPath = process.env.MYCLAW_CLI_READY;
const browserReadyPath = process.env.MYCLAW_JOINT_BROWSER_READY;
const cliForegroundReadyPath = process.env.MYCLAW_CLI_FOREGROUND_READY;
const cliRemovalDonePath = process.env.MYCLAW_CLI_REMOVAL_DONE;
const cliSettingsStartPath = process.env.MYCLAW_CLI_SETTINGS_START;
const cliSettingsReadyPath = process.env.MYCLAW_CLI_SETTINGS_READY;
const cliSettingsDonePath = process.env.MYCLAW_CLI_SETTINGS_DONE;
const cliDonePath = process.env.MYCLAW_CLI_DONE;
const settingsReleasePath = process.env.MYCLAW_CLI_SETTINGS_RELEASE;
let csrfToken = null;
let webControlCredential = null;
assert.ok(
  baseUrl && ticket && secondTicket && workspace && output && expectedInstance && observationPath
    && cliReadyPath && browserReadyPath && cliForegroundReadyPath && cliRemovalDonePath
    && cliSettingsStartPath && cliSettingsReadyPath && cliSettingsDonePath && cliDonePath
    && settingsReleasePath,
);

async function waitForJson(path, timeout = 90_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      const value = JSON.parse(await readFile(path, "utf8"));
      if (value?.status === "failed") throw new Error(`Scenario failed: ${JSON.stringify(value)}`);
      return value;
    } catch (error) {
      if (error.code !== "ENOENT" && error.name !== "SyntaxError") throw error;
      await delay(50);
    }
  }
  throw new Error(`Timed out waiting for evidence file: ${path}`);
}

async function waitForObservations(path, prompt, count) {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    try {
      const records = (await readFile(path, "utf8"))
        .split("\n")
        .filter((line) => line.trim().length > 0)
        .map((line) => JSON.parse(line));
      const sessions = new Set(records.filter((record) => (
        typeof record.prompt === "string" && record.prompt.includes(prompt)
        && Array.isArray(record.tools) && record.tools.length > 0
      )).map((record) => record.prompt.match(/Session ID: ([^\n]+)/)?.[1]).filter(Boolean));
      if (sessions.size >= count) return records;
    } catch (error) {
      if (error.code !== "ENOENT" && error.name !== "SyntaxError") throw error;
    }
    await delay(50);
  }
  throw new Error(`Timed out waiting for ${count} provider observations for ${prompt}`);
}

async function waitForPersistedConfirmationResult(workspacePath) {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try {
      const sessionDirectory = join(workspacePath, ".myclaw", "schedule-sessions");
      const names = await readdir(sessionDirectory);
      for (const name of names) {
        if (!name.endsWith(".jsonl")) continue;
        const records = (await readFile(join(sessionDirectory, name), "utf8"))
          .split("\n")
          .filter((line) => line.trim().length > 0)
          .map((line) => JSON.parse(line));
        const result = records.find((record) => (
          record?.role === "tool"
          && record.tool_call_id === "call-confirmation"
          && record.status === "success"
        ));
        if (result !== undefined) return result;
      }
    } catch (error) {
      if (error.code !== "ENOENT" && error.name !== "SyntaxError") throw error;
    }
    await delay(50);
  }
  throw new Error("Timed out waiting for the persisted background confirmation tool result");
}

async function requestJson(request, method, path, data) {
  const response = await request[method](path, {
    headers: {
      Origin: baseUrl,
      ...(method !== "get" ? { "X-MyClaw-CSRF": csrfToken } : {}),
      ...(webControlCredential === null ? {} : { "X-MyClaw-Control": webControlCredential }),
    },
    data,
  });
  const body = await response.json();
  assert.equal(response.status(), 200, `${method.toUpperCase()} ${path}: ${JSON.stringify(body)}`);
  return body;
}

function atTimePayload(requestId, message, title) {
  return {
    request_id: requestId,
    message,
    title,
    at_time: new Date(Date.now() + 900).toISOString(),
  };
}

function everyPayload(requestId, message, title) {
  return {
    request_id: requestId,
    message,
    title,
    every_seconds: 3600,
  };
}

async function scheduleJob(request, workspaceId, payload) {
  return requestJson(
    request,
    "post",
    `${baseUrl}/api/v1/workspaces/${workspaceId}/schedule/jobs`,
    payload,
  );
}

async function listSchedule(request, workspaceId) {
  return requestJson(request, "get", `${baseUrl}/api/v1/workspaces/${workspaceId}/schedule/jobs`);
}

async function resumeSchedule(request, projectId, jobIds) {
  return requestJson(
    request,
    "post",
    `${baseUrl}/api/v1/projects/${projectId}/schedule-resume`,
    { request_id: `installed-joint-resume-${Date.now()}`, job_ids: jobIds },
  );
}

async function ensureScheduleAdmitted(request, projectId, workspaceId) {
  const listing = await listSchedule(request, workspaceId);
  if (listing.status.admitted !== true) {
    await resumeSchedule(request, projectId, listing.jobs.map((job) => job.job_id));
  }
  return listing;
}

async function configResponse(request) {
  return requestJson(request, "get", `${baseUrl}/api/v1/config`);
}

async function persistedTerminal(path, prompt, expected) {
  return expect.poll(async () => {
    try {
      const records = (await readFile(path, "utf8")).split("\n").filter(Boolean)
        .map((line) => JSON.parse(line));
      const index = records.findIndex((record) => record.role === "user" && record.content === prompt);
      if (index < 0) return null;
      const result = records.slice(index + 1).find((record) => record.role === "assistant");
      if (!result) return null;
      return result.error?.code === "turn_cancelled" ? "cancelled"
        : result.status === "completed" && result.error === null ? "completed" : "failed";
    } catch (error) {
      if (error.code !== "ENOENT" && error.name !== "SyntaxError") throw error;
      return null;
    }
  }, { timeout: 90_000 }).toBe(expected);
}

async function decide(page, token, requestId) {
  return page.evaluate(({ token, requestId }) => new Promise((resolve, reject) => {
    const socket = window.__myclawJointSocket;
    const timer = setTimeout(() => {
      socket.removeEventListener("message", listener);
      reject(new Error("Confirmation response timed out"));
    }, 30_000);
    function listener(event) {
      const result = JSON.parse(event.data);
      if (result.request_id !== requestId) return;
      clearTimeout(timer);
      socket.removeEventListener("message", listener);
      resolve(result);
    }
    socket.addEventListener("message", listener);
    socket.send(JSON.stringify({ request_id: requestId, type: "confirmation_decide",
      workspace_id: null, session_id: null, claim_version: null,
      payload: { token, decision: "approved" } }));
  }), { token, requestId });
}

const browser = await chromium.launch({
  channel: process.env.MYCLAW_E2E_BROWSER_CHANNEL ?? "msedge",
});
const context = await browser.newContext();
const secondContext = await browser.newContext();
const page = await context.newPage();
const watchdog = setTimeout(() => { void browser.close(); }, 180_000);
const browserErrors = [];
const network = [];
const consoleMessages = [];
function redact(text) {
  let result = text;
  for (const secret of [ticket, secondTicket, csrfToken, webControlCredential]) {
    if (typeof secret === "string" && secret.length > 0) result = result.replaceAll(secret, "[redacted]");
  }
  return result;
}

const observeWebSocket = () => {
  const OriginalWebSocket = window.WebSocket;
  window.__myclawJointEvents = [];
  window.WebSocket = class extends OriginalWebSocket {
    constructor(...args) {
      super(...args);
      window.__myclawJointSocket = this;
      this.addEventListener("message", (event) => {
        try {
          window.__myclawJointEvents.push(JSON.parse(event.data));
        } catch {
          // Non-JSON frames are outside the service event contract.
        }
      });
    }
  };
};
await context.addInitScript(observeWebSocket);
await secondContext.addInitScript(observeWebSocket);
page.on("pageerror", (error) => browserErrors.push(error.message));
page.on("console", (message) => consoleMessages.push({ type: message.type(), text: redact(message.text()) }));
page.on("response", (response) => network.push({ path: new URL(response.url()).pathname, status: response.status() }));
page.on("requestfailed", (request) => network.push({ path: new URL(request.url()).pathname, failure: redact(request.failure()?.errorText ?? "unknown") }));

try {
  await writeFile(join(workspace, "fixture.txt"), "joint user file\n", "utf8");
  const webClientResponse = page.waitForResponse((response) => (
    response.request().method() === "POST" && response.url().endsWith("/api/v1/clients")
  ));
  await page.goto(`${baseUrl}/#ticket=${encodeURIComponent(ticket)}`);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await expect(page.getByRole("status").first()).toHaveText(/^(Online|在线)$/);
  await expect(page.getByRole("status").first()).toBeVisible();
  const webClient = await webClientResponse;
  assert.equal(webClient.status(), 200);
  webControlCredential = (await webClient.json()).web_control_credential;
  assert.equal(typeof webControlCredential, "string");
  const serviceResponse = await context.request.get(`${baseUrl}/api/v1/service`);
  assert.equal(serviceResponse.status(), 200);
  const service = await serviceResponse.json();
  assert.equal(service.service_instance_id, expectedInstance);
  const sessionResponse = await context.request.get(`${baseUrl}/api/v1/web/session`);
  assert.equal(sessionResponse.status(), 200);
  csrfToken = (await sessionResponse.json()).csrf_token;
  assert.equal(typeof csrfToken, "string");

  await page.getByRole("navigation").getByRole("link", { name: /Projects|项目/, exact: true }).click();
  await page.getByRole("heading", { name: /^(Projects|项目)$/ }).waitFor();

  async function registerProject() {
    await page.getByRole("button", { name: /Add project|登记项目/ }).first().click();
    const dialog = page.getByRole("dialog");
    await dialog.getByLabel(/Absolute local path|本地绝对路径/).fill(workspace);
    const responsePromise = page.waitForResponse((response) => (
      response.request().method() === "POST" && response.url().endsWith("/api/v1/projects")
    ));
    await dialog.getByRole("button", { name: /Register project|登记项目/ }).click();
    const response = await responsePromise;
    assert.equal(response.status(), 200, await response.text());
    const registered = await response.json();
    await dialog.waitFor({ state: "hidden" });
    await page.getByRole("heading", { name: "workspace", exact: true }).waitFor();
    return registered;
  }

  const first = await registerProject();
  const firstProjectId = first.project_id;
  const firstWorkspaceId = first.workspace_id;
  const secondPage = await secondContext.newPage();
  secondPage.on("pageerror", (error) => browserErrors.push(error.message));
  const secondClientResponse = secondPage.waitForResponse((response) => (
    response.request().method() === "POST" && response.url().endsWith("/api/v1/clients")
  ));
  await secondPage.goto(`${baseUrl}/#ticket=${encodeURIComponent(secondTicket)}`);
  await expect(secondPage.getByRole("status").first()).toHaveText(/^(Online|在线)$/);
  const secondClient = await (await secondClientResponse).json();
  assert.notEqual(secondClient.client_id, (await webClient.json()).client_id);
  const background = await scheduleJob(
    context.request,
    firstWorkspaceId,
    atTimePayload(
      "installed-joint-background-confirmation",
      "confirmation",
      "Installed background confirmation",
    ),
  );
  const backgroundJobId = background.job.job_id;
  await ensureScheduleAdmitted(context.request, firstProjectId, firstWorkspaceId);
  await expect.poll(async () => {
    const events = await page.evaluate(() => window.__myclawJointEvents);
    return events.find((event) => (
      event.type === "confirmation.requested"
      && event.payload?.origin === "background"
      && event.payload?.job_id === backgroundJobId
    ));
  }, { timeout: 90_000 }).toBeTruthy();
  const backgroundConfirmation = await page.evaluate((jobId) => (
    window.__myclawJointEvents.find((event) => (
      event.type === "confirmation.requested"
      && event.payload?.origin === "background"
      && event.payload?.job_id === jobId
    ))
  ), backgroundJobId);
  assert.ok(backgroundConfirmation);
  const confirmation = backgroundConfirmation;
  const backgroundDialog = page.locator('[role="dialog"][data-confirmation-origin="background"]');
  await backgroundDialog.waitFor();
  const backgroundText = await backgroundDialog.innerText();
  assert.match(backgroundText, /read_file/);
  assert.match(backgroundText, /confirmation-outside\.txt/);
  assert.match(backgroundText, new RegExp(backgroundJobId));
  const confirmationButtons = backgroundDialog.getByRole("button");
  await expect(confirmationButtons).toHaveCount(3);
  const secondDialog = secondPage.locator('[role="dialog"][data-confirmation-origin="background"]');
  await secondDialog.waitFor();
  const decisions = await Promise.all([
    decide(page, confirmation.payload.token, "installed-confirmation-race-one"),
    decide(secondPage, confirmation.payload.token, "installed-confirmation-race-two"),
  ]);
  assert.equal(decisions.filter((result) => result.accepted === true).length, 1);
  assert.equal(decisions.filter((result) => result.code === "confirmation_resolved").length, 1);
  await backgroundDialog.waitFor({ state: "hidden" });
  await secondDialog.waitFor({ state: "hidden" });
  const persistedConfirmation = await waitForPersistedConfirmationResult(workspace);
  assert.match(persistedConfirmation.content, /confirmation fixture content/);
  const toolRecords = (await readFile(join(workspace, ".myclaw", "schedule-sessions", `schedule_${backgroundJobId}.jsonl`), "utf8"))
    .split("\n").filter(Boolean).map((line) => JSON.parse(line));
  assert.equal(toolRecords.filter((record) => record.role === "tool" && record.tool_call_id === "call-confirmation").length, 1);
  await secondPage.close();
  const backgroundEvidence = {
    source: "background",
    job_id: backgroundJobId,
    title: "Installed background confirmation",
    tool_name: confirmation.payload.request.tool_name,
    decision: "approved",
    tool_execution: "success",
    exact_first_decision: true,
    competing_clients: 2,
    accepted_decisions: 1,
    rejected_decisions: 1,
    persisted_tool_results: 1,
  };
  await writeFile(browserReadyPath, JSON.stringify({
    status: "ready",
    service_instance_id: expectedInstance,
    project_id: firstProjectId,
    workspace_id: firstWorkspaceId,
    background_confirmation: backgroundEvidence,
  }));
  const cliReady = await waitForJson(cliReadyPath);
  assert.equal(cliReady.status, "ready");
  assert.equal(cliReady.service_instance_id, expectedInstance);
  assert.equal(cliReady.workspace_id, firstWorkspaceId);
  const foregroundReady = await waitForJson(cliForegroundReadyPath);
  assert.equal(foregroundReady.status, "ready");
  const removalPrompt = "project removal barrier schedule";
  const removalJob = await scheduleJob(
    context.request,
    firstWorkspaceId,
    atTimePayload("installed-joint-project-removal", removalPrompt, "Project removal barrier Job"),
  );
  const removalJobId = removalJob.job.job_id;
  const removalSavedJob = await scheduleJob(
    context.request,
    firstWorkspaceId,
    everyPayload(
      "installed-joint-project-removal-saved",
      "saved Job must survive project removal",
      "Project removal preserved Job",
    ),
  );
  const removalSavedJobId = removalSavedJob.job.job_id;
  await waitForObservations(observationPath, removalPrompt, 1);
  const removableProject = page.getByRole("list", { name: /Projects|项目/ }).filter({
    has: page.getByRole("heading", { name: "workspace", exact: true }),
  });
  const projectButtons = removableProject.getByRole("button");
  await expect(projectButtons).toHaveCount(1);
  await projectButtons.click();
  const removalDialog = page.getByRole("dialog", { name: /Remove project registration\?|移除项目登记[?？]/ });
  await removalDialog.waitFor();
  const removalResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "DELETE" && response.url().includes("/api/v1/projects/")
  ));
  const removalButtons = removalDialog.getByRole("button");
  await expect(removalButtons).toHaveCount(3);
  await removalButtons.nth(2).click();
  const removalResponse = await removalResponsePromise;
  assert.equal(removalResponse.status(), 200, await removalResponse.text());
  await removalDialog.waitFor({ state: "hidden" });
  await page.getByText(/Project registration removed\. The directory and saved work remain on disk\.|项目登记已移除，目录和已保存的工作仍保留在磁盘上。/).waitFor();
  await removableProject.waitFor({ state: "detached" });
  const cliRemoval = await waitForJson(cliRemovalDonePath);
  assert.equal(cliRemoval.claim_released, true);
  assert.equal(cliRemoval.terminal_state, "cancelled");
  assert.equal(cliRemoval.notification_received, true);
  await persistedTerminal(join(workspace, ".myclaw", "schedule-sessions", `schedule_${removalJobId}.jsonl`), removalPrompt, "cancelled");
  assert.equal(await readFile(join(workspace, "fixture.txt"), "utf8"), "joint user file\n");
  const savedSchedule = JSON.parse(await readFile(join(workspace, ".myclaw", "schedule.json"), "utf8"));
  assert.match(JSON.stringify(savedSchedule), /Project removal preserved Job/);
  assert.match(JSON.stringify(savedSchedule), new RegExp(removalSavedJobId));

  const second = await registerProject();
  const secondProjectId = second.project_id;
  const secondWorkspaceId = second.workspace_id;
  await writeFile(cliSettingsStartPath, "start\n", "utf8");
  const settingsReady = await waitForJson(cliSettingsReadyPath);
  assert.equal(settingsReady.service_instance_id, expectedInstance);
  assert.equal(settingsReady.workspace_id, secondWorkspaceId);
  const settingsPrompt = "settings generation barrier";
  const settingsJob = await scheduleJob(
    context.request,
    secondWorkspaceId,
    atTimePayload("installed-joint-settings-generation", settingsPrompt, "Settings generation Job"),
  );
  const settingsJobId = settingsJob.job.job_id;
  const settingsListing = await ensureScheduleAdmitted(context.request, secondProjectId, secondWorkspaceId);
  await waitForObservations(observationPath, settingsPrompt, 2);
  await page.getByRole("navigation").getByRole("link", { name: /Settings|设置/, exact: true }).click();
  await page.getByRole("heading", { name: /Settings|设置/ }).waitFor();
  const iterations = page.getByLabel(/Maximum iterations|最大迭代次数/);
  await expect(iterations).toBeEnabled();
  await iterations.fill("65");
  await page.locator('[id="settings-models-routes-default-model"]').fill("installed-new-model");
  const saveResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "PATCH" && response.url().endsWith("/api/v1/config")
  ));
  await page.getByRole("button", { name: /Save settings|保存设置/ }).click();
  const saveResponse = await saveResponsePromise;
  assert.equal(saveResponse.status(), 200, await saveResponse.text());
  const pending = await saveResponse.json();
  assert.equal(pending.application.status, "pending");
  assert.notEqual(pending.application.pending_revision, null);
  assert.notEqual(pending.application.pending_revision, pending.application.active_revision);
  assert.equal((await listSchedule(context.request, secondWorkspaceId)).status.active_job_count, 1);
  await writeFile(settingsReleasePath, "release\n", "utf8");
  await expect.poll(async () => (await configResponse(context.request)).application.status, {
    timeout: 90_000,
  }).toBe("active");
  const active = await configResponse(context.request);
  assert.equal(active.application.active_revision, pending.application.pending_revision);
  assert.equal(active.application.pending_revision, null);
  await persistedTerminal(join(workspace, ".myclaw", "schedule-sessions", `schedule_${settingsJobId}.jsonl`), settingsPrompt, "completed");
  const cliSettings = await waitForJson(cliSettingsDonePath);
  const cliDone = await waitForJson(cliDonePath);
  assert.equal(cliSettings.foreground_terminal, true);
  assert.equal(cliSettings.service_instance_id, expectedInstance);
  assert.equal(cliSettings.service_pid, cliReady.service_pid);
  assert.equal(cliSettings.workspace_id, secondWorkspaceId);
  assert.equal(cliDone.status, "passed");
  assert.equal(cliDone.settings_generation_completed, true);
  assert.equal(cliDone.project_removal_terminal, true);
  assert.equal(cliDone.claim_released, true);
  assert.equal(cliSettings.new_generation_model, "installed-new-model");
  assert.equal((await context.request.get(`${baseUrl}/api/v1/service`)).status(), 200);
  assert.equal(await readFile(join(workspace, "fixture.txt"), "utf8"), "joint user file\n");
  assert.equal(settingsListing.status.admitted, false);
  await writeFile(join(output, "joint.json"), `${JSON.stringify({
    marker: "INSTALLED_JOINT_E2E_OK",
    service_instance_id: expectedInstance,
    service_pid: cliReady.service_pid,
    same_service_instance: cliSettings.service_instance_id === expectedInstance,
    same_service_pid: cliSettings.service_pid === cliReady.service_pid,
    background_confirmation: backgroundEvidence,
    project_removal: {
      project_id: firstProjectId,
      workspace_id: firstWorkspaceId,
      removal_job_id: removalJobId,
      saved_job_id: removalSavedJobId,
      foreground_terminal: cliRemoval.terminal_state,
      claim_released: cliRemoval.claim_released,
      user_file_preserved: true,
      saved_jobs_preserved: true,
      foreground_cancelled_persisted: true,
      schedule_cancelled_persisted: true,
      cli_notification_received: true,
    },
    settings_generation: {
      project_id: secondProjectId,
      workspace_id: secondWorkspaceId,
      settings_job_id: settingsJobId,
      pending_revision: pending.application.pending_revision,
      active_revision: active.application.active_revision,
      final_status: "active",
      foreground_and_schedule_observed: true,
      foreground_completed_persisted: true,
      schedule_completed_persisted: true,
      new_generation_model: cliSettings.new_generation_model,
    },
  }, null, 2)}\n`);
  assert.deepEqual(browserErrors, []);
  console.log(JSON.stringify({ marker: "INSTALLED_JOINT_E2E_OK", background: backgroundEvidence }));
} catch (error) {
  const safeMessage = redact(`${error.name}: ${error.message}`);
  await page.screenshot({ path: join(output, "joint-failure.png"), fullPage: true }).catch(() => {});
  const statusText = await page.getByRole("status").allTextContents().catch(() => []);
  await writeFile(join(output, "joint-failure.json"), JSON.stringify({
    status: "failed",
    error: safeMessage,
    status_roles: statusText.map(redact),
    page_errors: browserErrors.map(redact),
    console: consoleMessages,
    network,
  })).catch(() => {});
  throw new Error(safeMessage);
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
