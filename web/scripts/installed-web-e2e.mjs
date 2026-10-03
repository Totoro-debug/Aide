import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { chromium, expect } from "@playwright/test";
import { URL } from "node:url";

if (process.platform !== "win32") {
  console.error("MyClaw requires Windows.");
  process.exit(1);
}

const baseUrl = process.env.MYCLAW_E2E_URL;
const ticket = process.env.MYCLAW_E2E_TICKET;
const workspace = process.env.MYCLAW_E2E_WORKSPACE;
const output = process.env.MYCLAW_E2E_OUTPUT;
const prompt = "installed package conversation\nstreaming markdown";
const crossClientReadyPath = process.env.MYCLAW_CROSS_CLIENT_READY;
const crossClientCliReadyPath = process.env.MYCLAW_CROSS_CLIENT_CLI_READY;
const crossClientCliDonePath = process.env.MYCLAW_CROSS_CLIENT_CLI_DONE;
const crossClientPrivateMarker = process.env.MYCLAW_CLI_PRIVATE_MARKER;
const observationPath = process.env.MYCLAW_PROVIDER_OBSERVATION_PATH;
const concurrencyReleasePath = process.env.MYCLAW_CONCURRENCY_RELEASE;
assert.ok(baseUrl && ticket && workspace);

async function waitForJson(path, timeout = 90_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      return JSON.parse(await readFile(path, "utf8"));
    } catch (error) {
      if (error.code !== "ENOENT" && error.name !== "SyntaxError") throw error;
      await delay(50);
    }
  }
  throw new Error(`Timed out waiting for evidence file: ${path}`);
}

const browser = await chromium.launch({
  channel: process.env.MYCLAW_E2E_BROWSER_CHANNEL ?? "msedge",
});
const context = await browser.newContext();
const page = await context.newPage();
const watchdog = setTimeout(() => { void browser.close(); }, 90000);
const errors = [];
let websocketObserved = false;
const events = [];
let stage = "assets";
page.on("pageerror", (error) => errors.push(error.message));
page.on("websocket", (socket) => {
  if (socket.url().includes("/api/v1/events")) {
    websocketObserved = true;
    socket.on("framereceived", ({ payload }) => {
      if (typeof payload === "string") events.push(JSON.parse(payload));
    });
  }
});

async function createClaimedDraft(button) {
  const responses = Promise.all([
    page.waitForResponse((response) => (
      response.request().method() === "POST" && response.url().endsWith("/sessions")
    )),
    page.waitForResponse((response) => (
      response.request().method() === "POST" && response.url().endsWith("/claim")
    )),
  ]);
  await button.click();
  const [createdResponse, claimedResponse] = await responses;
  assert.equal(createdResponse.status(), 200, "Installed draft creation failed");
  assert.equal(claimedResponse.status(), 200, "Installed draft Claim failed");
  const created = await createdResponse.json();
  const claimed = await claimedResponse.json();
  assert.equal(claimed.claim.session_id, created.session_id);
  await expect(button).toBeEnabled();
  return created;
}

try {
  const documentResponse = await context.request.get(baseUrl);
  assert.equal(documentResponse.status(), 200, "Installed package did not serve index.html");
  assert.match(
    documentResponse.headers()["cache-control"] ?? "",
    /no-store/,
    "Installed package document did not disable HTML caching",
  );
  assert.match(
    documentResponse.headers()["content-security-policy"] ?? "",
    /default-src 'self'/,
    "Installed package document did not include CSP",
  );
  const html = await documentResponse.text();
  const javascriptAsset = html.match(/src="(\/assets\/[^"']+\.js)"/)?.[1];
  const stylesheetAsset = html.match(/href="(\/assets\/[^"']+\.css)"/)?.[1];
  assert.ok(javascriptAsset && stylesheetAsset, "Installed package HTML did not reference JS/CSS");
  for (const [asset, contentType] of [[javascriptAsset, /javascript/], [stylesheetAsset, /text\/css/], ["/favicon.svg", /image\/svg\+xml/]]) {
    const response = await context.request.get(`${baseUrl}${asset}`);
    assert.equal(response.status(), 200, `Installed asset did not load: ${asset}`);
    assert.match(response.headers()["content-type"] ?? "", contentType, `Wrong MIME for ${asset}`);
    assert.match(response.headers()["cache-control"] ?? "", /immutable/, `Wrong cache policy for ${asset}`);
  }
  for (const missing of ["/assets/missing.js", "/missing.css", "/api/v1/missing"]) {
    const response = await context.request.get(`${baseUrl}${missing}`);
    assert.equal(response.status(), 404, `Missing resource became a SPA document: ${missing}`);
  }

  stage = "authentication";
  await page.goto(`${baseUrl}/#ticket=${encodeURIComponent(ticket)}`);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await expect(page.getByRole("status").first()).toHaveText(/^(Online|在线)$/);
  assert.match(page.url(), /\/status$/);
  assert.ok(websocketObserved, "Installed Web app did not open its authenticated WebSocket");
  const serviceResponse = await context.request.get(`${baseUrl}/api/v1/service`);
  assert.equal(serviceResponse.status(), 200, "Installed Web app API did not respond");
  const service = await serviceResponse.json();
  assert.equal(service.service_instance_id, process.env.MYCLAW_E2E_INSTANCE);

  await page.goto(`${baseUrl}/status`);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await expect(page.getByRole("status").first()).toHaveText(/^(Online|在线)$/);

  await page.getByRole("navigation").getByRole("link", { name: /Settings|设置/, exact: true }).click();
  await page.getByRole("heading", { name: /Settings|设置/ }).waitFor();
  await page.getByRole("heading", { name: /Runtime|运行时/ }).waitFor();
  await page.screenshot({ path: join(output, "settings.png"), fullPage: true });
  await page.goto(`${baseUrl}/settings`);
  await page.getByRole("heading", { name: /Runtime|运行时/ }).waitFor();

  const projectsLink = page.getByRole("navigation").getByRole("link", { name: /Projects|项目/, exact: true });
  await projectsLink.click();
  await page.getByRole("heading", { name: /^(Projects|项目)$/ }).waitFor();
  const addProject = page.getByRole("button", { name: /Add project|添加项目|登记项目/ }).first();
  await addProject.click();
  const dialog = page.getByRole("dialog");
  await dialog.getByLabel(/Absolute local path|本地绝对路径/).fill(workspace);
  const registered = page.waitForResponse((response) => (
    response.url() === `${baseUrl}/api/v1/projects` && response.request().method() === "POST"
  ));
  await dialog.getByRole("button", { name: /Register project|注册项目|登记项目/ }).click();
  const registeredResponse = await registered;
  assert.equal(registeredResponse.status(), 200, await registeredResponse.text());
  await dialog.waitFor({ state: "hidden" });

  await page.getByRole("heading", { name: "workspace", exact: true }).waitFor();
  await page.getByRole("link", { name: /Open sessions|打开会话/ }).click();
  const newSession = page.getByRole("button", { name: /New session|新建会话/ });
  stage = "first-draft";
  const firstSession = await createClaimedDraft(newSession);
  assert.equal(typeof firstSession.session_id, "string");
  const input = page.getByLabel(/Message input|消息输入/);
  await input.waitFor();
  const answer = page.locator('article[data-role="assistant"]').getByRole("heading", { name: "Streamed answer", exact: true }).last();
  let crossClientEvidence = null;
  let completedRunId = null;
  if (crossClientReadyPath) {
    assert.ok(crossClientCliReadyPath && crossClientCliDonePath && crossClientPrivateMarker);
    const privatePrompt = `${crossClientPrivateMarker} streaming markdown`;
    await input.fill(privatePrompt);
    await input.press("Enter");
    stage = "private-input-accepted";
    await expect.poll(() => events.filter((event) => (
      event.type === "input.accepted" && event.payload?.text === privatePrompt
    )).length, { timeout: 30000 }).toBe(1);
    const privateAccepted = events.find((event) => (
      event.type === "input.accepted" && event.payload?.text === privatePrompt
    ));
    assert.equal(privateAccepted.session_id, firstSession.session_id);
    await expect.poll(() => events.some((event) => (
      event.type === "run.completed" && event.run_id === privateAccepted.run_id
    )), { timeout: 30000 }).toBe(true);
    await page.getByText("Streamed answer", { exact: true }).first().waitFor();
    await writeFile(crossClientReadyPath, JSON.stringify({
      status: "ready",
      browser_session_id: firstSession.session_id,
      private_marker: crossClientPrivateMarker,
      workspace,
    }));
    const cliReady = await waitForJson(crossClientCliReadyPath);
    assert.equal(cliReady.status, "ready");
    assert.equal(cliReady.contested_session_id, firstSession.session_id);
    assert.equal(cliReady.workspace_id, firstSession.workspace_id);

    stage = "second-draft";
    const secondSession = await createClaimedDraft(newSession);
    const concurrentPrompt = "browser concurrent session streaming markdown";
    await input.fill(concurrentPrompt);
    await input.press("Enter");
    stage = "concurrent-input-accepted";
    await expect.poll(() => events.filter((event) => (
      event.type === "input.accepted" && event.payload?.text === concurrentPrompt
    )).length, { timeout: 30000 }).toBe(1);
    const concurrentAccepted = events.find((event) => (
      event.type === "input.accepted" && event.payload?.text === concurrentPrompt
    ));
    assert.equal(concurrentAccepted.session_id, secondSession.session_id);
    stage = "concurrent-provider-barrier";
    assert.ok(observationPath && concurrencyReleasePath);
    const cliPrompt = "installed CLI concurrent session streaming markdown";
    await expect.poll(async () => {
      const records = (await readFile(observationPath, "utf8")).split("\n")
        .filter(Boolean).map((line) => JSON.parse(line));
      return [cliPrompt, concurrentPrompt].every((text) => records.some((record) => (
        record.prompt.includes(text) && record.tools.length > 0
      )));
    }, { timeout: 30000 }).toBe(true);
    assert.equal(events.some((event) => event.type === "run.completed" && event.run_id === concurrentAccepted.run_id), false);
    await page.reload();
    await expect(page.getByRole("log").getByText(concurrentPrompt, { exact: true }))
      .toBeVisible({ timeout: 5000 });
    await expect(page.getByRole("button", { name: /Cancel run|取消运行/, exact: true })).toBeEnabled();
    assert.ok(events.some((event) => event.type === "snapshot.required"
      && event.payload?.snapshot?.sessions?.some((entry) => (
        entry.snapshot.live_state?.runs?.some((run) => run.run_id === concurrentAccepted.run_id)
      ))));
    await assert.rejects(readFile(crossClientCliDonePath, "utf8"), { code: "ENOENT" });
    await writeFile(concurrencyReleasePath, "release\n", "utf8");
    stage = "concurrent-completion";
    completedRunId = concurrentAccepted.run_id;
    await expect.poll(() => events.some((event) => (
      event.type === "run.completed" && event.run_id === concurrentAccepted.run_id
    )), { timeout: 30000 }).toBe(true);
    await page.getByText("Streamed answer", { exact: true }).last().waitFor();
    const cliDone = await waitForJson(crossClientCliDonePath);
    assert.equal(cliDone.status, "passed");
    assert.equal(cliDone.workspace_id, firstSession.workspace_id);
    assert.notEqual(cliDone.session_id, secondSession.session_id);
    assert.notEqual(cliDone.session_id, firstSession.session_id);
    assert.equal(cliDone.claim_error_contains_private_marker, false);
    crossClientEvidence = {
      browser_session_id: secondSession.session_id,
      browser_private_session_id: firstSession.session_id,
      cli_session_id: cliDone.session_id,
      workspace_id: firstSession.workspace_id,
      distinct_sessions: true,
      browser_run_completed: true,
      cli_run_completed: cliDone.assistant_persisted === true,
      same_claim_denied: cliDone.claim_denied_code === "session_claimed",
      claim_error_contains_private_marker: false,
      cli_adapter: cliDone.adapter ?? "installed console entry headless adapter",
      both_runs_waiting_at_provider: true,
      body_read_denied: cliDone.body_read_denied === true,
    };
  } else {
    stage = "single-input-accepted";
    await input.fill(prompt);
    await input.press("Enter");
    await expect.poll(() => events.filter((event) => (
      event.type === "input.accepted" && event.payload?.text === prompt
    )).length, { timeout: 30000 }).toBe(1);
    const accepted = events.find((event) => event.type === "input.accepted" && event.payload?.text === prompt);
    assert.equal(accepted.session_id, firstSession.session_id);
    completedRunId = accepted.run_id;
    await expect.poll(() => events.some((event) => (
      event.type === "run.completed" && event.run_id === accepted.run_id
    )), { timeout: 30000 }).toBe(true);
    await answer.waitFor();
  }
  await page.screenshot({ path: join(output, "conversation.png"), fullPage: true });
  await page.goto(page.url());
  await answer.waitFor();

  stage = "confirmation-refresh";
  await createClaimedDraft(newSession);
  await input.fill("confirmation");
  await input.press("Enter");
  const confirmationDialog = page.locator('[role="dialog"][data-confirmation-origin="foreground"]');
  await expect(confirmationDialog).toBeVisible();
  const originalConfirmation = [...events].reverse().find((event) => event.type === "confirmation.requested");
  const confirmedRun = [...events].reverse().find((event) => event.type === "input.accepted"
    && event.session_id === originalConfirmation.session_id);
  await page.reload();
  await expect(confirmationDialog).toBeVisible({ timeout: 5000 });
  const pending = [...events].reverse().find((event) => event.type === "snapshot.required"
    && event.payload?.snapshot?.pending_confirmation)?.payload.snapshot.pending_confirmation;
  assert.equal(pending?.payload.token, originalConfirmation.payload.token);
  assert.deepEqual(pending?.payload.request, originalConfirmation.payload.request);
  await page.keyboard.press("Enter");
  await expect(confirmationDialog).toBeHidden();
  await expect.poll(() => events.some((event) => event.type === "run.completed"
    && event.run_id === confirmedRun.run_id)).toBe(true);

  stage = "active-refresh-cancel";
  const canceledSession = await createClaimedDraft(newSession);
  const cancelPrompt = "installed expiry barrier";
  await input.fill(cancelPrompt);
  await input.press("Enter");
  await expect.poll(() => events.filter((event) => event.type === "input.accepted"
    && event.payload?.text === cancelPrompt).length).toBe(1);
  const acceptedCancel = events.find((event) => event.type === "input.accepted"
    && event.payload?.text === cancelPrompt);
  await page.reload();
  await expect(page.getByRole("log").getByText(cancelPrompt, { exact: true }))
    .toBeVisible({ timeout: 5000 });
  await page.getByRole("button", { name: /Cancel run|取消运行/, exact: true }).click();
  await expect.poll(() => events.some((event) => event.type === "run.completed"
    && event.run_id === acceptedCancel.run_id)).toBe(true);
  const canceledRecords = (await readFile(join(workspace, ".myclaw", "sessions",
    `${canceledSession.session_id}.jsonl`), "utf8")).split("\n").filter(Boolean).map(JSON.parse);
  assert.equal(canceledRecords.filter((record) => record.role === "user" && record.content === cancelPrompt).length, 1);

  assert.deepEqual(errors, [], `Installed Web app reported browser errors: ${errors.join("; ")}`);
  const evidence = {
    marker: "INSTALLED_WEB_E2E_OK",
    base_url: baseUrl,
    route: new URL(page.url()).pathname,
    api: "passed",
    assets: "mime-cache-verified",
    websocket: "passed",
    settings: "passed",
    conversation: "passed",
    recovery: { active_refresh: "passed", confirmation_refresh: "passed", targeted_cancel: "passed" },
    accepted_runs: crossClientEvidence === null ? 1 : 2,
    completed_run: completedRunId,
    service_instance_id: service.service_instance_id,
    missing_resources: "404",
    ...(crossClientEvidence === null ? {} : { cross_client: crossClientEvidence }),
  };
  await writeFile(join(output, "browser.json"), `${JSON.stringify(evidence, null, 2)}\n`);
  console.log(JSON.stringify(evidence));
} catch (error) {
  const redact = (value) => String(value).replaceAll(ticket, "[redacted]");
  const stack = redact(error.stack ?? `${error.name}: ${error.message}`);
  const diagnostics = {
    stage,
    stack,
    input: await page.getByLabel(/Message input|消息输入/).inputValue().catch(() => null),
    alerts: await page.getByRole("alert").allTextContents().catch(() => []),
    events: events.slice(-20).map((event) => ({
      type: event.type,
      session_id: event.session_id,
      run_id: event.run_id,
      text: event.payload?.text,
      code: event.error?.code,
    })),
  };
  await writeFile(join(output, "browser-failure.json"), redact(JSON.stringify(diagnostics, null, 2))).catch(() => {});
  await page.screenshot({ path: join(output, "failure.png"), fullPage: true }).catch(() => {});
  error.message = redact(error.message);
  error.stack = `Installed browser stage: ${stage}\n${stack}`;
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
