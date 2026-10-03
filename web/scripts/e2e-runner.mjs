import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { link, mkdir, readFile, readdir, rm, unlink, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { URL } from "node:url";
import { chromium, expect } from "@playwright/test";

import setup from "./e2e-setup.mjs";
import settingsAcceptance, { settingsConfirmationAcceptance, settingsModelMcpAcceptance } from "./settings-e2e.mjs";

if (process.platform !== "win32") {
  console.error("Omni requires Windows.");
  process.exit(1);
}

const viewports = [
  { width: 1440, height: 900 },
  { width: 1024, height: 768 },
  { width: 768, height: 1024 },
];
const output = resolve("test-results");
let control;
let browser;
let secondContext;
let acceptanceError;

async function shutdownControl() {
  try {
    await control?.shutdown();
  } catch (error) {
    if (acceptanceError === undefined) throw error;
    console.error("E2E cleanup also failed:", error.message);
  }
}

try {
  control = await setup();
  browser = await chromium.launch({
    channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge",
  });
  const primaryContext = await browser.newContext();
  const page = await primaryContext.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  async function waitForRecordedEvent(matches, description) {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      const messages = await page.evaluate(() => window.__omniTestMessages);
      if (matches(messages)) return;
      await delay(50);
    }
    const events = await page.evaluate(() => window.__omniTestMessages.slice(-12).map(
      (event) => ({ type: event.type, text: event.payload?.text, code: event.error?.code }),
    ));
    const alerts = await page.getByRole("alert").allTextContents();
    throw new Error(`Timed out waiting for ${description}: ${JSON.stringify({ events, alerts })}`);
  }
  async function waitForConfirmationRunCompletion() {
    let acceptedRun;
    await waitForRecordedEvent((messages) => {
      const accepted = [...messages].reverse().find((event) => (
        event.type === "input.accepted" && event.payload?.text === "confirmation"
      ));
      acceptedRun = accepted;
      return accepted !== undefined && messages.some((event) => (
        event.type === "run.completed" && event.run_id === accepted.run_id
      ));
    }, "confirmation Run completion");
    return acceptedRun;
  }
  await page.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    window.__omniTestMessages = [];
    window.WebSocket = class extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        window.__omniTestControlCredential = Array.isArray(args[1]) ? args[1][1] : null;
        window.__omniTestSocket = this;
        this.addEventListener("message", (event) => {
          try {
            window.__omniTestMessages.push(JSON.parse(event.data));
          } catch {
            // Only JSON service messages are relevant to this test.
          }
        });
      }
    };
  });
  const url = process.env.OMNI_E2E_URL;
  const launch = spawnSync("python", ["-c", [
    "import sys, webbrowser",
    "from omni.terminal.process_entry import run",
    "webbrowser.open_new_tab = lambda _url: False",
    "sys.argv = ['omni', 'web']",
    "run()",
  ].join("; ")], {
    cwd: resolve(process.cwd(), ".."),
    env: {
      ...process.env,
      USERPROFILE: process.env.OMNI_E2E_HOME_ROOT,
      HOME: process.env.OMNI_E2E_HOME_ROOT,
    },
    encoding: "utf8",
    timeout: 30000,
  });
  assert.equal(launch.status, 0, `omni web failed: ${launch.stderr}`);
  const launchUrl = launch.stdout.match(/http:\/\/127\.0\.0\.1:\d+\/#ticket=[\w-]+/)?.[0];
  assert.ok(launchUrl?.startsWith(`${url}/#ticket=`), "omni web did not attach to the isolated service");
  const documentResponse = await primaryContext.request.get(url);
  assert.equal(documentResponse.status(), 200, "The production document did not load");
  assert.match(
    documentResponse.headers()["content-security-policy"] ?? "",
    /default-src 'self'/,
    "The production document did not include the expected CSP",
  );
  await page.goto(launchUrl);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor();
  assert.match(page.url(), /\/status$/);
  secondContext = await browser.newContext();
  await secondContext.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    window.__omniTestMessages = [];
    window.WebSocket = class extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        window.__omniTestControlCredential = Array.isArray(args[1]) ? args[1][1] : null;
        window.__omniTestSocket = this;
        this.addEventListener("message", (event) => {
          try {
            window.__omniTestMessages.push(JSON.parse(event.data));
          } catch {
            // Only JSON service messages are relevant to this test.
          }
        });
      }
    };
  });
  const secondPage = await secondContext.newPage();
  await secondPage.goto(`${url}/#ticket=${encodeURIComponent(control.details.second_ticket)}`);
  await secondPage.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await secondPage.getByRole("status").first().getByText(/Online|在线/).waitFor();
  const replay = await browser.newContext();
  const reusedTicket = await replay.request.post(`${url}/api/v1/web/ticket`, {
    headers: { Origin: url },
    data: { ticket: launchUrl.split("#ticket=")[1] },
  });
  assert.equal(reusedTicket.status(), 401, "A consumed browser ticket was accepted again");
  await replay.close();

  await settingsAcceptance({ page, secondPage, control, output, viewports });
  await page.getByRole("navigation").getByRole("link", { name: "Status", exact: true }).click();

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    assert.equal(await page.locator("html").getAttribute("lang"), language);
    await page.getByRole("heading", { name: language === "en" ? "Service status" : "服务状态" }).waitFor();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      assert.equal(await page.locator("html").getAttribute("data-theme"), theme);
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        await page.getByRole("main").waitFor();
        await page.getByRole("navigation").getByRole("link", {
          name: language === "en" ? "Status" : "状态",
        }).waitFor();
        const layout = await page.evaluate(() => {
          const aside = document.querySelector("aside").getBoundingClientRect();
          const main = document.querySelector("main").getBoundingClientRect();
          return { width: document.documentElement.scrollWidth, asideRight: aside.right, mainLeft: main.left };
        });
        assert.ok(layout.width <= viewport.width, `Horizontal overflow at ${viewport.width}x${viewport.height}`);
        assert.ok(layout.mainLeft >= layout.asideRight - 1, `Sidebar overlaps content at ${viewport.width}x${viewport.height}`);
        await mkdir(output, { recursive: true });
        await page.screenshot({ path: resolve(output, `workbench-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }

  const statusLink = page.getByRole("navigation").getByRole("link", { name: "状态" });
  await statusLink.focus();
  await statusLink.press("Enter");
  assert.match(page.url(), /\/status$/);

  const details = page.getByRole("button", { name: /连接详情|Connection details/ });
  await details.focus();
  await details.press("Enter");
  await page.getByRole("dialog").waitFor();
  await page.keyboard.press("Escape");
  await expect(details).toBeFocused();
  assert.equal(await details.evaluate((element) => element === document.activeElement), true);

  await page.reload();
  await page.getByRole("heading", { name: "服务状态" }).waitFor();
  assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
  await page.getByRole("status").first().getByText("在线").waitFor();

  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("link", { name: "Projects" }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();
  await page.getByRole("heading", { name: "No projects registered" }).waitFor();

  const firstProject = process.env.OMNI_E2E_FIRST_PROJECT;
  const projectAlias = process.env.OMNI_E2E_PROJECT_ALIAS;
  const secondProject = process.env.OMNI_E2E_SECOND_PROJECT;
  const cliWorkspace = process.env.OMNI_E2E_CLI_WORKSPACE;
  assert.ok(firstProject && projectAlias && secondProject && cliWorkspace);
  const projectItems = page.locator('ul[aria-label="Projects"] > li');

  async function registerProject(path, name) {
    await page.getByRole("button", { name: "Add project" }).first().click();
    const dialog = page.getByRole("dialog");
    await dialog.getByLabel("Absolute local path").fill(path);
    await dialog.getByRole("button", { name: "Register project" }).click();
    await dialog.waitFor({ state: "hidden" });
    await page.getByRole("heading", { name }).waitFor();
  }

  await page.getByRole("button", { name: "Add project" }).first().click();
  const invalidDialog = page.getByRole("dialog");
  await invalidDialog.getByLabel("Absolute local path").fill("relative/project");
  await invalidDialog.getByRole("button", { name: "Register project" }).click();
  await invalidDialog.getByRole("alert").waitFor();
  await invalidDialog.getByRole("button", { name: "Cancel" }).click();

  await page.clock.install();
  await page.clock.pauseAt(new Date());
  await registerProject(firstProject, "project-one");
  const registrationNotice = page.getByText(
    "Project registered. Saved Schedule Jobs remain paused until resumed.",
    { exact: true },
  );
  await registrationNotice.waitFor();
  await page.clock.runFor(9999);
  assert.equal(await registrationNotice.count(), 1, "Registration feedback expired before 10 seconds");
  const projectRefresh = page.waitForResponse((response) => (
    response.request().method() === "GET"
    && response.url().endsWith("/api/v1/projects")
  ));
  await page.getByRole("button", { name: "Refresh projects" }).click();
  await projectRefresh;
  await page.clock.runFor(1);
  await registrationNotice.waitFor({ state: "hidden" });
  await page.getByText("E2E saved project job").waitFor();
  await page.getByText("Schedule paused for review").waitFor();
  assert.ok(
    await page.getByRole("button", { name: "Resume schedule" }).count() > 0,
    "The explicit resume entry disappeared with the transient feedback",
  );
  await page.clock.resume();

  await page.getByRole("navigation").getByRole("link", { name: "Projects", exact: true }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();

  const firstProjectItem = projectItems.filter({ hasText: firstProject });
  const scheduleResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "GET"
    && response.url().endsWith("/schedule/jobs")
  ));
  await firstProjectItem.getByRole("link", { name: "Open schedule" }).click();
  const scheduleResponse = await scheduleResponsePromise;
  assert.equal(scheduleResponse.status(), 200, "Schedule page did not load its real job response");
  const schedulePayload = await scheduleResponse.json();
  assert.equal(schedulePayload.status.admitted, false, "Schedule page reported a false paused state");
  assert.equal(schedulePayload.status.status, "available", "Schedule page reported a false health state");
  await page.getByRole("heading", { name: "Schedule Jobs", exact: true }).waitFor();
  const scheduleStatus = page.locator('dl[aria-label="Schedule status"]');
  await expect(scheduleStatus).toContainText("Paused");
  await expect(scheduleStatus).toContainText("Available");
  await expect(scheduleStatus).toContainText("Active Jobs");
  await expect(scheduleStatus).not.toContainText("{{");

  const historyJobTitle = "E2E schedule history job";
  const historyJobItem = page.getByRole("listitem").filter({ hasText: historyJobTitle });
  const historyResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "GET" && /\/schedule\/jobs\/[^/]+\/history(?:\?|$)/.test(response.url())
  ));
  await historyJobItem.getByRole("link", { name: "History", exact: true }).click();
  const historyResponse = await historyResponsePromise;
  assert.equal(historyResponse.status(), 200, "Schedule history did not load from the real service");
  const historyPayload = await historyResponse.json();
  assert.equal(historyPayload.groups.length, 20, "History did not honor the first page limit");
  assert.equal(typeof historyPayload.next_cursor, "string", "History did not return a cursor");
  assert.equal(historyPayload.job.state.last_status, null, "History reused the latest Job state");
  assert.equal(historyPayload.groups[0].result_state, "success");
  assert.equal(historyPayload.groups.some((group) => "occurrence_id" in group), false);
  await page.getByRole("heading", { name: historyJobTitle, exact: true }).first().waitFor();
  await page.getByText("Historical execution 1", { exact: true }).waitFor();
  await expect(page.getByText("Historical execution without a terminal result", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("textbox")).toHaveCount(0);

  const failHistoryLoad = (route) => route.fulfill({
    status: 500,
    json: { code: "persistence_error", message: "Schedule history could not be loaded safely." },
  });
  await page.route("**/schedule/jobs/*/history*", failHistoryLoad);
  const loadMoreHistory = page.getByRole("button", { name: "Load more executions", exact: true });
  await loadMoreHistory.focus();
  await loadMoreHistory.press("Enter");
  const historyLoadError = page.getByRole("alert").filter({ hasText: "Schedule history" });
  await historyLoadError.waitFor();
  await expect(page.getByText("Historical execution 1", { exact: true })).toBeVisible();
  await expect(loadMoreHistory).toBeFocused();
  await page.unroute("**/schedule/jobs/*/history*", failHistoryLoad);

  const moreHistoryResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "GET" && response.url().includes("/history?")
  ));
  await historyLoadError.getByRole("button", { name: "Retry", exact: true }).click();
  const moreHistoryResponse = await moreHistoryResponsePromise;
  assert.equal(moreHistoryResponse.status(), 200, "Schedule history pagination failed");
  await page.getByText("Historical execution without a terminal result", { exact: true }).waitFor();
  await page.getByText("Outcome unknown", { exact: true }).waitFor();

  const refreshHistoryButton = page.getByRole("button", { name: "Refresh Schedule history", exact: true });
  await page.route("**/schedule/jobs/*/history*", failHistoryLoad);
  await refreshHistoryButton.click();
  await historyLoadError.waitFor();
  await expect(page.getByText("Historical execution without a terminal result", { exact: true })).toBeVisible();
  await page.unroute("**/schedule/jobs/*/history*", failHistoryLoad);
  await historyLoadError.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(historyLoadError).toHaveCount(0);
  await expect(page.getByText("Historical execution without a terminal result", { exact: true })).toHaveCount(0);
  await expect(refreshHistoryButton).toBeFocused();

  const renderHistoryTools = async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.groups[0].messages.splice(1, 0,
      { role: "assistant", content: "", status: "completed", tool_calls: [
        { id: "history-failed-tool", name: "read_file", arguments: "<img src=x onerror=alert(1)>" },
      ] },
      { role: "tool", name: "read_file", tool_call_id: "history-failed-tool", status: "error", content: "Recorded tool failure" },
      { role: "tool", name: "legacy_tool", tool_call_id: "legacy-tool", content: "Legacy tool without a result state" },
    );
    body.groups[0].messages.at(-1).content += `\n\n[Unsafe link](javascript:alert(1))\n\n![Remote image](https://example.com/history.png)\n\n<img src=x onerror=alert(1)>\n\n\`\`\`text\n${"long-history-code ".repeat(200)}\n\`\`\``;
    await route.fulfill({ response, json: body });
  };
  await page.route("**/schedule/jobs/*/history*", renderHistoryTools);
  await refreshHistoryButton.click();
  const failedHistoryTool = page.locator('article[data-role="tool"]').filter({ hasText: "Recorded tool failure" });
  await failedHistoryTool.locator("summary").first().click();
  await expect(failedHistoryTool.getByText("Failed", { exact: true })).toBeVisible();
  const unknownHistoryTool = page.locator('article[data-role="tool"]').filter({ hasText: "Legacy tool without a result state" });
  await unknownHistoryTool.locator("summary").first().click();
  await expect(unknownHistoryTool.getByText("Outcome unknown", { exact: true })).toBeVisible();
  const historyToolRequest = page.locator('article[data-role="assistant"]').filter({ has: page.getByText("Arguments", { exact: true }) });
  await historyToolRequest.locator("summary").first().click();
  await expect(historyToolRequest.getByText("Completed", { exact: true })).toHaveCount(0);
  await historyToolRequest.getByText("Arguments", { exact: true }).click();
  await expect(historyToolRequest.locator("pre")).toHaveText("<img src=x onerror=alert(1)>");
  await expect(page.locator('img[src="https://example.com/history.png"], a[href^="javascript:"]')).toHaveCount(0);
  await page.unroute("**/schedule/jobs/*/history*", renderHistoryTools);
  await refreshHistoryButton.focus();
  const refreshHistoryResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "GET" && /\/schedule\/jobs\/[^/]+\/history(?:\?|$)/.test(response.url())
  ));
  await refreshHistoryButton.press("Enter");
  assert.equal((await refreshHistoryResponsePromise).status(), 200, "History refresh failed");
  await expect(refreshHistoryButton).toBeFocused();

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    await page.getByRole("heading", { name: historyJobTitle, exact: true }).first().waitFor();
    const listTitle = language === "en" ? "Recorded executions" : "已记录的执行";
    await page.getByRole("heading", { name: listTitle, exact: true }).waitFor();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        assert.ok(
          await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
          `History horizontal overflow at ${language}/${theme}/${viewport.width}x${viewport.height}`,
        );
        await page.screenshot({ path: resolve(output, `schedule-history-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("link", { name: "Back to Schedule Jobs", exact: true }).first().click();
  await page.getByRole("heading", { name: "Schedule Jobs", exact: true }).waitFor();

  let releaseHistoryLoad;
  let historyLoadArrived;
  const historyLoadGate = new Promise((done) => { releaseHistoryLoad = done; });
  const historyLoadArrival = new Promise((done) => { historyLoadArrived = done; });
  const delayHistoryLoad = async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.job.title = "STALE Schedule history response";
    historyLoadArrived();
    await historyLoadGate;
    await route.fulfill({ response, json: body });
  };
  await page.route("**/schedule/jobs/*/history*", delayHistoryLoad);
  await historyJobItem.getByRole("link", { name: "History", exact: true }).click();
  await historyLoadArrival;
  await page.getByRole("link", { name: "Back to Schedule Jobs", exact: true }).first().click();
  releaseHistoryLoad();
  await expect(page.getByText("STALE Schedule history response", { exact: true })).toHaveCount(0);
  await page.unroute("**/schedule/jobs/*/history*", delayHistoryLoad);

  async function assertHistoryScopeChange(scopeChange) {
    let releaseScopedHistory;
    let notifyScopedHistory;
    let heldScopedHistory = false;
    const scopedHistoryGate = new Promise((done) => { releaseScopedHistory = done; });
    const scopedHistoryArrived = new Promise((done) => { notifyScopedHistory = done; });
    const delayScopedHistory = async (route) => {
      if (heldScopedHistory) return route.continue();
      heldScopedHistory = true;
      const response = await route.fetch();
      const body = await response.json();
      body.job.title = `STALE history after ${scopeChange}`;
      notifyScopedHistory();
      await scopedHistoryGate;
      await route.fulfill({ response, json: body });
    };
    await page.route("**/schedule/jobs/*/history*", delayScopedHistory);
    await historyJobItem.getByRole("link", { name: "History", exact: true }).click();
    await scopedHistoryArrived;
    if (scopeChange === "job") {
      await page.getByRole("link", { name: "Back to Schedule Jobs", exact: true }).first().click();
      await page.getByRole("listitem").filter({ hasText: "E2E saved project job" })
        .getByRole("link", { name: "History", exact: true }).click();
      await page.getByRole("heading", { name: "E2E saved project job", exact: true }).first().waitFor();
    } else if (scopeChange === "project") {
      await page.locator("aside").getByRole("link", { name: "project-two", exact: true }).click();
      await page.getByRole("heading", { name: "project-two", exact: true }).waitFor();
    } else {
      await page.route("**/api/v1/clients", (route) => route.abort());
      await page.evaluate(() => window.__omniTestSocket.close());
      await page.getByRole("status").filter({ hasText: "Showing the last received Job status." }).waitFor();
      await expect(page.getByRole("button", { name: "Refresh Schedule history", exact: true })).toBeDisabled();
    }
    const scopedHistoryResponse = page.waitForResponse((response) => response.url().includes(`/schedule/jobs/${historyPayload.job_id}/history`));
    releaseScopedHistory();
    await scopedHistoryResponse;
    await expect(page.getByText(`STALE history after ${scopeChange}`, { exact: true })).toHaveCount(0);
    await page.unroute("**/schedule/jobs/*/history*", delayScopedHistory);
    if (scopeChange === "disconnect") {
      await page.unroute("**/api/v1/clients");
      await expect(page.getByRole("button", { name: "Refresh Schedule history", exact: true })).toBeEnabled({ timeout: 10000 });
      await page.getByRole("heading", { name: historyJobTitle, exact: true }).first().waitFor();
    }
    await page.getByRole("navigation").getByRole("link", { name: "Projects", exact: true }).click();
    await firstProjectItem.getByRole("link", { name: "Open schedule" }).click();
    await page.getByRole("heading", { name: "Schedule Jobs", exact: true }).waitFor();
    await expect(page.getByRole("button", { name: "Refresh schedule", exact: true })).toBeEnabled();
    await historyJobItem.getByRole("link", { name: "History", exact: true }).waitFor();
  }
  for (const scopeChange of ["job", "disconnect"]) await assertHistoryScopeChange(scopeChange);

  const scheduleCreate = page.getByRole("button", { name: "Create Job", exact: true });
  await expect(scheduleCreate).toBeEnabled();
  await scheduleCreate.click();
  const scheduleValidation = page.getByRole("alert").filter({ hasText: "Review the highlighted fields." });
  await scheduleValidation.waitFor();
  await expect(scheduleValidation.getByRole("link", { name: "Message", exact: true })).toBeVisible();
  await expect(scheduleValidation.getByRole("link", { name: "Run at", exact: true })).toBeVisible();
  await expect(page.getByLabel("Message")).toHaveAttribute("aria-invalid", "true");
  await expect(scheduleValidation).toBeFocused();

  async function createBrowserScheduleJob({
    title,
    message,
    kind,
    value,
  }) {
    await page.getByLabel("Message").fill(message);
    await page.getByLabel("Title").fill(title);
    await page.getByRole("button", { name: kind, exact: true }).click();
    if (kind === "At") await page.getByLabel("Run at").fill(value);
    if (kind === "Every") await page.getByLabel("Interval (seconds)").fill(value);
    if (kind === "Cron") {
      await page.getByLabel("Cron expression").fill(value);
      await page.getByLabel("Timezone").fill("UTC");
    }
    const createResponse = page.waitForResponse((response) => (
      response.request().method() === "POST"
      && response.url().endsWith("/schedule/jobs")
    ));
    await scheduleCreate.click();
    const response = await createResponse;
    assert.equal(response.status(), 200, `${kind} Schedule Job creation failed`);
    await page.getByText("Schedule Job created.", { exact: true }).waitFor();
    await page.getByRole("heading", { name: title, exact: true }).waitFor();
  }

  const browserAtTitle = "E2E browser at job";
  const browserEveryTitle = "E2E browser every job";
  const browserCronTitle = "E2E browser cron job";
  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
  await createBrowserScheduleJob({
    title: browserAtTitle,
    message: "E2E browser at message",
    kind: "At",
    value: "2099-01-01T00:00:00Z",
  });
  const createdScheduleNotice = page.getByRole("status").filter({ hasText: "Schedule Job created." });
  await page.clock.runFor(9999);
  await expect(createdScheduleNotice).toBeVisible();
  await page.clock.runFor(1);
  await expect(createdScheduleNotice).toHaveCount(0);
  await page.clock.resume();
  await createBrowserScheduleJob({
    title: browserEveryTitle,
    message: "E2E browser every message",
    kind: "Every",
    value: "60",
  });
  await createdScheduleNotice.getByRole("button", { name: "Close", exact: true }).click();
  await expect(createdScheduleNotice).toHaveCount(0);
  await createBrowserScheduleJob({
    title: browserCronTitle,
    message: "E2E browser cron message",
    kind: "Cron",
    value: "0 0 * * *",
  });

  const retryScheduleTitle = "E2E accepted Schedule create retry";
  const acceptedCreateRequests = [];
  let acceptedCreateHeaders;
  let acceptedScheduleJob;
  let notifyAcceptedCreate;
  let releaseAcceptedCreate;
  const acceptedCreateArrived = new Promise((done) => { notifyAcceptedCreate = done; });
  const acceptedCreateGate = new Promise((done) => { releaseAcceptedCreate = done; });
  const loseAcceptedCreateResponse = async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    acceptedCreateRequests.push(route.request().postDataJSON());
    acceptedCreateHeaders = await route.request().allHeaders();
    const response = await route.fetch();
    assert.equal(response.status(), 200, "The create retry fixture must first be accepted by the real service");
    acceptedScheduleJob = (await response.json()).job;
    if (acceptedCreateRequests.length === 1) {
      notifyAcceptedCreate();
      await acceptedCreateGate;
      return route.abort("failed");
    }
    return route.fulfill({ response });
  };
  await page.route("**/schedule/jobs", loseAcceptedCreateResponse);
  await page.getByLabel("Message", { exact: true }).fill("E2E accepted retry message");
  await page.getByLabel("Title", { exact: true }).fill(retryScheduleTitle);
  await page.getByRole("button", { name: "At", exact: true }).click();
  await page.getByLabel("Run at", { exact: true }).fill("2099-02-01T00:00:00Z");
  await scheduleCreate.click();
  await acceptedCreateArrived;
  for (const label of ["Message", "Title", "Run at"]) {
    await expect(page.getByLabel(label, { exact: true })).toBeDisabled();
  }
  await expect(page.getByRole("button", { name: "Every", exact: true })).toBeDisabled();
  releaseAcceptedCreate();
  await page.getByRole("alert").filter({ hasText: "The request may have completed, but its result is unknown." }).waitFor();
  await expect(page.getByLabel("Message", { exact: true })).toHaveValue("E2E accepted retry message");
  await expect(page.getByLabel("Message", { exact: true })).toBeDisabled();
  const retryAcceptedResponse = page.waitForResponse((response) => (
    response.request().method() === "POST" && response.url().endsWith("/schedule/jobs")
  ));
  await page.getByRole("button", { name: "Retry create", exact: true }).click();
  assert.equal((await retryAcceptedResponse).status(), 200);
  await page.getByRole("heading", { name: retryScheduleTitle, exact: true }).waitFor();
  assert.equal(acceptedCreateRequests.length, 2);
  assert.deepEqual(acceptedCreateRequests[1], acceptedCreateRequests[0], "Unknown create retry changed request_id or payload");
  delete acceptedCreateHeaders["content-length"];
  const acceptedListResponse = await primaryContext.request.get(scheduleResponse.url(), { headers: acceptedCreateHeaders });
  assert.equal(acceptedListResponse.status(), 200);
  const acceptedList = await acceptedListResponse.json();
  assert.equal(acceptedList.jobs.filter((job) => job.title === retryScheduleTitle).length, 1,
    "The two accepted create requests persisted more than one Job");
  assert.equal(acceptedList.jobs.find((job) => job.title === retryScheduleTitle).job_id, acceptedScheduleJob.job_id);
  await expect(page.getByLabel("Message", { exact: true })).toBeEnabled();
  await page.unroute("**/schedule/jobs", loseAcceptedCreateResponse);

  const browserAtItem = page.getByRole("listitem").filter({ hasText: browserAtTitle });
  await expect(browserAtItem.getByRole("status")).toHaveText("Scheduled");
  const detailResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "GET"
    && /\/schedule\/jobs\/[^/]+$/.test(response.url())
  ));
  await browserAtItem.getByRole("button", { name: "Inspect", exact: true }).click();
  const detailResponse = await detailResponsePromise;
  assert.equal(detailResponse.status(), 200, "Schedule detail did not load the authoritative response");
  const detailPayload = await detailResponse.json();
  assert.equal(detailPayload.job.schedule.kind, "at", "Schedule detail returned the wrong rule kind");
  const scheduleDetail = page.getByRole("dialog", { name: "Schedule Job details" });
  await scheduleDetail.getByText("E2E browser at message", { exact: true }).waitFor();
  await scheduleDetail.getByText(`At ${detailPayload.job.schedule.at_time}`, { exact: true }).waitFor();
  await scheduleDetail.getByText("Scheduled", { exact: true }).waitFor();
  await scheduleDetail.getByRole("button", { name: "Close", exact: true }).last().click();
  await scheduleDetail.waitFor({ state: "hidden" });

  let notifyDelayedDetail;
  let releaseDelayedDetail;
  const delayedDetailArrived = new Promise((done) => { notifyDelayedDetail = done; });
  const delayedDetailGate = new Promise((done) => { releaseDelayedDetail = done; });
  const delayScheduleDetail = async (route) => {
    const response = await route.fetch();
    notifyDelayedDetail();
    await delayedDetailGate;
    await route.fulfill({ response });
  };
  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
  await page.route("**/schedule/jobs/*", delayScheduleDetail);
  const inspectScheduleTrigger = browserAtItem.getByRole("button", { name: "Inspect", exact: true });
  await inspectScheduleTrigger.focus();
  await inspectScheduleTrigger.press("Enter");
  await delayedDetailArrived;
  await scheduleDetail.getByText("Loading Job details", { exact: true }).waitFor();
  await page.keyboard.press("Escape");
  await expect(scheduleDetail).toHaveCount(0);
  await page.clock.runFor(32);
  await expect(inspectScheduleTrigger).toBeFocused();
  const delayedDetailResponse = page.waitForResponse((response) => response.url() === detailResponse.url());
  releaseDelayedDetail();
  await delayedDetailResponse;
  await page.clock.runFor(32);
  await expect(scheduleDetail).toHaveCount(0);
  await expect(inspectScheduleTrigger).toBeFocused();
  await page.unroute("**/schedule/jobs/*", delayScheduleDetail);

  // Only the HTTP status projection is simulated here; CRUD and persistence above use the real service.
  let projectedScheduleStatus = "running";
  const projectScheduleStatus = async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    const response = await route.fetch();
    const body = await response.json();
    body.jobs = body.jobs.map((job) => job.job_id === detailPayload.job.job_id ? {
      ...job,
      active: projectedScheduleStatus === "running",
      status: projectedScheduleStatus,
      state: projectedScheduleStatus === "ok" ? {
        last_finished_at_ms: Date.now(), last_status: "ok", last_error: null,
      } : job.state,
    } : job);
    await route.fulfill({ response, json: body });
  };
  await page.route("**/schedule/jobs", projectScheduleStatus);
  await inspectScheduleTrigger.click();
  await scheduleDetail.getByText("Scheduled", { exact: true }).waitFor();
  await page.clock.runFor(5000);
  await scheduleDetail.getByText("Running", { exact: true }).waitFor();
  projectedScheduleStatus = "ok";
  await page.clock.runFor(5000);
  await scheduleDetail.getByText("Succeeded", { exact: true }).waitFor();
  await page.keyboard.press("Escape");
  await expect(scheduleDetail).toHaveCount(0);
  await page.clock.runFor(32);
  await expect(inspectScheduleTrigger).toBeFocused();
  await page.unroute("**/schedule/jobs", projectScheduleStatus);
  await page.clock.resume();

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    const labels = language === "en" ? {
      heading: "Schedule Jobs", create: "Create Job", message: "Message", at: "Run at",
      summary: "Review the highlighted fields.", inspect: "Inspect", delete: "Delete",
    } : {
      heading: "Schedule Job", create: "创建 Job", message: "消息", at: "执行时间",
      summary: "请检查标记出的字段。", inspect: "查看详情", delete: "删除",
    };
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const createTrigger = page.getByRole("button", { name: labels.create, exact: true });
        await expect(page.getByRole("main")).not.toContainText("{{");
        await createTrigger.focus();
        await createTrigger.press("Enter");
        const summary = page.getByRole("alert").filter({ hasText: labels.summary });
        await expect(summary).toBeFocused();
        await summary.getByRole("link", { name: labels.message, exact: true }).focus();
        await page.keyboard.press("Enter");
        await expect(page.getByLabel(labels.message, { exact: true })).toBeFocused();
        await expect(page.getByLabel(labels.at, { exact: true })).toHaveAttribute("aria-invalid", "true");
        for (const target of [createTrigger, browserAtItem.getByRole("button", { name: labels.inspect, exact: true }),
          browserAtItem.getByRole("button", { name: labels.delete, exact: true })]) {
          await target.scrollIntoViewIfNeeded();
          await expect(target).toBeVisible();
          const bounds = await target.boundingBox();
          assert.ok(bounds && bounds.x >= 0 && bounds.y >= 0
            && bounds.x + bounds.width <= viewport.width + 1 && bounds.y + bounds.height <= viewport.height + 1,
          `Schedule action unreachable at ${language}/${theme}/${viewport.width}x${viewport.height}`);
        }
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
          `Schedule horizontal overflow at ${language}/${theme}/${viewport.width}x${viewport.height}`);
        await page.getByRole("heading", { name: labels.heading, exact: true }).scrollIntoViewIfNeeded();
        await page.screenshot({ path: resolve(output, `schedule-jobs-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();

  let notifyOldScheduleLoad;
  let releaseOldScheduleLoad;
  const oldScheduleLoadArrived = new Promise((done) => { notifyOldScheduleLoad = done; });
  const oldScheduleLoadGate = new Promise((done) => { releaseOldScheduleLoad = done; });
  const delayOldScheduleLoad = async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.jobs[0] = { ...body.jobs[0], title: "STALE Schedule page response" };
    notifyOldScheduleLoad();
    await oldScheduleLoadGate;
    await route.fulfill({ response, json: body });
  };
  await page.route("**/schedule/jobs", delayOldScheduleLoad);
  await page.getByRole("button", { name: "Refresh schedule", exact: true }).click();
  await oldScheduleLoadArrived;
  await page.getByRole("link", { name: "Open sessions", exact: true }).click();
  await page.getByRole("heading", { name: "project-one", exact: true }).waitFor();
  const oldScheduleLoadResponse = page.waitForResponse((response) => response.url() === scheduleResponse.url());
  releaseOldScheduleLoad();
  await oldScheduleLoadResponse;
  await expect(page.getByText("STALE Schedule page response", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Schedule Jobs", exact: true })).toHaveCount(0);
  await page.unroute("**/schedule/jobs", delayOldScheduleLoad);
  await page.getByRole("navigation").getByRole("link", { name: "Projects", exact: true }).click();
  await firstProjectItem.getByRole("link", { name: "Open schedule" }).click();
  await page.getByRole("heading", { name: browserAtTitle, exact: true }).waitFor();

  let notifyDisconnectedLoad;
  let releaseDisconnectedLoad;
  let heldDisconnectedLoad = false;
  const disconnectedLoadArrived = new Promise((done) => { notifyDisconnectedLoad = done; });
  const disconnectedLoadGate = new Promise((done) => { releaseDisconnectedLoad = done; });
  const delayDisconnectedLoad = async (route) => {
    if (heldDisconnectedLoad || route.request().method() !== "GET") return route.continue();
    heldDisconnectedLoad = true;
    const response = await route.fetch();
    const body = await response.json();
    body.jobs[0] = { ...body.jobs[0], title: "STALE disconnected Schedule response" };
    notifyDisconnectedLoad();
    await disconnectedLoadGate;
    await route.fulfill({ response, json: body });
  };
  await page.route("**/schedule/jobs", delayDisconnectedLoad);
  await page.getByRole("button", { name: "Refresh schedule", exact: true }).click();
  await disconnectedLoadArrived;
  await page.route("**/api/v1/clients", (route) => route.abort());
  await page.evaluate(() => window.__omniTestSocket.close());
  const scheduleDisconnected = page.getByRole("status").filter({ hasText: "Showing the last received Job status." });
  await scheduleDisconnected.waitFor();
  await expect(page.getByRole("button", { name: "Create Job", exact: true })).toBeDisabled();
  await expect(page.getByRole("heading", { name: browserAtTitle, exact: true })).toBeVisible();
  const disconnectedLoadResponse = page.waitForResponse((response) => response.url() === scheduleResponse.url());
  releaseDisconnectedLoad();
  await disconnectedLoadResponse;
  await expect(page.getByText("STALE disconnected Schedule response", { exact: true })).toHaveCount(0);
  await page.unroute("**/api/v1/clients");
  await expect(scheduleDisconnected).toHaveCount(0, { timeout: 10000 });
  await expect(page.getByRole("button", { name: "Create Job", exact: true })).toBeEnabled();
  await page.getByRole("heading", { name: browserAtTitle, exact: true }).waitFor();
  await expect(page.getByText("STALE disconnected Schedule response", { exact: true })).toHaveCount(0);
  await page.unroute("**/schedule/jobs", delayDisconnectedLoad);

  async function deleteBrowserScheduleJob(title) {
    const job = page.getByRole("listitem").filter({ hasText: title });
    await job.getByRole("button", { name: "Delete", exact: true }).click();
    const deleteDialog = page.getByRole("dialog", { name: "Delete this Schedule Job?" });
    const deleteResponse = page.waitForResponse((response) => (
      response.request().method() === "DELETE"
      && /\/schedule\/jobs\/[^/]+$/.test(response.url())
    ));
    await deleteDialog.getByRole("button", { name: "Delete Job", exact: true }).click();
    const response = await deleteResponse;
    assert.equal(response.status(), 200, `${title} Schedule Job deletion failed`);
    await deleteDialog.waitFor({ state: "hidden" });
    await page.getByText("Schedule Job deleted.", { exact: true }).waitFor();
    await expect(job).toHaveCount(0);
  }

  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
  await deleteBrowserScheduleJob(browserAtTitle);
  const deletedScheduleNotice = page.getByRole("status").filter({ hasText: "Schedule Job deleted." });
  await page.clock.runFor(9999);
  await expect(deletedScheduleNotice).toBeVisible();
  await page.clock.runFor(1);
  await expect(deletedScheduleNotice).toHaveCount(0);
  await page.clock.resume();
  await deleteBrowserScheduleJob(browserEveryTitle);
  await deletedScheduleNotice.getByRole("button", { name: "Close", exact: true }).click();
  await expect(deletedScheduleNotice).toHaveCount(0);
  await deleteBrowserScheduleJob(browserCronTitle);
  await deleteBrowserScheduleJob(retryScheduleTitle);
  await expect(page.getByText("E2E browser at message", { exact: true })).toHaveCount(0);
  await expect(scheduleStatus).toContainText("Paused");
  await expect(scheduleStatus).toContainText("Available");
  await page.getByRole("link", { name: "Open sessions", exact: true }).click();
  await page.getByRole("heading", { name: "project-one", exact: true }).waitFor();
  const sessionList = page.getByRole("list", { name: "Conversation Sessions" });
  await sessionList.getByRole("button", { name: /Web available history/ }).click();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  await page.reload();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  assert.equal(await page.getByText("schedule-only content", { exact: true }).count(), 0);

  await sessionList.getByRole("button", { name: /Web restore history/ }).click();
  await page.getByText("Restore branch should disappear from history", { exact: true }).waitFor();
  const restoreTrigger = page.getByRole("button", { name: "Restore", exact: true });
  let releaseInspection;
  let inspectionReceived;
  const inspectionReleased = new Promise((done) => { releaseInspection = done; });
  const inspectionFetched = new Promise((done) => { inspectionReceived = done; });
  const delayInspection = async (route) => {
    const response = await route.fetch();
    inspectionReceived();
    await inspectionReleased;
    await route.fulfill({ response });
  };
  await page.route("**/management/restore/inspect", delayInspection);
  await restoreTrigger.click();
  await page.getByRole("button", { name: "Inspect restore" }).click();
  await inspectionFetched;
  await page.getByRole("button", { name: /Web available history/, includeHidden: true }).evaluate((element) => element.click());
  await page.getByRole("log", { includeHidden: true }).getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  releaseInspection();
  await page.getByRole("dialog").waitFor({ state: "hidden" });
  assert.equal(await page.getByText("Restore preview", { exact: true }).count(), 0);
  await page.unroute("**/management/restore/inspect", delayInspection);
  await sessionList.getByRole("button", { name: /Web restore history/ }).click();
  await page.getByText("Restore branch should disappear from history", { exact: true }).waitFor();
  for (const closeWithEscape of [true, false]) {
    await restoreTrigger.click();
    await page.getByRole("button", { name: "Inspect restore" }).click();
    await page.getByText("Restore preview", { exact: true }).waitFor();
    const cancelled = page.waitForResponse((response) => response.url().endsWith("/management/restore/cancel"));
    if (closeWithEscape) await page.keyboard.press("Escape");
    else await page.getByRole("dialog").getByRole("button", { name: "Cancel" }).click();
    assert.equal((await cancelled).status(), 200);
    await page.getByRole("dialog").waitFor({ state: "hidden" });
    await expect(restoreTrigger).toBeFocused();
  }
  await restoreTrigger.click();
  await page.locator("#restore-anchor-select").selectOption("1");
  await page.getByRole("button", { name: "Inspect restore" }).click();
  await page.getByText("Restore preview", { exact: true }).waitFor();
  await writeFile(resolve(control.details.restore_target), "changed by another Session\n", "utf8");
  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
  await page.getByRole("button", { name: "Restore session" }).click();
  const restoreNotice = page.getByRole("status").filter({ hasText: "Restore completed" });
  await restoreNotice.waitFor();
  await restoreNotice.getByText(control.details.restore_target, { exact: true }).waitFor();
  assert.equal(
    (await readFile(resolve(control.details.restore_target), "utf8")).replaceAll("\r\n", "\n"),
    "content before Restore\n",
  );
  await expect(page.locator("#sessions-heading")).toBeFocused();
  assert.equal(await page.getByText("Restore branch should disappear from history", { exact: true }).count(), 0);
  await page.clock.runFor(9999);
  assert.equal(await restoreNotice.count(), 1, "Restore feedback expired before 10 seconds");
  await page.clock.runFor(1);
  await restoreNotice.waitFor({ state: "hidden" });
  await page.clock.resume();
  const refreshResult = page.waitForResponse((response) => response.url().endsWith("/management/restore/result"));
  await page.reload();
  assert.equal((await refreshResult).status(), 200);
  await restoreNotice.waitFor();
  await restoreNotice.getByRole("button", { name: "Close" }).click();

  await sessionList.getByRole("button", { name: /Web manual restore history/ }).click();
  await page.getByText("Manual Restore branch should disappear from history", { exact: true }).waitFor();
  const manualRestoreTrigger = page.getByRole("button", { name: "Restore", exact: true });
  await manualRestoreTrigger.click();
  await page.getByRole("button", { name: "Inspect restore" }).click();
  await page.getByText("Restore preview", { exact: true }).waitFor();
  await writeFile(resolve(control.details.manual_restore_target), "changed by another Session\n", "utf8");
  let releaseExecution;
  let executionReceived;
  const executionReleased = new Promise((done) => { releaseExecution = done; });
  const executionFetched = new Promise((done) => { executionReceived = done; });
  const delayExecution = async (route) => {
    const response = await route.fetch();
    executionReceived();
    await executionReleased;
    await route.fulfill({ response });
  };
  await page.route("**/management/restore/execute", delayExecution);
  await page.getByRole("button", { name: "Restore session" }).click();
  await executionFetched;
  await page.getByRole("button", { name: /Web available history/, includeHidden: true }).evaluate((element) => element.click());
  await page.getByRole("log", { includeHidden: true }).getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  releaseExecution();
  await page.getByRole("dialog").waitFor({ state: "hidden" });
  assert.equal(await page.getByRole("status").filter({ hasText: "Restore completed" }).count(), 0);
  await page.unroute("**/management/restore/execute", delayExecution);
  await sessionList.getByRole("button", { name: /Web manual restore history/ }).click();
  const manualRestoreNotice = page.getByRole("status").filter({ hasText: "Restore completed" });
  await manualRestoreNotice.waitFor();
  assert.equal(
    (await readFile(resolve(control.details.manual_restore_target), "utf8")).replaceAll("\r\n", "\n"),
    "content before Restore\n",
  );
  await manualRestoreNotice.getByRole("button", { name: "Close" }).click();
  await manualRestoreNotice.waitFor({ state: "hidden" });

  await sessionList.getByRole("button", { name: /Web failed restore history/ }).click();
  await page.getByText("Failed Restore branch", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Restore", exact: true }).click();
  await page.getByRole("button", { name: "Inspect restore" }).click();
  await page.getByText("Restore preview", { exact: true }).waitFor();
  await unlink(control.details.failure_restore_target);
  await mkdir(control.details.failure_restore_target);
  await page.getByRole("button", { name: "Restore session" }).click();
  const failedRestoreNotice = page.getByRole("status").filter({ hasText: "Restore completed" });
  await failedRestoreNotice.getByText("Failed", { exact: true }).waitFor();
  await failedRestoreNotice.getByText(control.details.failure_restore_target, { exact: true }).waitFor();
  await failedRestoreNotice.getByRole("button", { name: "Close" }).click();
  await page.getByRole("button", { name: "Review restore failure" }).click();
  await failedRestoreNotice.getByRole("button", { name: "Acknowledge" }).waitFor();
  const failureRefreshResult = page.waitForResponse((response) => response.url().endsWith("/management/restore/result"));
  await page.reload();
  assert.equal((await failureRefreshResult).status(), 200);
  await failedRestoreNotice.getByRole("button", { name: "Acknowledge" }).waitFor();
  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
  await page.clock.runFor(10_000);
  await failedRestoreNotice.waitFor({ state: "hidden" });
  await page.clock.resume();
  await page.getByRole("button", { name: "Review restore failure" }).click();
  await failedRestoreNotice.getByRole("button", { name: "Acknowledge" }).click();
  await page.getByRole("button", { name: "Review restore failure" }).waitFor({ state: "hidden" });
  const durableRestore = JSON.parse(await readFile(resolve(firstProject, ".omni", "restore", control.details.failure_restore_session_id, "pending.json"), "utf8"));
  assert.equal(durableRestore.failure_notification_acknowledged, true);

  const draftResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "POST"
    && response.url().includes("/api/v1/projects/")
    && response.url().endsWith("/sessions")
  ));
  await page.getByRole("button", { name: "New session" }).click();
  const draftResponse = await draftResponsePromise;
  const draftId = (await draftResponse.json()).session_id;
  assert.equal(typeof draftId, "string");
  const sessionPanel = page.locator('aside[aria-label="Conversation Sessions"]');
  await sessionPanel.getByText("Empty draft", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Release session" }).click();
  await sessionPanel.getByText("Empty draft", { exact: true }).waitFor({ state: "detached" });
  const sessionFiles = await readdir(resolve(firstProject, ".omni", "sessions"));
  assert.equal(sessionFiles.includes(`${draftId}.jsonl`), false, "Released empty draft was persisted");

  await sessionList.getByRole("button", { name: /Web available history/ }).click();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  await sessionList.getByRole("button", { name: /CLI occupied history/ }).click();
  await page.getByRole("alert").filter({ hasText: "occupied" }).waitFor();
  assert.equal(await page.getByText("CLI-only history must remain private", { exact: true }).count(), 0);

  await sessionList.getByRole("button", { name: /Web available history/ }).click();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();

  const composer = page.getByLabel("Message input");
  await composer.fill("tool states");
  await composer.press("Enter");
  await waitForRecordedEvent((messages) => messages.some((event) => (
    event.type === "input.accepted" && event.payload?.text === "tool states"
  )), "Tool Run acceptance");
  const toolGroup = page.locator("article[data-run-id] details").filter({ hasText: "Tool activity" }).first();
  await toolGroup.locator("summary").first().waitFor();
  assert.equal(await toolGroup.getAttribute("open"), null, "Tool activity should default to collapsed");
  await toolGroup.locator("summary").first().click();
  await toolGroup.getByText("Completed", { exact: true }).waitFor();
  await toolGroup.getByText("Failed", { exact: true }).waitFor();
  await toolGroup.getByText("Rejected", { exact: true }).waitFor();
  await toolGroup.getByText("Running", { exact: true }).waitFor();
  const beforeToolRefresh = await page.evaluate(() => window.__omniTestMessages);
  await page.reload();
  await expect(page.getByRole("log").getByText("tool states", { exact: true })).toHaveCount(1);
  await page.evaluate((messages) => {
    window.__omniTestMessages = [...messages, ...window.__omniTestMessages];
  }, beforeToolRefresh);
  await toolGroup.locator("summary").first().click();
  for (const status of ["Completed", "Failed", "Rejected", "Running"]) {
    await toolGroup.getByText(status, { exact: true }).waitFor();
  }

  const newSessionResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "POST"
    && response.url().includes("/api/v1/projects/")
    && response.url().endsWith("/sessions")
  ));
  await page.getByRole("button", { name: "New session" }).click();
  const newSession = await (await newSessionResponsePromise).json();
  const conversationSessionId = newSession.session_id;
  const conversationWorkspaceId = newSession.workspace_id;
  assert.equal(typeof conversationSessionId, "string");
  assert.equal(typeof conversationWorkspaceId, "string");
  await page.getByText("Empty draft", { exact: true }).waitFor();
  const managementTrigger = page.getByRole("button", { name: "Runtime status and controls", exact: true });
  await managementTrigger.click();
  const managementDialog = page.getByRole("dialog", { name: "Runtime status", exact: true });
  await managementDialog.getByText("primary/small-model", { exact: true }).waitFor();
  await managementDialog.getByText("Next request context", { exact: true }).waitFor();
  await managementDialog.getByText("Client permission", { exact: true }).waitFor();
  await managementDialog.getByText("Active work", { exact: true }).waitFor();
  const permissionControl = managementDialog.getByLabel("Tool permission level");
  await permissionControl.selectOption("read-only");
  await permissionControl.locator("xpath=..")
    .getByRole("button", { name: "Save", exact: true }).click();
  await managementDialog.getByRole("status").getByText("Client permission updated.", { exact: true }).waitFor();
  const effortControl = managementDialog.getByLabel("Chat reasoning effort");
  await effortControl.selectOption("high");
  await effortControl.locator("xpath=..")
    .getByRole("button", { name: "Save", exact: true }).click();
  await managementDialog.getByRole("status").getByText("Reasoning effort updated.", { exact: true }).waitFor();
  await managementDialog.getByRole("button", { name: "View Memory", exact: true }).click();
  const memoryRegion = managementDialog.getByRole("region", { name: "Long-term Memory", exact: true });
  await memoryRegion.getByRole("heading", { level: 4, name: "Long-term Memory", exact: true }).waitFor();
  await managementDialog.getByRole("button", { name: "Run Dream", exact: true }).click();
  await managementDialog.getByRole("region", { name: "Dream", exact: true })
    .getByRole("status").getByText("No pending summaries.", { exact: true }).waitFor();
  let releaseDream;
  const dreamGate = new Promise((resolveGate) => { releaseDream = resolveGate; });
  let dreamArrived;
  const dreamArrival = new Promise((resolveGate) => { dreamArrived = resolveGate; });
  const delayDreamResponse = async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    dreamArrived();
    await dreamGate;
    await route.fulfill({
      status: response.status(), contentType: "application/json",
      body: JSON.stringify(body),
    });
  };
  await page.route("**/management/dream", delayDreamResponse);
  await managementDialog.getByRole("button", { name: "Run Dream", exact: true }).click();
  await dreamArrival;
  await managementDialog.press("Escape");
  await expect(managementTrigger).toBeFocused();
  await managementTrigger.press("Enter");
  await expect(managementDialog.getByRole("button", { name: "Running Dream...", exact: true })).toBeDisabled();
  releaseDream();
  await managementDialog.getByRole("region", { name: "Dream", exact: true })
    .getByRole("status").getByText("No pending summaries.", { exact: true }).waitFor();
  await page.unroute("**/management/dream", delayDreamResponse);
  for (const [code, message, notice] of [
    ["memory_task_running", "A Memory Task is already running.", "Dream is already running for this Workspace."],
    ["persistence_error", "Long-term Memory could not be written.", "Dream failed."],
  ]) {
    const failDream = async (route) => {
      const response = await route.fetch();
      const body = await response.json();
      body.result.dream_result.error = { code, message, retryable: false, retry_after_seconds: null };
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    };
    await page.route("**/management/dream", failDream);
    await managementDialog.getByRole("button", { name: "Run Dream", exact: true }).click();
    const result = managementDialog.getByRole("region", { name: "Dream", exact: true }).getByRole("status");
    await result.getByText(notice, { exact: true }).waitFor();
    await result.getByText(`${code}: ${message}`, { exact: true }).waitFor();
    await page.unroute("**/management/dream", failDream);
  }
  await managementDialog.getByRole("button", { name: "Run Dream", exact: true }).click();
  await managementDialog.getByRole("region", { name: "Dream", exact: true })
    .getByRole("status").getByText("No pending summaries.", { exact: true }).waitFor();
  await managementDialog.getByRole("button", { name: "Reload Skills", exact: true }).click();
  await managementDialog.getByRole("status").getByText("Skills reloaded: 0.", { exact: true }).waitFor();
  await managementDialog.getByRole("button", { name: "Close", exact: true }).click();
  await managementDialog.waitFor({ state: "hidden" });
  await expect(managementTrigger).toBeFocused();

  await managementTrigger.press("Enter");
  await expect(permissionControl).toHaveValue("read-only");
  const failSkillReload = async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    await route.fulfill({
      status: response.status(),
      contentType: "application/json",
      body: JSON.stringify({
        ...body,
        result: {
          ...body.result,
          skill_metadata: null,
          management_error: {
            code: "skill_reload_failed",
            message: "Skill reload failed.",
            retryable: false,
            retry_after_seconds: null,
          },
        },
      }),
    });
  };
  await page.route("**/management/skills/reload", failSkillReload);
  await managementDialog.getByRole("button", { name: "Reload Skills", exact: true }).click();
  await managementDialog.getByRole("alert").getByText("Skills could not be reloaded.", { exact: true }).waitFor();
  await page.unroute("**/management/skills/reload", failSkillReload);

  let releaseOldMemory;
  const oldMemoryGate = new Promise((resolveGate) => { releaseOldMemory = resolveGate; });
  let oldMemoryArrived;
  const oldMemoryArrival = new Promise((resolveGate) => { oldMemoryArrived = resolveGate; });
  let delayFirstMemory = true;
  const delayFirstMemoryResponse = async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    if (delayFirstMemory) {
      delayFirstMemory = false;
      oldMemoryArrived();
      await oldMemoryGate;
    }
    await route.fulfill({
      status: response.status(),
      contentType: "application/json",
      body: JSON.stringify({
        ...body,
        result: { ...body.result, memory_content: "STALE_MEMORY_RESPONSE" },
      }),
    });
  };
  await managementDialog.press("Escape");
  await expect(managementTrigger).toBeFocused();
  await page.route("**/management/memory", delayFirstMemoryResponse);
  await managementTrigger.click();
  await expect(permissionControl).toHaveValue("read-only");
  await managementDialog.getByRole("button", { name: "View Memory", exact: true }).click();
  await oldMemoryArrival;
  await managementDialog.press("Escape");
  await expect(managementTrigger).toBeFocused();
  await managementTrigger.click();
  await expect(permissionControl).toHaveValue("read-only");
  releaseOldMemory();
  await page.unroute("**/management/memory", delayFirstMemoryResponse);
  await expect(managementDialog.getByText("STALE_MEMORY_RESPONSE", { exact: true })).toHaveCount(0);

  for (const [action, label, value] of [
    ["permission", "Tool permission level", "full-access"],
    ["effort", "Chat reasoning effort", "max"],
  ]) {
    const actionUrl = `**/management/${action}`;
    await page.route(actionUrl, (route) => route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ request_id: "failed", result: {
        handled: true, output: "config_invalid: invalid selection",
        published_permission_level: null, published_effort: null,
      } }),
    }));
    const select = managementDialog.getByLabel(label);
    await select.selectOption(value);
    const save = select.locator("xpath=..").getByRole("button", { name: "Save", exact: true });
    await save.focus();
    await save.press("Enter");
    await managementDialog.getByRole("alert").waitFor();
    await expect(select).toHaveValue(value);
    await expect(managementDialog.getByText(/updated\./)).toHaveCount(0);
    await page.unroute(actionUrl);
  }
  await managementDialog.press("Escape");
  await expect(managementTrigger).toBeFocused();

  let releaseOldSave;
  const oldSaveGate = new Promise((resolveGate) => { releaseOldSave = resolveGate; });
  let oldSaveArrived;
  const oldSaveArrival = new Promise((resolveGate) => { oldSaveArrived = resolveGate; });
  const delayManagementSave = async (route) => {
    const response = await route.fetch();
    oldSaveArrived();
    await oldSaveGate;
    await route.fulfill({ response });
  };
  await page.route("**/management/permission", delayManagementSave);
  await managementTrigger.click();
  await expect(permissionControl).toHaveValue("read-only");
  await permissionControl.selectOption("full-access");
  await permissionControl.locator("xpath=..").getByRole("button", { name: "Save", exact: true }).click();
  await oldSaveArrival;
  await managementDialog.press("Escape");
  await managementTrigger.click();
  await expect(permissionControl).toHaveValue("full-access");
  await permissionControl.selectOption("workspace-write");
  releaseOldSave();
  await page.unroute("**/management/permission", delayManagementSave);
  await expect(permissionControl).toHaveValue("workspace-write");
  await expect(managementDialog.getByText("Client permission updated.", { exact: true })).toHaveCount(0);
  await permissionControl.locator("xpath=..").getByRole("button", { name: "Save", exact: true }).click();
  await managementDialog.getByText("Client permission updated.", { exact: true }).waitFor();
  await managementDialog.press("Escape");

  const memoryPath = resolve(firstProject, ".omni", "memory", "memory.md");
  const originalMemory = await readFile(memoryPath, "utf8");
  const longMemory = `# Inspected memory\n<script>window.__unsafeMemory = true</script>\n${"unbroken-memory".repeat(1500)}\n`;
  await writeFile(memoryPath, longMemory, "utf8");
  const skillRoot = resolve(control.details.home_root, ".omni", "skills");
  for (const [directory, document] of [
    ["web-review", "---\nname: web-review\ndescription: Browser reload metadata\n---\nPrivate instructions excluded from metadata.\n"],
    ["invalid", "---\nname: INVALID\ndescription: PRIVATE_BAD_SKILL_SECRET\n---\nPrivate bad document.\n"],
  ]) {
    await mkdir(resolve(skillRoot, directory), { recursive: true });
    await writeFile(resolve(skillRoot, directory, "SKILL.md"), document, "utf8");
  }
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of [...viewports, { width: 375, height: 812 }, { width: 812, height: 375 }]) {
        await page.setViewportSize(viewport);
        const trigger = page.getByRole("button", { name: language === "en" ? "Runtime status and controls" : "运行状态与控制", exact: true });
        await trigger.focus();
        await trigger.press("Enter");
        const dialog = page.getByRole("dialog", { name: language === "en" ? "Runtime status" : "运行状态", exact: true });
        await dialog.getByText("primary/small-model", { exact: true }).waitFor();
        const bounds = await dialog.boundingBox();
        assert.ok(bounds && bounds.x >= 0 && bounds.y >= 0
          && bounds.x + bounds.width <= viewport.width + 1
          && bounds.y + bounds.height <= viewport.height + 1, "Runtime dialog escaped viewport");
        const select = dialog.getByLabel(language === "en" ? "Chat reasoning effort" : "聊天推理强度");
        await select.focus();
        await expect(select).toBeFocused();
        for (const [label, result] of language === "en" ? [
          ["View Memory", "Long-term Memory loaded."],
          ["Run Dream", "No pending summaries."],
          ["Reload Skills", "Skills reloaded: 1."],
        ] : [
          ["查看记忆", "长期记忆已加载。"],
          ["运行 Dream", "没有待处理的摘要。"],
          ["重新加载 Skills", "Skills 已重新加载：1 个。"],
        ]) {
          const button = dialog.getByRole("button", { name: label, exact: true });
          await button.focus();
          await expect(button).toBeFocused();
          await button.press("Enter");
          await dialog.getByRole("status").getByText(result, { exact: true }).first().waitFor();
        }
        const memory = dialog.getByRole("region", { name: language === "en" ? "Long-term Memory" : "长期记忆", exact: true }).locator("pre");
        await expect(memory).toHaveText(longMemory);
        assert.equal(await page.evaluate(() => window.__unsafeMemory), undefined, "Memory executed HTML");
        await dialog.getByText("Browser reload metadata", { exact: true }).waitFor();
        await expect(dialog.getByText("PRIVATE_BAD_SKILL_SECRET", { exact: true })).toHaveCount(0);
        await dialog.evaluate((element) => {
          if (element.scrollWidth > element.clientWidth + 1) throw new Error("Runtime content overflows horizontally");
        });
        await page.screenshot({ path: resolve(output, `runtime-${language}-${theme}-${viewport.width}x${viewport.height}.png`) });
        await dialog.press("Escape");
        await expect(trigger).toBeFocused();
      }
    }
  }
  await writeFile(memoryPath, originalMemory, "utf8");
  await rm(resolve(skillRoot, "web-review"), { recursive: true });
  await rm(resolve(skillRoot, "invalid"), { recursive: true });
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.setViewportSize(viewports.at(-1));
  await control.command("settings-arm");
  await page.getByLabel("Message input").fill("recovery streaming markdown");
  await page.getByLabel("Message input").press("Shift+Enter");
  await page.getByLabel("Message input").type("second line");
  const multilinePrompt = "recovery streaming markdown\nsecond line";
  assert.equal(await page.getByLabel("Message input").inputValue(), multilinePrompt);
  await page.getByLabel("Message input").evaluate((element) => {
    element.dispatchEvent(new element.ownerDocument.defaultView.KeyboardEvent("keydown", {
      key: "Enter", bubbles: true, isComposing: true,
    }));
  });
  assert.equal(await page.getByLabel("Message input").inputValue(), multilinePrompt, "IME Enter submitted the prompt");
  await page.getByLabel("Message input").press("Enter");
  await page.getByText("Streamed answer", { exact: true }).waitFor();
  assert.equal(await page.getByText("The response arrived in multiple chunks.", { exact: true }).count(), 0,
    "The complete answer appeared before its first streamed frame was observed");
  await control.command("settings-wait");
  const beforeStreamRefresh = await page.evaluate(() => window.__omniTestMessages);
  await page.reload();
  await expect(page.getByRole("log").getByText(multilinePrompt, { exact: true })).toHaveCount(1);
  await expect(page.getByRole("heading", { name: "Streamed answer", exact: true })).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Cancel run", exact: true })).toBeEnabled();
  await page.evaluate((messages) => {
    window.__omniTestMessages = [...messages, ...window.__omniTestMessages];
  }, beforeStreamRefresh);
  await sessionList.getByRole("button", { name: /Web available history/ }).click();
  await expect(page.getByRole("log").getByText("tool states", { exact: true })).toHaveCount(1);
  const backgroundDraft = page.getByRole("button", { name: /New Session draft/ });
  await backgroundDraft.getByText("Running", { exact: true }).waitFor();
  let releaseLateClaim;
  const lateClaimGate = new Promise((resolveGate) => { releaseLateClaim = resolveGate; });
  let lateClaimArrived;
  const lateClaimArrival = new Promise((resolveArrival) => { lateClaimArrived = resolveArrival; });
  const lateClaimRoute = `**/sessions/${conversationSessionId}/claim`;
  await page.route(lateClaimRoute, async (route) => {
    const response = await route.fetch();
    lateClaimArrived();
    await lateClaimGate;
    await route.fulfill({ response });
  });
  await backgroundDraft.click();
  await lateClaimArrival;
  await control.command("settings-release");
  await waitForRecordedEvent((messages) => messages.some((event) => event.type === "run.completed"
    && event.session_id === conversationSessionId), "Run completion while Claim response is held");
  releaseLateClaim();
  await page.unroute(lateClaimRoute);
  await page.getByText("Persisted Markdown", { exact: true }).waitFor();
  await expect(page.getByRole("log").getByText(multilinePrompt, { exact: true })).toHaveCount(1);
  await expect(page.getByRole("heading", { name: "Streamed answer", exact: true })).toHaveCount(1);
  await waitForRecordedEvent((messages) => messages.some((event) => (
    event.type === "run.completed" && messages.some((accepted) => (
      accepted.type === "input.accepted" && accepted.payload?.text === multilinePrompt && accepted.run_id === event.run_id
    ))
  )), "multiline Run completion");
  const acceptedConversation = await page.evaluate((prompt) => window.__omniTestMessages.filter((event) => (
    event.type === "input.accepted" && event.payload?.text === prompt
  )), multilinePrompt);
  assert.equal(acceptedConversation.length, 1, "Multiline prompt was accepted more than once");
  const conversationRunId = acceptedConversation[0].run_id;
  const frames = await page.evaluate((runId) => window.__omniTestMessages.filter((event) => (
    event.type === "run.output" && event.run_id === runId
    && event.payload?.message?.metadata?._stream_delta === true
  )), conversationRunId);
  assert.ok(frames.length >= 3, `Expected progressive Markdown frames, received ${frames.length}`);
  assert.equal(await page.locator('img[src="https://example.com/remote.png"]').count(), 0,
    "Remote Markdown image was loaded");
  assert.equal(await page.locator('a[href^="javascript:"]').count(), 0, "Unsafe Markdown link survived rendering");
  const codeBlock = page.locator("pre").filter({ hasText: "x".repeat(100) }).first();
  await expect(codeBlock).toBeVisible();
  await expect.poll(() => codeBlock.evaluate((element) => element.scrollWidth > element.clientWidth), {
    message: "Long code block did not scroll locally",
  }).toBe(true);
  let persistedConversation;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const records = (await readFile(resolve(firstProject, ".omni", "sessions", `${conversationSessionId}.jsonl`), "utf8"))
        .trim().split("\n").map((line) => JSON.parse(line));
      if (records.some((record) => record.role === "assistant" && String(record.content).includes("Persisted Markdown"))) {
        persistedConversation = records;
        break;
      }
    } catch { /* Persistence may still be in progress. */ }
    await delay(50);
  }
  assert.ok(persistedConversation, "Completed Session JSONL was not persisted");
  assert.equal(persistedConversation.filter((record) => record.role === "user" && record.content === multilinePrompt).length, 1);
  await page.getByText("Empty draft", { exact: true }).waitFor({ state: "detached" });

  await page.evaluate(() => {
    const socket = window.__omniTestSocket;
    window.__omniRetrySocket = socket;
    const send = socket.send.bind(socket);
    socket.send = (value) => {
      send(value);
      if (JSON.parse(value).type === "input") socket.close();
    };
  });
  await page.getByLabel("Message input").fill("retry once");
  await page.getByLabel("Message input").press("Enter");
  let reconnected = false;
  for (let attempt = 0; attempt < 200; attempt += 1) {
    reconnected = await page.evaluate(() => window.__omniRetrySocket.readyState === 3
      && window.__omniTestSocket !== window.__omniRetrySocket
      && window.__omniTestSocket.readyState === 1);
    if (reconnected) break;
    await delay(50);
  }
  assert.equal(reconnected, true, "The browser did not reconnect after the input socket closed");
  await waitForRecordedEvent((messages) => messages.some((event) => (
    event.type === "input.accepted" && event.payload?.text === "retry once"
  )), "retried Run acceptance");
  await waitForRecordedEvent((messages) => messages.some((event) => (
    event.type === "run.completed" && messages.some((accepted) => (
      accepted.type === "input.accepted" && accepted.payload?.text === "retry once" && accepted.run_id === event.run_id
    ))
  )), "retried Run completion");
  assert.equal(await page.evaluate(() => new Set(window.__omniTestMessages.filter((event) => (
    event.type === "input.accepted" && event.payload?.text === "retry once"
  )).map((event) => event.run_id)).size), 1, "Reconnect accepted a duplicate Run");

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const input = page.getByLabel(language === "en" ? "Message input" : "消息输入");
        await input.scrollIntoViewIfNeeded();
        const send = page.getByRole("button", { name: language === "en" ? "Send" : "发送" });
        await send.scrollIntoViewIfNeeded();
        const bounds = await page.evaluate(() => {
          const input = document.querySelector("textarea").getBoundingClientRect();
          const send = document.querySelector("form button[type='submit']").getBoundingClientRect();
          return {
            inputWidth: input.width,
            sendWidth: send.width,
            sendBottom: send.bottom,
            width: document.documentElement.scrollWidth,
          };
        });
        assert.ok(bounds.width <= viewport.width, `Conversation overflow at ${viewport.width}x${viewport.height}`);
        assert.ok(bounds.inputWidth > 0 && bounds.sendWidth > 0 && bounds.sendBottom <= viewport.height + 1,
          `Composer unreachable at ${viewport.width}x${viewport.height}`);
        await page.screenshot({ path: resolve(output, `conversation-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }

  await page.getByRole("button", { name: /Web available history/ }).click();
  await page.getByRole("button", { name: /Runtime status and controls|运行状态与控制/, exact: true }).click();
  const activeRuntimeDialog = page.getByRole("dialog", { name: /Runtime status|运行状态/, exact: true });
  await activeRuntimeDialog.getByText(/1 active|1 个运行中/, { exact: true }).waitFor();
  await activeRuntimeDialog.press("Escape");
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const cancel = page.getByRole("button", { name: language === "en" ? "Cancel run" : "取消运行" });
        await cancel.scrollIntoViewIfNeeded();
        const box = await cancel.boundingBox();
        assert.ok(box && box.width > 0 && box.y >= 0 && box.y + box.height <= viewport.height + 1,
          `Cancel unreachable at ${viewport.width}x${viewport.height}`);
        await page.screenshot({ path: resolve(output, `cancel-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("button", { name: "Cancel run" }).click();
  const canceledGroup = page.locator("article[data-run-id] details").filter({ hasText: /Tool activity|工具活动/ }).first();
  await canceledGroup.locator("summary").first().click();
  await canceledGroup.getByText("Canceled", { exact: true }).waitFor();
  await page.screenshot({ path: resolve(output, "canceled-en-dark-768.png") });
  await waitForRecordedEvent((messages) => messages.some((event) => event.type === "run.cancelled"),
    "Tool Run cancellation");
  const acceptedToolRuns = await page.evaluate(() => window.__omniTestMessages.filter((event) => (
    event.type === "input.accepted" && event.payload?.text === "tool states"
  )));
  assert.equal(new Set(acceptedToolRuns.map((event) => event.run_id)).size, 1,
    "Tool prompt was accepted into more than one Run");
  const canceledRunIds = await page.evaluate(() => window.__omniTestMessages
    .filter((event) => event.type === "run.cancelled").map((event) => event.run_id));
  assert.ok(canceledRunIds.includes(acceptedToolRuns[0].run_id), "Cancel did not terminate the selected Run");
  assert.equal(canceledRunIds.includes(conversationRunId), false, "Cancel affected the other Session");
  await page.reload();
  const canceledHistoryTool = page.locator('article[data-role="tool"]')
    .filter({ hasText: "Tool call interrupted because the turn was cancelled." });
  await canceledHistoryTool.waitFor();
  await canceledHistoryTool.locator("summary").first().click();
  await canceledHistoryTool.getByText("Canceled", { exact: true }).waitFor();
  assert.equal(await page.getByRole("log").getByText("tool states", { exact: true }).count(), 1,
    "Reload duplicated the persisted Tool Run prompt");
  await page.getByRole("button", { name: /New session/ }).waitFor();

  const duplicatePage = await page.context().newPage();
  try {
    await duplicatePage.goto(page.url());
    const duplicateResult = await duplicatePage.evaluate(async (sessionId) => {
      const browserSession = await window.fetch("/api/v1/web/session", { credentials: "include" });
      const { csrf_token: csrf } = await browserSession.json();
      const projectId = window.location.pathname.split("/").at(-1);
      const response = await window.fetch(`/api/v1/projects/${projectId}/sessions/${sessionId}/claim`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json", "X-Omni-CSRF": csrf },
        body: JSON.stringify({ request_id: window.crypto.randomUUID() }),
      });
      return { status: response.status, body: await response.text() };
    }, control.details.available_session_id);
    assert.equal(duplicateResult.status, 403, "Copied tab loaded the active Claim");
    assert.equal(duplicateResult.body.includes("Available history loaded after a successful Claim"), false);
  } finally {
    await duplicatePage.close();
  }
  await secondPage.getByRole("link", { name: /Projects|项目/ }).click();
  await secondPage.getByRole("heading", { name: /^(Projects|项目)$/ }).waitFor();
  await secondPage.locator("aside").getByRole("link", { name: "project-one", exact: true }).click();
  await secondPage.getByRole("heading", { name: "project-one", exact: true }).waitFor();
  const secondSessionList = secondPage.getByRole("list", { name: /Conversation Sessions|对话会话/ });
  await secondSessionList.getByRole("button", { name: /CLI occupied history/ }).waitFor();
  await secondSessionList
    .getByRole("button", { name: /CLI occupied history/ })
    .getByText(/Occupied|已占用/, { exact: true })
    .waitFor();
  await secondSessionList.getByRole("button", { name: /Web available history/ }).click();
  await secondPage.getByRole("alert").filter({ hasText: /occupied|占用/ }).waitFor();
  assert.equal(
    await secondPage.getByText("Available history loaded after a successful Claim", { exact: true }).count(),
    0,
  );

  const confirmationPath = control.details.confirmation_path;
  assert.ok(confirmationPath.endsWith("confirmation-outside.txt"));
  const settingsConfirmationRunId = await settingsConfirmationAcceptance({ page, control });
  await page.locator("aside").getByRole("link", { name: "project-one", exact: true }).click();
  const availableHistory = page.getByRole("list", { name: "Conversation Sessions", exact: true })
    .getByRole("button", { name: /Web available history/ });
  let historyOpened = false;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    await availableHistory.click();
    try {
      await page.locator("textarea").waitFor({ timeout: 1000 });
      historyOpened = true;
      break;
    } catch (error) {
      const alerts = await page.getByRole("alert").allTextContents();
      if (!alerts.some((text) => text.includes("local service is closing"))) throw error;
      await delay(500);
    }
  }
  assert.equal(historyOpened, true, "Available history remained blocked by service cleanup");
  const confirmationCombinations = [];
  const confirmationRuns = [];
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    await secondPage.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      await secondPage.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        await secondPage.setViewportSize(viewport);
        confirmationCombinations.push({ language, theme, viewport });
        const primaryTitle = language === "en" ? "Tool Confirmation" : "工具确认";
        const secondaryTitle = primaryTitle;
        const primaryDialog = page.getByRole("dialog", { name: primaryTitle, exact: true });
        const secondaryDialog = secondPage.getByRole("dialog", { name: secondaryTitle, exact: true });
        const input = page.locator("textarea");
        const previousPromptCount = await page.getByRole("log", { includeHidden: true })
          .getByText("confirmation", { exact: true }).count();
        await input.fill("confirmation");
        await input.press("Enter");
        await primaryDialog.waitFor();
        await secondaryDialog.waitFor();
        const originalRequest = await page.evaluate(() => [...window.__omniTestMessages]
          .reverse().find((event) => event.type === "confirmation.requested"));
        const priorMessages = await page.evaluate(() => window.__omniTestMessages);
        if (viewport.width === 1440) {
          await page.reload();
          await expect(primaryDialog).toBeVisible({ timeout: 5000 });
          await page.evaluate((messages) => {
            window.__omniTestMessages = [...messages, ...window.__omniTestMessages];
          }, priorMessages);
        } else if (viewport.width === 1024) {
          await page.evaluate(() => window.__omniTestSocket.close());
          await expect(primaryDialog).toBeHidden();
          await expect(primaryDialog).toBeVisible({ timeout: 5000 });
        } else {
          const beforeSnapshots = await page.evaluate(() => window.__omniTestMessages
            .filter((event) => event.type === "snapshot.required").length);
          await page.evaluate(() => {
            const latest = [...window.__omniTestMessages].reverse().find((event) =>
              typeof event.seq === "number");
            window.__omniTestSocket.dispatchEvent(new globalThis.MessageEvent("message", {
              data: JSON.stringify({ ...latest, type: "test.gap", seq: latest.seq + 2, payload: {} }),
            }));
          });
          await expect.poll(() => page.evaluate(() => window.__omniTestMessages
            .filter((event) => event.type === "snapshot.required").length))
            .toBeGreaterThan(beforeSnapshots);
        }
        {
          const recoveredRequest = await page.evaluate(() => [...window.__omniTestMessages]
            .reverse().find((event) => event.type === "snapshot.required"
              && event.payload?.snapshot?.pending_confirmation)?.payload.snapshot.pending_confirmation);
          assert.equal(recoveredRequest?.payload.token, originalRequest.payload.token);
          assert.deepEqual(recoveredRequest?.payload.request, originalRequest.payload.request);
          await expect(page.getByRole("log", { includeHidden: true }).getByText("confirmation", { exact: true }))
            .toHaveCount(previousPromptCount + 1);
          await expect(page.getByRole("button", { name: /Cancel run|取消运行/, exact: true, includeHidden: true })).toBeVisible();
        }
        assert.equal(await page.getByRole("status").filter({ hasText: /resolved by another client|其他客户端/ }).count(), 0,
          "A previous confirmation notice overlaps the active dialog");

        const primaryText = await primaryDialog.innerText();
        const secondaryText = await secondaryDialog.innerText();
        assert.equal(primaryText, secondaryText, "Clients received different confirmation facts");
        assert.ok(primaryText.includes("read_file"), "Confirmation omitted the exact Tool name");
        assert.ok(primaryText.includes("confirmation-outside.txt"), "Confirmation omitted exact parameters");
        assert.ok(primaryText.includes(control.details.available_session_id), "Confirmation omitted its Session source");
        assert.equal(
          await primaryDialog.getByRole("button", { name: language === "en" ? "Decline" : "拒绝" }).evaluate(
            (element) => element === document.activeElement,
          ),
          true,
          "Confirmation did not place focus on the safe default",
        );
        const bounds = await primaryDialog.boundingBox();
        const layout = await page.evaluate(() => ({
          width: document.documentElement.scrollWidth,
          height: document.documentElement.scrollHeight,
        }));
        assert.ok(bounds && bounds.x >= 0 && bounds.y >= 0
          && bounds.x + bounds.width <= viewport.width + 1
          && bounds.y + bounds.height <= viewport.height + 1,
        `Confirmation dialog escaped the viewport at ${viewport.width}x${viewport.height}`);
        assert.ok(layout.width <= viewport.width, `Confirmation overflow at ${viewport.width}x${viewport.height}`);
        await page.screenshot({ path: resolve(output, `confirmation-${language}-${theme}-${viewport.width}.png`) });

        const combinationIndex = confirmationCombinations.length - 1;
        if (combinationIndex === 0) {
          await secondaryDialog.getByRole("button", { name: language === "en" ? "Approve" : "批准" }).focus();
          await secondPage.keyboard.press("Enter");
        } else if (combinationIndex === 1) {
          await primaryDialog.getByRole("button", { name: language === "en" ? "Close" : "关闭" }).click();
        } else if (combinationIndex === 2) {
          await page.keyboard.press("Escape");
        } else if (combinationIndex === 3) {
          await page.keyboard.press("Enter");
        } else {
          await primaryDialog.getByRole("button", { name: language === "en" ? "Decline" : "拒绝" }).click();
        }
        await primaryDialog.waitFor({ state: "hidden" });
        await secondaryDialog.waitFor({ state: "hidden" });
        const completedRun = await waitForConfirmationRunCompletion();
        const expectedStatus = combinationIndex === 0 ? "success" : "refused";
        const finishedStatuses = await page.evaluate((runId) => window.__omniTestMessages
          .filter((event) => event.type === "run.output" && event.run_id === runId
            && event.payload?.message?.type === "tool_call"
            && event.payload?.message?.metadata?.tool_call_id === "call-confirmation"
            && typeof event.payload?.message?.metadata?.status === "string")
          .map((event) => event.payload.message.metadata.status), completedRun.run_id);
        assert.deepEqual(finishedStatuses, [expectedStatus],
          `Unexpected Tool Gateway result for confirmation Run ${completedRun.run_id}`);
        confirmationRuns.push({
          runId: completedRun.run_id,
          sessionId: completedRun.session_id,
          expectedStatus,
        });
        let composerFocused = false;
        for (let attempt = 0; attempt < 100; attempt += 1) {
          composerFocused = await input.evaluate((element) => element === document.activeElement);
          if (composerFocused) break;
          await delay(50);
        }
        assert.equal(await input.evaluate((element) => element === document.activeElement), true,
          "Confirmation did not restore focus to the triggering input");
      }
    }
  }
  const confirmationToolRuns = await page.evaluate((settingsRunId) => window.__omniTestMessages
    .filter((event) => event.type === "run.output"
      && event.run_id !== settingsRunId
      && event.payload?.message?.type === "tool_call"
      && event.payload?.message?.metadata?.tool_call_id === "call-confirmation")
    .map((event) => event.run_id), settingsConfirmationRunId);
  assert.equal(new Set(confirmationToolRuns).size, confirmationCombinations.length,
    "Confirmation workflow executed more than once for a Run");
  assert.equal(new Set(confirmationRuns.map((run) => run.runId)).size, confirmationCombinations.length);
  let persistedConfirmationResults = [];
  for (let attempt = 0; attempt < 100; attempt += 1) {
    persistedConfirmationResults = [];
    for (const sessionId of new Set(confirmationRuns.map((run) => run.sessionId))) {
      const records = (await readFile(resolve(firstProject, ".omni", "sessions", `${sessionId}.jsonl`), "utf8"))
        .trim().split("\n").map((line) => JSON.parse(line));
      persistedConfirmationResults.push(...records.filter((record) => (
        record.role === "tool" && record.tool_call_id === "call-confirmation"
      )));
    }
    if (persistedConfirmationResults.length === confirmationRuns.length) break;
    await delay(50);
  }
  assert.equal(persistedConfirmationResults.length, confirmationRuns.length,
    "The real service did not persist exactly one Tool result per confirmed Run");
  assert.deepEqual(persistedConfirmationResults.map((result) => result.status),
    confirmationRuns.map((run) => run.expectedStatus));
  assert.equal(persistedConfirmationResults.filter((result) => (
    result.status === "success" && result.content.includes("confirmation fixture content")
  )).length, 1, "The approved exact read did not execute exactly once");
  assert.equal(persistedConfirmationResults.filter((result) => (
    result.status === "refused" && result.content.includes("confirmation fixture content")
  )).length, 0, "Declined confirmations exposed Tool output");

  await page.getByRole("button", { name: "EN", exact: true }).click();
  const releaseResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "POST" && response.url().includes("/release")
  ));
  const releaseButton = page.getByRole("button", { name: /Release session|释放会话/ });
  await releaseButton.click();
  assert.equal((await releaseResponsePromise).status(), 200);
  const refreshResponsePromise = secondPage.waitForResponse((response) => (
    response.request().method() === "GET"
    && response.url().includes("/api/v1/projects/")
    && new URL(response.url()).pathname.endsWith("/sessions")
  ));
  await secondPage.getByRole("button", { name: /Refresh sessions|刷新会话/ }).click();
  assert.equal((await refreshResponsePromise).status(), 200);
  const releasedSession = secondSessionList.getByRole("button", { name: /Web available history/ });
  await releasedSession.getByText(/Occupied|已占用/, { exact: true }).waitFor({ state: "detached" });
  const handoffClaimPromise = secondPage.waitForResponse((response) => (
    response.request().method() === "POST" && response.url().includes("/claim")
  ));
  await releasedSession.click();
  assert.equal((await handoffClaimPromise).status(), 200);
  await secondPage.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  await secondPage.getByRole("button", { name: /Release session|释放会话/ }).click();
  await secondPage.getByRole("button", { name: /Release session|释放会话/ }).waitFor({ state: "detached" });

  const sessionSearch = page.getByLabel("Search by title");
  await sessionSearch.fill("Web available");
  await sessionList.getByRole("button", { name: /Web available history/ }).waitFor();
  await sessionSearch.fill("no matching session");
  await page.getByText("No Sessions match this title.", { exact: true }).waitFor();
  for (const failOldRequest of [false, true]) {
    let releaseOldRequest;
    let oldRequestReceived;
    const released = new Promise((resolveRelease) => { releaseOldRequest = resolveRelease; });
    const received = new Promise((resolveReceived) => { oldRequestReceived = resolveReceived; });
    const oldTitle = failOldRequest ? "stale failure" : "Web available";
    const interceptSearch = async (route) => {
      if (new URL(route.request().url()).searchParams.get("title") !== oldTitle) {
        await route.continue();
        return;
      }
      const response = await route.fetch();
      oldRequestReceived();
      await released;
      if (failOldRequest) await route.fulfill({ status: 503, json: {} });
      else await route.fulfill({ response });
    };
    await page.route("**/api/v1/projects/*/sessions?*", interceptSearch);
    await sessionSearch.fill(oldTitle);
    await received;
    const currentSearchResponse = page.waitForResponse((response) => new URL(response.url()).searchParams.get("title") === "no matching session");
    await sessionSearch.fill("no matching session");
    await currentSearchResponse;
    const oldSearchResponse = page.waitForResponse((response) => new URL(response.url()).searchParams.get("title") === oldTitle);
    releaseOldRequest();
    await oldSearchResponse;
    await page.evaluate(() => new Promise((resolveFrame) => window.requestAnimationFrame(() => window.requestAnimationFrame(resolveFrame))));
    await page.getByText("No Sessions match this title.", { exact: true }).waitFor();
    assert.equal(await page.getByRole("alert").count(), 0, "An obsolete search failure replaced the current results");
    await page.unroute("**/api/v1/projects/*/sessions?*", interceptSearch);
  }
  await sessionSearch.fill("");
  const renameTarget = sessionList.getByRole("button", { name: /Web available history/ });
  await renameTarget.click();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  await sessionSearch.fill("Web available");
  await sessionList.getByRole("button", { name: /Web available history/ }).waitFor();
  const renameButton = page.getByRole("button", { name: "Rename session" });
  const renameDialog = page.getByRole("dialog");
  for (const closeWithEscape of [true, false]) {
    await renameButton.click();
    await renameDialog.getByLabel("Session title").waitFor();
    if (closeWithEscape) await page.keyboard.press("Escape");
    else await renameDialog.getByRole("button", { name: "Cancel" }).click();
    await renameDialog.waitFor({ state: "hidden" });
    assert.equal(await renameButton.evaluate((element) => element === document.activeElement), true,
      "Rename dialog did not return focus to its trigger");
  }
  await renameButton.click();
  await renameDialog.getByLabel("Session title").fill("Renamed available history");
  const interceptRenameConflict = async (route) => {
    if (route.request().method() !== "PATCH") {
      await route.continue();
      return;
    }
    const body = route.request().postDataJSON();
    const concurrent = await route.fetch({
      postData: JSON.stringify({ ...body, request_id: `${body.request_id}-concurrent`, title: "Concurrent renamed history" }),
    });
    assert.equal(concurrent.status(), 200);
    const conflict = await route.fetch();
    assert.equal(conflict.status(), 409);
    await route.fulfill({ response: conflict });
  };
  await page.route("**/api/v1/projects/*/sessions/*", interceptRenameConflict);
  await renameDialog.getByRole("button", { name: "Save" }).click();
  await renameDialog.getByText("This Session changed elsewhere. Reload it and try again.", { exact: true }).waitFor();
  await renameDialog.getByRole("button", { name: "Save" }).waitFor({ state: "visible" });
  await expect(renameDialog.getByRole("button", { name: "Save" })).toBeEnabled();
  assert.equal(await renameDialog.getByLabel("Session title").inputValue(), "Renamed available history",
    "A metadata conflict discarded the user's title");
  await page.unroute("**/api/v1/projects/*/sessions/*", interceptRenameConflict);
  const renameResponse = page.waitForResponse((response) => (
    response.request().method() === "PATCH"
    && response.url().includes("/sessions/")
  ));
  await renameDialog.getByRole("button", { name: "Save" }).click();
  assert.equal((await renameResponse).status(), 200);
  await page.getByRole("heading", { name: "Renamed available history", exact: true }).waitFor();
  await page.getByText("No Sessions match this title.", { exact: true }).waitFor();
  assert.equal(await renameButton.evaluate((element) => element === document.activeElement), true,
    "Saving a title outside the filter lost the selected Session or its trigger focus");
  await renameButton.click();
  await renameDialog.getByLabel("Session title").waitFor();
  assert.equal(await renameDialog.getByLabel("Session title").inputValue(), "Renamed available history");
  await page.keyboard.press("Escape");
  const deleteButton = page.getByRole("button", { name: "Delete session" });
  await deleteButton.click();
  const deleteDialog = page.getByRole("dialog");
  await deleteDialog.getByText("Delete this Session permanently?", { exact: true }).waitFor();
  await deleteDialog.getByText("The conversation JSONL, Session logs, tool Artifacts, and Restore backups will be removed.", { exact: false }).waitFor();
  await page.keyboard.press("Escape");
  await deleteDialog.waitFor({ state: "hidden" });
  assert.equal(await deleteButton.evaluate((element) => element === document.activeElement), true,
    "Delete dialog did not return focus to its trigger");
  let releaseRestoredClaim;
  let restoredClaimReceived;
  const restoredClaimReleased = new Promise((resolveRelease) => { releaseRestoredClaim = resolveRelease; });
  const restoredClaimStarted = new Promise((resolveReceived) => { restoredClaimReceived = resolveReceived; });
  const interceptRestoredClaim = async (route) => {
    const response = await route.fetch();
    restoredClaimReceived();
    await restoredClaimReleased;
    await route.fulfill({ response });
  };
  const restoredClaimUrl = `**/api/v1/projects/*/sessions/${control.details.available_session_id}/claim`;
  await page.route(restoredClaimUrl, interceptRestoredClaim);
  await page.reload();
  await restoredClaimStarted;
  const restoredSearchResponse = page.waitForResponse((response) => new URL(response.url()).searchParams.get("title") === "no matching session");
  await page.getByLabel("Search by title").fill("no matching session");
  await restoredSearchResponse;
  await page.getByText("No Sessions match this title.", { exact: true }).waitFor();
  releaseRestoredClaim();
  await page.getByRole("heading", { name: "Renamed available history", exact: true }).waitFor();
  await page.evaluate(() => new Promise((resolveFrame) => window.requestAnimationFrame(() => window.requestAnimationFrame(resolveFrame))));
  await page.getByText("No Sessions match this title.", { exact: true }).waitFor();
  await page.unroute(restoredClaimUrl, interceptRestoredClaim);
  await page.getByRole("button", { name: /Release session|释放会话/ }).click();

  await page.getByLabel("Search by title").fill("");
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const creation = page.waitForResponse((response) => response.request().method() === "POST"
          && /\/projects\/[^/]+\/sessions$/.test(new URL(response.url()).pathname));
        await page.getByRole("button", { name: language === "en" ? "New session" : "新建会话" }).click();
        const { session_id: deleteId } = await (await creation).json();
        await page.getByText(language === "en" ? "Empty draft" : "空白草稿", { exact: true }).waitFor();
        await page.getByRole("button", { name: language === "en" ? "Release session" : "释放会话", exact: true }).waitFor();
        const prompt = `retry once delete review ${language} ${theme} ${viewport.width}`;
        await page.getByLabel(language === "en" ? "Message input" : "消息输入").fill(prompt);
        await page.getByLabel(language === "en" ? "Message input" : "消息输入").press("Enter");
        await waitForRecordedEvent((messages) => messages.some((completed) => completed.type === "run.completed"
          && messages.some((accepted) => accepted.type === "input.accepted"
            && accepted.payload?.text === prompt && accepted.run_id === completed.run_id)), "deletion fixture completion");
        const deleteLabel = language === "en" ? "Delete session" : "删除会话";
        const deleteDialog = page.getByRole("dialog", { name: language === "en"
          ? "Delete this Session permanently?" : "永久删除此会话？" });
        const deleteTrigger = page.getByRole("button", { name: deleteLabel, exact: true });
        await deleteTrigger.waitFor();
        const extendedCase = language === "en" && theme === "light"
          && [1440, 768].includes(viewport.width);
        if (extendedCase) {
          await page.getByRole("button", { name: "Rename session", exact: true }).click();
          const titleDialog = page.getByRole("dialog");
          await titleDialog.getByLabel("Session title").fill("A".repeat(128));
          await titleDialog.getByRole("button", { name: "Save", exact: true }).click();
          await page.getByRole("heading", { name: "A".repeat(60), exact: true }).waitFor();
        }
        const ownedRoot = resolve(firstProject, ".omni");
        const artifactRoot = resolve(ownedRoot, "artifacts", deleteId);
        const restoreRoot = resolve(ownedRoot, "restore", deleteId);
        await mkdir(artifactRoot, { recursive: true });
        await mkdir(restoreRoot, { recursive: true });
        const protectedFile = resolve(firstProject, `delete-protected-${deleteId}.txt`);
        await writeFile(protectedFile, "user data survives", "utf8");
        const unsafeArtifact = resolve(artifactRoot, "linked.txt");
        await writeFile(resolve(restoreRoot, "backup.bin"), "owned backup", "utf8");
        const deleteUrl = `**/api/v1/projects/*/sessions/${deleteId}`;
        const requestIds = [];
        let mode = "conflict";
        let releaseLostResponse;
        let notifyLostResponse;
        const lostResponseReady = new Promise((resolveReady) => { notifyLostResponse = resolveReady; });
        const lostResponseRelease = new Promise((resolveRelease) => { releaseLostResponse = resolveRelease; });
        await page.route(deleteUrl, async (route) => {
          if (route.request().method() !== "DELETE") return route.continue();
          requestIds.push(route.request().postDataJSON().request_id);
          if (mode === "conflict") {
            const headers = { ...route.request().headers(), "x-omni-claim": "invalid-claim" };
            const rejected = await route.fetch({ headers });
            assert.equal(rejected.status(), 409, "Invalid Claim deletion did not return a conflict");
            mode = "failure";
            return route.fulfill({ response: rejected });
          }
          if (mode === "lost-response") {
            const success = await route.fetch();
            assert.equal(success.status(), 200);
            mode = "retry";
            notifyLostResponse();
            await lostResponseRelease;
            return route.abort("failed");
          }
          return route.continue();
        });
        await deleteTrigger.click();
        assert.equal(requestIds.length, 0, "Opening confirmation issued DELETE");
        await page.keyboard.press("Escape");
        await deleteDialog.waitFor({ state: "hidden" });
        assert.equal(await deleteTrigger.evaluate((element) => element === document.activeElement), true);
        await deleteTrigger.click();
        if (extendedCase) {
          await page.setViewportSize({ width: 375, height: 812 });
          const mobileBounds = await deleteDialog.boundingBox();
          assert.ok(mobileBounds && mobileBounds.x >= 0 && mobileBounds.y >= 0
            && mobileBounds.x + mobileBounds.width <= 376 && mobileBounds.y + mobileBounds.height <= 813,
          "Mobile deletion dialog is outside the viewport");
          assert.equal(await deleteDialog.evaluate((element) => element.scrollWidth <= element.clientWidth), true,
            "Long Session title overflows the delete dialog");
          await page.screenshot({ path: resolve(output, "delete-en-light-375.png") });
          await page.setViewportSize({ width: 812, height: 375 });
          await deleteDialog.getByRole("button", { name: deleteLabel, exact: true }).scrollIntoViewIfNeeded();
          const landscapeBounds = await deleteDialog.boundingBox();
          assert.ok(landscapeBounds && landscapeBounds.y >= 0 && landscapeBounds.y + landscapeBounds.height <= 376,
            "Landscape deletion dialog is outside the viewport");
          await page.screenshot({ path: resolve(output, "delete-en-light-812-landscape.png") });
          await page.setViewportSize(viewport);
        }
        await deleteDialog.getByRole("button", { name: deleteLabel, exact: true }).click();
        await deleteDialog.getByRole("alert").waitFor();
        assert.equal(await readFile(protectedFile, "utf8"), "user data survives");
        await deleteDialog.getByRole("button", { name: language === "en" ? "Cancel" : "取消", exact: true }).click();
        await deleteDialog.waitFor({ state: "hidden" });
        assert.equal(await deleteTrigger.evaluate((element) => element === document.activeElement), true);
        await expect(page.getByRole("button", { name: language === "en" ? "New session" : "新建会话" })).toBeEnabled();
        await deleteTrigger.click();
        await link(protectedFile, unsafeArtifact);
        await deleteDialog.getByRole("button", { name: deleteLabel, exact: true }).click();
        await deleteDialog.getByText(language === "en"
          ? "Session deletion did not finish. Retry to continue cleanup."
          : "会话删除尚未完成，请重试以继续清理。", { exact: true }).waitFor();
        let otherPrompt = null;
        let otherSessionId = null;
        if (extendedCase) {
          await deleteDialog.getByRole("button", { name: "Cancel", exact: true }).click();
          await deleteDialog.waitFor({ state: "hidden" });
          const otherSessionCreation = page.waitForResponse((response) => (
            response.request().method() === "POST"
            && /\/api\/v1\/projects\/[^/]+\/sessions$/.test(new URL(response.url()).pathname)
          ));
          await page.getByRole("button", { name: "New session", exact: true }).click();
          otherSessionId = (await (await otherSessionCreation).json()).session_id;
          await page.getByRole("heading", { name: "New Session draft", exact: true }).waitFor();
          otherPrompt = "retry once another session stays selected";
          await page.getByLabel("Message input").fill(otherPrompt);
          await page.getByLabel("Message input").press("Enter");
          await waitForRecordedEvent((messages) => messages.some((completed) => completed.type === "run.completed"
            && messages.some((accepted) => accepted.type === "input.accepted"
              && accepted.payload?.text === otherPrompt && accepted.run_id === completed.run_id)), "unrelated Session completion");
        }
        let releaseRestoredSessionClaim;
        let notifyRestoredSessionClaim;
        let notifyRestoredSessionClaimResponse;
        const restoredSessionClaimStarted = new Promise((resolveStarted) => {
          notifyRestoredSessionClaim = resolveStarted;
        });
        const restoredSessionClaimReleased = new Promise((resolveReleased) => {
          releaseRestoredSessionClaim = resolveReleased;
        });
        const restoredSessionClaimResponse = new Promise((resolveResponse) => {
          notifyRestoredSessionClaimResponse = resolveResponse;
        });
        const delayRestoredSessionClaim = async (route) => {
          const response = await route.fetch();
          const pathname = new URL(route.request().url()).pathname;
          if (otherSessionId !== null && pathname.endsWith(`/sessions/${otherSessionId}/claim`)) {
            notifyRestoredSessionClaim();
            await restoredSessionClaimReleased;
          }
          await route.fulfill({ response });
          if (otherSessionId !== null && pathname.endsWith(`/sessions/${otherSessionId}/claim`)) {
            notifyRestoredSessionClaimResponse();
          }
        };
        if (otherSessionId !== null) {
          await page.route("**/api/v1/projects/*/sessions/*/claim", delayRestoredSessionClaim);
        }
        const deletionRetry = page.getByRole("button", { name: language === "en" ? "Retry" : "重试", exact: true });
        await page.reload();
        await expect(deletionRetry).toBeEnabled();
        if (otherSessionId !== null) {
          await Promise.race([
            restoredSessionClaimStarted,
            delay(10_000).then(() => { throw new Error("Reload did not issue the expected restored Session Claim"); }),
          ]);
        }
        await deletionRetry.focus();
        await expect(deletionRetry).toBeFocused();
        await page.keyboard.press("Enter");
        await deleteDialog.waitFor();
        if (otherSessionId !== null) {
          releaseRestoredSessionClaim();
          await restoredSessionClaimResponse;
          await page.getByRole("button", { name: "Delete session", exact: true }).waitFor();
          await page.unroute("**/api/v1/projects/*/sessions/*/claim", delayRestoredSessionClaim);
        }
        await page.keyboard.press("Escape");
        await deleteDialog.waitFor({ state: "hidden" });
        await expect(deletionRetry, "Retry deletion dialog did not restore focus to Retry").toBeFocused();
        await page.screenshot({ path: resolve(output, `delete-retry-focus-${language}-${theme}-${viewport.width}.png`) });
        await deletionRetry.click();
        if (otherPrompt !== null) {
          await page.getByRole("log", { includeHidden: true }).getByText(otherPrompt, { exact: true }).waitFor({ state: "attached" });
          await page.keyboard.press("Escape");
          await page.getByLabel("Message input").fill("unsent input stays selected");
          await deletionRetry.click();
        }
        const dialogBounds = await deleteDialog.boundingBox();
        assert.ok(dialogBounds && dialogBounds.x >= 0 && dialogBounds.y >= 0
          && dialogBounds.x + dialogBounds.width <= viewport.width + 1
          && dialogBounds.y + dialogBounds.height <= viewport.height + 1, "Delete dialog overflow");
        await page.screenshot({ path: resolve(output, `delete-${language}-${theme}-${viewport.width}.png`) });
        await unlink(unsafeArtifact);
        mode = "lost-response";
        await deleteDialog.getByRole("button", { name: deleteLabel, exact: true }).click();
        await lostResponseReady;
        await page.keyboard.press("Escape");
        assert.equal(await deleteDialog.isVisible(), true, "In-flight deletion closed on Escape");
        assert.equal(await deleteDialog.getByRole("button", { name: deleteLabel, exact: true }).isDisabled(), true,
          "In-flight deletion allowed a duplicate request");
        releaseLostResponse();
        await deleteDialog.getByRole("alert").waitFor();
        await deleteDialog.getByRole("button", { name: deleteLabel, exact: true }).click();
        await deleteDialog.waitFor({ state: "hidden" });
        await page.unroute(deleteUrl);
        assert.equal(new Set(requestIds.slice(1)).size, 1, "Deletion retry changed request_id");
        const remaining = await readdir(resolve(ownedRoot, "sessions"));
        assert.equal(remaining.includes(`${deleteId}.jsonl`), false);
        assert.ok(remaining.includes(`${control.details.available_session_id}.jsonl`), "Another Session was deleted");
        await assert.rejects(readFile(resolve(restoreRoot, "backup.bin")), { code: "ENOENT" });
        await assert.rejects(readdir(artifactRoot), { code: "ENOENT" });
        assert.equal(await readFile(protectedFile, "utf8"), "user data survives");
        if (otherPrompt !== null) {
          await page.getByRole("log").getByText(otherPrompt, { exact: true }).waitFor();
          assert.equal(await page.getByLabel("Message input").inputValue(), "unsent input stays selected",
            "Deleting another Session cleared the selected Session input");
        }
        assert.equal(await page.getByRole("heading", { name: "project-one", exact: true }).evaluate(
          (element) => element === document.activeElement), true, "Deletion did not restore focus");
        if (otherPrompt !== null) await page.getByRole("button", { name: "Release session", exact: true }).click();
      }
    }
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("navigation").getByRole("link", { name: "Projects", exact: true }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();

  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1000));
  await registerProject(secondProject, "project-two");
  assert.equal(await projectItems.count(), 2);
  await projectItems.filter({ hasText: secondProject }).getByText("Schedule active").waitFor();

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    await page.getByText(language === "en" ? "Project registered." : "项目已登记。", { exact: true }).waitFor();
  }
  await page.clock.resume();
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await firstProjectItem.getByRole("link", { name: "Open schedule" }).click();
  await page.getByRole("heading", { name: "E2E saved project job", exact: true }).waitFor();
  await assertHistoryScopeChange("project");
  let notifySwitchingScheduleLoad;
  let releaseSwitchingScheduleLoad;
  const switchingScheduleLoadArrived = new Promise((done) => { notifySwitchingScheduleLoad = done; });
  const switchingScheduleLoadGate = new Promise((done) => { releaseSwitchingScheduleLoad = done; });
  const delaySwitchingScheduleLoad = async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.jobs[0] = { ...body.jobs[0], title: "STALE previous Project Schedule response" };
    notifySwitchingScheduleLoad();
    await switchingScheduleLoadGate;
    await route.fulfill({ response, json: body });
  };
  await page.route("**/schedule/jobs", delaySwitchingScheduleLoad);
  await page.getByRole("button", { name: "Refresh schedule", exact: true }).click();
  await switchingScheduleLoadArrived;
  await page.locator("aside").getByRole("link", { name: "project-two", exact: true }).click();
  await page.getByRole("heading", { name: "project-two", exact: true }).waitFor();
  const switchingScheduleLoadResponse = page.waitForResponse((response) => response.url() === scheduleResponse.url());
  releaseSwitchingScheduleLoad();
  await switchingScheduleLoadResponse;
  await expect(page.getByText("STALE previous Project Schedule response", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Schedule Jobs", exact: true })).toHaveCount(0);
  await page.unroute("**/schedule/jobs", delaySwitchingScheduleLoad);
  await page.getByRole("navigation").getByRole("link", { name: "Projects", exact: true }).click();
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    await page.locator("header").getByText(language === "en" ? "Projects" : "项目", { exact: true }).waitFor();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const layout = await page.evaluate(() => {
          const aside = document.querySelector("aside").getBoundingClientRect();
          const main = document.querySelector("main").getBoundingClientRect();
          return { width: document.documentElement.scrollWidth, asideRight: aside.right, mainLeft: main.left };
        });
        assert.ok(layout.width <= viewport.width, `Project horizontal overflow at ${viewport.width}x${viewport.height}`);
        assert.ok(layout.mainLeft >= layout.asideRight - 1, `Project sidebar overlaps content at ${viewport.width}x${viewport.height}`);
        await page.getByRole("button", { name: language === "en" ? "Add project" : "登记项目" }).first().waitFor();
        await mkdir(output, { recursive: true });
        await page.screenshot({ path: resolve(output, `projects-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.locator("aside").getByRole("link", { name: "project-one", exact: true }).click();
  await page.getByRole("button", { name: /Renamed available history/ }).click();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const layout = await page.evaluate(() => {
          const list = document.querySelector("main aside").getBoundingClientRect();
          const content = document.querySelector("main [role='log']").parentElement.getBoundingClientRect();
          const horizontal = Math.max(0, Math.min(list.right, content.right) - Math.max(list.left, content.left));
          const vertical = Math.max(0, Math.min(list.bottom, content.bottom) - Math.max(list.top, content.top));
          return { width: document.documentElement.scrollWidth, overlap: horizontal * vertical };
        });
        assert.ok(layout.width <= viewport.width, `Session horizontal overflow at ${viewport.width}x${viewport.height}`);
        assert.ok(layout.overlap < 1, `Session list overlaps history at ${viewport.width}x${viewport.height}`);
        await page.screenshot({ path: resolve(output, `sessions-${language}-${theme}-${viewport.width}.png`) });
        const restore = page.getByRole("button", { name: "Restore", exact: true });
        await restore.click();
        await page.getByRole("button", { name: /Inspect restore|检查 Restore/ }).click();
        await page.getByText(language === "en" ? "Restore preview" : "Restore 预览", { exact: true }).waitFor();
        const restoreBounds = await page.getByRole("dialog").evaluate((element) => {
          const rect = element.getBoundingClientRect();
          return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
        });
        assert.ok(restoreBounds.left >= 0 && restoreBounds.right <= viewport.width);
        assert.ok(restoreBounds.top >= 0 && restoreBounds.bottom <= viewport.height);
        await page.screenshot({ path: resolve(output, `restore-${language}-${theme}-${viewport.width}.png`) });
        await page.keyboard.press("Escape");
        await page.getByRole("dialog").waitFor({ state: "hidden" });
        await expect(restore).toBeFocused();
      }
    }
  }
  await page.locator("aside").getByRole("link", { name: "project-two", exact: true }).click();
  await page.getByRole("heading", { name: "project-two", exact: true }).waitFor();
  assert.equal(await page.getByText("Available history loaded after a successful Claim", { exact: true }).count(), 0);
  await page.getByRole("navigation").getByRole("link", { name: /Projects|项目/ }).click();
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.setViewportSize({ width: 768, height: 1024 });

  await registerProject(projectAlias, "project-one");
  assert.equal(await projectItems.count(), 2, "directory alias created a duplicate Project");

  await rm(secondProject, { recursive: true, force: true });
  await page.getByRole("button", { name: "Refresh projects" }).click();
  const missingProject = projectItems.filter({ hasText: secondProject });
  await missingProject.getByText("Unavailable", { exact: true }).waitFor();
  assert.equal(await page.getByText(cliWorkspace).count(), 0, "unregistered CLI Workspace leaked into Project list");

  const restarted = await control.restart();
  await page.goto(`${restarted.url}/#ticket=${encodeURIComponent(restarted.ticket)}`);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await secondPage.goto(`${restarted.url}/#ticket=${encodeURIComponent(restarted.second_ticket)}`);
  await secondPage.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await secondPage.getByRole("status").first().getByText(/Online|在线/).waitFor();
  await page.getByRole("link", { name: "Projects" }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();
  await page.getByRole("heading", { name: "project-one" }).waitFor();
  await page.getByText("Schedule paused for review").waitFor();
  await page.locator('ul[aria-label="Projects"] > li').filter({ hasText: secondProject }).getByText("Unavailable", { exact: true }).waitFor();
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    const resumeLabel = language === "en" ? "Resume schedule" : "恢复调度";
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const resumeButton = page.getByRole("button", { name: resumeLabel });
        await resumeButton.focus();
        await resumeButton.press("Enter");
        const review = page.getByRole("dialog", {
          name: language === "en" ? "Review Schedule Jobs" : "检查定时任务",
        });
        await review.getByText("E2E saved project job").waitFor();
        await review.getByRole("listitem").filter({ hasText: "E2E saved project job" }).getByText(
          language === "en" ? "Upcoming" : "尚未到期",
          { exact: true },
        ).waitFor();
        for (const title of ["E2E overdue at job", "E2E overdue every job"]) {
          const job = review.getByRole("listitem").filter({ hasText: title });
          await job.getByText(language === "en"
            ? "Overdue; may run when resumed" : "已到期，恢复后可能立即执行",
          { exact: true }).waitFor();
        }
        await review.getByRole("listitem").filter({ hasText: "E2E next cron job" }).getByText(
          language === "en" ? "Next matching time after resume" : "恢复后在下次匹配时间执行",
          { exact: true },
        ).waitFor();
        const layout = await review.evaluate((element) => {
          const bounds = element.getBoundingClientRect();
          return {
            left: bounds.left, right: bounds.right, top: bounds.top, bottom: bounds.bottom,
            width: document.documentElement.scrollWidth,
          };
        });
        assert.ok(layout.left >= 0 && layout.right <= viewport.width
          && layout.top >= 0 && layout.bottom <= viewport.height && layout.width <= viewport.width,
        `Schedule review overflow at ${language}/${theme}/${viewport.width}x${viewport.height}`);
        await review.getByRole("button", { name: resumeLabel }).focus();
        assert.equal(await review.getByRole("button", { name: resumeLabel }).evaluate(
          (element) => element === document.activeElement,
        ), true);
        assert.equal(await review.getByRole("button", { name: resumeLabel }).evaluate(
          (element) => window.getComputedStyle(element).outlineStyle,
        ), "solid", "Schedule decision has no visible keyboard focus outline");
        await page.screenshot({ path: resolve(output, `schedule-review-${language}-${theme}-${viewport.width}.png`) });
        await page.keyboard.press("Escape");
        await review.waitFor({ state: "hidden" });
        await expect(resumeButton).toBeFocused();
        assert.equal(await resumeButton.evaluate((element) => element === document.activeElement), true);
        assert.equal(await resumeButton.evaluate((element) => window.getComputedStyle(element).outlineStyle), "solid",
          "The persistent resume entry has no visible keyboard focus outline");
      }
    }
  }
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("button", { name: "Refresh projects" }).click();
  await page.getByText("Schedule paused for review").waitFor();
  const resumeButton = page.getByRole("button", { name: "Resume schedule" });
  await resumeButton.click();
  const review = page.getByRole("dialog", { name: "Review Schedule Jobs" });
  await review.getByRole("button", { name: "Resume schedule" }).click();
  await page.locator('ul[aria-label="Projects"] > li').filter({ hasText: firstProject }).getByText("Schedule active").waitFor();

  const resumedScheduleResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "GET"
    && response.url().endsWith("/schedule/jobs")
  ));
  await page.locator('ul[aria-label="Projects"] > li').filter({ hasText: firstProject })
    .getByRole("link", { name: "Open schedule" }).click();
  const resumedScheduleResponse = await resumedScheduleResponsePromise;
  assert.equal(resumedScheduleResponse.status(), 200, "Resumed Schedule page did not load its real response");
  const resumedSchedulePayload = await resumedScheduleResponse.json();
  assert.equal(resumedSchedulePayload.status.admitted, true, "Resumed Schedule page reported a false admission state");
  assert.equal(resumedSchedulePayload.status.status, "available", "Resumed Schedule page reported a false health state");
  const resumedScheduleStatus = page.locator('dl[aria-label="Schedule status"]');
  await expect(resumedScheduleStatus).toContainText("Admitted");
  await expect(resumedScheduleStatus).toContainText("Available");
  await page.getByRole("navigation").getByRole("link", { name: "Projects", exact: true }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();

  const removableProject = page.locator('ul[aria-label="Projects"] > li').filter({ hasText: firstProject });
  await removableProject.getByRole("button", { name: "Remove registration" }).click();
  const removalDialog = page.getByRole("dialog", { name: "Remove project registration?" });
  await removalDialog.getByText("saved Schedule Jobs stay on disk").waitFor();
  const removalResponsePromise = page.waitForResponse((response) => (
    response.request().method() === "DELETE"
    && response.url().includes("/api/v1/projects/")
  ));
  await removalDialog.getByRole("button", { name: "Remove registration" }).click();
  const removalResponse = await removalResponsePromise;
  assert.equal(removalResponse.status(), 200);
  const removalOperation = await removalResponse.json();
  assert.ok(removalOperation.operation_id);
  await removalDialog.waitFor({ state: "hidden" });
  await page.getByText("Project registration removed. The directory and saved work remain on disk.").waitFor();
  await removableProject.waitFor({ state: "detached" });
  await readdir(resolve(firstProject, ".omni"));

  await registerProject(firstProject, "project-one");
  await page.getByText("Schedule paused for review").waitFor();
  await page.getByRole("button", { name: "Resume schedule" }).click();
  const savedReview = page.getByRole("dialog", { name: "Review Schedule Jobs" });
  await savedReview.getByText("E2E saved project job").waitFor();
  await page.keyboard.press("Escape");
  await savedReview.waitFor({ state: "hidden" });
  await page.reload();
  await page.getByText("Schedule paused for review").waitFor();
  await page.getByRole("button", { name: "Resume schedule" }).waitFor();

  await settingsModelMcpAcceptance({ page: secondPage, secondPage: page, control, output });

  await page.getByRole("navigation").getByRole("link", { name: "Status" }).click();
  await page.getByRole("heading", { name: "Service status", exact: true }).waitFor();
  await page.route("**/api/v1/clients", (route) => route.abort());
  await page.evaluate(() => window.__omniTestSocket.close());
  await page.getByRole("status").first().getByText(/Reconnecting|恢复连接中/).waitFor();
  await page.getByRole("status").first().getByText(/Offline|离线/).waitFor({ timeout: 10000 });
  await page.unroute("**/api/v1/clients");
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor({ timeout: 10000 });
  assert.deepEqual(browserErrors, [], "Browser JavaScript errors were reported");
  console.log("Playwright production E2E: 4 locale/theme combinations x 3 viewports; Schedule CRUD, accepted-create lost-ack retry, locked fields, delayed detail focus, simulated status polling, stale page/Project/disconnected responses, keyboard validation and 9999/10000ms feedback; Restore overwrite, cancel, stale responses, refresh, failure acknowledgement; delete, ticket, focus, reconnect passed");
} catch (error) {
  acceptanceError = error;
  throw error;
} finally {
  await secondContext?.close();
  await browser?.close();
  await shutdownControl();
}
