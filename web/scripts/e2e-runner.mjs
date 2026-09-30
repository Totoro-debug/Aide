import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { link, mkdir, readFile, readdir, rm, unlink, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { URL } from "node:url";
import { chromium, expect } from "@playwright/test";

import setup from "./e2e-setup.mjs";

const viewports = [
  { width: 1440, height: 900 },
  { width: 1024, height: 768 },
  { width: 768, height: 1024 },
];
const output = resolve("test-results");
let control;
let browser;
let secondContext;

try {
  control = await setup();
  browser = await chromium.launch({
    channel: process.env.MYCLAW_E2E_BROWSER_CHANNEL ?? (process.platform === "win32" ? "msedge" : undefined),
  });
  const primaryContext = await browser.newContext();
  const page = await primaryContext.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  async function waitForRecordedEvent(matches, description) {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      const messages = await page.evaluate(() => window.__myclawTestMessages);
      if (matches(messages)) return;
      await delay(50);
    }
    const events = await page.evaluate(() => window.__myclawTestMessages.slice(-12).map(
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
    window.__myclawTestMessages = [];
    window.WebSocket = class extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        window.__myclawTestControlCredential = Array.isArray(args[1]) ? args[1][1] : null;
        window.__myclawTestSocket = this;
        this.addEventListener("message", (event) => {
          try {
            window.__myclawTestMessages.push(JSON.parse(event.data));
          } catch {
            // Only JSON service messages are relevant to this test.
          }
        });
      }
    };
  });
  const url = process.env.MYCLAW_E2E_URL;
  const launch = spawnSync("python", ["-c", [
    "import sys, webbrowser",
    "from myclaw.terminal.process_entry import run",
    "webbrowser.open_new_tab = lambda _url: False",
    "sys.argv = ['myclaw', 'web']",
    "run()",
  ].join("; ")], {
    cwd: resolve(process.cwd(), ".."),
    env: {
      ...process.env,
      USERPROFILE: process.env.MYCLAW_E2E_HOME_ROOT,
      HOME: process.env.MYCLAW_E2E_HOME_ROOT,
    },
    encoding: "utf8",
    timeout: 30000,
  });
  assert.equal(launch.status, 0, `myclaw web failed: ${launch.stderr}`);
  const launchUrl = launch.stdout.match(/http:\/\/127\.0\.0\.1:\d+\/#ticket=[\w-]+/)?.[0];
  assert.ok(launchUrl?.startsWith(`${url}/#ticket=`), "myclaw web did not attach to the isolated service");
  await page.goto(launchUrl);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor();
  assert.match(page.url(), /\/status$/);
  secondContext = await browser.newContext();
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
  await page.waitForFunction(() => document.activeElement?.textContent?.includes("连接详情"));
  assert.equal(await details.evaluate((element) => element === document.activeElement), true);

  await page.reload();
  await page.getByRole("heading", { name: "服务状态" }).waitFor();
  assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
  await page.getByRole("status").first().getByText("在线").waitFor();

  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("link", { name: "Projects" }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();
  await page.getByRole("heading", { name: "No projects registered" }).waitFor();

  const firstProject = process.env.MYCLAW_E2E_FIRST_PROJECT;
  const projectAlias = process.env.MYCLAW_E2E_PROJECT_ALIAS;
  const secondProject = process.env.MYCLAW_E2E_SECOND_PROJECT;
  const cliWorkspace = process.env.MYCLAW_E2E_CLI_WORKSPACE;
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

  const firstProjectItem = projectItems.filter({ hasText: firstProject });
  await firstProjectItem.getByRole("link", { name: "Open sessions" }).click();
  await page.getByRole("heading", { name: "project-one", exact: true }).waitFor();
  const sessionList = page.getByRole("list", { name: "Conversation Sessions" });
  await sessionList.getByRole("button", { name: /Web available history/ }).click();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  await page.reload();
  await page.getByText("Available history loaded after a successful Claim", { exact: true }).waitFor();
  assert.equal(await page.getByText("schedule-only content", { exact: true }).count(), 0);

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
  const sessionFiles = await readdir(resolve(firstProject, ".myclaw", "sessions"));
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
  const permissionChange = await page.evaluate(async ({ activeWorkspaceId, sessionId }) => {
    const browserSession = await window.fetch("/api/v1/web/session", { credentials: "include" });
    const { csrf_token: csrf } = await browserSession.json();
    const control = window.__myclawTestControlCredential;
    if (typeof control !== "string") throw new Error("The Web control credential was not captured");
    const response = await window.fetch(
      `/api/v1/workspaces/${activeWorkspaceId}/management/permission`,
      {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          "X-MyClaw-CSRF": csrf,
          "X-MyClaw-Control": control,
        },
        body: JSON.stringify({
          request_id: window.crypto.randomUUID(),
          current_session_id: sessionId,
          permission_level: "read-only",
        }),
      },
    );
    return { status: response.status, body: await response.text() };
  }, { activeWorkspaceId: conversationWorkspaceId, sessionId: conversationSessionId });
  assert.equal(permissionChange.status, 200, `Could not select read-only E2E permission: ${permissionChange.body}`);
  await page.getByLabel("Message input").fill("streaming markdown");
  await page.getByLabel("Message input").press("Shift+Enter");
  await page.getByLabel("Message input").type("second line");
  const multilinePrompt = "streaming markdown\nsecond line";
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
  await sessionList.getByRole("button", { name: /Web available history/ }).click();
  const backgroundDraft = page.getByRole("button", { name: /New Session draft/ });
  await backgroundDraft.getByText("Running", { exact: true }).waitFor();
  await backgroundDraft.click();
  await page.getByText("Persisted Markdown", { exact: true }).waitFor();
  await waitForRecordedEvent((messages) => messages.some((event) => (
    event.type === "run.completed" && messages.some((accepted) => (
      accepted.type === "input.accepted" && accepted.payload?.text === multilinePrompt && accepted.run_id === event.run_id
    ))
  )), "multiline Run completion");
  const acceptedConversation = await page.evaluate((prompt) => window.__myclawTestMessages.filter((event) => (
    event.type === "input.accepted" && event.payload?.text === prompt
  )), multilinePrompt);
  assert.equal(acceptedConversation.length, 1, "Multiline prompt was accepted more than once");
  const conversationRunId = acceptedConversation[0].run_id;
  const frames = await page.evaluate((runId) => window.__myclawTestMessages.filter((event) => (
    event.type === "run.output" && event.run_id === runId
    && event.payload?.message?.metadata?._stream_delta === true
  )), conversationRunId);
  assert.ok(frames.length >= 3, `Expected progressive Markdown frames, received ${frames.length}`);
  assert.equal(await page.locator('img[src="https://example.com/remote.png"]').count(), 0,
    "Remote Markdown image was loaded");
  assert.equal(await page.locator('a[href^="javascript:"]').count(), 0, "Unsafe Markdown link survived rendering");
  const codeBlock = page.locator("pre").filter({ hasText: "x".repeat(100) }).first();
  assert.equal(await codeBlock.evaluate((element) => element.scrollWidth > element.clientWidth), true,
    "Long code block did not scroll locally");
  let persistedConversation;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const records = (await readFile(resolve(firstProject, ".myclaw", "sessions", `${conversationSessionId}.jsonl`), "utf8"))
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
    const socket = window.__myclawTestSocket;
    window.__myclawRetrySocket = socket;
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
    reconnected = await page.evaluate(() => window.__myclawRetrySocket.readyState === 3
      && window.__myclawTestSocket !== window.__myclawRetrySocket
      && window.__myclawTestSocket.readyState === 1);
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
  assert.equal(await page.evaluate(() => new Set(window.__myclawTestMessages.filter((event) => (
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
  const acceptedToolRuns = await page.evaluate(() => window.__myclawTestMessages.filter((event) => (
    event.type === "input.accepted" && event.payload?.text === "tool states"
  )));
  assert.equal(new Set(acceptedToolRuns.map((event) => event.run_id)).size, 1,
    "Tool prompt was accepted into more than one Run");
  const canceledRunIds = await page.evaluate(() => window.__myclawTestMessages
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
        headers: { "Content-Type": "application/json", "X-MyClaw-CSRF": csrf },
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
        await input.fill("confirmation");
        await input.press("Enter");
        await primaryDialog.waitFor();
        await secondaryDialog.waitFor();
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
        const finishedStatuses = await page.evaluate((runId) => window.__myclawTestMessages
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
  const confirmationToolRuns = await page.evaluate(() => window.__myclawTestMessages
    .filter((event) => event.type === "run.output"
      && event.payload?.message?.type === "tool_call"
      && event.payload?.message?.metadata?.tool_call_id === "call-confirmation")
    .map((event) => event.run_id));
  assert.equal(new Set(confirmationToolRuns).size, confirmationCombinations.length,
    "Confirmation workflow executed more than once for a Run");
  assert.equal(new Set(confirmationRuns.map((run) => run.runId)).size, confirmationCombinations.length);
  let persistedConfirmationResults = [];
  for (let attempt = 0; attempt < 100; attempt += 1) {
    persistedConfirmationResults = [];
    for (const sessionId of new Set(confirmationRuns.map((run) => run.sessionId))) {
      const records = (await readFile(resolve(firstProject, ".myclaw", "sessions", `${sessionId}.jsonl`), "utf8"))
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
  await secondContext.close();
  secondContext = undefined;

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
  await page.waitForFunction(() => !document.querySelector("[role='dialog'] button[type='submit']")?.disabled);
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
        const extendedCase = language === "en" && theme === "light" && viewport.width === 1440;
        if (extendedCase) {
          await page.getByRole("button", { name: "Rename session", exact: true }).click();
          const titleDialog = page.getByRole("dialog");
          await titleDialog.getByLabel("Session title").fill("A".repeat(128));
          await titleDialog.getByRole("button", { name: "Save", exact: true }).click();
          await page.getByRole("heading", { name: "A".repeat(60), exact: true }).waitFor();
        }
        const ownedRoot = resolve(firstProject, ".myclaw");
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
            const headers = { ...route.request().headers(), "x-myclaw-claim": "invalid-claim" };
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
        if (extendedCase) {
          await deleteDialog.getByRole("button", { name: "Cancel", exact: true }).click();
          await deleteDialog.waitFor({ state: "hidden" });
          await page.getByRole("button", { name: "New session", exact: true }).click();
          await page.getByRole("heading", { name: "New Session draft", exact: true }).waitFor();
          otherPrompt = "retry once another session stays selected";
          await page.getByLabel("Message input").fill(otherPrompt);
          await page.getByLabel("Message input").press("Enter");
          await waitForRecordedEvent((messages) => messages.some((completed) => completed.type === "run.completed"
            && messages.some((accepted) => accepted.type === "input.accepted"
              && accepted.payload?.text === otherPrompt && accepted.run_id === completed.run_id)), "unrelated Session completion");
        }
        await page.reload();
        const deletionRetry = page.getByRole("button", { name: language === "en" ? "Retry" : "重试", exact: true });
        await expect(deletionRetry).toBeEnabled();
        await deletionRetry.click();
        await deleteDialog.waitFor();
        await page.keyboard.press("Escape");
        await deleteDialog.waitFor({ state: "hidden" });
        assert.equal(await deletionRetry.evaluate((element) => element === document.activeElement), true,
          "Retry deletion dialog did not restore focus to Retry");
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
        await review.getByText(language === "en" ? "Upcoming" : "尚未到期", { exact: true }).waitFor();
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
        await page.waitForFunction((label) => document.activeElement?.textContent?.includes(label), resumeLabel);
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
  await readdir(resolve(firstProject, ".myclaw"));

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

  await page.getByRole("navigation").getByRole("link", { name: "Status" }).click();
  await page.getByRole("heading", { name: "Service status", exact: true }).waitFor();
  await page.route("**/api/v1/clients", (route) => route.abort());
  await page.evaluate(() => window.__myclawTestSocket.close());
  await page.getByRole("status").first().getByText(/Reconnecting|恢复连接中/).waitFor();
  await page.getByRole("status").first().getByText(/Offline|离线/).waitFor({ timeout: 10000 });
  await page.unroute("**/api/v1/clients");
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor({ timeout: 10000 });
  assert.deepEqual(browserErrors, [], "Browser JavaScript errors were reported");
  console.log("Playwright production E2E: 4 locale/theme combinations x 3 viewports, session delete conflict/failure/refresh/lost-response retry, ticket, focus, reconnect passed");
} finally {
  await secondContext?.close();
  await browser?.close();
  await control?.shutdown();
}
