import assert from "node:assert/strict";
import { mkdir, readFile, unlink } from "node:fs/promises";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { openLatestRestore, registerProjectFromSidebar } from "./project-ui.mjs";
import { setInterfaceLanguage } from "./settings-e2e.mjs";

const expect = playwrightExpect.configure({ timeout: 30000 });
const control = await setup();
let browser;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  let restoreTaskRemoved = false;
  page.on("pageerror", error => errors.push(error.message));
  await page.route(/\/api\/v1\/workspaces\/[^/]+\/sessions\/[^/]+\/subagents(?:\?.*)?$/, async route => {
    const url = new globalThis.URL(route.request().url());
    const match = url.pathname.match(/\/workspaces\/([^/]+)\/sessions\/([^/]+)\/subagents$/);
    assert.ok(match, `Unexpected SubAgent list path: ${url.pathname}`);
    const [, workspaceId, sessionId] = match;
    if (sessionId !== control.details.restore_session_id) {
      await route.fulfill({ json: {
        workspace_id: workspaceId,
        session_id: sessionId,
        items: [],
        next_cursor: null,
      } });
      return;
    }
    await route.fulfill({ json: {
      workspace_id: workspaceId,
      session_id: sessionId,
      items: restoreTaskRemoved ? [] : [{
        agent_id: "00000000-0000-4000-8000-000000000335",
        title: "Restore branch task",
        status: "completed",
        created_at: "2026-10-08T00:00:00+00:00",
        finished_at: "2026-10-08T00:01:00+00:00",
        result_preview: "Result from discarded restore branch",
        error: null,
        usage: { model_calls: 1, input_tokens: 12, output_tokens: 5, total_tokens: 17 },
      }],
      next_cursor: null,
    } });
  });
  await page.addInitScript(() => {
    window.restoreInputs = [];
    const send = window.WebSocket.prototype.send;
    window.WebSocket.prototype.send = function (data) {
      const command = JSON.parse(data);
      if (command.type === "input") window.restoreInputs.push(command.payload.text);
      return send.call(this, data);
    };
  });
  await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
  await setInterfaceLanguage(page, "en");
  const project = await registerProjectFromSidebar(page, control.details.first_project);
  await page.goto(`${control.details.url}/projects/${project.project_id}`);
  const sessions = page.locator("#app-sidebar").getByRole("list", { name: "project-one Sessions", exact: true, includeHidden: true });
  const input = page.getByLabel("Message input", { exact: true });
  const dialog = page.getByRole("dialog");
  const notice = page.getByRole("status").filter({ hasText: "Restore completed" });
  const sessionIds = {
    "Web restore history": control.details.restore_session_id,
    "Web available history": control.details.available_session_id,
    "Web manual restore history": control.details.manual_restore_session_id,
    "Web failed restore history": control.details.failure_restore_session_id,
  };

  async function selectSession(title) {
    await sessions.getByRole("button", { name: new RegExp(title) }).click();
    await expect.poll(() => page.evaluate(() => (
      JSON.parse(window.localStorage.getItem("aide.browser-recovery") ?? "null")?.session_id
    ))).toBe(sessionIds[title]);
    await expect(input).toBeEnabled();
  }
  async function inspect(mode) {
    await openLatestRestore(page);
    await page.getByRole("button", { name: "Inspect restore", exact: true }).click();
    await expect(page.getByText("Restore preview", { exact: true })).toBeVisible();
    await page.locator(`input[name="restore-mode"][value="${mode}"]`).check();
  }
  async function execute() {
    const response = page.waitForResponse(response => response.url().endsWith("/management/restore/execute"));
    await page.getByRole("button", { name: "Restore session", exact: true }).click();
    assert.equal((await response).status(), 200);
    await expect(dialog).toBeHidden();
    await expect(notice).toBeVisible();
  }
  async function assertRecovery(text) {
    await expect(input).toHaveValue(text);
    await expect.poll(() => page.evaluate(() => (
      JSON.parse(window.localStorage.getItem("aide.browser-recovery") ?? "null")?.input_text
    ))).toBe(text);
  }

  await selectSession("Web restore history");
  const subagentTrigger = page.getByRole("button", { name: "SubAgent tasks", exact: true });
  await expect(subagentTrigger).toBeVisible();
  await subagentTrigger.click();
  const subagentPanel = page.getByRole("dialog", { name: "SubAgent tasks", exact: true });
  await expect(subagentPanel.locator('[data-agent-id="00000000-0000-4000-8000-000000000335"]')).toBeVisible();
  await expect(subagentPanel.getByText("SubAgents", { exact: true }).locator("..")).toContainText("total 17");
  await page.keyboard.press("Escape");
  await expect(subagentPanel).toBeHidden();

  await input.fill("Draft preserved on cancel and failure");
  for (const useEscape of [true, false]) {
    await inspect("conversation-only");
    if (useEscape) await page.keyboard.press("Escape");
    else await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(dialog).toBeHidden();
    await assertRecovery("Draft preserved on cancel and failure");
  }
  await inspect("conversation-only");
  await page.route("**/management/restore/execute", route => route.fulfill({
    status: 409,
    json: { code: "stale_restore_plan", message: "Restore plan is stale.", field_errors: {}, retryable: false, request_id: "restore-test" },
  }), { times: 1 });
  await page.getByRole("button", { name: "Restore session", exact: true }).click();
  await expect(dialog.getByRole("alert")).toBeVisible();
  await assertRecovery("Draft preserved on cancel and failure");
  await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(dialog).toBeHidden();

  const multiline = "Restore 回填全文\n第二行：保留 Unicode 和空格  \n最后一行";
  await input.fill(multiline);
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect.poll(() => page.getByRole("log").locator("details").filter({
    has: page.getByRole("button", { name: "Restore conversation to before this message", exact: true, includeHidden: true }),
  }).count()).toBe(2);
  await inspect("conversation-only");
  await expect(page.locator("#restore-anchor-select")).toHaveValue("2");
  const inputCount = await page.evaluate(() => window.restoreInputs.length);
  await execute();
  restoreTaskRemoved = true;
  await subagentTrigger.click();
  await expect(subagentPanel.locator("li[data-agent-id]")).toHaveCount(0);
  await expect(subagentPanel.getByText("SubAgents", { exact: true }).locator("..")).toContainText("calls 0 · input 0 · output 0 · total 0");
  await page.keyboard.press("Escape");
  await expect(subagentPanel).toBeHidden();
  await assertRecovery(multiline);
  await expect(page.getByRole("log").getByText(multiline, { exact: true })).toHaveCount(0);
  await expect(page.getByRole("log").getByText("Restore branch should disappear from history", { exact: true })).toBeVisible();
  assert.equal((await readFile(control.details.restore_target, "utf8")).replaceAll("\r\n", "\n"), "current branch\n");
  assert.equal(await page.evaluate(() => window.restoreInputs.length), inputCount, "Restore automatically submitted the anchor");
  await notice.getByRole("button", { name: "Close", exact: true }).click();
  await selectSession("Web available history");
  await subagentTrigger.click();
  await expect(subagentPanel.locator("li[data-agent-id]")).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(subagentPanel).toBeHidden();
  await input.fill("Other Session draft");
  await selectSession("Web restore history");
  await assertRecovery(multiline);
  await page.reload();
  await assertRecovery(multiline);
  await selectSession("Web available history");
  await input.fill("Other Session draft");
  await selectSession("Web restore history");

  // A completed transaction whose response arrives after switching Sessions must not replace the new draft.
  await inspect("conversation-only");
  let releaseExecution;
  let executionReceived;
  const released = new Promise(resolve => { releaseExecution = resolve; });
  const received = new Promise(resolve => { executionReceived = resolve; });
  await page.route("**/management/restore/execute", async route => {
    const response = await route.fetch();
    assert.equal(response.status(), 200);
    executionReceived();
    await released;
    await route.fulfill({ response });
  }, { times: 1 });
  await page.getByRole("button", { name: "Restore session", exact: true }).click();
  await received;
  await sessions.getByRole("button", { name: /Web available history/, includeHidden: true }).evaluate(element => element.click());
  await expect(page.getByRole("log", { includeHidden: true }).getByText("Available history loaded after a successful Claim", { exact: true })).toBeVisible();
  const delayedResponse = page.waitForResponse(response => response.url().endsWith("/management/restore/execute"));
  releaseExecution();
  await delayedResponse;
  await expect(dialog).toBeHidden();
  await assertRecovery("Other Session draft");
  await expect(notice).toHaveCount(0);

  await selectSession("Web manual restore history");
  await input.fill("Old draft to replace");
  await inspect("files");
  await execute();
  const fileAnchor = "Manual Restore branch should disappear from history";
  await assertRecovery(fileAnchor);
  assert.equal(await page.evaluate(() => window.restoreInputs.length), 0, "File Restore automatically submitted the anchor");
  assert.equal((await readFile(control.details.manual_restore_target, "utf8")).replaceAll("\r\n", "\n"), "content before Restore\n");
  await expect(page.getByRole("log").getByText(fileAnchor, { exact: true })).toHaveCount(0);
  await notice.getByRole("button", { name: "Close", exact: true }).click();
  await selectSession("Web available history");
  await assertRecovery("Other Session draft");
  await selectSession("Web manual restore history");
  await assertRecovery(fileAnchor);
  await page.reload();
  await assertRecovery(fileAnchor);
  await selectSession("Web available history");
  await input.fill("Other Session draft");

  await selectSession("Web failed restore history");
  await input.fill("Draft before partial file failure");
  await inspect("files");
  await unlink(control.details.failure_restore_target);
  await mkdir(control.details.failure_restore_target);
  await execute();
  await expect(notice.getByText("Failed", { exact: true })).toBeVisible();
  await assertRecovery("Failed Restore branch");
  assert.equal(await page.evaluate(() => window.restoreInputs.length), 0, "Partial File Restore automatically submitted the anchor");
  await expect(page.getByRole("log").getByText("Failed Restore branch", { exact: true })).toHaveCount(0);
  await notice.getByRole("button", { name: "Close", exact: true }).click();
  await selectSession("Web available history");
  await assertRecovery("Other Session draft");
  await selectSession("Web failed restore history");
  await assertRecovery("Failed Restore branch");
  await page.reload();
  await assertRecovery("Failed Restore branch");
  assert.deepEqual(errors, []);
  console.log("Restore regression passed: conversation-only, files, partial file failure, draft persistence, cancel, execution failure, and delayed response isolation.");
} finally {
  try {
    await browser?.close();
  } finally {
    await control.shutdown();
  }
}
