import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { setInterfaceLanguage } from "./settings-e2e.mjs";

const expect = playwrightExpect.configure({ timeout: 30000 });
const control = await setup({ executionMode: "agent" });
let browser;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage();
  const commandErrors = [];
  page.on("websocket", (socket) => socket.on("framereceived", ({ payload }) => {
    const frame = JSON.parse(String(payload));
    if (typeof frame.code === "string" && typeof frame.request_id === "string") {
      commandErrors.push({ code: frame.code, message: frame.message });
      console.error("Service command rejected:", frame.code, frame.message);
    }
  }));
  const openedConversation = page.waitForResponse((response) =>
    new globalThis.URL(response.url()).pathname.endsWith("/conversations/open")
      && response.request().method() === "POST");
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  const initialOpen = await openedConversation;
  assert.equal(initialOpen.status(), 200);
  const initialConversation = await initialOpen.json();
  assert.equal(typeof initialConversation.session_id, "string");
  assert.equal(typeof initialConversation.directory, "string");

  async function readProviderObservationRows() {
    const content = await readFile(control.details.provider_observation_path, "utf8");
    return content.trim()
      ? content.trim().split("\n").map((line) => JSON.parse(line))
      : [];
  }
  async function navigateWithinApp(path) {
    await page.evaluate((nextPath) => {
      window.history.pushState({}, "", nextPath);
      window.dispatchEvent(new window.PopStateEvent("popstate"));
    }, path);
  }
  await setInterfaceLanguage(page, "en");
  const input = page.getByRole("textbox", { name: "Message input", exact: true });
  const fixtureResponse = page.getByText("Fixture response.", { exact: true });
  const cancelRun = page.getByRole("button", { name: "Cancel run", exact: true });
  async function waitForRunResponse(previousCount) {
    await expect.poll(async () => fixtureResponse.count()).toBeGreaterThan(previousCount);
    await expect(cancelRun).toHaveCount(0);
  }
  await expect(input).toBeEnabled();
  await input.fill("foreground question acceptance");
  await input.press("Enter");
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  let responseCount = await fixtureResponse.count();
  await input.fill("independent supplement");
  await input.press("Enter");
  await expect(page.getByRole("log")).toContainText("independent supplement");
  await waitForRunResponse(responseCount);
  const beforeRestart = await readProviderObservationRows();
  const replayableRequestsBeforeRestart = beforeRestart.filter((item) =>
    item.prompt.includes("foreground question acceptance")
      || item.prompt.includes("independent supplement")).length;
  const restarted = await control.restart();
  await page.goto(`${restarted.url}/status#ticket=${encodeURIComponent(restarted.ticket)}`);
  await expect(page.getByRole("heading", { name: "Service status", exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "Service status" }).getByRole("status")
    .filter({ hasText: /^Online$/ })).toBeVisible({ timeout: 30000 });
  const resumedQuery = new globalThis.URLSearchParams({
    directory: initialConversation.directory,
    session: initialConversation.session_id,
  });
  const reopenedConversation = page.waitForResponse((response) =>
    new globalThis.URL(response.url()).pathname.endsWith("/conversations/open")
      && response.request().method() === "POST"
      && response.request().postDataJSON()?.session_id === initialConversation.session_id);
  await navigateWithinApp(`/chat?${resumedQuery.toString()}`);
  const reopened = await reopenedConversation;
  assert.equal(reopened.status(), 200);
  assert.equal(reopened.request().postDataJSON()?.session_id, initialConversation.session_id);
  assert.equal((await reopened.json()).session_id, initialConversation.session_id);
  const answeredTaskId = (await reopened.json()).snapshot.foreground_questions.find(
    (question) => question.text === "Which output directory?",
  ).task_id;
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  await expect(page.locator("#app-sidebar")).toContainText("Questions: 1 · Main 1 · SubAgent 0");
  await expect(page.locator("#app-sidebar")).not.toContainText("Interrupted tasks: 1");
  const afterRestart = await readProviderObservationRows();
  assert.equal(
    afterRestart.filter((item) =>
      item.prompt.includes("foreground question acceptance")
        || item.prompt.includes("independent supplement")).length,
    replayableRequestsBeforeRestart,
  );
  const answer = page.getByRole("textbox", { name: "Answer to Aide's question", exact: true });
  const taskPanel = page.getByRole("region", { name: "Foreground tasks" });
  await expect(taskPanel).toContainText("Interrupted");
  responseCount = await fixtureResponse.count();
  await answer.fill("reports-directory");
  const answerSubmit = page.getByRole("button", { name: "Send answer", exact: true });
  await expect(answerSubmit).toBeEnabled();
  await answerSubmit.click();
  await waitForRunResponse(responseCount);
  await expect(answer).toHaveCount(0);
  await expect(page.getByRole("log")).toContainText("reports-directory");
  await expect(page.locator("#app-sidebar")).not.toContainText("Questions: 1 · Main 1 · SubAgent 0");
  await expect(page.locator("#app-sidebar")).not.toContainText("Interrupted tasks: 1");
  await assert.rejects(access(join(control.details.home_root, ".aide", "chat", "after-question.txt")));
  await expect(taskPanel).toContainText("Completed");
  responseCount = await fixtureResponse.count();
  await taskPanel.getByRole("button", { name: `Continue task ${answeredTaskId}`, exact: true }).click();
  await expect(page.getByRole("log")).toContainText("Continue the current foreground task.");
  await waitForRunResponse(responseCount);

  responseCount = await fixtureResponse.count();
  await input.fill("foreground question choice acceptance");
  await input.press("Enter");
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  await expect(page.getByText("reports", { exact: true })).toBeVisible();
  await expect(page.getByText("Archives folder", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  await expect(page.getByText("reports", { exact: true })).toBeVisible();
  const reportsOption = page.getByRole("radio", { name: "Reports folder", exact: true });
  await reportsOption.check();
  responseCount = await fixtureResponse.count();
  await page.getByRole("button", { name: "Send answer", exact: true }).click();
  await waitForRunResponse(responseCount);
  await expect(page.getByRole("log")).toContainText("Reports folder");
  await assert.rejects(access(join(control.details.home_root, ".aide", "chat", "after-question.txt")));
  await expect(taskPanel).toContainText("Completed");
  await page.getByRole("button", { name: /Cancel task/ }).last().click();
  await expect(taskPanel).toContainText("Cancelled");
  await expect(page.getByRole("textbox", { name: "Answer to Aide's question", exact: true })).toHaveCount(0);
  const newConversation = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && new globalThis.URL(response.url()).pathname.endsWith("/conversations/open"));
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  assert.equal((await newConversation).status(), 200);
  await expect(taskPanel).toHaveCount(0);
  await expect(input).toBeEnabled();
  await input.fill("foreground question acceptance");
  await input.press("Enter");
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  await taskPanel.getByRole("button", { name: /Cancel task/ }).last().click();
  await expect(page.getByText("Which output directory?", { exact: true })).toHaveCount(0);
  const activeSessionReminder = page.locator("#app-sidebar button[aria-current='page']");
  await expect(activeSessionReminder).not.toContainText("Questions:");
  await expect(activeSessionReminder).not.toContainText("Interrupted tasks:");
  await page.reload();
  await expect(taskPanel).toContainText("Cancelled");
  await expect(page.getByRole("textbox", { name: "Answer to Aide's question", exact: true })).toHaveCount(0);
  await input.fill("subagent question acceptance");
  await input.press("Enter");
  await expect(page.getByText("Which child output?", { exact: true })).toBeVisible();
  await expect(cancelRun).toHaveCount(0);
  await page.reload();
  await expect(page.getByText("Which child output?", { exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: "Answer to Aide's question", exact: true }).fill("child reports");
  await page.getByRole("button", { name: "Send answer", exact: true }).click();
  await expect(page.getByText("Which child output?", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("log")).toContainText("child reports");
  await expect(cancelRun).toHaveCount(0);
  const autoOpen = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && new globalThis.URL(response.url()).pathname.endsWith("/conversations/open"));
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  const openedAutoConversation = await autoOpen;
  assert.equal(openedAutoConversation.status(), 200);
  const autoConversation = await openedAutoConversation.json();
  assert.equal(typeof autoConversation.workspace_id, "string");
  assert.equal(typeof autoConversation.session_id, "string");
  await expect(input).toBeEnabled();
  await input.fill("automatic continuation acceptance");
  await input.press("Enter");
  // Keep Main busy until all three terminal records are publicly observable.
  await expect.poll(async () => {
    const batch = await control.command(
      `batch-subagents ${autoConversation.workspace_id} ${autoConversation.session_id}`,
    );
    return batch.items.filter((item) => item.status === "completed").length;
  }, { timeout: 30000 }).toBe(3);
  await control.command("batch-main-release");
  await expect(page.getByRole("log")).toContainText("Automatic result handled in browser.", { timeout: 30000 });
  await expect(cancelRun).toHaveCount(0);
  const subagents = await control.command(
    `batch-subagents ${autoConversation.workspace_id} ${autoConversation.session_id}`,
  );
  assert.equal(subagents.items.length, 3);
  assert.deepEqual(
    subagents.items.map((item) => item.status).sort(),
    ["completed", "completed", "completed"],
  );
  assert.deepEqual(
    subagents.items.map((item) => item.result_preview).sort(),
    [
      "Durable browser batch child result 1.",
      "Durable browser batch child result 2.",
      "Durable browser batch child result 3.",
    ],
  );
  const providerObservations = (await readFile(control.details.provider_observation_path, "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  const automaticRequests = providerObservations.filter((item) =>
    item.prompt.includes("Runtime Event: SubAgent Result"));
  assert.equal(automaticRequests.length, 1);
  for (const result of [1, 2, 3]) {
    assert.ok(automaticRequests[0].prompt.includes(`Durable browser batch child result ${result}.`));
  }
  await page.reload();
  await expect(page.getByRole("log")).toContainText("Automatic result handled in browser.");
  await expect(page.locator('[data-role="user"]', { hasText: "SubAgent result:" })).toHaveCount(0);
  const singleOpen = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && new globalThis.URL(response.url()).pathname.endsWith("/conversations/open"));
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  const singleResponse = await singleOpen;
  assert.equal(singleResponse.status(), 200);
  const original = await singleResponse.json();
  await expect(input).toBeEnabled();
  await input.fill("claimless continuation acceptance");
  await input.press("Enter");
  await expect(page.getByRole("log")).toContainText("Main delegated the browser task.");
  await expect(cancelRun).toHaveCount(0);
  const switchOpen = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && new globalThis.URL(response.url()).pathname.endsWith("/conversations/open"));
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  const switched = await switchOpen;
  assert.equal(switched.status(), 200);
  assert.notEqual((await switched.json()).session_id, original.session_id);
  await control.command("auto-child-release");
  await expect.poll(async () => {
    const rows = (await readFile(control.details.provider_observation_path, "utf8"))
      .trim().split("\n").map((line) => JSON.parse(line));
    return rows.filter((row) => row.prompt.includes("Runtime Event: SubAgent Result")
      && row.prompt.includes("Durable browser claimless child result.")).length;
  }, { timeout: 30000 }).toBe(1);
  const originalUrl = new globalThis.URL("/chat", control.details.url);
  originalUrl.search = new globalThis.URLSearchParams({
    directory: original.directory, session: original.session_id,
  }).toString();
  const returnOpen = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && new globalThis.URL(response.url()).pathname.endsWith("/conversations/open"));
  await page.goto(originalUrl.toString());
  const returned = await returnOpen;
  assert.equal(returned.status(), 200);
  assert.equal((await returned.json()).session_id, original.session_id);
  await expect(page.getByRole("log")).toContainText("Automatic result handled in browser.");
  await expect(cancelRun).toHaveCount(0);
  await page.reload();
  await expect(page.getByRole("log")).toContainText("Automatic result handled in browser.");
  await expect(cancelRun).toHaveCount(0);
  await input.fill("multi subagent question acceptance");
  await input.press("Enter");
  await expect(page.getByText("Which output should child 1 prepare?", { exact: true })).toBeVisible();
  await expect(page.getByText("Which output should child 2 prepare?", { exact: true })).toBeVisible();
  await expect(page.locator("#app-sidebar")).toContainText("Questions: 2 · Main 0 · SubAgent 2");
  const newQuestionSession = page.waitForResponse((response) =>
    response.request().method() === "POST"
      && new globalThis.URL(response.url()).pathname.endsWith("/conversations/open"));
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  assert.equal((await newQuestionSession).status(), 200);
  await page.locator("#app-sidebar").getByRole("button")
    .filter({ hasText: "Questions: 2 · Main 0 · SubAgent 2" }).first().click();
  await expect(page.getByText("Which output should child 1 prepare?", { exact: true })).toBeVisible();
  await expect(page.getByText("Which output should child 2 prepare?", { exact: true })).toBeVisible();
  const firstChildQuestion = page.locator("section").filter({
    hasText: "Which output should child 1 prepare?",
  }).last();
  await firstChildQuestion.getByRole("radio", { name: "Reports folder", exact: true }).check();
  await firstChildQuestion.getByRole("button", { name: "Send answer", exact: true }).click();
  await expect(page.getByText("Which output should child 1 prepare?", { exact: true })).toHaveCount(0);
  await expect(page.locator("#app-sidebar")).toContainText("Questions: 1 · Main 0 · SubAgent 1");
  let answeredChildren;
  await expect.poll(async () => {
    const children = await control.command(`batch-subagents ${original.workspace_id} ${original.session_id}`);
    answeredChildren = children;
    return children.items.filter((item) => item.title === "Browser question child 1"
      && item.status === "running").length;
  }, { timeout: 30000 }).toBe(1).catch((error) => {
    throw new Error(`${error.message}\n${JSON.stringify(answeredChildren)}\n${control.serviceLog()}`);
  });
  assert.equal(answeredChildren.items.find(
    (item) => item.title === "Browser question child 2",
  ).status, "waiting_for_answer");
  await expect.poll(async () => (await readProviderObservationRows()).filter(
    (row) => row.prompt.includes("Human Answer: Reports folder")
      && row.tool_results.includes("multi-child-question-1"),
  ).length).toBe(1);
  await expect(page.getByText("Which output should child 1 prepare?", { exact: true })).toHaveCount(0);
  await expect(page.getByText("Which output should child 2 prepare?", { exact: true })).toBeVisible();
  const secondChildQuestion = page.locator("section").filter({
    hasText: "Which output should child 2 prepare?",
  }).last();
  await secondChildQuestion.getByRole("textbox", { name: "Answer to Aide's question", exact: true })
    .fill("child two output");
  await secondChildQuestion.getByRole("button", { name: "Send answer", exact: true }).click();
  await expect(page.locator("[role='alert']")).toHaveCount(0);
  await expect(page.getByText("Which output should child 2 prepare?", { exact: true })).toHaveCount(0)
    .catch((error) => { throw new Error(`${error.message}\n${JSON.stringify(commandErrors)}`); });
  await expect(page.locator("#app-sidebar")).not.toContainText("Questions: 1 · Main 0 · SubAgent 1");
  await control.command("multi-answer-release");
  await expect.poll(async () => {
    const children = await control.command(`batch-subagents ${original.workspace_id} ${original.session_id}`);
    return children.items.filter((item) => item.title.startsWith("Browser question child")
      && item.status === "completed").length;
  }, { timeout: 30000 }).toBe(2);
  console.log("Foreground question and task operations: Web answer/continue/cancel passed");
} finally {
  await browser?.close();
  await control.shutdown();
}
