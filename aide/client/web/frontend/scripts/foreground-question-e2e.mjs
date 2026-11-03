import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { setInterfaceLanguage } from "./settings-e2e.mjs";

const control = await setup();
let browser;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage();
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
  await expect(page.getByRole("region", { name: "Service status" }).getByRole("status"))
    .toContainText("Online", { timeout: 30000 });
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
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
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
  await page.getByRole("button", { name: "Send answer", exact: true }).click();
  await waitForRunResponse(responseCount);
  await expect(answer).toHaveCount(0);
  await expect(page.getByRole("log")).toContainText("reports-directory");
  await assert.rejects(access(join(control.details.home_root, ".aide", "chat", "after-question.txt")));
  await expect(taskPanel).toContainText("Completed");
  responseCount = await fixtureResponse.count();
  await taskPanel.getByRole("button", { name: /Continue task/ }).first().click();
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
  console.log("Foreground question and task operations: Web answer/continue/cancel passed");
} finally {
  await browser?.close();
  await control.shutdown();
}
