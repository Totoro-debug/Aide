import assert from "node:assert/strict";
import { access } from "node:fs/promises";
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
  assert.equal((await openedConversation).status(), 200);
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
  await page.reload();
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  const answer = page.getByRole("textbox", { name: "Answer to Aide's question", exact: true });
  responseCount = await fixtureResponse.count();
  await answer.fill("reports-directory");
  await page.getByRole("button", { name: "Send answer", exact: true }).click();
  await waitForRunResponse(responseCount);
  await expect(answer).toHaveCount(0);
  await expect(page.getByRole("log")).toContainText("reports-directory");
  await assert.rejects(access(join(control.details.home_root, ".aide", "chat", "after-question.txt")));
  const taskPanel = page.getByRole("region", { name: "Foreground tasks" });
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
  console.log("Foreground question and task operations: Web answer/continue/cancel passed");
} finally {
  await browser?.close();
  await control.shutdown();
}
