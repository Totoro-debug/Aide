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
  await expect(input).toBeEnabled();
  await input.fill("foreground question acceptance");
  await input.press("Enter");
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  await input.fill("independent supplement");
  await input.press("Enter");
  await expect(page.getByRole("log")).toContainText("independent supplement");
  await page.reload();
  await expect(page.getByText("Which output directory?", { exact: true })).toBeVisible();
  const answer = page.getByRole("textbox", { name: "Answer to Aide's question", exact: true });
  await answer.fill("reports-directory");
  await page.getByRole("button", { name: "Send answer", exact: true }).click();
  await expect(answer).toHaveCount(0);
  await expect(page.getByRole("log")).toContainText("reports-directory");
  await assert.rejects(access(join(control.details.home_root, ".aide", "chat", "after-question.txt")));

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
  await page.getByRole("button", { name: "Send answer", exact: true }).click();
  await expect(page.getByRole("log")).toContainText("Reports folder");
  await assert.rejects(access(join(control.details.home_root, ".aide", "chat", "after-question.txt")));
  console.log("Foreground question: Web text/choice/reload/supplement passed");
} finally {
  await browser?.close();
  await control.shutdown();
}
