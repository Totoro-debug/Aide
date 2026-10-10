import assert from "node:assert/strict";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { setInterfaceLanguage } from "./settings-e2e.mjs";

const control = await setup({ shutdownTimeoutMs: 60000 });
let browser;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage();
  await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
  await setInterfaceLanguage(page, "en");
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();

  const settings = page.getByRole("main");
  for (const label of ["Configuration versions", "Saved version", "Active version"]) {
    await expect(settings.getByText(label, { exact: true })).toHaveCount(0);
  }
  const config = await control.command("config-read");
  assert.ok(config.application.saved_revision);

  const restart = settings.getByRole("button", { name: "Restart Aide", exact: true });
  await expect(restart).toBeEnabled();
  await restart.click();
  const confirmation = page.getByRole("dialog", { name: "Restart Aide", exact: true });
  await confirmation.getByRole("button", { name: "Restart Aide", exact: true }).click();
  const notice = settings.getByRole("status").filter({ hasText: "Aide restarted; saved settings are active." });
  await expect(notice).toBeVisible({ timeout: 60000 });
  await expect(notice).toBeHidden({ timeout: 12000 });
  await expect(restart).toBeEnabled();
  console.log("Settings notices E2E: configuration revisions remain in service state but not the UI, and restart success closes after ten seconds.");
} catch (error) {
  console.error(control.serviceLog());
  throw error;
} finally {
  await browser?.close();
  await control.shutdown();
}
