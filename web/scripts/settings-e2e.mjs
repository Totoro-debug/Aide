import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { expect } from "@playwright/test";

export async function settingsConfirmationAcceptance({ page, control }) {
  await page.bringToFront();
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("button", { name: "New session", exact: true }).click();
  await control.command("settings-arm");
  await page.locator("textarea").fill("settings generation barrier confirmation");
  await page.locator("textarea").press("Enter");
  await control.command("settings-wait");
  await page.getByRole("navigation").getByRole("link", { name: "Settings", exact: true }).click();
  const field = page.getByLabel("Maximum iterations", { exact: true });
  await expect(field).toBeEnabled();
  await field.fill("65");
  const savedResponse = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  const saved = await savedResponse;
  assert.equal(saved.status(), 200);
  assert.equal((await saved.json()).application.status, "pending");
  await control.command("settings-release");
  const dialog = page.getByRole("dialog", { name: "Tool Confirmation", exact: true });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("confirmation-outside.txt");
  await dialog.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByText("Active generation", { exact: true })).toBeVisible({ timeout: 15000 });
  console.log("Settings pending generation: existing browser Run Tool confirmation remains usable and finishes naturally after save");
  return page.evaluate(() => [...window.__myclawTestMessages].reverse().find((event) => (
    event.type === "input.accepted" && event.payload?.text === "settings generation barrier confirmation"
  )).run_id);
}

export default async function settingsAcceptance({ page, secondPage, control, output, viewports }) {
  const settings = async (target) => {
    await target.bringToFront();
    await target.getByRole("button", { name: "EN", exact: true }).click();
    await target.getByRole("navigation").getByRole("link", { name: "Settings", exact: true }).click();
    await expect(target.getByLabel("Maximum iterations", { exact: true })).toBeEnabled();
  };
  const save = async (target, expectedStatus = 200) => {
    const received = target.waitForResponse((response) => (
      response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
    ));
    await target.getByRole("button", { name: "Save settings", exact: true }).click();
    const response = await received;
    assert.equal(response.status(), expectedStatus);
    return response.json();
  };
  const configPath = resolve(control.details.home_root, ".myclaw", "config.toml");
  await settings(page);
  const original = await readFile(configPath);
  await page.getByLabel("Maximum iterations", { exact: true }).fill("1");
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  const summary = page.getByRole("alert").filter({ hasText: "Settings need attention" });
  await expect(summary).toContainText("Review the highlighted settings.");
  await expect(summary).toBeFocused();
  await summary.getByRole("link").click();
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toBeFocused();
  assert.deepEqual(await readFile(configPath), original, "Invalid browser edits changed config bytes");

  await page.getByLabel("Maximum iterations", { exact: true }).fill("61");
  await settings(secondPage);
  await secondPage.getByLabel("Memory batch size", { exact: true }).fill("12");
  const competing = await save(secondPage);
  await page.bringToFront();
  await expect(page.getByRole("definition").filter({ hasText: competing.revision })).toHaveCount(2);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("61");
  const beforeConflict = await readFile(configPath);
  await save(page, 409);
  await expect(page.getByText("These settings changed elsewhere. Your edits are still here.", { exact: true })).toBeVisible();
  assert.deepEqual(await readFile(configPath), beforeConflict, "Stale browser save changed config bytes");
  await page.getByRole("button", { name: "Reload saved values", exact: true }).click();
  await expect(page.getByLabel("Memory batch size", { exact: true })).toHaveValue("12");
  await page.getByLabel("Maximum iterations", { exact: true }).fill("62");
  const saved = await save(page);
  await expect(page.getByText("Active generation", { exact: true })).toBeVisible();
  assert.match(await readFile(configPath, "utf8"), /max_iterations = 62/);
  assert.match(await readFile(configPath, "utf8"), /e2e-fixture-only/);

  const hold = await control.command("settings-hold");
  await page.evaluate(() => { window.__settingsSocketBefore = window.__myclawTestSocket; });
  await page.getByLabel("Memory batch size", { exact: true }).fill("13");
  const pending = await save(page);
  assert.equal(pending.application.status, "pending");
  assert.equal(pending.application.active_revision, saved.revision);
  await expect(page.getByText("Waiting for", { exact: true })).toBeVisible();
  await page.getByRole("navigation").getByRole("link", { name: "Status", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Service status", exact: true })).toBeVisible();
  await settings(page);
  const released = await control.command("settings-release");
  assert.equal(released.pid, hold.pid, "Configuration application restarted the service");
  await expect(page.getByText("Active generation", { exact: true })).toBeVisible({ timeout: 15000 });
  assert.equal(await page.evaluate(() => window.__settingsSocketBefore === window.__myclawTestSocket), true,
    "Configuration application replaced the browser connection");

  const memoryPath = resolve(control.details.cli_workspace, ".myclaw", "memory", "memory.md");
  const memory = await readFile(memoryPath);
  try {
    await writeFile(memoryPath, Buffer.from([0xff, 0xfe]));
    await page.getByLabel("Maximum iterations", { exact: true }).fill("63");
    const failedSave = await save(page);
    await expect(page.getByRole("button", { name: "Retry application", exact: true })).toBeVisible({ timeout: 15000 });
    const versions = page.getByRole("definition");
    await expect(versions.filter({ hasText: failedSave.revision })).toHaveCount(2);
    await expect(versions.filter({ hasText: pending.revision })).toHaveCount(1);
    await writeFile(memoryPath, memory);
    const retryResponse = page.waitForResponse((response) => response.url().endsWith("/api/v1/config/retry"));
    await page.getByRole("button", { name: "Retry application", exact: true }).click();
    assert.equal((await retryResponse).status(), 200);
    await expect(page.getByText("Active generation", { exact: true })).toBeVisible({ timeout: 15000 });
  } finally {
    await writeFile(memoryPath, memory);
  }

  let releasePoll;
  let pollArrived;
  const gate = new Promise((done) => { releasePoll = done; });
  const arrival = new Promise((done) => { pollArrived = done; });
  const delayedPoll = async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    const response = await route.fetch();
    pollArrived();
    await gate;
    await route.fulfill({ response });
  };
  await page.route("**/api/v1/config", delayedPoll);
  await arrival;
  await page.getByLabel("Maximum iterations", { exact: true }).fill("64");
  const deliveredPoll = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "GET"
  ));
  releasePoll();
  await (await deliveredPoll).finished();
  await page.evaluate(() => new Promise((done) => window.requestAnimationFrame(done)));
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("64");
  await page.unroute("**/api/v1/config", delayedPoll);
  await save(page);

  let releaseSave;
  let saveArrived;
  const saveGate = new Promise((done) => { releaseSave = done; });
  const saveArrival = new Promise((done) => { saveArrived = done; });
  const delayedSave = async (route) => {
    if (route.request().method() !== "PATCH") return route.continue();
    const response = await route.fetch();
    assert.equal(response.status(), 200);
    saveArrived();
    await saveGate;
    await route.fulfill({ response });
  };
  await page.route("**/api/v1/config", delayedSave);
  await page.getByLabel("Maximum iterations", { exact: true }).fill("66");
  await page.getByRole("button", { name: "Save settings", exact: true }).click();
  await saveArrival;
  await page.getByRole("navigation").getByRole("link", { name: "Status", exact: true }).click();
  await settings(page);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("66");
  await page.getByLabel("Maximum iterations", { exact: true }).fill("67");
  const deliveredSave = page.waitForResponse((response) => (
    response.url().endsWith("/api/v1/config") && response.request().method() === "PATCH"
  ));
  releaseSave();
  await (await deliveredSave).finished();
  await page.evaluate(() => new Promise((done) => window.requestAnimationFrame(done)));
  await page.unroute("**/api/v1/config", delayedSave);
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  const saveButton = page.getByRole("button", { name: "Save settings", exact: true });
  await expect(saveButton).toBeEnabled();
  await page.route("**/api/v1/clients", (route) => route.abort());
  await page.evaluate(() => window.__myclawTestSocket.close());
  await expect(saveButton).toBeDisabled({ timeout: 10000 });
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  await page.unroute("**/api/v1/clients");
  await expect(saveButton).toBeEnabled({ timeout: 10000 });
  await expect(page.getByLabel("Maximum iterations", { exact: true })).toHaveValue("67");
  await save(page);
  await expect(page.getByText("Active generation", { exact: true })).toBeVisible({ timeout: 15000 });

  await mkdir(output, { recursive: true });
  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of [...viewports, { width: 390, height: 844 }]) {
        await page.setViewportSize(viewport);
        const label = language === "en" ? "Maximum iterations" : "最大迭代次数";
        await page.getByLabel(label, { exact: true }).focus();
        await expect(page.getByLabel(label, { exact: true })).toBeFocused();
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
          `Settings overflow at ${language}/${theme}/${viewport.width}`);
        await page.screenshot({ path: resolve(output, `settings-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.setViewportSize(viewports[0]);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  console.log("Settings production CSP E2E: global without Claim, invalid bytes, cross-client stale CAS and explicit reload, dirty late poll, real active Run pending/activation with same PID/WS, real candidate resource failure/retry, versions, keyboard and 4 locale/theme x 4 viewports passed");
}
