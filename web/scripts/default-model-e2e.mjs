import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";

const expect = playwrightExpect.configure({ timeout: 10000 });
const control = await setup({ shutdownTimeoutMs: 60000 });
let browser;
try {
  browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
  const context = await browser.newContext({ locale: "en" });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const Original = window.WebSocket;
    window.WebSocket = class extends Original {
      constructor(...args) {
        super(...args);
        window.modelControl = args[1][1];
      }
    };
  });
  const effort = page.getByLabel("Reasoning effort", { exact: true });
  const model = page.getByLabel("Session model", { exact: true });
  let currentClaim;
  async function snapshot() {
    assert.ok(currentClaim);
    return page.evaluate(async claim => {
      const path = `/api/v1/workspaces/${claim.workspace_id}/sessions/${claim.session_id}?claim_version=${claim.claim_version}`;
      const response = await window.fetch(path, { headers: {
        "X-Omni-Control": window.modelControl,
        "X-Omni-Claim": claim.reconnect_credential,
      } });
      if (!response.ok) throw new Error(`Session snapshot failed (${response.status})`);
      return (await response.json()).snapshot;
    }, currentClaim);
  }
  page.on("response", async response => {
    if (new globalThis.URL(response.url()).pathname.endsWith("/conversations/open") && response.ok()) {
      currentClaim = (await response.json()).claim;
    }
  });
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await expect(model).toBeEnabled();
  await expect(effort).toHaveValue("medium");
  await control.command("effort high");
  await expect(effort).toHaveValue("high");

  const configPath = join(control.details.home_root, ".omni", "config.toml");
  const config = await readFile(configPath);
  try {
    await writeFile(configPath, "[broken\n");
    const result = await control.command("effort max");
    assert.equal(result.published_effort, "max");
  } finally {
    await writeFile(configPath, config);
  }
  await expect(effort).toHaveValue("max");
  assert.deepEqual(await readFile(configPath), config);

  // A failed background read must keep the last successful projection usable.
  let failedRead;
  const failed = new Promise(resolve => { failedRead = resolve; });
  await page.route("**/api/v1/models/available", async route => {
    await route.fulfill({ status: 500, contentType: "application/json", body: "{}" });
    failedRead();
  });
  await failed;
  await expect(model).toBeEnabled();
  await expect(effort).toHaveValue("max");
  await page.unroute("**/api/v1/models/available");

  await page.locator("#composer-model-trigger").click();
  await effort.selectOption("medium");
  await expect(model).toHaveValue(JSON.stringify(["primary", "small-model"]));
  const explicit = await snapshot();
  await control.command("effort high");
  await delay(5500);
  await expect(effort).toHaveValue("medium");
  assert.equal((await snapshot()).model_configuration_version, explicit.model_configuration_version);

  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  await expect(model).toHaveValue("");
  await expect(effort).toHaveValue("high");
  let releaseOlder;
  let olderStarted;
  let newerFinished;
  const olderGate = new Promise(resolve => { releaseOlder = resolve; });
  const older = new Promise(resolve => { olderStarted = resolve; });
  const newer = new Promise(resolve => { newerFinished = resolve; });
  let reads = 0;
  await page.route("**/api/v1/models/available", async route => {
    const response = await route.fetch();
    if (++reads === 1) {
      olderStarted();
      await olderGate;
      await route.fulfill({ response });
    } else {
      await route.fulfill({ response });
      newerFinished();
    }
  });
  await older;
  await control.command("effort max");
  await newer;
  await expect(effort).toHaveValue("max");
  const olderResponse = page.waitForResponse(response => response.url().endsWith("/models/available"));
  releaseOlder();
  await olderResponse;
  await delay(300);
  await expect(effort).toHaveValue("max");
  await page.unroute("**/api/v1/models/available");
  assert.deepEqual(errors, []);
  console.log("Default model E2E: live CLI control, persistence failure, background failure, explicit Session isolation and stale response ordering passed");
} finally {
  try {
    await browser?.close();
  } finally {
    await control.shutdown();
  }
}
