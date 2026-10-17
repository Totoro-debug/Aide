import assert from "node:assert/strict";
import { URL } from "node:url";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";

const control = await setup();
let browser;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  const errors = [];
  const traceRequests = [];
  let credential;
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    credential = request.headers()["x-aide-control"] ?? credential;
    if (new URL(request.url()).pathname.startsWith("/api/v1/traces")) traceRequests.push(request.url());
  });
  await page.addInitScript(() => window.localStorage.setItem("aide.language", "en"));
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await expect(page.locator("#conversation-input")).toBeEnabled();
  const execution = await control.command("trace-run");
  assert.equal(execution.session_id, control.details.occupied_session_id);

  async function records() {
    return page.evaluate(async ({ sessionId, token }) => {
      const response = await window.fetch(`/api/v1/traces?owner_session_id=${sessionId}`, {
        headers: { "X-Aide-Control": token },
      });
      if (response.status !== 200) throw new Error(`Trace API returned ${response.status}`);
      return (await response.json()).items;
    }, { sessionId: execution.session_id, token: credential });
  }
  await expect.poll(async () => (await records()).filter(record => record.kind === "tool").length,
    { timeout: 30000 }).toBe(2);
  await expect.poll(async () => (await records()).filter(record => record.kind === "model"
    && record.purpose === "agent_run").length, { timeout: 30000 }).toBe(3);
  const persisted = await records();
  const tool = persisted.find(record => record.kind === "tool");
  const model = persisted.find(record => record.record_id === tool.parent_record_id);
  assert.ok(persisted.some(record => record.record_id === tool.parent_record_id));
  assert.equal(tool.tool_name, "read_file");
  assert.equal(model.capture_state, "complete");
  const before = await control.command("trace-claim");
  const url = `${control.details.url}/traces/sessions/${model.workspace_key}/${execution.session_id}`;
  await page.goto(`${url}?recordId=${model.record_id}`);
  await expect(page.getByRole("heading", { name: "Execution Trace", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "small-model", exact: true })).toBeVisible();
  await page.getByRole("link", { name: /read_file · Succeeded/ }).click();
  await expect(page.getByRole("heading", { name: "read_file", exact: true })).toBeVisible();
  await expect(page.locator("pre").filter({ hasText: "fixture.txt" }).first()).toBeVisible();
  await page.getByRole("link", { name: "View initiating model call", exact: true }).click();
  await expect(page.getByRole("heading", { name: "small-model", exact: true })).toBeVisible();
  assert.equal(traceRequests.filter(url => new URL(url).pathname.endsWith("/raw")).length, 0);
  const count = traceRequests.length;
  await page.waitForTimeout(500);
  assert.equal(traceRequests.length, count);
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect.poll(() => traceRequests.length).toBeGreaterThan(count);
  const after = await control.command("trace-claim");
  assert.deepEqual(after.claim, before.claim, "Reading Trace changed the CLI Claim");
  assert.deepEqual(errors, []);
  console.log("Real Service execution → SQLite → authenticated API → Trace Web passed.");
} finally {
  if (browser !== undefined) await browser.close();
  await control.shutdown();
}
