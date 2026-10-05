import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";

const expect = playwrightExpect.configure({ timeout: 15000 });
const control = await setup({ shutdownTimeoutMs: 60000, sessionModels: true });
let browser;
try {
  browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
  const context = await browser.newContext({ locale: "en" });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    const Original = window.WebSocket;
    window.modelCommands = [];
    window.WebSocket = class extends Original {
      constructor(...args) {
        super(...args);
        window.modelControl = args[1][1];
      }
      send(value) {
        const command = JSON.parse(value);
        if (command.type === "session_model_configure") {
          if (window.injectModelConflict) {
            window.injectModelConflict = false;
            command.payload.expected_model_configuration_version = 0;
          }
          window.modelCommands.push(command);
          super.send(JSON.stringify(command));
        } else {
          super.send(value);
        }
      }
    };
  });
  let claim;
  page.on("response", async response => {
    if (/\/sessions\/[^/]+\/claim$/.test(response.url()) && response.ok()) {
      claim = (await response.json()).claim;
    }
  });
  async function snapshot() {
    assert.ok(claim);
    return page.evaluate(async current => {
      const response = await window.fetch(
        `/api/v1/workspaces/${current.workspace_id}/sessions/${current.session_id}?claim_version=${current.claim_version}`,
        { headers: { "X-Omni-Control": window.modelControl, "X-Omni-Claim": current.reconnect_credential } },
      );
      if (!response.ok) throw new Error(`Session snapshot failed (${response.status})`);
      return (await response.json()).snapshot;
    }, claim);
  }
  const model = page.getByLabel("Session model", { exact: true });
  const effort = page.getByLabel("Reasoning effort", { exact: true });
  const defaultValue = JSON.stringify(["primary", "small-model"]);
  const otherValue = JSON.stringify(["primary", "large-model"]);
  const commands = () => page.evaluate(() => window.modelCommands.length);
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await expect(model).toBeEnabled();
  await expect(model).toHaveValue("");
  await expect(model.locator('option[value=""]')).toHaveAttribute("disabled", "");
  assert.equal(await model.locator('option[value=""]').evaluate(option => option.matches(":disabled")), true);
  await expect(model.locator("option")).toHaveCount(3);
  await model.focus();
  await page.keyboard.press("End");
  await expect(model).toHaveValue(otherValue);
  await expect.poll(commands).toBe(1);
  await expect(model.locator('option[value=""]')).toHaveCount(0);
  await effort.selectOption("high");
  await expect(effort).toHaveValue("high");
  await expect.poll(commands).toBe(2);
  await model.selectOption(defaultValue);
  await expect(model).toHaveValue(defaultValue);
  await expect.poll(commands).toBe(3);
  const expected = { provider_id: "primary", model: "small-model", reasoning_effort: "high" };
  const selected = await snapshot();
  assert.deepEqual(selected.model_configuration, expected);
  assert.equal(selected.model_configuration_version, 3);

  await page.getByLabel("Message input", { exact: true }).fill("model selection persistence");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
  const persisted = JSON.parse((await readFile(join(control.details.home_root, ".omni", "chat", ".omni", "sessions", `${claim.session_id}.jsonl`), "utf8")).split("\n")[0]);
  assert.deepEqual(persisted.metadata.model_configuration, expected);
  assert.equal(persisted.metadata.model_configuration_version, 3);
  await page.reload();
  await expect(model).toHaveValue(defaultValue);
  await expect(effort).toHaveValue("high");
  assert.deepEqual((await snapshot()).model_configuration, expected);
  await page.evaluate(() => { window.injectModelConflict = true; });
  await model.selectOption(otherValue);
  await expect(page.getByText("The Session model changed elsewhere. Its latest value was loaded.", { exact: true })).toBeVisible();
  await expect(model).toHaveValue(defaultValue);
  await expect(effort).toHaveValue("high");
  const conflicted = await snapshot();
  assert.deepEqual(conflicted.model_configuration, expected);
  assert.equal(conflicted.model_configuration_version, 3);
  assert.equal(await commands(), 1);
  assert.deepEqual(errors, []);
  console.log("Session model E2E: disabled inherited placeholder, keyboard selection, switch back to default, single commands, preserved effort, durable reload and real version conflict passed");
} finally {
  try {
    await browser?.close();
  } finally {
    await control.shutdown();
  }
}
