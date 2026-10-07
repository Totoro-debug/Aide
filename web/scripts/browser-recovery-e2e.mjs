import { registerProjectFromSidebar } from "./project-ui.mjs";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { URL, URLSearchParams, pathToFileURL } from "node:url";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";

const expect = playwrightExpect.configure({ timeout: 30000 });

export default async function browserRecoveryAcceptance({ page: initialPage, control }) {
  let page = initialPage;
  const context = page.context();
  const origin = new URL(page.url()).origin;
  const errors = [];
  async function preparePage(target) {
    target.on("pageerror", error => errors.push(error.message));
    target.on("console", message => {
      if (message.type() === "error" && message.text().includes("Maximum update depth")) {
        errors.push(message.text());
      }
    });
    await target.addInitScript(() => {
      if (window.top === window && window.location.protocol === "http:") {
        const staged = window.sessionStorage.getItem("omni.test-recovery");
        if (staged !== null) {
          window.localStorage.setItem("omni.browser-recovery", staged);
          window.sessionStorage.removeItem("omni.test-recovery");
        }
      }
      const Original = window.WebSocket;
      window.recoveryInputAttempts = [];
      window.WebSocket = class extends Original {
        constructor(...args) {
          super(...args);
          window.recoverySocket = this;
          window.recoveryControl = args[1][1];
        }
        send(value) {
          const command = JSON.parse(value);
          if (command.type === "input") {
            window.recoveryInputAttempts.push(command);
            if (window.dropRecoveryInput) {
              window.dropRecoveryInput = false;
              this.close();
              return;
            }
          }
          super.send(value);
        }
      };
    });
  }
  await preparePage(page);
  function freshAuthorizationUrl() {
    const launch = spawnSync("python", ["-c", [
      "import asyncio, sys",
      "from pathlib import Path",
      "from omni.config.agent_home import AgentHome",
      "from omni.service.client import ServiceClient",
      "async def run():",
      "    client = await ServiceClient.connect_or_start(AgentHome(Path(sys.argv[1]) / '.omni'), Path(sys.argv[2]))",
      "    try:",
      "        print(await client.create_web_ticket())",
      "    finally:",
      "        await client.close()",
      "asyncio.run(run())",
    ].join("\n"), control.details.home_root, control.details.cli_workspace], {
      cwd: "..", encoding: "utf8", timeout: 30000,
    });
    assert.equal(launch.status, 0, "The CLI could not issue a fresh browser authorization ticket");
    const launchUrl = launch.stdout.match(/http:\/\/127\.0\.0\.1:\d+\/#ticket=[\w-]+/)?.[0];
    assert.ok(launchUrl, "The CLI returned no browser authorization URL");
    return launchUrl;
  }
  const input = () => page.getByRole("textbox", { name: "Message input", exact: true });
  const model = () => page.getByLabel("Session model", { exact: true });
  const effort = () => page.getByLabel("Reasoning effort", { exact: true });
  const snapshot = () => page.evaluate(() => JSON.parse(window.localStorage.getItem("omni.browser-recovery")));
  async function printRecoveryDiagnostic(label) {
    console.error(label, await page.evaluate(() => ({
      route: window.location.pathname + window.location.search,
      main: document.querySelector("main")?.innerText.slice(0, 2000),
      recovery: JSON.parse(window.localStorage.getItem("omni.browser-recovery") ?? "null"),
    })));
  }
  async function clientId() {
    return page.evaluate(async () => (await (await window.fetch("/api/v1/web/session")).json()).client_id);
  }
  async function assertRecovered(saved, text) {
    try {
      await expect(input()).toHaveValue(text, { timeout: 15000 });
    } catch (error) {
      await printRecoveryDiagnostic("Browser recovery diagnostic");
      throw error;
    }
    await expect(input()).toBeEnabled({ timeout: 15000 });
    const recovered = await snapshot();
    assert.equal(recovered.service_instance_id, saved.service_instance_id);
    assert.deepEqual(recovered.target, saved.target);
    if (!saved.draft) assert.equal(recovered.session_id, saved.session_id);
    if (saved.model_configuration !== null) {
      await expect(model()).toHaveValue(JSON.stringify([
        saved.model_configuration.provider_id, saved.model_configuration.model,
      ]));
      await expect(effort()).toHaveValue(saved.model_configuration.reasoning_effort);
    }
    assert.equal(await page.getByRole("log").getByText(text, { exact: true }).count(), 0);
    assert.equal("workspace_id" in recovered, false);
  }
  await page.reload();
  await expect(input()).toBeEnabled();
  const projectId = (await registerProjectFromSidebar(page, control.details.first_project)).project_id;
  const chat = await page.evaluate(async () => (await (await window.fetch("/api/v1/chat/sessions?limit=100", { headers: { "X-Omni-Control": window.recoveryControl } })).json()).sessions
    .find(session => session.available));
  assert.ok(chat, "The fixture must contain an available chat history");
  for (const scope of ["project", "chat"]) {
    for (const draft of [false, true]) {
      console.log(`Browser recovery E2E: ${scope} ${draft ? "draft" : "history"}`);
      if (scope === "project") {
        await page.goto(`${origin}/projects/${encodeURIComponent(projectId)}?session=${control.details.available_session_id}`);
        await expect(input()).toBeEnabled();
        if (draft) await page.getByRole("button", { name: "New session", exact: true }).click();
      } else if (draft) {
        await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
      } else {
        const params = new URLSearchParams({ directory: chat.directory, session: chat.id });
        await page.goto(`${origin}/chat?${params}`);
      }
      await expect(input()).toBeEnabled();
      if (draft) {
        await page.locator("#composer-model-trigger").click();
        const selectedModel = JSON.stringify(["primary", "small-model"]);
        await model().selectOption(selectedModel);
        await expect(page.locator("#composer-model-menu")).toBeHidden();
        await expect(model()).toHaveValue(selectedModel);
        await expect(model()).toBeEnabled();
        await page.locator("#composer-model-trigger").click();
        await effort().selectOption("xhigh");
        await expect(page.locator("#composer-model-menu")).toBeHidden();
        await expect(effort()).toHaveValue("xhigh");
        await expect(effort()).toBeEnabled();
      }
      const text = `Unsent ${scope} ${draft ? "draft" : "history"}\n逐字恢复 😀`;
      await input().fill(text);
      const saved = await snapshot();
      await page.reload();
      await assertRecovered(saved, text);
      const shortClient = await clientId();
      await context.setOffline(true);
      await delay(1500);
      await context.setOffline(false);
      await assertRecovered(saved, text);
      assert.equal(await clientId(), shortClient, "Short disconnect replaced the original Client");
      const expiredClient = await clientId();
      await page.close();
      await delay(31000);
      page = await context.newPage();
      await preparePage(page);
      await page.goto(origin);
      await assertRecovered(saved, text);
      assert.notEqual(await clientId(), expiredClient, "Expired Client was reused");
    }
  }

  const beforeAuthorization = await snapshot();
  await context.clearCookies();
  await page.reload();
  await page.getByText("Open this workbench from the local Omni command.", { exact: true }).waitFor();
  await delay(31000);
  await page.goto(freshAuthorizationUrl());
  await assertRecovered(beforeAuthorization, beforeAuthorization.input_text);

  // Drop the input before delivery and recover the same Client without resending it.
  await page.evaluate(() => { window.dropRecoveryInput = true; });
  await input().fill("unknown input must never be resent");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.recoverySocket.readyState)).toBe(1);
  await delay(2000);
  assert.equal(await page.evaluate(() => window.recoveryInputAttempts.length), 1);
  await expect(input()).toBeEnabled();
  await page.reload();
  await expect(input()).toHaveValue("");
  assert.equal(await page.getByRole("log").getByText("unknown input must never be resent", { exact: true }).count(), 0);

  // Keep an existing Claim while restoring a different, occupied target.
  const occupiedContext = await context.browser().newContext({ locale: "en" });
  let availableRecovery;
  try {
    const occupant = await occupiedContext.newPage();
    const occupiedUrl = new URL(freshAuthorizationUrl());
    occupiedUrl.pathname = `/projects/${encodeURIComponent(projectId)}`;
    occupiedUrl.searchParams.set("session", control.details.restore_session_id);
    await occupant.goto(occupiedUrl.toString());
    await expect(occupant.getByRole("log").getByText("Restore branch should disappear from history", { exact: true })).toBeVisible();
    await page.goto(`${origin}/projects/${encodeURIComponent(projectId)}?session=${control.details.available_session_id}`);
    await expect(page.getByRole("log").getByText("Available history loaded after a successful Claim", { exact: true })).toBeVisible();
    await expect.poll(async () => {
      const saved = await snapshot();
      return saved?.target.kind === "project" && saved.target.project_id === projectId
        && saved.session_id === control.details.available_session_id;
    }, { timeout: 15000 }).toBe(true);
    availableRecovery = await snapshot();
    await page.evaluate(saved => window.sessionStorage.setItem("omni.test-recovery", JSON.stringify(saved)), {
      ...availableRecovery, session_id: control.details.restore_session_id, draft: false, input_text: "", model_configuration: null,
    });
    await page.goto(origin);
    try {
      await page.getByRole("region", { name: "Conversation", exact: true })
        .getByText("This Session is occupied by another client.", { exact: true }).waitFor();
    } catch (error) {
      await printRecoveryDiagnostic("Occupied recovery diagnostic");
      throw error;
    }
    assert.equal(await page.getByRole("log").count(), 0, "Occupied recovery retained old conversation body");
    await page.getByRole("region", { name: "Conversation", exact: true })
      .getByRole("button", { name: "New session", exact: true }).click();
    await expect(input()).toBeEnabled();
  } finally {
    await occupiedContext.close();
  }

  for (const target of [
    { kind: "project", project_id: "removed-recovery-project" },
    { kind: "chat", directory: chat.directory },
  ]) {
    const saved = await snapshot() ?? availableRecovery;
    await page.evaluate(value => window.sessionStorage.setItem("omni.test-recovery", JSON.stringify(value)), {
      ...saved, target, session_id: "20261005-000000-000000_00000000-0000-4000-8000-000000000322", draft: false, input_text: "", model_configuration: null,
    });
    await page.goto(origin);
    try {
      await page.getByText("This Session is no longer available.", { exact: true }).waitFor();
    } catch (error) {
      await printRecoveryDiagnostic(`Deleted recovery diagnostic: ${target.kind}`);
      throw error;
    }
    await expect(input()).toBeEnabled();
    await expect(page.getByRole("region", { name: "Conversation", exact: true })
      .getByRole("heading", { name: "New Session draft", exact: true })).toBeVisible();
    await expect(input()).toHaveValue("");
    assert.equal(new URL(page.url()).pathname, "/");
    const recovered = await snapshot();
    assert.ok(recovered === null || recovered.target.kind === "chat");
  }
  await input().fill("Discard this draft after a new service starts");
  const beforeRestart = await snapshot();
  const restarted = await control.restart();
  const staleParams = new URLSearchParams({ directory: beforeRestart.target.directory, session: beforeRestart.session_id });
  await page.goto(`${restarted.url}/chat?${staleParams}#ticket=${encodeURIComponent(restarted.ticket)}`);
  await expect(page.getByRole("region", { name: "Conversation", exact: true })
    .getByRole("heading", { name: "New Session draft", exact: true })).toBeVisible();
  await expect(input()).toHaveValue("");
  assert.equal(new URL(page.url()).pathname, "/");
  const afterRestart = await snapshot();
  assert.ok(afterRestart === null || (afterRestart.service_instance_id !== beforeRestart.service_instance_id
    && afterRestart.session_id !== beforeRestart.session_id && afterRestart.target.kind === "chat"));
  assert.deepEqual(errors, []);
  console.log("Browser recovery E2E: project/chat × history/draft refresh, short disconnect, expired Client, reauthorization, unknown input, occupied/deleted targets and new service passed");
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const control = await setup({ shutdownTimeoutMs: 60000 });
  let browser;
  try {
    browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
    const context = await browser.newContext({ locale: "en", viewport: { width: 1280, height: 900 } });
    const page = await context.newPage();
    await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
    await expect(page.getByLabel("Message input")).toBeEnabled();
    await page.getByLabel("Message input").fill("Browser recovery history fixture");
    await page.getByRole("button", { name: "Send", exact: true }).click();
    await expect(page.getByRole("log").getByText("Fixture response.", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "Release session", exact: true }).click();
    if (process.env.OMNI_E2E_RECOVERY_SHARED_DIRECTORY === "1") {
      await mkdir(`${control.details.home_root}\\chat-next`);
      await control.command("project-history-seed");
      await registerProjectFromSidebar(page, `${control.details.home_root}\\chat-next`);
      await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
      await page.getByRole("navigation", { name: "Settings sections", exact: true })
        .getByRole("button", { name: "General & appearance", exact: true }).click();
      const changed = page.waitForResponse(response => response.url().endsWith("/api/v1/config")
        && response.request().method() === "PATCH");
      await page.getByLabel("Default conversation workspace", { exact: true })
        .fill(`${control.details.home_root}\\chat-next`);
      await page.getByLabel("Default conversation workspace", { exact: true }).blur();
      assert.equal((await changed).status(), 200);
    }
    await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
    await expect(page.getByLabel("Message input")).toBeEnabled();
    await browserRecoveryAcceptance({ page, control });
  } finally {
    try {
      await browser?.close();
    } finally {
      await control.shutdown();
    }
  }
}
