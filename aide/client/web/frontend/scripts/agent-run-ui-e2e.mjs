import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { newProjectConversation } from "./project-ui.mjs";
import { setInterfaceLanguage, setInterfaceTheme } from "./settings-e2e.mjs";

const control = await setup();
let browser;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    window.WebSocket = class extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        window.agentRunTestControl = args[1][1];
      }
    };
  });
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await expect(page.locator("#conversation-input")).toBeEnabled();
  const project = await page.evaluate(async (path) => {
    const session = await (await globalThis.fetch("/api/v1/web/session")).json();
    const response = await globalThis.fetch("/api/v1/projects", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Aide-Control": window.agentRunTestControl,
        "X-Aide-CSRF": session.csrf_token,
      },
      body: JSON.stringify({ request_id: globalThis.crypto.randomUUID(), path }),
    });
    return { status: response.status, body: await response.json() };
  }, control.details.first_project);
  assert.equal(project.status, 200, JSON.stringify(project.body));
  await page.goto(`${control.details.url}/projects/${project.body.project_id}`);
  await newProjectConversation(page);
  await setInterfaceLanguage(page, "en");
  await setInterfaceTheme(page, "light");
  await mkdir(resolve("test-results"), { recursive: true });

  await control.command("process-arm");
  const input = page.locator("#conversation-input");
  await input.fill("process cycles");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await control.command("process-wait");
  const activeActivity = page.locator("article[data-run-id]").filter({ hasText: "process cycles" })
    .getByRole("group", { name: "Run activity", exact: true });
  await expect(activeActivity).toHaveAttribute("open", "");
  const firstTool = activeActivity.locator(":scope > ul > li > details").first();
  await expect(firstTool.locator(":scope > summary")).toContainText("read_file");
  await expect(firstTool).not.toHaveAttribute("open");
  await expect(firstTool.getByText("fixture content", { exact: true })).toBeHidden();
  await firstTool.locator(":scope > summary").focus();
  await firstTool.locator(":scope > summary").press("Enter");
  await expect(firstTool.getByText("fixture content", { exact: true })).toBeVisible();
  const cancel = page.getByRole("button", { name: "Cancel run", exact: true });
  await expect(cancel).toHaveAttribute("type", "button");
  await expect(page.getByRole("button", { name: "Send", exact: true })).toHaveCount(0);
  const inComposer = await cancel.evaluate((element) => element.closest("form")?.contains(element)
    && element.closest("article[data-run-id]") === null);
  assert.equal(inComposer, true, "Run cancellation is outside the composer");
  await page.screenshot({ path: resolve("test-results", "agent-run-ui-running-desktop.png") });

  await page.reload();
  await expect(activeActivity).toHaveAttribute("open", "");
  await expect(firstTool).not.toHaveAttribute("open");
  await control.command("process-release");
  await page.getByRole("log").getByText("Process final reply.", { exact: true }).waitFor();
  const completedActivity = page.getByRole("log").getByRole("group", { name: "Run activity", exact: true })
    .filter({ hasText: "Before first tool." }).last();
  await expect(completedActivity).not.toHaveAttribute("open");
  await expect(completedActivity).not.toContainText("Process final reply.");
  await expect(page.getByRole("button", { name: "Send", exact: true })).toBeVisible();
  await completedActivity.locator(":scope > summary").click();
  const completedTool = completedActivity.locator(":scope > ul > li > details").first();
  await expect(completedTool).not.toHaveAttribute("open");
  await completedTool.locator(":scope > summary").click();
  await expect(completedTool.getByText("fixture content", { exact: true })).toBeVisible();

  await setInterfaceLanguage(page, "zh-CN");
  await setInterfaceTheme(page, "dark");
  await page.setViewportSize({ width: 390, height: 844 });
  await newProjectConversation(page);
  await control.command("process-arm");
  await input.fill("process cancel gap");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await control.command("process-wait");
  const chineseActivity = page.locator("article[data-run-id]").filter({ hasText: "process cancel gap" })
    .getByRole("group", { name: "运行过程", exact: true });
  await expect(chineseActivity).toHaveAttribute("open", "");
  const chineseCancel = page.getByRole("button", { name: "取消运行", exact: true });
  await expect(chineseCancel).toBeEnabled();
  const bounds = await chineseCancel.boundingBox();
  assert.ok(bounds && bounds.x >= 0 && bounds.x + bounds.width <= 390
    && bounds.y >= 0 && bounds.y + bounds.height <= 844, "Mobile cancel button is clipped");
  await page.screenshot({ path: resolve("test-results", "agent-run-ui-running-mobile.png") });
  await chineseCancel.click();
  await expect(chineseCancel).toBeHidden();
  await control.command("process-release");
  await expect(page.getByRole("button", { name: "发送", exact: true })).toBeVisible();
  const canceledActivity = page.getByRole("log").getByRole("group", { name: "运行过程", exact: true })
    .filter({ hasText: "Before first tool." }).last();
  await expect(canceledActivity.locator(":scope > summary")).toContainText("已取消");
  await expect(canceledActivity).not.toHaveAttribute("open");
  assert.deepEqual(errors, []);
  console.log("Agent Run UI E2E: composer cancel, nested Tool cards, active recovery, terminal collapse, desktop/mobile and English/Chinese passed");
} finally {
  await control.command("process-release").catch(() => {});
  await browser?.close();
  await control.shutdown();
}
