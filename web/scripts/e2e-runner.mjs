import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

import setup from "./e2e-setup.mjs";

const viewports = [
  { width: 1440, height: 900 },
  { width: 1024, height: 768 },
  { width: 768, height: 1024 },
];
const output = resolve("test-results");
let control;
let browser;

try {
  control = await setup();
  browser = await chromium.launch({
    channel: process.env.MYCLAW_E2E_BROWSER_CHANNEL ?? (process.platform === "win32" ? "msedge" : undefined),
  });
  const page = await browser.newPage();
  await page.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    window.WebSocket = class extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        window.__myclawTestSocket = this;
      }
    };
  });
  const url = process.env.MYCLAW_E2E_URL;
  const launch = spawnSync("python", ["-c", [
    "import sys, webbrowser",
    "from myclaw.terminal.process_entry import run",
    "webbrowser.open_new_tab = lambda _url: False",
    "sys.argv = ['myclaw', 'web']",
    "run()",
  ].join("; ")], {
    cwd: resolve(process.cwd(), ".."),
    env: {
      ...process.env,
      USERPROFILE: process.env.MYCLAW_E2E_HOME_ROOT,
      HOME: process.env.MYCLAW_E2E_HOME_ROOT,
    },
    encoding: "utf8",
    timeout: 30000,
  });
  assert.equal(launch.status, 0, `myclaw web failed: ${launch.stderr}`);
  const launchUrl = launch.stdout.match(/http:\/\/127\.0\.0\.1:\d+\/#ticket=[\w-]+/)?.[0];
  assert.ok(launchUrl?.startsWith(`${url}/#ticket=`), "myclaw web did not attach to the isolated service");
  await page.goto(launchUrl);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor();
  assert.match(page.url(), /\/status$/);
  const replay = await browser.newContext();
  const reusedTicket = await replay.request.post(`${url}/api/v1/web/ticket`, {
    headers: { Origin: url },
    data: { ticket: launchUrl.split("#ticket=")[1] },
  });
  assert.equal(reusedTicket.status(), 401, "A consumed browser ticket was accepted again");
  await replay.close();

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文" }).click();
    assert.equal(await page.locator("html").getAttribute("lang"), language);
    await page.getByRole("heading", { name: language === "en" ? "Service status" : "服务状态" }).waitFor();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      assert.equal(await page.locator("html").getAttribute("data-theme"), theme);
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        await page.getByRole("main").waitFor();
        await page.getByRole("navigation").getByRole("link", {
          name: language === "en" ? "Status" : "状态",
        }).waitFor();
        const layout = await page.evaluate(() => {
          const aside = document.querySelector("aside").getBoundingClientRect();
          const main = document.querySelector("main").getBoundingClientRect();
          return { width: document.documentElement.scrollWidth, asideRight: aside.right, mainLeft: main.left };
        });
        assert.ok(layout.width <= viewport.width, `Horizontal overflow at ${viewport.width}x${viewport.height}`);
        assert.ok(layout.mainLeft >= layout.asideRight - 1, `Sidebar overlaps content at ${viewport.width}x${viewport.height}`);
        await mkdir(output, { recursive: true });
        await page.screenshot({ path: resolve(output, `workbench-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }

  const statusLink = page.getByRole("navigation").getByRole("link", { name: "状态" });
  await statusLink.focus();
  await statusLink.press("Enter");
  assert.match(page.url(), /\/status$/);

  const details = page.getByRole("button", { name: /连接详情|Connection details/ });
  await details.focus();
  await details.press("Enter");
  await page.getByRole("dialog").waitFor();
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => document.activeElement?.textContent?.includes("连接详情"));
  assert.equal(await details.evaluate((element) => element === document.activeElement), true);

  await page.reload();
  await page.getByRole("heading", { name: "服务状态" }).waitFor();
  assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
  await page.getByRole("status").first().getByText("在线").waitFor();

  await page.getByRole("button", { name: "EN" }).click();
  await page.getByRole("link", { name: "Projects" }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();
  await page.getByRole("heading", { name: "No projects registered" }).waitFor();

  const firstProject = process.env.MYCLAW_E2E_FIRST_PROJECT;
  const projectAlias = process.env.MYCLAW_E2E_PROJECT_ALIAS;
  const secondProject = process.env.MYCLAW_E2E_SECOND_PROJECT;
  const cliWorkspace = process.env.MYCLAW_E2E_CLI_WORKSPACE;
  assert.ok(firstProject && projectAlias && secondProject && cliWorkspace);
  const projectItems = page.locator('ul[aria-label="Projects"] > li');

  async function registerProject(path, name) {
    await page.getByRole("button", { name: "Add project" }).first().click();
    const dialog = page.getByRole("dialog");
    await dialog.getByLabel("Absolute local path").fill(path);
    await dialog.getByRole("button", { name: "Register project" }).click();
    await dialog.waitFor({ state: "hidden" });
    await page.getByRole("heading", { name }).waitFor();
  }

  await page.getByRole("button", { name: "Add project" }).first().click();
  const invalidDialog = page.getByRole("dialog");
  await invalidDialog.getByLabel("Absolute local path").fill("relative/project");
  await invalidDialog.getByRole("button", { name: "Register project" }).click();
  await invalidDialog.getByRole("alert").waitFor();
  await invalidDialog.getByRole("button", { name: "Cancel" }).click();

  await registerProject(firstProject, "project-one");
  await page.getByText("E2E saved project job").waitFor();
  await page.getByText("Schedule paused for review").waitFor();
  await registerProject(secondProject, "project-two");
  assert.equal(await projectItems.count(), 2);

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文" }).click();
    await page.getByText(language === "en" ? "Project registered." : "项目已登记。", { exact: true }).waitFor();
    await page.locator("header").getByText(language === "en" ? "Projects" : "项目", { exact: true }).waitFor();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        const layout = await page.evaluate(() => {
          const aside = document.querySelector("aside").getBoundingClientRect();
          const main = document.querySelector("main").getBoundingClientRect();
          return { width: document.documentElement.scrollWidth, asideRight: aside.right, mainLeft: main.left };
        });
        assert.ok(layout.width <= viewport.width, `Project horizontal overflow at ${viewport.width}x${viewport.height}`);
        assert.ok(layout.mainLeft >= layout.asideRight - 1, `Project sidebar overlaps content at ${viewport.width}x${viewport.height}`);
        await page.getByRole("button", { name: language === "en" ? "Add project" : "登记项目" }).first().waitFor();
        await mkdir(output, { recursive: true });
        await page.screenshot({ path: resolve(output, `projects-${language}-${theme}-${viewport.width}.png`) });
      }
    }
  }
  await page.getByRole("button", { name: "EN" }).click();
  await page.setViewportSize({ width: 768, height: 1024 });

  await registerProject(projectAlias, "project-one");
  assert.equal(await projectItems.count(), 2, "directory alias created a duplicate Project");

  await rm(secondProject, { recursive: true, force: true });
  await page.getByRole("button", { name: "Refresh projects" }).click();
  const missingProject = projectItems.filter({ hasText: secondProject });
  await missingProject.getByText("Unavailable", { exact: true }).waitFor();
  assert.equal(await page.getByText(cliWorkspace).count(), 0, "unregistered CLI Workspace leaked into Project list");

  const restarted = await control.restart();
  await page.goto(`${restarted.url}/#ticket=${encodeURIComponent(restarted.ticket)}`);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await page.getByRole("link", { name: "Projects" }).click();
  await page.getByRole("heading", { name: "Projects", exact: true }).waitFor();
  await page.getByRole("heading", { name: "project-one" }).waitFor();
  await page.getByText("Schedule paused for review").waitFor();
  await page.locator('ul[aria-label="Projects"] > li').filter({ hasText: secondProject }).getByText("Unavailable", { exact: true }).waitFor();
  const resumeButton = page.getByRole("button", { name: "Resume schedule" });
  await resumeButton.focus();
  await resumeButton.press("Enter");
  const review = page.getByRole("dialog", { name: "Review Schedule Jobs" });
  await review.getByText("E2E saved project job").waitFor();
  await review.getByText("Upcoming").waitFor();
  await page.keyboard.press("Escape");
  await review.waitFor({ state: "hidden" });
  assert.equal(await resumeButton.evaluate((element) => element === document.activeElement), true);
  await resumeButton.click();
  await review.getByRole("button", { name: "Resume schedule" }).click();
  await page.locator('ul[aria-label="Projects"] > li').filter({ hasText: firstProject }).getByText("Schedule active").waitFor();

  await page.getByRole("navigation").getByRole("link", { name: "Status" }).click();
  await page.getByRole("heading", { name: "Service status", exact: true }).waitFor();
  await page.route("**/api/v1/clients", (route) => route.abort());
  await page.evaluate(() => window.__myclawTestSocket.close());
  await page.getByRole("status").first().getByText(/Reconnecting|恢复连接中/).waitFor();
  await page.getByRole("status").first().getByText(/Offline|离线/).waitFor({ timeout: 10000 });
  await page.unroute("**/api/v1/clients");
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor({ timeout: 10000 });
  console.log("Playwright production E2E: 4 locale/theme combinations x 3 viewports, ticket, refresh, focus, reconnect passed");
} finally {
  await browser?.close();
  await control?.shutdown();
}
