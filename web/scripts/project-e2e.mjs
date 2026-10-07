import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { URL } from "node:url";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { registerProjectFromSidebar, selectProjectDirectory, projectItemByPath, projectMenuAction, openProjectMenu } from "./project-ui.mjs";

const expect = playwrightExpect.configure({ timeout: 30000 });
const control = await setup();
const browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));
try {
  await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  await expect(page.getByLabel("Message input")).toBeEnabled();
  const origin = new URL(page.url()).origin;
  await page.goto(`${origin}/projects`);
  await expect(page).toHaveURL(url => url.pathname === "/" || url.pathname === "/chat");
  await expect(page.getByLabel("Message input")).toBeEnabled();
  await expect(page.getByRole("main").getByRole("heading", { name: "Projects", exact: true })).toHaveCount(0);

  const input = page.getByLabel("Message input");
  await input.fill("Keep my current conversation draft");
  const before = page.url();
  let registrations = 0;
  page.on("request", request => {
    if (request.method() === "POST" && request.url().endsWith("/api/v1/projects")) registrations += 1;
  });
  await selectProjectDirectory(page, null);
  assert.equal(registrations, 0);
  assert.equal(page.url(), before);
  await expect(input).toHaveValue("Keep my current conversation draft");

  await page.route("**/api/v1/projects/directory-picker", route => route.fulfill({
    status: 500,
    json: { code: "directory_picker_unavailable", message: "Folder dialog unavailable.", field_errors: {}, retryable: true },
  }), { times: 1 });
  await page.locator("#add-project-button").click();
  const error = page.getByRole("alert").filter({ hasText: "folder selection window" });
  await expect(error).toBeVisible();
  await page.route("**/api/v1/projects/directory-picker", route => route.fulfill({
    json: { request_id: route.request().postDataJSON().request_id, path: null },
  }), { times: 1 });
  await error.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(error).toBeHidden();
  await expect(page.locator("#add-project-button")).toBeEnabled();
  assert.equal(registrations, 0);

  const first = await registerProjectFromSidebar(page, control.details.first_project);
  assert.equal(first.schedule_state, "awaiting_resume");
  await expect(input).toHaveValue("Keep my current conversation draft");
  await expect(page.getByLabel("Absolute local path")).toHaveCount(0);
  const unicodePath = join(control.details.home_root, "中文 空文件夹");
  await mkdir(unicodePath);
  const unicodeProject = await registerProjectFromSidebar(page, unicodePath);
  const duplicate = await registerProjectFromSidebar(page, unicodePath);
  assert.equal(unicodeProject.project_id, duplicate.project_id);
  await expect(page.locator('#app-sidebar ul[aria-label="Projects"] > li')).toHaveCount(2);
  const item = projectItemByPath(page, control.details.first_project);
  const trigger = await projectMenuAction(item, "View workspace directory");
  const directory = page.getByRole("dialog", { name: "Workspace directory", exact: true });
  await expect(directory).toContainText(control.details.first_project);
  await directory.press("Escape");
  await expect(trigger).toBeFocused();

  for (const language of ["en", "zh-CN"]) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
    for (const theme of ["light", "dark"]) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      for (const width of [1440, 375]) {
        await page.setViewportSize({ width, height: 900 });
        const localizedItem = page.locator('#app-sidebar ul[aria-label="Projects"], #app-sidebar ul[aria-label="项目"]')
          .getByRole("listitem").filter({ has: page.getByTitle(control.details.first_project, { exact: true }) });
        const summary = await openProjectMenu(localizedItem);
        const menu = localizedItem.locator("details").first().getByRole("group");
        const bounds = await menu.boundingBox();
        assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width, "Project menu overflowed the viewport");
        await menu.getByRole("button", { name: language === "en" ? "Resume schedule" : "恢复调度", exact: true }).click();
        const review = page.getByRole("dialog", { name: language === "en" ? "Review Schedule Jobs" : "检查定时任务", exact: true });
        await expect(review.getByText("E2E saved project job", { exact: true })).toBeVisible();
        await review.press("Escape");
        await expect(summary).toBeFocused();
        if (width === 375) await summary.press("Escape");
      }
    }
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await projectMenuAction(item, "Resume schedule");
  await page.getByRole("dialog").getByRole("button", { name: "Resume schedule", exact: true }).click();
  await expect(item.getByTitle("Schedule active", { exact: true })).toBeVisible();
  await projectMenuAction(item, "Open schedule");
  await expect(page.getByRole("heading", { name: "Schedule Jobs", exact: true })).toBeVisible();
  await item.getByRole("button", { name: "project-one", exact: true }).click();
  await expect(page.getByRole("heading", { name: "project-one", exact: true })).toBeVisible();
  await projectMenuAction(item, "Remove registration");
  const removal = page.getByRole("dialog", { name: "Remove project registration?", exact: true });
  await removal.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(item).toBeVisible();
  await projectMenuAction(item, "Remove registration");
  await removal.getByRole("button", { name: "Remove registration", exact: true }).click();
  await expect(item).toHaveCount(0);
  await expect(page).toHaveURL(url => url.pathname === "/" || url.pathname === "/chat");
  const restored = await registerProjectFromSidebar(page, control.details.first_project);
  assert.equal(restored.schedule_state, "awaiting_resume");
  await expect(page.getByText("Schedule paused for review", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "中文", exact: true }).click();
  await page.getByRole("button", { name: "浅色", exact: true }).click();
  await page.screenshot({ path: resolve("test-results/project-sidebar.png"), fullPage: true });
  assert.deepEqual(errors, []);
  console.log("Project sidebar acceptance: passed (selection, cancel, retry, Unicode, deduplication, resume, removal, 8 appearance/viewport combinations).");
} catch (error) {
  await page.screenshot({ path: resolve("test-results/project-sidebar-failure.png"), fullPage: true });
  console.error("Project acceptance failure:", await page.getByRole("main").innerText(), errors);
  throw error;
} finally {
  await browser.close();
  await control.shutdown();
}
