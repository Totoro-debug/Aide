import assert from "node:assert/strict";
import { expect } from "@playwright/test";

// Only the OS choice is stubbed; registration still crosses the real service.
export async function showProjectNavigation(page) {
  const toggle = page.getByRole("button", { name: /^(Open navigation|打开导航)$/ });
  if (await toggle.isVisible() && await page.locator("#app-sidebar").getAttribute("data-open") !== "true") {
    await toggle.click();
  }
}

export async function selectProjectDirectory(page, path) {
  const pattern = "**/api/v1/projects/directory-picker";
  await page.route(pattern, route => route.fulfill({
    json: { request_id: route.request().postDataJSON().request_id, path },
  }), { times: 1 });
  const sidebar = page.locator("#app-sidebar");
  const add = sidebar.getByRole("button", { name: /^(Add project|添加项目|登记项目)$/ });
  await showProjectNavigation(page);
  const result = path === null ? null : page.waitForResponse(response => (
    response.request().method() === "POST" && response.url().endsWith("/api/v1/projects")
  ));
  const selected = page.waitForResponse(response => response.url().endsWith("/directory-picker"));
  await add.click();
  await selected;
  const response = result === null ? null : await result;
  await expect(add).toBeEnabled();
  return response;
}

export async function registerProjectFromSidebar(page, path) {
  const before = page.url();
  const response = await selectProjectDirectory(page, path);
  assert.equal(response.status(), 200, await response.text());
  assert.equal(page.url(), before, "Adding a project navigated away from the current view");
  return response.json();
}

export function projectItemByPath(page, path) {
  return page.locator('#app-sidebar ul[aria-label="Projects"] > li, #app-sidebar ul[aria-label="项目"] > li').filter({
    has: page.getByTitle(path, { exact: true }),
  });
}

export async function openProjectMenu(item) {
  const menu = item.locator("details").first();
  const trigger = menu.locator("summary");
  await showProjectNavigation(item.page());
  if (await menu.getAttribute("open") === null) await trigger.click();
  return trigger;
}

export async function projectMenuAction(item, label) {
  const trigger = await openProjectMenu(item);
  await item.locator("details").first().getByRole("button", { name: label, exact: true }).click();
  return trigger;
}

export async function openWorkspaceAction(page, label) {
  if (/Runtime status|运行状态/.test(String(label))) {
    const trigger = page.getByRole("button", { name: label, exact: true });
    await trigger.click();
    return trigger;
  }
  await showProjectNavigation(page);
  const match = new globalThis.URL(page.url()).pathname.match(/^\/projects\/([^/]+)/);
  const scope = match === null ? page.locator("#app-sidebar").locator("section").last()
    : page.locator(`button[aria-controls="project-sessions-${match[1]}"]`).locator("..");
  const menu = scope.locator("details").filter({
    has: page.getByRole("button", { name: label, exact: true, includeHidden: true }),
  });
  const trigger = menu.locator("summary");
  if (await menu.getAttribute("open") === null) await trigger.click();
  await menu.getByRole("button", { name: label, exact: true }).click();
  return trigger;
}

export async function newProjectConversation(page) {
  await showProjectNavigation(page);
  const match = new globalThis.URL(page.url()).pathname.match(/^\/projects\/([^/]+)/);
  assert.ok(match, "Expected a selected Project");
  const opened = page.waitForResponse(response => response.url().endsWith("/conversations/open")
    && response.request().method() === "POST"
    && response.request().postDataJSON()?.project_id === match[1]
    && response.request().postDataJSON()?.create_new === true);
  await page.locator(`button[aria-controls="project-sessions-${match[1]}"]`)
    .locator("..").getByRole("button").nth(2).click();
  const response = await opened;
  if (response.ok()) {
    const { session_id } = await response.json();
    await expect.poll(() => page.evaluate(() => (
      JSON.parse(window.localStorage.getItem("omni.browser-recovery") ?? "null")?.session_id
    ))).toBe(session_id);
  }
}

export async function openActiveSessionActions(page) {
  await showProjectNavigation(page);
  const active = page.locator('#app-sidebar button[data-active="true"]').last();
  const menu = active.locator("..").locator("details");
  const trigger = menu.locator("summary");
  if (await menu.getAttribute("open") === null) await trigger.click();
  return trigger;
}

export function latestRestoreMenu(page) {
  return page.getByRole("log").locator("details").filter({
    has: page.getByRole("button", {
      name: /^(Restore conversation and files to before this message|恢复对话和文件到此消息之前)$/,
      includeHidden: true,
    }),
  }).last();
}

export async function openLatestRestore(page) {
  const menu = latestRestoreMenu(page);
  const trigger = menu.locator("summary");
  await trigger.click();
  await menu.getByRole("button", {
    name: /^(Restore conversation and files to before this message|恢复对话和文件到此消息之前)$/,
  }).click();
  return trigger;
}
