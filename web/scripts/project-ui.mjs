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
