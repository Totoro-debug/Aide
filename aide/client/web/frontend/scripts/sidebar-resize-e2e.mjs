import assert from "node:assert/strict";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";

const control = await setup();
let browser;
try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => window.localStorage.setItem("aide.language", "en"));
  await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
  const input = page.getByLabel("Message input", { exact: true });
  const sidebar = page.locator("#app-sidebar");
  const handle = page.getByRole("separator", { name: "Drag to resize navigation; double-click to reset" });
  const sidebarWidth = () => sidebar.evaluate(element => element.getBoundingClientRect().width);
  const savedWidth = () => page.evaluate(() => window.localStorage.getItem("aide.sidebar.width"));
  const expectWidth = width => expect.poll(sidebarWidth).toBe(width);
  const assertNoOverflow = async () => assert.ok(await page.evaluate(
    () => document.documentElement.scrollWidth <= window.innerWidth), "Page overflowed horizontally");
  async function dragTo(width) {
    const box = await handle.boundingBox();
    const before = await sidebarWidth();
    await page.mouse.move(box.x + box.width / 2, 180);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width / 2 + width - before, 200, { steps: 6 });
    await page.mouse.up();
  }

  await expect(input).toBeEnabled();
  await expectWidth(232);
  assert.equal(await handle.getAttribute("tabindex"), null);
  assert.equal(await handle.evaluate(element => window.getComputedStyle(element).cursor), "col-resize");
  await input.fill("Unsent draft survives resizing");
  await dragTo(360);
  await expectWidth(360);
  assert.equal(await savedWidth(), "360");
  await expect(input).toHaveValue("Unsent draft survives resizing");
  await page.mouse.move(700, 300);
  await expectWidth(360);
  assert.equal(await page.evaluate(() => document.body.style.userSelect), "");
  assert.equal(await page.evaluate(() => document.body.style.cursor), "");
  await page.reload();
  await expect(input).toBeEnabled();
  await expectWidth(360);

  await dragTo(900);
  await expectWidth(480);
  await assertNoOverflow();
  await page.setViewportSize({ width: 1024, height: 768 });
  await expectWidth(384);
  assert.ok(await page.locator("#main-content").evaluate(element => element.getBoundingClientRect().width >= 640));
  assert.equal(await savedWidth(), "480", "Viewport resizing overwrote the preferred width");
  await page.setViewportSize({ width: 1440, height: 900 });
  await expectWidth(480);
  await dragTo(50);
  await expectWidth(200);
  await assertNoOverflow();
  await handle.dblclick();
  await expectWidth(232);
  assert.equal(await savedWidth(), "232");
  await dragTo(400);
  await page.locator("#app-sidebar").getByRole("link", { name: "Settings", exact: true }).click();
  await expect(handle).toBeHidden();
  await expect(sidebar).toBeHidden();
  assert.equal(await page.locator("#main-content").evaluate(element => element.getBoundingClientRect().width), 1440);
  await page.getByRole("button", { name: "Back to app", exact: true }).click();
  await expect(handle).toBeVisible();
  await expectWidth(400);

  for (const width of [1023, 768, 375]) {
    await page.setViewportSize({ width, height: 800 });
    await expect(handle).toBeHidden();
    await page.getByRole("button", { name: "Open navigation", exact: true }).click();
    await expect(sidebar).toBeVisible();
    await expectWidth(Math.min(280, width - 48));
    await page.locator("#app-sidebar").getByRole("button", { name: "Close navigation", exact: true }).click();
    await expect(sidebar).toBeHidden();
    await assertNoOverflow();
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await expectWidth(400);

  // Losing window focus must release capture and restore normal mouse behavior.
  const box = await handle.boundingBox();
  await page.mouse.move(box.x + box.width / 2, 180);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 50, 180);
  await expectWidth(350);
  await page.evaluate(() => window.dispatchEvent(new globalThis.Event("blur")));
  await page.mouse.move(700, 300);
  await page.mouse.up();
  await expectWidth(350);
  assert.equal(await page.evaluate(() => document.body.style.userSelect), "");
  assert.equal(await page.evaluate(() => document.body.style.cursor), "");

  for (const value of ["invalid", "9999", "199"]) {
    await page.evaluate(value => window.localStorage.setItem("aide.sidebar.width", value), value);
    await page.reload();
    await expect(input).toBeEnabled();
    await expectWidth(232);
  }
  await page.addInitScript(() => {
    const storage = globalThis.Storage.prototype;
    const getItem = storage.getItem;
    const setItem = storage.setItem;
    storage.getItem = function (key) {
      if (key === "aide.sidebar.width") throw new Error("Storage unavailable");
      return getItem.call(this, key);
    };
    storage.setItem = function (key, value) {
      if (key === "aide.sidebar.width") throw new Error("Storage unavailable");
      return setItem.call(this, key, value);
    };
  });
  await page.reload();
  await expect(input).toBeEnabled();
  await expectWidth(232);
  await dragTo(320);
  await expectWidth(320);
  await handle.dblclick();
  await expectWidth(232);
  assert.deepEqual(errors, []);
  console.log("Sidebar mouse resizing, limits, persistence, reset, storage fallback and responsive layout: passed");
} finally {
  await browser?.close();
  await control.shutdown();
}
