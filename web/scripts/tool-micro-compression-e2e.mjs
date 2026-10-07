import assert from "node:assert/strict";
import { join } from "node:path";
import { chromium } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { settingsMicroCompressionAcceptance } from "./settings-e2e.mjs";

const control = await setup();
let browser;
try {
  browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
  const context = await browser.newContext({ locale: "en" });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    const OriginalWebSocket = window.WebSocket;
    window.WebSocket = class extends OriginalWebSocket {
      constructor(...args) {
        super(...args);
        window.__omniTestControlCredential = Array.isArray(args[1]) ? args[1][1] : null;
      }
    };
  });
  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await page.goto(new globalThis.URL("/settings", control.details.url).href);
  await settingsMicroCompressionAcceptance({
    page, configPath: join(control.details.home_root, ".omni", "config.toml"),
  });
  assert.deepEqual(errors, []);
} finally {
  try {
    await browser?.close();
  } finally {
    await control.shutdown();
  }
}
