import assert from "node:assert/strict";
import { URL } from "node:url";
import { chromium, expect } from "@playwright/test";
import { createServer } from "vite";
import setup from "./e2e-setup.mjs";
import { openServiceStatus } from "./settings-e2e.mjs";

const control = await setup();
let dev;
let browser;
try {
  const origin = new URL(control.details.url).origin;
  dev = await createServer({
    server: {
      host: "127.0.0.1", port: 0,
      proxy: {
        "/api": {
          target: origin, changeOrigin: true, ws: true,
          configure(proxy) {
            // The test proxy forwards the local service's authenticated Origin.
            const setOrigin = (request) => request.setHeader("Origin", origin);
            proxy.on("proxyReq", setOrigin);
            proxy.on("proxyReqWs", setOrigin);
          },
        },
      },
    },
  });
  await dev.listen();
  const address = dev.httpServer.address();
  assert.ok(address && typeof address === "object");
  browser = await chromium.launch({ channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage({ locale: "en" });
  await page.addInitScript(() => window.localStorage.setItem("omni.language", "en"));
  const errors = [];
  const exchanges = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("response", response => {
    if (new URL(response.url()).pathname === "/api/v1/web/ticket") exchanges.push(response.status());
  });
  await page.goto(`http://127.0.0.1:${address.port}/settings#ticket=${encodeURIComponent(control.details.ticket)}`);
  await openServiceStatus(page);
  await expect(page.getByRole("status").first().getByText("Online", { exact: true })).toBeVisible({ timeout: 30000 });
  assert.deepEqual(exchanges, [200], "StrictMode consumed the launch ticket more than once");
  assert.deepEqual(errors, []);
  console.log("Client development E2E: StrictMode authenticates once and connects successfully");
} finally {
  await browser?.close();
  await dev?.close();
  await control.shutdown();
}
