import { setInterfaceLanguage } from "./settings-e2e.mjs";
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
  const context = await browser.newContext({ locale: "en" });
  const page = await context.newPage();
  await page.addInitScript(() => {
    if (window.localStorage.getItem("omni.language") === null) window.localStorage.setItem("omni.language", "en");
  });
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
  const appOrigin = `http://127.0.0.1:${address.port}`;
  await page.goto(appOrigin);
  await expect(page.getByLabel("Message input")).toBeEnabled();
  const originalClient = await page.evaluate(async () => (
    await (await window.fetch("/api/v1/web/session")).json()
  ).client_id);
  const cookies = await page.context().cookies();
  const duplicate = await page.context().newPage();
  try {
    await duplicate.goto(appOrigin);
    await expect(duplicate.getByRole("alert").filter({ hasText: "Another Web client is already open" })).toBeVisible();
    const repeatedLaunch = await control.command("web-ticket");
    const repeatedTicket = new URL(repeatedLaunch.url).hash;
    await duplicate.goto(`${appOrigin}/${repeatedTicket}`);
    await expect(duplicate.getByRole("alert").filter({ hasText: "Another Web client is already open" })).toBeVisible();
    assert.deepEqual(await page.context().cookies(), cookies, "Duplicate launch replaced the original browser Cookie");
  } finally {
    await duplicate.close();
  }
  const foreign = await browser.newContext({ locale: "en" });
  try {
    const foreignPage = await foreign.newPage();
    await foreignPage.addInitScript(() => window.localStorage.setItem("omni.language", "en"));
    const foreignLaunch = await control.command("web-ticket");
    await foreignPage.goto(`${appOrigin}/${new URL(foreignLaunch.url).hash}`);
    await expect(foreignPage.getByRole("alert").filter({ hasText: "Another Web client is already open" })).toBeVisible();
    assert.equal(await foreignPage.getByRole("log").count(), 0);
  } finally {
    await foreign.close();
  }
  for (const language of ["en", "zh-CN"]) {
    await setInterfaceLanguage(page, language);
    await expect(page.getByRole("button", { name: /Release session|释放会话/ })).toHaveCount(0);
    await page.reload();
    await expect(page.getByLabel(language === "en" ? "Message input" : "消息输入")).toBeEnabled();
    assert.equal(await page.evaluate(async () => (
      await (await window.fetch("/api/v1/web/session")).json()
    ).client_id), originalClient);
  }
  assert.deepEqual(errors, []);
  console.log("Client development E2E: one Web client, blocked tabs and browsers, preserved Cookies, refresh recovery and no release control passed");
} finally {
  await browser?.close();
  await dev?.close();
  await control.shutdown();
}
