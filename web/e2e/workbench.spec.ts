import { expect, test } from "@playwright/test";

test("production workbench opens from one-time ticket and remains usable", async ({ page }, testInfo) => {
  const url = process.env.OMNI_E2E_URL;
  const ticket = process.env.OMNI_E2E_TICKET;
  if (!url || !ticket) throw new Error("The isolated E2E service did not provide a launch ticket.");

  await page.goto(`${url}/#ticket=${encodeURIComponent(ticket)}`);
  await expect(page.getByRole("heading", { name: /Service status|服务状态/ })).toBeVisible();
  await expect(page.getByRole("status").first()).toContainText(/Online|在线/);
  await expect(page).toHaveURL(/\/status$/);

  for (const language of ["en", "zh-CN"] as const) {
    await page.getByRole("button", { name: language === "en" ? "EN" : "中文" }).click();
    await expect(page.locator("html")).toHaveAttribute("lang", language);
    await expect(page.getByRole("heading", { name: language === "en" ? "Service status" : "服务状态" })).toBeVisible();

    for (const theme of ["light", "dark"] as const) {
      await page.getByRole("button", { name: theme === "light" ? /Light|浅色/ : /Dark|深色/ }).click();
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);

      for (const viewport of [
        { width: 1440, height: 900 },
        { width: 1024, height: 768 },
        { width: 768, height: 1024 },
      ]) {
        await page.setViewportSize(viewport);
        await expect(page.getByRole("main")).toBeVisible();
        await expect(page.getByRole("navigation").getByRole("link", { name: language === "en" ? "Status" : "状态" })).toBeVisible();
        const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
        expect(pageWidth).toBeLessThanOrEqual(viewport.width);
        if (viewport.width === 1440) {
          await testInfo.attach(`workbench-${language}-${theme}`, {
            body: await page.screenshot(),
            contentType: "image/png",
          });
        }
      }
    }
  }

  const details = page.getByRole("button", { name: /连接详情|Connection details/ });
  await details.focus();
  await details.press("Enter");
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(details).toBeFocused();

  await page.reload();
  await expect(page.getByRole("heading", { name: "服务状态" })).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("status").first()).toContainText("在线");
});
