import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { expect as playwrightExpect } from "@playwright/test";

const expect = playwrightExpect.configure({ timeout: 10000 });

export default async function composerAcceptance(page) {
  const permission = page.locator("#composer-permission-trigger");
  const model = page.locator("#composer-model-trigger");
  const permissionMenu = page.locator("#composer-permission-menu");
  const modelMenu = page.locator("#composer-model-menu");
  const effortLabels = ["Low", "Medium", "High", "Very high", "Maximum"];
  const output = resolve("test-results", "composer");
  await mkdir(output, { recursive: true });
  const originalViewport = page.viewportSize();

  await permission.focus();
  await permission.press("ArrowDown");
  await expect(permissionMenu.getByRole("menuitemradio").first()).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(permissionMenu.getByRole("menuitemradio").nth(1)).toBeFocused();
  await page.keyboard.press("End");
  await expect(permissionMenu.getByRole("menuitemradio").last()).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(permissionMenu).toBeHidden();
  await expect(permission).toBeFocused();

  for (const language of ["en", "zh-CN"]) {
    for (const theme of ["light", "dark"]) {
      await page.setViewportSize({ width: 1440, height: 900 });
      await page.getByRole("button", { name: language === "en" ? "EN" : "中文", exact: true }).click();
      await page.getByRole("button", {
        name: language === "en" ? (theme === "light" ? "Light" : "Dark") : (theme === "light" ? "浅色" : "深色"),
        exact: true,
      }).click();
      for (const width of [1440, 768, 375]) {
        await page.setViewportSize({ width, height: 900 });
        const input = page.getByLabel(language === "en" ? "Message input" : "消息输入", { exact: true });
        await input.scrollIntoViewIfNeeded();
        const form = page.locator("form").filter({ has: input });
        const layout = await form.evaluate(element => {
          const bounds = target => {
            const rect = target.getBoundingClientRect();
            return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
          };
          return {
            input: bounds(element.querySelector("textarea")),
            box: bounds(element.querySelector("textarea").parentElement),
            permission: bounds(element.querySelector("#composer-permission-trigger")),
            model: bounds(element.querySelector("#composer-model-trigger")),
            send: bounds(element.querySelector('button[type="submit"]')),
            documentWidth: document.documentElement.scrollWidth,
          };
        });
        assert.ok(layout.documentWidth <= width + 1, `Horizontal overflow at ${language}/${theme}/${width}`);
        assert.ok(layout.permission.left >= layout.box.left && layout.send.right <= layout.box.right,
          `Controls escaped the input box: ${JSON.stringify(layout)}`);
        assert.ok(layout.permission.right <= layout.model.left && layout.model.right <= layout.send.left,
          `Composer controls overlap: ${JSON.stringify(layout)}`);
        assert.ok(layout.input.bottom <= layout.permission.top + 1
          && layout.permission.bottom <= layout.box.bottom && layout.send.bottom <= layout.box.bottom,
        `Composer toolbar must stay below the text and inside the input box: ${JSON.stringify(layout)}`);
        await expect(model).toContainText("High");

        await model.click();
        await expect(modelMenu).toBeVisible();
        const effort = page.getByLabel(language === "en" ? "Reasoning effort" : "推理强度", { exact: true });
        assert.deepEqual(await effort.locator("option").allTextContents(), effortLabels);
        const menuBounds = await modelMenu.boundingBox();
        assert.ok(menuBounds.x >= 0 && menuBounds.x + menuBounds.width <= width + 1
          && menuBounds.y + menuBounds.height <= layout.model.top,
        `Model menu is clipped or failed to open upward: ${JSON.stringify(menuBounds)}`);
        await page.screenshot({ path: resolve(output, `${language}-${theme}-${width}-model.png`) });
        await page.keyboard.press("Escape");
        await expect(modelMenu).toBeHidden();
        await expect(model).toBeFocused();

        await permission.click();
        await expect(permissionMenu.getByRole("menuitemradio")).toHaveCount(3);
        await expect(permissionMenu.locator('[aria-checked="true"]')).toHaveAttribute("data-value", "full-access");
        await page.screenshot({ path: resolve(output, `${language}-${theme}-${width}-permission.png`) });
        await input.click({ position: { x: (await input.boundingBox()).width - 10, y: 8 } });
        await expect(permissionMenu).toBeHidden();
        await model.click();
        await page.keyboard.press("Tab");
        await page.keyboard.press("Tab");
        await expect(modelMenu).toBeHidden();
      }
    }
  }

  await page.setViewportSize(originalViewport);
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await page.getByRole("button", { name: "Light", exact: true }).click();
  const modelSelect = page.getByLabel("Session model", { exact: true });
  const originalModel = await modelSelect.inputValue();
  const modelValues = await modelSelect.locator("option").evaluateAll(options => options.filter(option => !option.disabled).map(option => option.value));
  for (const value of [...modelValues, originalModel]) {
    await model.click();
    await modelSelect.selectOption(value);
    await expect(modelSelect).toHaveValue(value);
    await expect(modelSelect).toBeEnabled();
  }
  const effort = page.getByLabel("Reasoning effort", { exact: true });
  for (const value of ["low", "medium", "high", "xhigh", "max", "high"]) {
    await model.click();
    await effort.selectOption(value);
    await expect(effort).toHaveValue(value);
    await expect(effort).toBeEnabled();
  }
  console.log("Composer E2E: 12 layout/language/theme combinations, English effort labels, five effort selections, keyboard and menu dismissal passed");
}
