import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  reporter: "list",
  globalSetup: "./scripts/e2e-setup.mjs",
  use: {
    channel: process.env.OMNI_E2E_BROWSER_CHANNEL ?? "msedge",
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
});
