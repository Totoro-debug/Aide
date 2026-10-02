import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium, expect } from "@playwright/test";
import { URL } from "node:url";

const baseUrl = process.env.MYCLAW_E2E_URL;
const ticket = process.env.MYCLAW_E2E_TICKET;
const workspace = process.env.MYCLAW_E2E_WORKSPACE;
const output = process.env.MYCLAW_E2E_OUTPUT;
const prompt = "installed package conversation\nstreaming markdown";
assert.ok(baseUrl && ticket && workspace);

const browser = await chromium.launch({
  channel: process.env.MYCLAW_E2E_BROWSER_CHANNEL ?? (process.platform === "win32" ? "msedge" : undefined),
});
const context = await browser.newContext();
const page = await context.newPage();
const watchdog = setTimeout(() => { void browser.close(); }, 90000);
const errors = [];
let websocketObserved = false;
const events = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("websocket", (socket) => {
  if (socket.url().includes("/api/v1/events")) {
    websocketObserved = true;
    socket.on("framereceived", ({ payload }) => {
      if (typeof payload === "string") events.push(JSON.parse(payload));
    });
  }
});

try {
  const documentResponse = await context.request.get(baseUrl);
  assert.equal(documentResponse.status(), 200, "Installed package did not serve index.html");
  assert.match(
    documentResponse.headers()["cache-control"] ?? "",
    /no-store/,
    "Installed package document did not disable HTML caching",
  );
  assert.match(
    documentResponse.headers()["content-security-policy"] ?? "",
    /default-src 'self'/,
    "Installed package document did not include CSP",
  );
  const html = await documentResponse.text();
  const javascriptAsset = html.match(/src="(\/assets\/[^"']+\.js)"/)?.[1];
  const stylesheetAsset = html.match(/href="(\/assets\/[^"']+\.css)"/)?.[1];
  assert.ok(javascriptAsset && stylesheetAsset, "Installed package HTML did not reference JS/CSS");
  for (const [asset, contentType] of [[javascriptAsset, /javascript/], [stylesheetAsset, /text\/css/], ["/favicon.svg", /image\/svg\+xml/]]) {
    const response = await context.request.get(`${baseUrl}${asset}`);
    assert.equal(response.status(), 200, `Installed asset did not load: ${asset}`);
    assert.match(response.headers()["content-type"] ?? "", contentType, `Wrong MIME for ${asset}`);
    assert.match(response.headers()["cache-control"] ?? "", /immutable/, `Wrong cache policy for ${asset}`);
  }
  for (const missing of ["/assets/missing.js", "/missing.css", "/api/v1/missing"]) {
    const response = await context.request.get(`${baseUrl}${missing}`);
    assert.equal(response.status(), 404, `Missing resource became a SPA document: ${missing}`);
  }

  await page.goto(`${baseUrl}/#ticket=${encodeURIComponent(ticket)}`);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor();
  assert.match(page.url(), /\/status$/);
  assert.ok(websocketObserved, "Installed Web app did not open its authenticated WebSocket");
  const serviceResponse = await context.request.get(`${baseUrl}/api/v1/service`);
  assert.equal(serviceResponse.status(), 200, "Installed Web app API did not respond");
  const service = await serviceResponse.json();
  assert.equal(service.service_instance_id, process.env.MYCLAW_E2E_INSTANCE);

  await page.goto(`${baseUrl}/status`);
  await page.getByRole("heading", { name: /Service status|服务状态/ }).waitFor();
  await page.getByRole("status").first().getByText(/Online|在线/).waitFor();

  await page.getByRole("navigation").getByRole("link", { name: /Settings|设置/, exact: true }).click();
  await page.getByRole("heading", { name: /Settings|设置/ }).waitFor();
  await page.getByRole("heading", { name: /Runtime|运行时/ }).waitFor();
  await page.screenshot({ path: join(output, "settings.png"), fullPage: true });
  await page.goto(`${baseUrl}/settings`);
  await page.getByRole("heading", { name: /Runtime|运行时/ }).waitFor();

  const projectsLink = page.getByRole("navigation").getByRole("link", { name: /Projects|项目/, exact: true });
  await projectsLink.click();
  await page.getByRole("heading", { name: /^(Projects|项目)$/ }).waitFor();
  const addProject = page.getByRole("button", { name: /Add project|添加项目|登记项目/ }).first();
  await addProject.click();
  const dialog = page.getByRole("dialog");
  await dialog.getByLabel(/Absolute local path|本地绝对路径/).fill(workspace);
  const registered = page.waitForResponse((response) => (
    response.url() === `${baseUrl}/api/v1/projects` && response.request().method() === "POST"
  ));
  await dialog.getByRole("button", { name: /Register project|注册项目|登记项目/ }).click();
  const registeredResponse = await registered;
  assert.equal(registeredResponse.status(), 200, await registeredResponse.text());
  await dialog.waitFor({ state: "hidden" });

  await page.getByRole("heading", { name: "workspace", exact: true }).waitFor();
  await page.getByRole("link", { name: /Open sessions|打开会话/ }).click();
  const newSession = page.getByRole("button", { name: /New session|新建会话/ });
  await newSession.click();
  const input = page.getByLabel(/Message input|消息输入/);
  await input.waitFor();
  await input.fill(prompt);
  await input.press("Enter");
  await expect.poll(() => events.filter((event) => (
    event.type === "input.accepted" && event.payload?.text === prompt
  )).length, { timeout: 30000 }).toBe(1);
  const accepted = events.find((event) => event.type === "input.accepted" && event.payload?.text === prompt);
  await expect.poll(() => events.some((event) => (
    event.type === "run.completed" && event.run_id === accepted.run_id
  )), { timeout: 30000 }).toBe(true);
  const answer = page.locator('article[data-role="assistant"]').getByRole("heading", { name: "Streamed answer", exact: true });
  await answer.waitFor();
  await page.screenshot({ path: join(output, "conversation.png"), fullPage: true });
  await page.goto(page.url());
  await answer.waitFor();

  assert.deepEqual(errors, [], `Installed Web app reported browser errors: ${errors.join("; ")}`);
  const evidence = {
    marker: "INSTALLED_WEB_E2E_OK",
    base_url: baseUrl,
    route: new URL(page.url()).pathname,
    api: "passed",
    assets: "mime-cache-verified",
    websocket: "passed",
    settings: "passed",
    conversation: "passed",
    accepted_runs: 1,
    completed_run: accepted.run_id,
    service_instance_id: service.service_instance_id,
    missing_resources: "404",
  };
  await writeFile(join(output, "browser.json"), `${JSON.stringify(evidence, null, 2)}\n`);
  console.log(JSON.stringify(evidence));
} catch (error) {
  await page.screenshot({ path: join(output, "failure.png"), fullPage: true }).catch(() => {});
  throw error;
} finally {
  clearTimeout(watchdog);
  await browser.close();
}
