import assert from "node:assert/strict";
import { chromium, expect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { setInterfaceLanguage } from "./settings-e2e.mjs";

const control = await setup();
const requests = [];
let browser;
let statsMode = "populated";
let refreshRevision = 0;
const initialTime = "2026-10-12T04:00:00.000Z";
const newRecordTime = "2026-10-12T04:30:00.000Z";
let holdNextPage = false;
let releaseNextPage;
let failNextPage = false;

const tokens = {
  input_tokens: 120,
  output_tokens: 60,
  total_tokens: 180,
  cached_input_tokens: 20,
  cache_creation_input_tokens: 5,
};

function statsFixture(query) {
  const empty = statsMode === "empty";
  const workspaceBeta = query.get("workspace_key") === "workspace-b";
  const visibleRevision = Date.parse(query.get("to_at")) >= Date.parse(newRecordTime) ? refreshRevision : 0;
  const modelCalls = empty ? 0 : workspaceBeta ? 1 + visibleRevision : 4 + visibleRevision;
  const toolCalls = empty ? 0 : workspaceBeta ? 1 : 2;
  const failedCount = empty ? 0 : 1;
  const successCount = modelCalls + toolCalls - failedCount;
  const filteredTokens = workspaceBeta ? {
    input_tokens: 30,
    output_tokens: 20,
    total_tokens: 50,
    cached_input_tokens: 5,
    cache_creation_input_tokens: 1,
  } : tokens;
  return {
    from_at: query.get("from_at"),
    to_at: query.get("to_at"),
    workspace_key: query.get("workspace_key"),
    totals: {
      model_calls: modelCalls,
      tool_calls: toolCalls,
      success_count: successCount,
      failed_count: failedCount,
      failure_rate: empty ? null : failedCount / (successCount + failedCount),
      retry_count: empty ? 0 : workspaceBeta ? 1 : 3,
      retried_calls: empty ? 0 : workspaceBeta ? 1 : 2,
      unknown_usage_count: empty ? 0 : workspaceBeta ? 0 : 1,
      average_duration_ms: empty ? null : workspaceBeta ? 500 : 850,
    },
    tokens: empty ? { input_tokens: 0, output_tokens: 0, total_tokens: 0,
      cached_input_tokens: 0, cache_creation_input_tokens: 0 } : filteredTokens,
    by_model: empty ? [] : [{ key: "model-gpt", calls: modelCalls, tokens: filteredTokens }],
    by_tool: empty ? [] : [{ key: "exec", calls: toolCalls, tokens: filteredTokens }],
    by_purpose: empty ? [] : [{ key: "agent_run", calls: modelCalls + toolCalls, tokens: filteredTokens }],
    by_error_category: empty ? [] : [{ category: "cancelled", calls: 1 }],
    daily: empty ? [] : [{
      date: "2026-10-12",
      model_calls: modelCalls,
      tool_calls: toolCalls,
      successful_calls: successCount,
      failed_calls: failedCount,
      tokens: filteredTokens,
    }],
    methodology: "Retained finalized records only.",
  };
}

function traceRecord(workspaceKey, workspacePath, sessionId, recordId) {
  return {
    record_id: recordId,
    root_id: `root-${recordId}`,
    workspace_key: workspaceKey,
    workspace_path: workspacePath,
    owner_session_id: sessionId,
    subject_session_id: null,
    origin: "foreground",
    run_id: `run-${recordId}`,
    schedule_run_id: null,
    restore_run_token: null,
    agent_id: null,
    parent_root_id: null,
    parent_record_id: null,
    tool_call_id: null,
    kind: "model",
    purpose: "agent_run",
    tool_name: null,
    model: "model-gpt",
    reasoning_effort: "mid",
    endpoint: null,
    status: "success",
    error: null,
    retry_count: 2,
    started_at: "2026-10-10T10:00:00+00:00",
    finished_at: "2026-10-10T10:00:01+00:00",
    duration_ms: 1000,
    final_attempt_duration_ms: 800,
    usage: {
      state: "reported",
      input_tokens: 120,
      output_tokens: 60,
      total_tokens: 180,
      cached_input_tokens: 20,
      cache_creation_input_tokens: 5,
    },
    capture_state: "complete",
    restored: false,
  };
}

try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, timezoneId: "Asia/Shanghai" });
  await page.clock.setFixedTime(new Date(initialTime));
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.addInitScript(() => window.localStorage.setItem("aide.language", "en"));
  await page.route((url) => url.pathname.startsWith("/api/v1/traces"), async (route) => {
    const url = new globalThis.URL(route.request().url());
    requests.push({ method: route.request().method(), path: url.pathname, query: url.searchParams });
    if (url.pathname === "/api/v1/traces/stats") {
      if (statsMode === "unavailable") {
        await route.fulfill({ status: 503, json: { code: "trace_unavailable", message: "Unavailable" } });
      } else if (statsMode === "error") {
        await route.fulfill({ status: 500, json: { code: "internal_error", message: "Failed" } });
      } else {
        await route.fulfill({ status: 200, json: statsFixture(url.searchParams) });
      }
      return;
    }
    if (url.pathname === "/api/v1/traces") {
      const cursor = url.searchParams.get("cursor");
      if (cursor !== null && holdNextPage) {
        holdNextPage = false;
        await new Promise((resolve) => { releaseNextPage = resolve; });
      }
      if (cursor !== null && failNextPage) {
        failNextPage = false;
        await route.fulfill({ status: 500, json: { code: "internal_error", message: "Failed" } });
        return;
      }
      const isWorkspacePage = url.searchParams.get("limit") === "200" && !url.searchParams.has("from_at");
      if (isWorkspacePage) {
        await route.fulfill({
          status: 200,
          json: {
            items: [
              traceRecord("workspace-a", "D:\\Projects\\Alpha", "session-a", "record-a"),
              traceRecord("workspace-b", "D:\\Projects\\Beta", "session-b", "record-b"),
              traceRecord("workspace-dream", "D:\\Projects\\Dream", null, "record-dream"),
            ],
            next_cursor: null,
            snapshot_sequence: 12,
          },
        });
        return;
      }
      const workspace = url.searchParams.get("workspace_key");
      const record = workspace === "workspace-b"
        ? traceRecord("workspace-b", "D:\\Projects\\Beta", "session-b", "record-b")
        : traceRecord("workspace-a", "D:\\Projects\\Alpha", "session-a", "record-a");
      await route.fulfill({
        status: 200,
        json: {
          items: statsMode === "empty" ? [] : cursor !== null
            ? [traceRecord(record.workspace_key, record.workspace_path, "session-next", "record-next")]
            : [record, { ...traceRecord("workspace-dream", "D:\\Projects\\Dream", null, "record-dream"),
              origin: "auxiliary", purpose: "dream" }],
          next_cursor: statsMode === "empty" || cursor !== null ? null : "next-page",
          snapshot_sequence: 12,
        },
      });
      return;
    }
    await route.fallback();
  });

  await page.goto(`${control.details.url}/#ticket=${control.details.ticket}`);
  await expect(page.locator("#conversation-input")).toBeEnabled();
  await page.waitForTimeout(250);
  assert.equal(requests.length, 0, "Trace APIs are not queried before opening Statistics");

  await page.locator("#app-sidebar").getByRole("link", { name: "Statistics", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Statistics", exact: true })).toBeVisible();
  await expect(page.getByText(/retained finalized records/)).toBeVisible();
  await expect.poll(() => requests.length).toBe(3);
  const metrics = page.getByRole("region", { name: "Execution metrics" });
  await expect(metrics.getByText("4", { exact: true })).toBeVisible();
  const metricText = await metrics.innerText();
  assert.match(metricText, /Model calls\s+4/);
  assert.match(metricText, /Tool calls\s+2/);
  assert.match(metricText, /Succeeded\s+5/);
  assert.match(metricText, /Failed\s+1/);
  assert.match(metricText, /Failure rate\s+16\.7%/);
  assert.match(metricText, /Retry attempts\s+3\s+2 retried calls/);
  assert.match(metricText, /Known tokens\s+180\s+120 input \/ 60 output/);
  assert.match(metricText, /Unknown usage calls\s+1/);
  assert.match(metricText, /Cached input tokens\s+20\s+5 cache creation tokens/);
  assert.match(metricText, /Average duration\s+850 ms/);
  await expect(page.getByRole("button", { name: "model-gpt, 4 calls", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "exec, 2 calls", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Agent Run, 6 calls", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Oct 12, 2026: 4 model calls, 2 Tool calls", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Oct 12, 2026: 180 known tokens", exact: true })).toBeVisible();
  const initialStats = requests.find((request) => request.path.endsWith("/stats"));
  assert.ok(initialStats.query.has("from_at"));
  assert.ok(initialStats.query.has("to_at"));
  assert.equal(initialStats.query.get("from_at"), "2026-10-05T16:00:00.000Z");
  assert.equal(initialStats.query.get("to_at"), initialTime);
  assert.equal(new globalThis.URL(page.url()).pathname, "/statistics");
  await expect(page.getByLabel("Workspace", { exact: true }).locator('option[value="workspace-dream"]'))
    .toHaveText("D:\\Projects\\Dream");
  const auxiliarySources = page.getByRole("region", { name: "Work without a Session" });
  await expect(auxiliarySources.getByText("Dream", { exact: true })).toBeVisible();
  await expect(auxiliarySources.getByText("D:\\Projects\\Dream", { exact: true })).toBeVisible();

  const idleRequestCount = requests.length;
  await page.waitForTimeout(500);
  assert.equal(requests.length, idleRequestCount, "Trace APIs do not poll while the page is idle");

  await page.getByLabel("Workspace", { exact: true }).selectOption("workspace-b");
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).length).toBe(2);
  assert.equal(requests.filter((request) => request.path.endsWith("/stats")).at(-1).query.get("workspace_key"), "workspace-b");
  await expect(metrics.locator("dd").first()).toHaveText("1");
  await expect(page.getByRole("link", { name: /session-b/ })).toHaveAttribute(
    "href", "/traces/sessions/workspace-b/session-b",
  );

  await page.getByRole("button", { name: /model-gpt/ }).click();
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).length).toBe(3);
  const dimensionQuery = requests.filter((request) => request.path.endsWith("/stats")).at(-1).query;
  assert.equal(dimensionQuery.get("workspace_key"), "workspace-b");
  assert.equal(dimensionQuery.get("model"), "model-gpt");

  await page.getByRole("button", { name: "Today", exact: true }).click();
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).length).toBe(4);
  const todayQuery = requests.filter((request) => request.path.endsWith("/stats")).at(-1).query;
  assert.equal(todayQuery.get("from_at"), "2026-10-11T16:00:00.000Z");

  await page.getByRole("button", { name: "30 days", exact: true }).click();
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).length).toBe(5);
  assert.equal(requests.filter((request) => request.path.endsWith("/stats")).at(-1).query.get("from_at"), "2026-09-12T16:00:00.000Z");

  await page.getByRole("button", { name: "Custom", exact: true }).click();
  const beforeInvalidRange = requests.length;
  await page.getByLabel("From", { exact: true }).fill("2026-10-12");
  await page.getByLabel("To", { exact: true }).fill("2026-10-11");
  await page.getByRole("button", { name: "Apply", exact: true }).click();
  await expect(page.getByText("Choose a valid date range.", { exact: true })).toBeVisible();
  assert.equal(requests.length, beforeInvalidRange);
  await page.getByLabel("From", { exact: true }).fill("2026-10-09");
  await page.getByLabel("To", { exact: true }).fill("2026-10-11");
  await page.getByRole("button", { name: "Apply", exact: true }).click();
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).length).toBe(6);
  const customQuery = requests.filter((request) => request.path.endsWith("/stats")).at(-1).query;
  assert.equal(customQuery.get("from_at"), "2026-10-08T16:00:00.000Z");
  assert.equal(customQuery.get("to_at"), "2026-10-11T15:59:59.999Z");

  await page.getByRole("button", { name: "Load more sessions", exact: true }).click();
  await expect.poll(() => requests.filter((request) => request.path === "/api/v1/traces").length).toBe(8);
  assert.equal(requests.filter((request) => request.path === "/api/v1/traces").at(-1).query.get("cursor"), "next-page");
  await expect(page.getByRole("link", { name: /session-next/ })).toBeVisible();

  await page.getByRole("button", { name: "Today", exact: true }).click();
  await expect(metrics.locator("dd").first()).toHaveText("1");
  refreshRevision = 1;
  await page.clock.setFixedTime(new Date("2026-10-12T05:00:00.000Z"));
  const beforeRefreshValue = await metrics.locator("dd").first().innerText();
  const beforeRefreshRequests = requests.length;
  await page.waitForTimeout(500);
  assert.equal(await metrics.locator("dd").first().innerText(), beforeRefreshValue);
  assert.equal(requests.length, beforeRefreshRequests, "New records remain hidden until manual refresh");
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(metrics.locator("dd").first()).toHaveText("2");

  // Replacing the query while a page is in flight must leave the new page usable.
  holdNextPage = true;
  await page.getByRole("button", { name: "Load more sessions", exact: true }).click();
  await expect.poll(() => typeof releaseNextPage).toBe("function");
  await page.getByLabel("Workspace", { exact: true }).selectOption("workspace-a");
  await expect(metrics.locator("dd").first()).toHaveText("5");
  releaseNextPage();
  await expect(page.getByRole("button", { name: "Load more sessions", exact: true })).toBeEnabled();
  failNextPage = true;
  await page.getByRole("button", { name: "Load more sessions", exact: true }).click();
  await expect(page.getByText("Related Sessions could not be loaded.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Load more sessions", exact: true }).click();
  await expect(page.getByRole("link", { name: /session-next/ })).toBeVisible();
  await expect(page.getByText("Related Sessions could not be loaded.", { exact: true })).toHaveCount(0);

  // T11 daily buckets are UTC, even when the browser date inputs use local time.
  await page.getByRole("button", { name: /Oct 12, 2026: .* model calls/ }).click();
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).at(-1).query.get("from_at"))
    .toBe("2026-10-12T00:00:00.000Z");
  assert.equal(requests.filter((request) => request.path.endsWith("/stats")).at(-1).query.get("to_at"), "2026-10-12T23:59:59.999Z");

  await page.getByRole("button", { name: /exec, .* calls/ }).click();
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).at(-1).query.get("tool_name")).toBe("exec");
  await page.getByRole("button", { name: /Agent Run, .* calls/ }).click();
  await expect.poll(() => requests.filter((request) => request.path.endsWith("/stats")).at(-1).query.get("purpose")).toBe("agent_run");
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  assert.equal(requests.filter((request) => request.path.endsWith("/stats")).at(-1).query.get("to_at"), "2026-10-12T23:59:59.999Z", "Refreshing a custom range preserves its boundary");

  statsMode = "empty";
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByText("No trace records for this range.", { exact: true })).toBeVisible();

  statsMode = "error";
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByText("Statistics could not be loaded.", { exact: true })).toBeVisible();

  statsMode = "unavailable";
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByText("Trace storage is unavailable.", { exact: true })).toBeVisible();

  statsMode = "populated";
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Statistics", exact: true })).toBeVisible();
  await setInterfaceLanguage(page, "zh-CN");
  await expect(page.getByRole("heading", { name: "统计", exact: true })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "今天", exact: true }).click();
  await page.getByLabel("工作区", { exact: true }).selectOption("");
  await expect(page.getByRole("region", { name: "执行指标" }).locator("dd").first()).toHaveText("5");
  const mobileLayout = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
    headingVisible: document.querySelector("main h1")?.getBoundingClientRect().width ?? 0,
  }));
  assert.ok(mobileLayout.documentWidth <= mobileLayout.viewportWidth, JSON.stringify(mobileLayout));
  assert.ok(mobileLayout.headingVisible > 0);

  // Exercise the authenticated T11 API as well as the fixed browser fixtures.
  await page.unrouteAll({ behavior: "wait" });
  const realStatsResponse = page.waitForResponse((response) => new globalThis.URL(response.url()).pathname === "/api/v1/traces/stats");
  await page.getByRole("button", { name: "刷新", exact: true }).click();
  const realStats = await realStatsResponse;
  assert.equal(realStats.status(), 200);
  const realStatsBody = await realStats.json();
  assert.equal(realStatsBody.totals.model_calls, 0);
  assert.equal(realStatsBody.totals.failure_rate, null);
  await expect(page.getByText("此范围内没有 Trace 记录。", { exact: true })).toBeVisible();
  assert.deepEqual(pageErrors, []);
  console.log("Statistics E2E passed.");
} finally {
  if (browser !== undefined) await browser.close();
  await control.shutdown();
}
