import assert from "node:assert/strict";
import { chromium, expect as playwrightExpect } from "@playwright/test";
import setup from "./e2e-setup.mjs";
import { newProjectConversation, registerProjectFromSidebar } from "./project-ui.mjs";
import { setInterfaceLanguage } from "./settings-e2e.mjs";

const expect = playwrightExpect.configure({ timeout: 15000 });
const control = await setup();
let browser;
let activeSocket;
let latestServiceEvent = null;
let emittedSequence = 0;
let workspaceId;
let sessionId;
const taskIds = [
  "00000000-0000-4000-8000-000000000001",
  "00000000-0000-4000-8000-000000000002",
  "00000000-0000-4000-8000-000000000003",
  "00000000-0000-4000-8000-000000000004",
  "00000000-0000-4000-8000-000000000005",
  "00000000-0000-4000-8000-000000000006",
];
const statuses = ["running", "running", "queued", "completed", "failed", "interrupted"];
const fixtureItems = Array.from({ length: 22 }, (_, index) => {
  const agentId = taskIds[index] ?? `00000000-0000-4000-8000-${String(index + 1).padStart(12, "0")}`;
  const status = statuses[index] ?? (index === 6 ? "cancelled" : index === 21 ? "running" : "completed");
  return {
    agent_id: agentId,
    title: index === 0 ? "Alpha task" : index === 1 ? "Beta task" : index === 2 ? "Gamma task" : `Task ${index + 1}`,
    status,
    created_at: "2026-10-08T00:00:00+00:00",
    finished_at: status === "running" || status === "queued" ? null : "2026-10-08T00:01:00+00:00",
    result_preview: status === "completed" ? `Result ${index + 1}` : null,
    error: status === "failed" || status === "interrupted"
      ? { code: status, message: `${status} detail` }
      : null,
    usage: { model_calls: 1, input_tokens: index + 1, output_tokens: 2, total_tokens: index + 3 },
  };
});
const fixtureDetails = new Map(fixtureItems.map((item) => [item.agent_id, {
  workspace_id: workspaceId,
  session_id: sessionId,
  agent_id: item.agent_id,
  title: item.title,
  task: `Input for ${item.title}`,
  source: "foreground",
  status: item.status,
  created_at: item.created_at,
  started_at: item.status === "queued" ? null : item.created_at,
  finished_at: item.finished_at,
  revision: 2,
  conversation: [
    { role: "user", content: `Input for ${item.title}` },
    { role: "assistant", content: `Stored output for ${item.title}` },
    ...Array.from({ length: 18 }, (_, index) => ({
      role: "assistant",
      content: `Additional ${item.title} output ${index + 1}. ${"Detailed output line. ".repeat(8)}`,
    })),
  ],
  result: item.status === "completed" ? `Result ${item.title}` : null,
  error: item.error,
  usage: item.usage,
}]));

function serviceEvent(type, agentId, revision, data) {
  assert.ok(activeSocket && latestServiceEvent, "The authenticated event stream is not ready.");
  const sequence = Math.max(emittedSequence, latestServiceEvent.seq) + 1;
  emittedSequence = sequence;
  activeSocket.send(JSON.stringify({
    protocol_version: 1,
    service_instance_id: latestServiceEvent.service_instance_id,
    stream_id: latestServiceEvent.stream_id,
    seq: sequence,
    type,
    workspace_id: type === "snapshot.required" ? null : workspaceId,
    project_id: null,
    session_id: type === "snapshot.required" ? null : sessionId,
    run_id: null,
    payload: { agent_id: agentId, revision, occurred_at: "2026-10-08T00:02:00+00:00", data },
  }));
}

try {
  browser = await chromium.launch({ channel: process.env.AIDE_E2E_BROWSER_CHANNEL ?? "msedge" });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const errors = [];
  let failGammaCancellation = true;
  let failBetaCancellation = true;
  let delayedPage = null;
  page.on("pageerror", (error) => errors.push(error.message));

  await page.route(/\/api\/v1\/workspaces\/[^/]+\/sessions\/[^/]+\/subagents(?:\/[^/?]+)?(?:\?.*)?$/, async (route) => {
    const request = route.request();
    const url = new globalThis.URL(request.url());
    const pathParts = url.pathname.split("/");
    workspaceId = pathParts[pathParts.indexOf("workspaces") + 1];
    sessionId = pathParts[pathParts.indexOf("sessions") + 1];
    const agentId = pathParts.at(-1);
    if (request.method() === "GET" && agentId === "subagents") {
      const offset = Number(url.searchParams.get("cursor") ?? 0);
      const limit = Number(url.searchParams.get("limit") ?? 20);
      const items = fixtureItems.slice(offset, offset + limit);
      const next = offset + items.length;
      if (offset > 0 && delayedPage !== null) {
        const gate = delayedPage;
        gate.arrived();
        await gate.wait;
      }
      await route.fulfill({ json: {
        workspace_id: workspaceId,
        session_id: sessionId,
        items,
        next_cursor: next < fixtureItems.length ? String(next) : null,
      } });
      return;
    }
    if (request.method() === "DELETE" && agentId === taskIds[2] && failGammaCancellation) {
      failGammaCancellation = false;
      await route.fulfill({ status: 503, json: {
        code: "subagent_unavailable",
        message: "Cancellation is temporarily unavailable.",
        retryable: true,
      } });
      return;
    }
    if (request.method() === "DELETE" && agentId === taskIds[1] && failBetaCancellation) {
      failBetaCancellation = false;
      await route.fulfill({ status: 503, json: {
        code: "subagent_unavailable", message: "Beta cancellation unavailable.", retryable: true,
      } });
      return;
    }
    if (request.method() === "DELETE") {
      const detail = fixtureDetails.get(agentId);
      assert.ok(detail, `Unknown fixture task ${agentId}`);
      detail.workspace_id = workspaceId;
      detail.session_id = sessionId;
      detail.status = "cancelled";
      detail.revision += 1;
      detail.finished_at = "2026-10-08T00:03:00+00:00";
      const item = fixtureItems.find((candidate) => candidate.agent_id === agentId);
      item.status = "cancelled";
      item.finished_at = detail.finished_at;
      await route.fulfill({ json: {
        workspace_id: workspaceId,
        session_id: sessionId,
        cancelled: true,
        agent: detail,
      } });
      return;
    }
    const detail = fixtureDetails.get(agentId);
    assert.ok(detail, `Unknown fixture task ${agentId}`);
    detail.workspace_id = workspaceId;
    detail.session_id = sessionId;
    await route.fulfill({ json: detail });
  });
  await page.route("**/api/v1/workspaces/*/runtime/status", (route) => route.fulfill({ json: {
    request_id: "fixture-status",
    workspace_id: workspaceId,
    status: {
      version: "test",
      chat_model: "fixture-model",
      chat_reasoning_effort: "mid",
      uptime_seconds: 1,
      context_window: 8192,
      max_output: 1024,
      available_context: 7168,
      compact_ratio: 0.9,
      compact_context_window: 6452,
      projected_next_request_tokens: 125,
      projection_source: "estimated",
      input_budget_used_percent: 1.7,
      session_message_count: 0,
      last_compacted: 0,
      cumulative_usage: { model_calls: 2, input_tokens: 100, output_tokens: 50, total_tokens: 150 },
      last_request_usage: null,
      configured_permission_level: "workspace-write",
      current_permission_level: "workspace-write",
    },
  } }));
  await page.routeWebSocket(/\/api\/v1\/events$/, (socket) => {
    activeSocket = socket;
    const server = socket.connectToServer();
    server.onMessage((message) => {
      const text = typeof message === "string" ? message : message.toString();
      try {
        const value = JSON.parse(text);
        if (typeof value?.seq === "number" && typeof value?.service_instance_id === "string") {
          latestServiceEvent = value;
        }
      } catch {
        // Command acknowledgements are not events.
      }
      socket.send(message);
    });
  });

  await page.goto(`${control.details.url}/#ticket=${encodeURIComponent(control.details.ticket)}`);
  await setInterfaceLanguage(page, "en");
  await page.locator("#app-sidebar").getByRole("link", { name: "New conversation", exact: true }).click();
  await expect(page.getByLabel("Message input", { exact: true })).toBeEnabled();
  const trigger = page.getByRole("button", { name: "SubAgent tasks", exact: true });
  await expect(trigger).toBeVisible();
  await trigger.focus();
  await page.keyboard.press("Enter");

  const panel = page.getByRole("dialog", { name: "SubAgent tasks", exact: true });
  await expect(panel).toBeVisible();
  for (const status of ["Queued", "Running", "Completed", "Failed", "Cancelled", "Interrupted"]) {
    await expect(panel.getByText(status, { exact: true }).first()).toBeVisible();
  }
  await expect(panel.getByText("Result 4", { exact: true })).toBeVisible();
  await expect(panel.getByText("failed detail", { exact: true })).toBeVisible();
  await expect(panel.getByText("interrupted detail", { exact: true })).toBeVisible();

  await expect.poll(() => latestServiceEvent !== null).toBeTruthy();
  const alphaTaskButton = panel.getByRole("button", { name: /Alpha task/ });
  await alphaTaskButton.focus();
  await page.keyboard.press("Enter");
  const taskInputHeading = panel.getByRole("heading", { name: "Task input", exact: true });
  await expect(taskInputHeading.locator("..").getByText("Input for Alpha task", { exact: true })).toBeVisible();
  await expect(panel.getByText("Stored output for Alpha task", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Conversation history").getByText("Stored output for Alpha task", { exact: true })).toHaveCount(0);
  const taskUsage = panel.getByRole("region", { name: "Task usage", exact: true });
  await expect(taskUsage).toContainText("calls 1 · input 1 · output 2 · total 3");
  const detailScroll = panel.getByRole("region", { name: "Scrollable task details", exact: true });
  await detailScroll.focus();
  await page.keyboard.press("PageDown");
  await expect.poll(() => detailScroll.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
  const alphaTask = panel.locator(`[data-agent-id="${taskIds[0]}"]`);
  serviceEvent("subagent.status", taskIds[0], 1, {
    status: "failed",
    error: { code: "stale", message: "Stale snapshot status" },
  });
  await expect(alphaTask.getByText("Running", { exact: true })).toBeVisible();
  await expect(alphaTask.getByText("Stale snapshot status", { exact: true })).toHaveCount(0);
  await panel.getByRole("button", { name: "Back to list", exact: true }).click();
  await expect(alphaTaskButton).toBeFocused();
  await alphaTaskButton.click();
  const usage = panel.getByRole("region", { name: "Session usage", exact: true });
  await expect(usage).toContainText("150");
  await expect(usage).toContainText("297");
  await expect(usage).toContainText("447");
  await expect(usage).toContainText("125 / 7,168");

  serviceEvent("subagent.output", taskIds[0], 10, { type: "text_delta", delta: "alpha live" });
  serviceEvent("subagent.output", taskIds[1], 10, { type: "text_delta", delta: "beta live" });
  serviceEvent("subagent.output", taskIds[0], 9, { type: "text_delta", delta: "stale output" });
  await expect(panel.getByText("alpha live", { exact: true })).toBeVisible();
  await expect(alphaTaskButton).toBeFocused();
  await expect(panel.getByText("beta live", { exact: true })).toHaveCount(0);
  await expect(panel.getByText("stale output", { exact: true })).toHaveCount(0);
  await panel.getByRole("button", { name: /Beta task/ }).click();
  await expect(panel.getByText("beta live", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Conversation history").getByText(/alpha live|beta live/)).toHaveCount(0);
  await expect(page.getByLabel("Message input", { exact: true })).toBeEnabled();

  await alphaTaskButton.click();
  serviceEvent("subagent.activity", taskIds[0], 11, {
    type: "tool_call_started", tool_call_id: "alpha-tool", tool_name: "read_file", arguments: "{}",
  });
  const alphaActivity = panel.getByRole("group", { name: "Run activity", exact: true });
  await expect(alphaActivity).toHaveCount(1);
  await alphaActivity.locator("summary").click();
  await expect(alphaActivity.getByText("read_file", { exact: true })).toBeVisible();
  const alphaCheckpoint = fixtureDetails.get(taskIds[0]);
  alphaCheckpoint.revision = 11;
  alphaCheckpoint.conversation.push({ role: "assistant", content: "alpha live", tool_calls: [
    { id: "alpha-tool", name: "read_file", arguments: {} },
  ] });
  serviceEvent("snapshot.required", taskIds[0], 11, {});
  await expect(alphaActivity.getByText("alpha live", { exact: true })).toHaveCount(1);
  if (!await alphaActivity.evaluate((element) => element.open)) await alphaActivity.locator("summary").click();
  serviceEvent("subagent.activity", taskIds[0], 12, {
    type: "tool_call_finished", tool_call_id: "alpha-tool", tool_name: "read_file", status: "success", result: "Alpha tool result",
  });
  await expect(alphaActivity.getByText("Alpha tool result", { exact: true })).toBeVisible();
  serviceEvent("subagent.output", taskIds[0], 13, { type: "text_delta", delta: "Alpha final output" });
  const completedAlpha = fixtureDetails.get(taskIds[0]);
  completedAlpha.conversation.push(
    { role: "tool", tool_call_id: "alpha-tool", name: "read_file", status: "success", content: "Alpha tool result" },
    { role: "assistant", content: "Alpha final output" },
    { role: "assistant", content: "Alpha checkpoint saved" },
  );
  completedAlpha.status = "completed";
  completedAlpha.revision = 14;
  completedAlpha.result = "Alpha completed result";
  fixtureItems[0].status = "completed";
  serviceEvent("subagent.status", taskIds[0], 14, { status: "completed", result: completedAlpha.result });
  await expect(panel.getByText("Alpha completed result", { exact: true }).last()).toBeVisible();
  await expect(panel.getByText("Alpha checkpoint saved", { exact: true })).toBeVisible();
  await expect(panel.getByText("alpha live", { exact: true })).toHaveCount(1);
  await expect(alphaTask.getByRole("button", { name: "Cancel task", exact: true })).toHaveCount(0);
  await panel.getByRole("button", { name: /Beta task/ }).click();

  const gamma = panel.locator(`[data-agent-id="${taskIds[2]}"]`);
  const gammaCancel = gamma.getByRole("button", { name: "Cancel task", exact: true });
  await gammaCancel.focus();
  await page.keyboard.press("Enter");
  await expect(panel.getByRole("alert")).toContainText("Cancellation is temporarily unavailable.");
  await gammaCancel.focus();
  await page.keyboard.press("Enter");
  await expect(gamma).toContainText("Cancelled");
  await expect(gamma.getByRole("button", { name: "Cancel task", exact: true })).toHaveCount(0);

  await expect(panel.getByRole("button", { name: "Load more tasks", exact: true })).toBeVisible();
  await panel.getByRole("button", { name: "Load more tasks", exact: true }).click();
  await expect(panel.getByRole("button", { name: /Task 22/ })).toBeVisible();
  const lastTaskButton = panel.getByRole("button", { name: /Task 22/ });
  await lastTaskButton.focus();
  let releaseNextPage;
  let notifyNextPage;
  const nextPageArrived = new Promise((resolve) => { notifyNextPage = resolve; });
  const nextPageReleased = new Promise((resolve) => { releaseNextPage = resolve; });
  delayedPage = { arrived: notifyNextPage, wait: nextPageReleased };
  serviceEvent("subagent.status", taskIds[3], 3, { status: "completed" });
  await nextPageArrived;
  try {
    await expect(lastTaskButton).toBeFocused();
  } finally {
    delayedPage = null;
    releaseNextPage();
  }
  await expect(lastTaskButton).toBeFocused();

  await page.setViewportSize({ width: 390, height: 844 });
  const bounds = await panel.boundingBox();
  assert.ok(bounds && bounds.width <= 390 && bounds.x >= 0, `Task panel overflowed mobile viewport: ${JSON.stringify(bounds)}`);
  await expect(panel.getByRole("heading", { name: "Beta task", exact: true })).toBeVisible();
  await expect(panel.getByRole("button", { name: "Cancel task", exact: true })).toBeVisible();
  await panel.getByRole("button", { name: "Cancel task", exact: true }).click();
  await expect(panel.getByRole("alert")).toContainText("Beta cancellation unavailable.");
  await expect(panel.getByRole("alert")).toBeVisible();
  await panel.getByRole("button", { name: "Back to list", exact: true }).click();
  const betaTaskButton = panel.getByRole("button", { name: /Beta task/ });
  await expect(betaTaskButton).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(panel.getByRole("button", { name: "Back to list", exact: true })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(panel).toBeHidden();
  await expect(trigger).toBeFocused();

  await page.setViewportSize({ width: 1440, height: 900 });
  await trigger.click();
  await panel.getByRole("button", { name: /Beta task/ }).click();
  serviceEvent("subagent.usage", taskIds[1], 11, {
    usage: { model_calls: 1, input_tokens: 4, output_tokens: 4, total_tokens: 8 },
  });
  serviceEvent("subagent.status", taskIds[1], 12, { status: "running" });
  await expect(taskUsage).toContainText("calls 1 · input 4 · output 4 · total 8");
  const recoveredBeta = fixtureDetails.get(taskIds[1]);
  recoveredBeta.status = "completed";
  recoveredBeta.revision = 15;
  recoveredBeta.usage = { model_calls: 3, input_tokens: 20, output_tokens: 10, total_tokens: 30 };
  recoveredBeta.conversation.push({ role: "assistant", content: "Beta checkpoint recovered" });
  fixtureItems[1].status = "completed";
  fixtureItems[1].usage = recoveredBeta.usage;
  serviceEvent("snapshot.required", taskIds[1], 15, {});
  await expect(panel.getByText("Beta checkpoint recovered", { exact: true })).toBeVisible();
  await expect(taskUsage).toContainText("calls 3 · input 20 · output 10 · total 30");
  await expect(usage).toContainText("323");
  await expect(usage).toContainText("473");
  await expect(usage).toContainText("125 / 7,168");
  await expect(detailScroll.getByRole("button", { name: "Cancel task", exact: true })).toHaveCount(0);
  await panel.getByRole("button", { name: /Task 22/ }).click();
  const lateTaskId = fixtureItems[21].agent_id;
  await expect(detailScroll.getByRole("button", { name: "Cancel task", exact: true })).toBeVisible();
  await panel.getByRole("button", { name: /Beta task/ }).click();
  const lateTask = fixtureDetails.get(lateTaskId);
  lateTask.status = "completed";
  lateTask.revision = 5;
  lateTask.conversation.push({ role: "assistant", content: "Late task checkpoint recovered" });
  fixtureItems[21].status = "completed";
  const disconnectedSocket = activeSocket;
  disconnectedSocket.close();
  await expect.poll(() => activeSocket !== disconnectedSocket).toBeTruthy();
  const lateTaskRow = panel.locator(`[data-agent-id="${lateTaskId}"]`);
  await expect(lateTaskRow.getByText("Completed", { exact: true })).toBeVisible();
  await expect(lateTaskRow.getByRole("button", { name: "Cancel task", exact: true })).toHaveCount(0);
  await panel.getByRole("button", { name: /Task 22/ }).click();
  await expect(panel.getByText("Late task checkpoint recovered", { exact: true })).toBeVisible();
  await expect(detailScroll.getByRole("button", { name: "Cancel task", exact: true })).toHaveCount(0);
  await page.keyboard.press("Escape");

  await page.reload();
  await page.setViewportSize({ width: 1440, height: 900 });
  await trigger.click();
  const reconnectedPanel = page.getByRole("dialog", { name: "SubAgent tasks", exact: true });
  await expect(reconnectedPanel).toBeVisible();
  await expect(reconnectedPanel.getByRole("button", { name: /Alpha task/ })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(reconnectedPanel).toBeHidden();

  fixtureItems[2].status = "queued";
  fixtureItems[2].finished_at = null;
  const projectGamma = fixtureDetails.get(taskIds[2]);
  projectGamma.status = "queued";
  projectGamma.finished_at = null;
  const project = await registerProjectFromSidebar(page, control.details.first_project);
  const origin = new globalThis.URL(page.url()).origin;
  await page.goto(`${origin}/projects/${project.project_id}`);
  await newProjectConversation(page);
  await expect(page.getByLabel("Message input", { exact: true })).toBeEnabled();
  const projectTrigger = page.getByRole("button", { name: "SubAgent tasks", exact: true });
  await expect(projectTrigger).toBeVisible();
  await projectTrigger.click();
  const projectPanel = page.getByRole("dialog", { name: "SubAgent tasks", exact: true });
  await expect(projectPanel.getByText("Queued", { exact: true }).first()).toBeVisible();
  await projectPanel.getByRole("button", { name: /Alpha task/ }).click();
  await expect(projectPanel.getByText("Stored output for Alpha task", { exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(projectPanel).toBeHidden();
  await expect(projectTrigger).toBeFocused();

  assert.deepEqual(errors, []);
  console.log("SubAgent Web acceptance: Chat and Project sessions, statuses, details, isolated Usage, event ordering, keyboard cancellation, pagination, responsive layout, focus, and reconnect passed");
} finally {
  await browser?.close();
  await control.shutdown();
}
