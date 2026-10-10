import assert from "node:assert/strict";
import test from "node:test";
import {
  readBrowserRecoverySnapshot,
  reconcileBrowserRecovery,
  writeBrowserRecoverySnapshot,
  clearBrowserRecoverySnapshot,
} from "../src/features/conversations/browserRecovery.ts";

function recovery(effort, draft = true) {
  return {
    version: 1,
    service_instance_id: "service-1",
    target: { kind: "chat", directory: "D:\\workspace" },
    session_id: "session-1",
    draft,
    input_text: "Unsent message",
    model_configuration: { provider_id: "primary", model: "model-1", reasoning_effort: effort },
    scroll_top: 24,
  };
}

function storage(t, snapshot) {
  const previous = globalThis.window;
  const state = { raw: JSON.stringify(snapshot), writes: 0 };
  globalThis.window = {
    localStorage: {
      getItem: () => state.raw,
      setItem: (_key, value) => { state.raw = value; state.writes += 1; },
    },
  };
  t.after(() => {
    if (previous === undefined) delete globalThis.window;
    else globalThis.window = previous;
  });
  return state;
}

for (const effort of ["low", "mid", "high", "xhigh", "max"]) {
  test(`reads canonical ${effort} recovery without writing`, t => {
    const expected = recovery(effort);
    const state = storage(t, expected);
    assert.deepEqual(readBrowserRecoverySnapshot(), expected);
    assert.equal(state.writes, 0);
  });
}

for (const effort of ["medium", "unknown", "MID", null, 1]) {
  test(`rejects malformed effort ${JSON.stringify(effort)} without writing`, t => {
    const state = storage(t, recovery(effort));
    assert.equal(readBrowserRecoverySnapshot(), null);
    assert.equal(state.writes, 0);
  });
}

function connection(overrides = {}) {
  return {
    currentInstanceId: "service-1",
    previousInstanceId: "service-1",
    initial: false,
    expectedRestart: false,
    location: { pathname: "/chat", search: "?session=session-1" },
    ...overrides,
  };
}

test("reconnect retains the draft and current route", () => {
  const saved = recovery("high");
  const decision = reconcileBrowserRecovery(saved, connection());
  assert.deepEqual(decision, {
    snapshot: saved, storage: "keep", serviceChanged: false, navigation: null,
  });
});

test("initial connection restores an unselected conversation", () => {
  const decision = reconcileBrowserRecovery(recovery("high"), connection({
    previousInstanceId: null, initial: true, location: { pathname: "/", search: "" },
  }));
  assert.deepEqual(decision.navigation, {
    kind: "conversation", route: "/chat?session=session-1&directory=D%3A%5Cworkspace",
  });
});

test("initial connection respects an explicit session selection", () => {
  const decision = reconcileBrowserRecovery(recovery("high"), connection({
    previousInstanceId: null, initial: true,
    location: { pathname: "/chat", search: "?session=explicit-session" },
  }));
  assert.equal(decision.navigation, null);
});

test("expected restart migrates identity while preserving unsent work", () => {
  const saved = recovery("high");
  const decision = reconcileBrowserRecovery(saved, connection({
    currentInstanceId: "service-2", expectedRestart: true,
  }));
  assert.equal(decision.storage, "write");
  assert.equal(decision.serviceChanged, true);
  assert.deepEqual(decision.snapshot, { ...saved, service_instance_id: "service-2" });
  assert.equal(saved.service_instance_id, "service-1");
  assert.equal(decision.navigation.kind, "conversation");
});

test("expected restart on settings updates the return destination", () => {
  const decision = reconcileBrowserRecovery(recovery("high"), connection({
    currentInstanceId: "service-2", expectedRestart: true,
    location: { pathname: "/settings", search: "" },
  }));
  assert.deepEqual(decision.navigation, {
    kind: "settings-return", route: "/chat?session=session-1&directory=D%3A%5Cworkspace",
  });
});

test("unexpected service replacement clears the stale conversation", () => {
  const decision = reconcileBrowserRecovery(recovery("high"), connection({
    currentInstanceId: "service-2",
  }));
  assert.equal(decision.snapshot, null);
  assert.equal(decision.storage, "clear");
  assert.deepEqual(decision.navigation, { kind: "conversation", route: "/" });
});

test("stale recovery on initial settings connection keeps the settings page", () => {
  const decision = reconcileBrowserRecovery(recovery("high"), connection({
    currentInstanceId: "service-2", previousInstanceId: null, initial: true,
    location: { pathname: "/settings", search: "" },
  }));
  assert.equal(decision.snapshot, null);
  assert.equal(decision.storage, "clear");
  assert.equal(decision.navigation, null);
});

test("initial project selection respects the requested project", () => {
  const saved = { ...recovery("high"), target: { kind: "project", project_id: "project-1" } };
  const decision = reconcileBrowserRecovery(saved, connection({
    previousInstanceId: null, initial: true,
    location: { pathname: "/projects/project-2", search: "" },
  }));
  assert.equal(decision.navigation, null);
});

test("unavailable browser storage leaves recovery optional", t => {
  storage(t, null);
  const unavailable = () => { throw new Error("Storage unavailable"); };
  Object.assign(globalThis.window.localStorage, {
    getItem: unavailable, setItem: unavailable, removeItem: unavailable,
  });
  assert.equal(readBrowserRecoverySnapshot(), null);
  assert.doesNotThrow(() => writeBrowserRecoverySnapshot(recovery("high")));
  assert.doesNotThrow(() => clearBrowserRecoverySnapshot());
  const decision = reconcileBrowserRecovery(null, connection());
  assert.equal(decision.snapshot, null);
  assert.equal(decision.navigation, null);
});
