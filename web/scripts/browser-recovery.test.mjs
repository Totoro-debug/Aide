import assert from "node:assert/strict";
import test from "node:test";
import {
  readBrowserRecoverySnapshot,
} from "../src/browserRecovery.ts";

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
