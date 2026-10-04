---
status: accepted
---

# Use Message Bus, Session Scheduling, and Agent Runner

AgentService retains an independent Message Bus per foreground Session and serializes its ordinary inputs with an on-demand processor. Inbound is an editable FIFO. Outbound is forwarded to scoped broker events during each Run; the CLI client projects those events into its terminal presentation bus. It carries sparse reasoning, response, Tool-call, and system-control messages. Ownership, concurrency, and Session Claims follow [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

Foreground model text is published as stream deltas. Exactly one terminal marker ends an Agent Run; it is not a replacement copy of the final response. Tool start notifications carry the Tool name as content and `tool_call_id` plus raw `arguments` as metadata. Tool completion notifications carry the same name and identifier with only `status` (`success`, `error`, or `refused`). They update the existing Tool row without exposing Tool Result content or Artifact references. Missing completion notifications never imply success. Tool Results remain in the Runner's model transcript and Session persistence. Schedule and Dream output never enter foreground Outbound.

Agent Runner is a reusable Session-independent ReAct engine whose constructor owns a Model Router and the narrow request-preparation collaborator required by its owning lane. Each invocation receives complete initial messages, a route, Tool Gateway, output and confirmation callbacks, result externalizer, cancellation policy, iteration limit, and failure policy. It returns the invocation's Provider-valid assistant/Tool increment, final content, four-field usage, finish reason, and optional Error Info. Foreground uses `chat` and User Schedule Jobs use `schedule`. Requests needing only a single completion call the Model Router directly; Dream follows that path as recorded in [ADR-0024](0024-use-one-shot-dream-model-request.md).

One iteration is one model call followed by every requested Tool call in Provider order. Provider retries do not consume iterations. The default and minimum limit is 50; the last allowed response completes its Tool calls before reporting `agent_iteration_limit`, unless normal completion or cancellation takes priority. Tool errors and refusals normally return to the model for continuation. Cancellation repairs incomplete assistant/Tool pairs without undoing accepted side effects.

Foreground and User Schedule execution use the same Agent Runner implementation through separate run-local instances, each bound to that run's Router and request preparer. User Schedule retains its dedicated Session and creates context, cancellation and externalization state for each Run, with its own Session Log. Its available Catalog uses `ToolGateway.for_run()` and excludes `ScheduleTool` while retaining the Workspace's MCP capabilities, as defined by [ADR-0021](0021-defer-tool-schema-exposure-per-agent-run.md). Its confirmation callback submits background envelopes to the service coordinator; it has no foreground output callback. Dream directly owns its one-shot Memory Model request and restricted edit execution.

Requirements: [Message Bus and Runner](https://github.com/Totoro-debug/OmniAgent/issues/162), [shared service](https://github.com/Totoro-debug/OmniAgent/issues/281).
