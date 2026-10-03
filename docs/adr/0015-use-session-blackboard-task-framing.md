---
status: accepted
---

# Use Session Blackboard Task Framing Before Foreground Agent Runs

A user's task can be continued, narrowed, expanded, replaced, or cancelled across
successive inputs. Blackboard provides one compact goal and completion boundary for
interpreting that continuity without introducing workflow control or a visible task manager.

## Decision

Before every nonempty ordinary foreground user input that is not a Manual Skill
Invocation, MyClaw performs an isolated Task Framing call through the configured `chat`
Model Route with no Tools. Task Framing uses the same Workspace Runtime-owned Model Router
instance supplied to the Agent Loop; it does not construct or own another router. The
call receives only the previous Blackboard, the complete content of the latest assistant
Session message, and the new raw user input. These values are embedded into one
parameterized System Prompt; the request contains no additional User Message. Its strict
JSON decision contains exactly `action`, `task_goal`, and `completion_boundary`.
`task_goal` maps to the Blackboard's internal `goal` field, so the decision keeps,
replaces, or clears one Blackboard containing exactly `goal` and
`completion_boundary`. The output parser may extract JSON from a raw response, one
Markdown fence, or surrounding prose, but the resulting object and action invariants
remain strict. The Runtime imposes no Blackboard character limit.

A Manual Skill Invocation and Task Framing are mutually exclusive within one foreground
Agent Run. A manually invoked Skill does not read, generate, project, update, or clear a
Blackboard and does not add Task Framing model usage. Any Blackboard already persisted on
the Conversation Session remains unchanged and becomes eligible for Task Framing again
on a later non-manual foreground input.

The resolved Blackboard is staged for one foreground Agent Run. The model-visible current
User Message retains the raw input first and appends the Blackboard as two Markdown
sections with the exact following shape; the two values are inserted verbatim after the
Blackboard's existing outer-whitespace trimming:

```markdown
## Task goal

{task_goal}

## Completion boundary

{completion_boundary}
```

The persisted user Session message remains the exact raw input. Blackboard state lives
only in the active Conversation Session's `metadata.blackboard` and has no history,
user-facing command, Tool, event, or presentation surface. It is state for task
interpretation only; it cannot authorize operations, bypass Tool Confirmation, order
work, or otherwise control execution. The foreground System Prompt contains no separate
Blackboard guidance.

Blackboard state and its model usage are committed only with a successfully accepted
foreground Agent Run Session increment. A normal failed, maximum-iteration, or cancelled
Runner result still has an accepted repaired increment and therefore commits the staged
state. Preparation cancellation or failure retains the previous state. A completed Task
Framing call whose response is invalid clears the state if the main increment commits;
a Task Framing model failure also degrades to no Blackboard and the raw input. Task
Framing is excluded from Schedule and Memory execution.

## Consequences

- Every eligible non-manual foreground input adds one synchronous Chat Route call,
  increasing latency, model usage, and cost. Its usage is included in Session cumulative
  usage when the turn commits. Manual Skill Invocations add no Task Framing call or usage.
- The complete latest assistant content can overflow the framing route's context. This is
  accepted; the call then degrades to no Blackboard and the raw Agent Run continues.
- A malformed persisted Blackboard does not make the Session unloadable. It is treated as
  absent and is naturally replaced or cleared by a later committed turn.
- The final Markdown Blackboard sections are not a security boundary because they remain
  inside a `user` role message. Tool schemas, Permission Policy, and Tool Confirmation
  remain authoritative without separate Blackboard guidance in the System Prompt.
- Session messages, Schedule behavior, Memory behavior, Message Bus output, Agent Runner,
  and Tool Gateway contracts otherwise remain unchanged.

The complete implementation contract and acceptance plan are recorded in
[GitHub Issue #174](https://github.com/Totoro-debug/OmniAgent/issues/174).
