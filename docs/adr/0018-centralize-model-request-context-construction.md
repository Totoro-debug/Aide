---
status: accepted
---

# Centralize Agent Loop Model Request Context Construction

`ContextBuilder` is the sole construction boundary for System Prompts and initial Model Request Contexts used by `AgentLoop`, including foreground chat, Schedule Jobs, Session title generation, preflight, and foreground summary-budget projections. It owns the Workspace, Agent Home, timezone, `MemoryManager`, and `SkillLoader`; it reads the current Long-term Memory snapshot when constructing a System Prompt and reads the Skill Loader's current frozen Skill state. It exposes both focused prompt methods and complete message-building methods and does not accept Tool schemas. Current User Messages use the configured local time. A Schedule projection captures its timestamp and System Prompt once so asynchronous budget checks and execution share those values; a foreground projection holds its captured Skill state stable across asynchronous preparation.

`SkillLoader` owns the frozen Skill state directly. Each successful `load()` atomically replaces that state, while failure preserves the previous state. The `/reload_skill` Management Command calls the current Agent Loop's loader directly: an active Agent Run keeps its captured Skill projection, and later requests use the reloaded state without rebuilding the Agent Loop or clearing its Conversation Session. A successful reload returns the new Skill metadata to the Terminal so its completion cache changes with the model-visible catalog and Manual Skill resolution; a failed reload leaves all three views unchanged.

`Blackboard` owns Task Framing generation, prompt construction, model invocation, parsing, and reduction through an asynchronous class method. `AgentRunContextController` and Dream retain locally owned System Prompt and request assembly because they are separate execution responsibilities. `AgentRunner` remains independent from `ContextBuilder`, consumes complete initial messages, and owns ReAct transcript increments and repair messages. Tool schemas remain owned by Tool Gateway and request/status call sites rather than Context Builder.

Model-visible structural wrappers use Markdown. Substantive versioned System Prompt content remains in the `myclaw.templates` package and is not asserted verbatim by tests; single-use assembly-only templates are inlined at their construction sites. Tests verify message roles, ordering, dynamic projection, isolation, escaping, Markdown structure, and execution behavior instead of static prompt wording.

Requirements: [context construction and Skill reload](https://github.com/Totoro-debug/myclaw/issues/203).
