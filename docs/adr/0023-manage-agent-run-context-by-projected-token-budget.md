---
status: accepted
---

# Manage Agent Run Context by Projected Token Budget

Chat, Schedule, and SubAgent Runs prepare their multi-turn context immediately before each logical model request. Startup and Skill loading do not check context budgets or trigger compaction. Token occupancy, rather than message count, controls compaction. One-shot requests, including Title, Dream, and Summary, retain their separate preparation boundaries. Definitions are in [CONTEXT](../../CONTEXT.md); defaults are in the [configuration template](../../aide/templates/default-config.md).

ContextBuilder owns model-visible context construction and projection for these multi-turn Runs. ContextController owns token estimation, budget decisions, compaction control, and staged context state. Its pure `estimate_request_tokens` method calculates local token usage; its `prepare` method coordinates compaction and final validation through the Agent Runner request-preparation interface. Dedicated data types are retained for shared contracts, while controller-local bundles remain internal state. This keeps request construction and control consistent across all three execution paths.

## Projection and limits

Budgeting includes the complete request: System Prompt, projected messages, and exposed Tool schemas. A compatible latest main-assistant usage anchor supplies reported usage. Preparation first compares that historical usage with the compaction threshold; when it already reaches the threshold, compaction starts without estimating the pre-compaction delta. Otherwise, a dedicated local tokenizer measures the net context change and combines it with reported historical usage. Missing or incompatible provenance triggers a complete local tokenizer estimate; projection never searches past the latest assistant or adds overlapping request usage. Character-count estimates are not used for multi-turn requests. Session cumulative usage remains observational.

Local tokenization uses tiktoken with its model-to-encoding mapping for recognized OpenAI models and `o200k_base` for Claude or unknown compatible models. Required vocabularies ship with the application so counting does not download data during request preparation. This replaces character heuristics with reproducible local tokenization while keeping reported usage authoritative for the measured history; a fallback encoding does not promise the Provider's exact token count. Usage anchors identify the estimator version and encoding, and incompatible anchors use a complete local estimate rather than mixing measurements from different tokenizers.

Each request revision is checked once during multi-turn request preparation; ModelRouter does not repeat that budget check. Reaching the Compaction Context Window triggers the maximum compaction permitted by retention rules. After summarization and micro-compression, ContextController checks the final request against Available Context using a complete local tokenizer estimate. A result still above the soft threshold may proceed below Available Context; a result at or above Available Context fails with `model_context_overflow`. Retries on the same route reuse the prepared context. One-shot requests retain their existing route-capacity checks, including checks after dynamic fallback. Only Title and Memory can fall back to Chat; fallback does not start nested compaction or extra Summary calls. Chat, Schedule, and SubAgent never fall back.

## Retention and summaries

Retention slices count only the target Run's uncompacted raw messages, excluding prompts and schemas. Before the first model request, the newest completed Run is retained only when its slice fits within 10% of Available Context; a sole eligible completed Run is selected in full. During subsequent ReAct requests, the current Run is retained when its slice fits within 50%; otherwise it may be selected, while the latest completed Cycle remains intact. A selected current User is re-projected once for the remainder of the Run.

Fact Summary receives newly selected original messages and appends to the Workspace stream. Action Summary combines that selection with its staged predecessor and replaces Session metadata. Both receive complete Tool Results. Cursor, Action Summary, and usage commit with the terminal Session increment; the Workspace Summary stream remains outside that transaction. Fact persistence can survive a later Action Summary failure without advancing the staged Session cursor.

Micro-compression runs after summarization. When more than ten eligible Tool Results remain, older results longer than 512 characters may be replaced in the Provider projection. The latest completed Cycle stays complete. Catalog membership determines eligibility; persisted messages, summaries, Runner results, and Artifacts retain original content.

## Preparation boundaries

A Run captures its Skill snapshot and current-input projection across asynchronous preparation. Task Framing is an isolated Tool-free request for eligible foreground input and is mutually exclusive with Manual Skill Invocation. Its Blackboard guides interpretation, never authorization or workflow control. Framing failure falls back to raw input; staged Blackboard state commits only with an accepted terminal increment. Host-projected Skill or Blackboard content does not replace the original persisted user input.

Runtime status projects the next independent foreground request from committed Session state. It uses the same pure estimation rules without triggering compaction, admission checks, model calls, or persistence. It does not expose uncommitted ReAct state. A previous Title or Memory fallback does not change the next initial route.

Dream remains a direct one-shot Memory request with hard capacity checks. It advances the Summary Cursor before processing and applies edits sequentially; a later failure does not roll back the cursor or completed edits.
