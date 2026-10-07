---
status: accepted
---

# Manage Agent Run Context by Projected Token Budget

Foreground and User Schedule Runs use one budget policy before execution and before each ReAct request. Token occupancy, rather than message count, controls compaction. Definitions are in [CONTEXT](../../CONTEXT.md); defaults are in the [configuration template](../../aide/templates/default-config.md).

## Projection and limits

Budgeting includes the complete request: System Prompt, projected messages, and exposed Tool schemas. A compatible latest main-assistant usage anchor supplies reported usage plus an estimated delta. Missing or incompatible provenance triggers a complete local estimate; projection never searches past the latest assistant or adds overlapping request usage. Session cumulative usage remains observational.

Each request revision is checked once. Reaching the Compaction Context Window triggers the maximum compaction permitted by retention rules. A result still above that soft threshold may proceed below Available Context; a result at or above Available Context fails with `model_context_overflow`. Every Provider attempt, including dynamic fallback, checks its actual route capacity. Fallback does not start nested compaction or extra Summary calls.

## Retention and summaries

Retention slices count only the target Run's uncompacted raw messages, excluding prompts and schemas. At Run start, the newest completed Run is retained only when its slice fits within 10% of Available Context; a sole eligible completed Run is selected in full. During ReAct, the current Run is retained when its slice fits within 50%; otherwise it may be selected, while the latest completed Cycle remains intact. A selected current User is re-projected once for the remainder of the Run.

Fact Summary receives newly selected original messages and appends to the Workspace stream. Action Summary combines that selection with its staged predecessor and replaces Session metadata. Both receive complete Tool Results. Cursor, Action Summary, and usage commit with the terminal Session increment; the Workspace Summary stream remains outside that transaction. Fact persistence can survive a later Action Summary failure without advancing the staged Session cursor.

Micro-compression runs after summarization. When more than ten eligible Tool Results remain, older results longer than 512 characters may be replaced in the Provider projection. The latest completed Cycle stays complete. Catalog membership determines eligibility; persisted messages, summaries, Runner results, and Artifacts retain original content.

## Preparation boundaries

A Run captures its Skill snapshot and current-input projection across asynchronous preparation. Task Framing is an isolated Tool-free request for eligible foreground input and is mutually exclusive with Manual Skill Invocation. Its Blackboard guides interpretation, never authorization or workflow control. Framing failure falls back to raw input; staged Blackboard state commits only with an accepted terminal increment. Host-projected Skill or Blackboard content does not replace the original persisted user input.

Runtime status projects the next independent foreground request from committed Session state. It neither exposes uncommitted ReAct state nor adopts a previous call's dynamic fallback as the next initial route.

Dream remains a direct one-shot Memory request with hard capacity checks. It advances the Summary Cursor before processing and applies edits sequentially; a later failure does not roll back the cursor or completed edits.
