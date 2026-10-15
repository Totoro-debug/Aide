---
status: accepted
---

# Match Complete Tool Search Expressions

Tool Search treats each keyword as one complete expression, including multiword phrases. A query matches an expression when the complete expression occurs at word boundaries, ignoring case; BM25 ranks the matched expressions. Splitting phrases into independently matching words broadens discovery into unrelated MCP capabilities, so we accept narrower recall for partial phrases in exchange for more precise discovery.

MCP Tool Remote Names and MCP Tool Model Names both participate as complete identity expressions. They remain distinct from the bounded supplementary keywords, which use one filtering policy whether configured by the user or previously generated. Built-in discovery vocabulary remains explicitly curated code constants. Identity expressions retain their complete names rather than inheriting restrictions intended for supplementary vocabulary.
