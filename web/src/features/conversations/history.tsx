import {
  Activity,
  Ban,
  ChevronDown,
  CircleCheck,
  FolderOpen,
  Info,
  MoreHorizontal,
  RotateCcw,
  ShieldX,
  TriangleAlert
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type {
  RestoreAnchor,
  RestoreMode
} from "../../shared/service/protocol";
import commonStyles from "../../shared/styles/controls.module.css";
import conversationsStyles from "./Conversations.module.css";
import { historyMessageText, historyRoleLabel, historyToolActivities, safeMarkdownUrl } from "./historyProjection.ts";
import type { RunStatus, ToolActivity, ToolStatus } from "./run.ts";
import { toolStatusKey } from "./run.ts";

const styles = { ...commonStyles, ...conversationsStyles };

export function StatusIcon(status: RunStatus | ToolStatus, size = 14) {
  if (status === "unknown") return <Info size={size} aria-hidden="true" />;
  if (status === "completed") return <CircleCheck size={size} aria-hidden="true" />;
  if (status === "failed") return <TriangleAlert size={size} aria-hidden="true" />;
  if (status === "rejected") return <ShieldX size={size} aria-hidden="true" />;
  if (status === "canceled") return <Ban size={size} aria-hidden="true" />;
  return <Activity size={size} aria-hidden="true" />;
}

export function MarkdownContent({ content }: { content: string }) {
  return (
    <div className={styles.markdownContent}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={safeMarkdownUrl}
        components={{
          img: ({ alt }) => alt ? <span className={styles.blockedMedia}>{alt}</span> : null,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function ToolActivityGroup({
  tools,
  t,
  showStatus = true,
}: {
  tools: ToolActivity[];
  t: (key: string) => string;
  showStatus?: boolean;
}) {
  return (
    <details className={styles.toolActivity}>
      <summary className={styles.toolActivityHeader}>
        <span className={styles.toolActivityTitle}>
          <Activity size={14} aria-hidden="true" />
          {t("conversation.toolActivity")}
        </span>
        <span className={styles.toolActivityCount}>{tools.length}</span>
        <ChevronDown className={styles.toolActivityChevron} size={14} aria-hidden="true" />
      </summary>
      <ul className={styles.toolActivityList}>
        {tools.map((tool) => (
          <li className={styles.toolActivityItem} key={tool.toolCallId}>
            <div className={styles.toolActivityItemHeader}>
              <span className={styles.toolName}>{tool.name || t("conversation.unknownTool")}</span>
              {showStatus ? <span className={`${styles.statusBadge} ${styles[`status${tool.status}`]}`}>
                {StatusIcon(tool.status, 12)}
                {t(toolStatusKey(tool.status))}
              </span> : null}
            </div>
            {tool.arguments ? (
              <details className={styles.toolArguments}>
                <summary>{t("conversation.toolArguments")}</summary>
                <pre>{tool.arguments}</pre>
              </details>
            ) : null}
          </li>
        ))}
      </ul>
    </details>
  );
}

export function HistoryMessageView({
  message,
  index,
  t,
  scheduleHistory = false,
  restoreAnchor,
  onRestoreAnchor,
}: {
  message: Record<string, unknown>;
  index: number;
  t: (key: string) => string;
  scheduleHistory?: boolean;
  restoreAnchor?: RestoreAnchor;
  onRestoreAnchor?: (anchorId: number, mode: RestoreMode, trigger: HTMLElement) => void;
}) {
  const role = message.role;
  const messageStatus = message.status;
  if (role === "tool") {
    const rawStatus = typeof messageStatus === "string" ? messageStatus : "error";
    const status: ToolStatus = scheduleHistory && !["success", "refused", "error"].includes(String(messageStatus))
      ? "unknown"
      : rawStatus === "success"
      ? "completed"
      : rawStatus === "refused"
        ? "rejected"
        : rawStatus === "error" && message.content === "Tool call interrupted because the turn was cancelled."
          ? "canceled"
          : "failed";
    const tool = {
      toolCallId: typeof message.tool_call_id === "string" ? message.tool_call_id : `tool-${index}`,
      name: typeof message.name === "string" ? message.name : t("conversation.unknownTool"),
      arguments: "",
      status,
    } satisfies ToolActivity;
    return (
      <article className={styles.historyMessage} data-role="tool" key={`${index}-tool`}>
        <div className={styles.historyMessageRole}>{historyRoleLabel(role, t)}</div>
        <ToolActivityGroup tools={[tool]} t={t} />
        <MarkdownContent content={historyMessageText(message.content)} />
      </article>
    );
  }
  const toolActivities = scheduleHistory && role === "assistant" ? historyToolActivities(message) : [];
  return (
    <article className={styles.historyMessage} data-role={typeof role === "string" ? role : "system"} key={`${index}-${String(role)}`}>
      <div className={styles.historyMessageHeader}>
        <div className={styles.historyMessageRole}>{historyRoleLabel(role, t)}</div>
        {role === "user" && restoreAnchor !== undefined && onRestoreAnchor !== undefined ? (
          <details className={styles.messageRestoreMenu}>
            <summary
              className={styles.messageRestoreTrigger}
              aria-label={t("controls.restoreOptionsForMessage")}
              title={t("controls.restoreOptionsForMessage")}
            >
              <MoreHorizontal size={15} aria-hidden="true" />
            </summary>
            <div className={styles.messageRestoreItems} role="group" aria-label={t("controls.restoreOptionsForMessage")}>
              <button className={styles.messageRestoreItem} type="button" onClick={(event) => {
                event.currentTarget.closest("details")?.removeAttribute("open");
                const trigger = event.currentTarget.closest("details")?.querySelector("summary");
                if (trigger instanceof HTMLElement) onRestoreAnchor(restoreAnchor.anchor_id, "conversation-only", trigger);
              }}>
                <RotateCcw size={14} aria-hidden="true" />
                {t("controls.restoreConversationBeforeMessage")}
              </button>
              <button className={styles.messageRestoreItem} type="button" onClick={(event) => {
                event.currentTarget.closest("details")?.removeAttribute("open");
                const trigger = event.currentTarget.closest("details")?.querySelector("summary");
                if (trigger instanceof HTMLElement) onRestoreAnchor(restoreAnchor.anchor_id, "files", trigger);
              }}>
                <FolderOpen size={14} aria-hidden="true" />
                {t("controls.restoreConversationAndFilesBeforeMessage")}
              </button>
            </div>
          </details>
        ) : null}
      </div>
      {toolActivities.length > 0 ? <ToolActivityGroup tools={toolActivities} t={t} showStatus={false} /> : null}
      <MarkdownContent content={historyMessageText(message.content)} />
    </article>
  );
}

import { useState } from "react";

import type { LiveRun, RunActivity, RunActivityPart, RunActivityStatus } from "./run.ts";
import { isLiveRunActive, runStatusKey } from "./run.ts";

type ConversationHistoryEntry =
  | { kind: "message"; key: string; message: Record<string, unknown>; index: number }
  | { kind: "activity"; key: string; activity: RunActivity };

function conversationHistoryEntries(messages: Record<string, unknown>[]): ConversationHistoryEntry[] {
  const entries: ConversationHistoryEntry[] = [];
  let activityParts: RunActivityPart[] | null = null;
  let activityStatus: RunActivityStatus | null = null;

  const startActivity = () => {
    activityParts ??= [];
  };
  const appendText = (key: string, content: string) => {
    if (!content) return;
    startActivity();
    activityParts?.push({ kind: "text", key, content });
  };
  const flushActivity = () => {
    if (activityParts === null) return;
    const index = entries.length;
    entries.push({
      kind: "activity",
      key: `activity-${index}`,
      activity: { status: activityStatus, parts: activityParts },
    });
    activityParts = null;
    activityStatus = null;
  };

  messages.forEach((message, index) => {
    const role = message.role;
    if (role === "user") {
      flushActivity();
      entries.push({ kind: "message", key: `message-${index}`, message, index });
      return;
    }

    if (role === "assistant") {
      const isInterrupted = message.status === "interrupted";
      const isFailed = message.status === "error";
      if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
        appendText(`assistant-${index}`, historyMessageText(message.content));
        for (const tool of historyToolActivities(message)) {
          startActivity();
          activityParts?.push({ kind: "tool", tool });
        }
        return;
      }
      if (isInterrupted || isFailed) {
        startActivity();
        appendText(`assistant-${index}`, historyMessageText(message.content));
        const error = message.error;
        if (typeof error === "object" && error !== null && "message" in error
          && typeof error.message === "string" && error.message !== message.content) {
          appendText(`error-${index}`, error.message);
        }
        activityStatus = isInterrupted ? "canceled" : "failed";
        flushActivity();
        return;
      }
      if (activityParts !== null) {
        activityStatus = "completed";
        flushActivity();
      }
      entries.push({ kind: "message", key: `message-${index}`, message, index });
      return;
    }

    if (role === "tool") {
      const rawStatus = typeof message.status === "string" ? message.status : "error";
      const status: ToolStatus = rawStatus === "success" ? "completed"
        : rawStatus === "refused" ? "rejected"
          : rawStatus === "error" && message.content === "Tool call interrupted because the turn was cancelled."
            ? "canceled" : "failed";
      const toolCallId = typeof message.tool_call_id === "string" ? message.tool_call_id : `tool-${index}`;
      const result = historyMessageText(message.content);
      startActivity();
      if (status === "canceled") activityStatus = "canceled";
      const existing = activityParts?.find((part) => part.kind === "tool" && part.tool.toolCallId === toolCallId);
      if (existing?.kind === "tool") {
        existing.tool.status = status;
        existing.tool.result = result;
      } else {
        activityParts?.push({ kind: "tool", tool: {
          toolCallId,
          name: typeof message.name === "string" ? message.name : "",
          arguments: "",
          status,
          result,
        } });
      }
      return;
    }

    flushActivity();
    entries.push({ kind: "message", key: `message-${index}`, message, index });
  });
  flushActivity();
  return entries;
}

function RunActivityGroup({
  activity,
  t,
  initiallyOpen = false,
}: {
  activity: RunActivity;
  t: (key: string) => string;
  initiallyOpen?: boolean;
}) {
  const [expanded, setExpanded] = useState(initiallyOpen);
  const toolCount = activity.parts.filter((part) => part.kind === "tool").length;
  return (
    <details className={styles.toolActivity} role="group" aria-label={t("conversation.runActivity")}
      open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
      <summary className={styles.toolActivityHeader}>
        <span className={styles.toolActivityTitle}>
          <Activity size={14} aria-hidden="true" />
          {t("conversation.runActivity")}
        </span>
        {toolCount > 0 ? <span className={styles.toolActivityCount}>{toolCount}</span> : null}
        {activity.status !== null ? (
          <span className={`${styles.statusBadge} ${styles[`status${activity.status}`]}`}>
            {StatusIcon(activity.status, 12)}
            {t(runStatusKey(activity.status))}
          </span>
        ) : null}
        <ChevronDown className={styles.toolActivityChevron} size={14} aria-hidden="true" />
      </summary>
      <ul className={styles.toolActivityList}>
        {activity.parts.map((part) => part.kind === "text" ? (
          <li className={styles.runActivityText} key={part.key}>
            <MarkdownContent content={part.content} />
          </li>
        ) : (
          <li className={styles.runToolActivityItem} key={part.tool.toolCallId}>
            <details className={styles.toolCallCard}>
              <summary className={styles.toolActivityItemHeader}>
                <span className={styles.toolName}>{part.tool.name || t("conversation.unknownTool")}</span>
                <span className={`${styles.statusBadge} ${styles[`status${part.tool.status}`]}`}>
                  {StatusIcon(part.tool.status, 12)}
                  {t(toolStatusKey(part.tool.status))}
                </span>
                <ChevronDown className={styles.toolActivityChevron} size={14} aria-hidden="true" />
              </summary>
              {part.tool.arguments || part.tool.result !== undefined && part.tool.result !== null ? (
                <div className={styles.toolCallBody}>
                {part.tool.arguments ? (
                  <div className={styles.runActivityDetail}>
                    <span>{t("conversation.toolArguments")}</span>
                    <pre>{part.tool.arguments}</pre>
                  </div>
                ) : null}
                {part.tool.result !== undefined && part.tool.result !== null ? (
                  <div className={styles.runActivityDetail}>
                    <span>{t("conversation.toolResult")}</span>
                    <MarkdownContent content={part.tool.result} />
                  </div>
                ) : null}
                </div>
              ) : null}
            </details>
          </li>
        ))}
      </ul>
    </details>
  );
}

export function ConversationHistoryView({
  messages,
  t,
  restoreAnchors = [],
  onRestoreAnchor,
}: {
  messages: Record<string, unknown>[];
  t: (key: string) => string;
  restoreAnchors?: RestoreAnchor[];
  onRestoreAnchor?: (anchorId: number, mode: RestoreMode, trigger: HTMLElement) => void;
}) {
  return <>
    {conversationHistoryEntries(messages).map((entry) => entry.kind === "activity" ? (
      <RunActivityGroup key={entry.key} activity={entry.activity} t={t} />
    ) : (
      <HistoryMessageView
        key={entry.key}
        message={entry.message}
        index={entry.index}
        t={t}
        restoreAnchor={restoreAnchors.find((anchor) => (
          entry.message.role === "user"
          && typeof entry.message.content === "string"
          && typeof entry.message.timestamp === "string"
          && anchor.content === entry.message.content
          && anchor.timestamp === entry.message.timestamp
        ))}
        onRestoreAnchor={onRestoreAnchor}
      />
    ))}
  </>;
}

export function LiveRunView({
  run,
  t,
}: {
  run: LiveRun;
  t: (key: string) => string;
}) {
  const active = isLiveRunActive(run);
  const activityParts: RunActivityPart[] = [];
  run.tools.forEach((tool, index) => {
    const content = run.responseSegments[index];
    if (content) activityParts.push({ kind: "text", key: `response-${index}`, content });
    activityParts.push({ kind: "tool", tool });
  });
  if ((active || run.status !== "completed") && run.assistantContent) {
    activityParts.push({ kind: "text", key: "current-content", content: run.assistantContent });
  }
  if (!active && run.error) activityParts.push({ kind: "text", key: "run-error", content: run.error });
  const activityStatus: RunActivityStatus = active ? "running" : run.status as RunActivityStatus;
  const showActivity = active || run.status !== "completed" || activityParts.length > 0;
  return (
    <article className={styles.liveRun} data-run-id={run.runId ?? run.localId}>
      <div className={styles.historyMessage} data-role="user">
        <div className={styles.historyMessageRole}>{t("sessions.userMessage")}</div>
        <div className={styles.livePrompt}>{run.prompt}</div>
      </div>
      <div className={styles.historyMessage} data-role="assistant">
        <div className={styles.liveRunHeader}>
          <span className={styles.historyMessageRole}>{t("sessions.assistantMessage")}</span>
          <span className={`${styles.statusBadge} ${styles[`status${run.status}`]}`} role="status" aria-live="polite">
            {StatusIcon(run.status)}
            {t(runStatusKey(run.status))}
          </span>
        </div>
        {showActivity ? (
          <RunActivityGroup
            key={`${run.localId}-${active ? "active" : "terminal"}`}
            activity={{ status: activityStatus, parts: activityParts }}
            t={t}
            initiallyOpen={active}
          />
        ) : null}
        {run.status === "completed" && run.assistantContent
          ? <MarkdownContent content={run.assistantContent} />
          : active && !run.assistantContent
            ? <p className={styles.pendingAnswer}>{t("conversation.assistantPending")}</p> : null}
      </div>
    </article>
  );
}
