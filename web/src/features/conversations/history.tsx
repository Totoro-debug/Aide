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
import styles from "../../App.module.css";
import type {
RestoreAnchor,
RestoreMode
} from "../../protocol";
import { historyMessageText,historyRoleLabel,historyToolActivities,safeMarkdownUrl } from "./historyProjection.ts";
import type { RunStatus,ToolActivity,ToolStatus } from "./run.ts";
import { toolStatusKey } from "./run.ts";

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
