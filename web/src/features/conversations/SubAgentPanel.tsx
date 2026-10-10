import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowLeft,
  Check,
  CircleAlert,
  CircleCheck,
  Clock3,
  ListTodo,
  LoaderCircle,
  RefreshCw,
  Square,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { cancelSubAgent, getRuntimeStatus, getSubAgent, getSubAgents } from "../../shared/service/api";
import type {
  RuntimeStatus,
  ServiceEvent,
  SessionClaim,
  SubAgentDetail,
  SubAgentEventKind,
  SubAgentListItem,
  SubAgentStatus,
  SubAgentUsage,
} from "../../shared/service/protocol";
import { isRecord } from "../../shared/validation";
import styles from "./SubAgentPanel.module.css";

const PAGE_LIMIT = 20;
const VISIBLE_PAGE_SIZE = 20;
const SUBAGENT_EVENT_KINDS = new Set<string>([
  "subagent.status",
  "subagent.output",
  "subagent.activity",
  "subagent.usage",
]);
const SUBAGENT_STATUSES = new Set<SubAgentStatus>([
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
  "interrupted",
]);

type LivePart =
  | { kind: "text"; content: string }
  | {
    kind: "tool";
    id: string;
    name: string;
    arguments: string;
    status: "running" | "completed" | "failed";
    result?: string;
  };

interface LiveSubAgentState {
  revision: number;
  parts: LivePart[];
  events: LiveSubAgentEvent[];
  status?: SubAgentStatus;
  result?: string | null;
  error?: { code: string; message: string } | null;
  usage?: SubAgentUsage;
}

interface LiveSubAgentEvent {
  kind: SubAgentEventKind;
  agentId: string;
  revision: number;
  data: Record<string, unknown>;
}

interface SubAgentPanelProps {
  claim: SessionClaim;
  connectionState: string;
  refreshVersion: number;
  subscribeServiceEvents: (listener: (event: ServiceEvent) => void) => () => void;
  renderConversation: (messages: Record<string, unknown>[]) => React.ReactNode;
}

function emptyUsage(): Required<SubAgentUsage> {
  return { model_calls: 0, input_tokens: 0, output_tokens: 0, total_tokens: 0 };
}

function addUsage(left: SubAgentUsage, right: SubAgentUsage): Required<SubAgentUsage> {
  const total = emptyUsage();
  for (const key of Object.keys(total) as (keyof Required<SubAgentUsage>)[]) {
    total[key] = (left[key] ?? 0) + (right[key] ?? 0);
  }
  return total;
}

function statusFrom(value: unknown): SubAgentStatus | null {
  return typeof value === "string" && SUBAGENT_STATUSES.has(value as SubAgentStatus)
    ? value as SubAgentStatus
    : null;
}

function usageFrom(value: unknown): SubAgentUsage | undefined {
  if (!isRecord(value)) return undefined;
  const usage = emptyUsage();
  for (const key of Object.keys(usage) as (keyof Required<SubAgentUsage>)[]) {
    const candidate = value[key];
    if (typeof candidate === "number" && Number.isFinite(candidate) && candidate >= 0) {
      usage[key] = candidate;
    }
  }
  return usage;
}

function eventData(event: ServiceEvent): LiveSubAgentEvent | null {
  if (!SUBAGENT_EVENT_KINDS.has(event.type)) return null;
  const payload = event.payload;
  if (typeof payload.agent_id !== "string"
    || typeof payload.revision !== "number"
    || !Number.isInteger(payload.revision)
    || !isRecord(payload.data)) return null;
  return {
    kind: event.type as SubAgentEventKind,
    agentId: payload.agent_id,
    revision: payload.revision,
    data: payload.data,
  };
}

function appendLivePart(parts: LivePart[], kind: "text", content: string): LivePart[];
function appendLivePart(parts: LivePart[], kind: "tool", content: LivePart): LivePart[];
function appendLivePart(parts: LivePart[], kind: "text" | "tool", content: string | LivePart): LivePart[] {
  const next = [...parts];
  if (kind === "text") {
    if (typeof content !== "string" || content.length === 0) return next;
    const last = next.at(-1);
    if (last?.kind === "text") next[next.length - 1] = { kind: "text", content: last.content + content };
    else next.push({ kind: "text", content });
    return next;
  }
  if (typeof content !== "string") next.push(content);
  return next;
}

function updateLiveState(current: LiveSubAgentState, event: LiveSubAgentEvent): LiveSubAgentState {
  const { kind, data } = event;
  const next: LiveSubAgentState = {
    ...current, parts: [...current.parts], events: [...current.events, event], revision: event.revision,
  };
  if (kind === "subagent.status") {
    const status = statusFrom(data.status);
    if (status !== null) next.status = status;
    if (typeof data.result === "string" || data.result === null) next.result = data.result;
    if (isRecord(data.error) && typeof data.error.code === "string" && typeof data.error.message === "string") {
      next.error = { code: data.error.code, message: data.error.message };
    } else if (data.error === null) {
      next.error = null;
    }
  } else if (kind === "subagent.output" && data.type === "text_delta" && typeof data.delta === "string") {
    next.parts = appendLivePart(next.parts, "text", data.delta);
  } else if (kind === "subagent.activity" && typeof data.tool_call_id === "string") {
    const index = next.parts.findIndex((part) => part.kind === "tool" && part.id === data.tool_call_id);
    const existing = index < 0 ? null : next.parts[index];
    if (data.type === "tool_call_started") {
      const tool: LivePart = {
        kind: "tool",
        id: data.tool_call_id,
        name: typeof data.tool_name === "string" ? data.tool_name : "",
        arguments: typeof data.arguments === "string" ? data.arguments : "",
        status: "running",
      };
      if (index >= 0) next.parts[index] = tool;
      else next.parts.push(tool);
    } else if (data.type === "tool_call_finished") {
      const tool: LivePart = {
        kind: "tool",
        id: data.tool_call_id,
        name: typeof data.tool_name === "string" ? data.tool_name : "",
        arguments: "",
        ...(existing?.kind === "tool" ? existing : {}),
        status: data.status === "success" ? "completed" : "failed",
        result: typeof data.result === "string" ? data.result : "",
      };
      if (index >= 0) next.parts[index] = tool;
      else next.parts.push(tool);
    }
  } else if (kind === "subagent.usage") {
    next.usage = usageFrom(data.usage) ?? next.usage;
  }
  return next;
}

function liveMessages(detail: SubAgentDetail, live: LiveSubAgentState | undefined): Record<string, unknown>[] {
  if (live === undefined) return detail.conversation;
  const persistedToolIds = new Set<string>();
  const persistedResults = new Set<string>();
  for (const message of detail.conversation) {
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (isRecord(call) && typeof call.id === "string") persistedToolIds.add(call.id);
      }
    }
    if (message.role === "tool" && typeof message.tool_call_id === "string") persistedResults.add(message.tool_call_id);
  }
  const additions: Record<string, unknown>[] = [];
  for (const part of live.parts) {
    if (part.kind === "text") {
      additions.push({ role: "assistant", content: part.content });
      continue;
    }
    if (!persistedToolIds.has(part.id)) additions.push({
      role: "assistant",
      content: "",
      tool_calls: [{
        id: part.id,
        name: part.name,
        arguments: part.arguments,
      }],
    });
    if (part.status !== "running" && !persistedResults.has(part.id)) {
      additions.push({
        role: "tool",
        tool_call_id: part.id,
        name: part.name,
        status: part.status === "completed" ? "success" : "error",
        content: part.result ?? "",
      });
    }
  }
  return [...detail.conversation, ...additions];
}

function statusIcon(status: SubAgentStatus) {
  if (status === "running") return <LoaderCircle className={styles.spinner} size={14} aria-hidden="true" />;
  if (status === "completed") return <Check size={14} aria-hidden="true" />;
  if (status === "failed" || status === "interrupted") return <CircleAlert size={14} aria-hidden="true" />;
  if (status === "cancelled") return <CircleCheck size={14} aria-hidden="true" />;
  return <Clock3 size={14} aria-hidden="true" />;
}

function isTerminal(status: SubAgentStatus): boolean {
  return status !== "queued" && status !== "running";
}

function usageLine(usage: SubAgentUsage, format: (value: number | undefined) => string, labels: {
  calls: string;
  input: string;
  output: string;
  total: string;
}): string {
  return `${labels.calls} ${format(usage.model_calls)} · ${labels.input} ${format(usage.input_tokens)} · ${labels.output} ${format(usage.output_tokens)} · ${labels.total} ${format(usage.total_tokens)}`;
}

export default function SubAgentPanel({
  claim,
  connectionState,
  refreshVersion,
  subscribeServiceEvents,
  renderConversation,
}: SubAgentPanelProps) {
  const { i18n, t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<SubAgentListItem[]>([]);
  const [visibleCount, setVisibleCount] = useState(VISIBLE_PAGE_SIZE);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, SubAgentDetail>>({});
  const [liveStates, setLiveStates] = useState<Record<string, LiveSubAgentState>>({});
  const [canceling, setCanceling] = useState<Record<string, boolean>>({});
  const [listLoading, setListLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [mainStatus, setMainStatus] = useState<RuntimeStatus | null>(null);
  const [mainStatusError, setMainStatusError] = useState(false);
  const [listRefreshVersion, setListRefreshVersion] = useState(0);
  const [detailRefreshVersion, setDetailRefreshVersion] = useState(0);
  const latestEventRevision = useRef<Record<string, number>>({});
  const latestSnapshotRevision = useRef<Record<string, number>>({});
  const selectedAgentRef = useRef(selectedAgentId);
  const liveStatesRef = useRef(liveStates);
  const taskButtons = useRef<Record<string, HTMLButtonElement | null>>({});
  const detailBackButton = useRef<HTMLButtonElement | null>(null);
  selectedAgentRef.current = selectedAgentId;
  liveStatesRef.current = liveStates;

  const applySnapshot = useCallback((detail: SubAgentDetail) => {
    const agentId = detail.agent_id;
    if (detail.workspace_id !== claim.workspace_id || detail.session_id !== claim.session_id
      || detail.revision < (latestSnapshotRevision.current[agentId] ?? -1)) return;
    latestSnapshotRevision.current[agentId] = detail.revision;
    latestEventRevision.current[agentId] = Math.max(latestEventRevision.current[agentId] ?? -1, detail.revision);
    setDetails((current) => ({ ...current, [agentId]: detail }));
    setLiveStates((current) => {
      const previous = current[agentId];
      if (previous === undefined) return current;
      const next = { ...current };
      const pending = previous.events.filter((event) => event.revision > detail.revision);
      if (pending.length === 0) delete next[agentId];
      else next[agentId] = pending.reduce(updateLiveState, { revision: detail.revision, parts: [], events: [] });
      return next;
    });
    setItems((current) => current.map((item) => item.agent_id === agentId
      ? { ...item, status: detail.status, finished_at: detail.finished_at, result_preview: detail.result?.slice(0, 240) ?? null, error: detail.error, usage: detail.usage }
      : item));
  }, [claim.session_id, claim.workspace_id]);

  const fetchAllItems = useCallback(async (active: () => boolean) => {
    let cursor: string | undefined;
    const allItems: SubAgentListItem[] = [];
    do {
      const page = await getSubAgents(claim.workspace_id, claim.session_id, {
        ...(cursor === undefined ? {} : { cursor }),
        limit: PAGE_LIMIT,
      });
      if (!active()) return null;
      if (page.workspace_id !== claim.workspace_id || page.session_id !== claim.session_id) {
        throw new Error("SubAgent list belongs to another Session.");
      }
      allItems.push(...page.items);
      cursor = page.next_cursor ?? undefined;
    } while (cursor !== undefined);
    setItems(allItems);
    const knownIds = new Set(Object.keys(liveStatesRef.current));
    if (selectedAgentRef.current !== null) knownIds.add(selectedAgentRef.current);
    await Promise.all(allItems.filter((item) => knownIds.has(item.agent_id)).map(async (item) => {
      try {
        const detail = await getSubAgent(claim.workspace_id, claim.session_id, item.agent_id);
        if (active()) applySnapshot(detail);
      } catch {
        if (active()) setActionError("subagents.detailFailed");
      }
    }));
    return allItems;
  }, [applySnapshot, claim.session_id, claim.workspace_id]);

  useEffect(() => {
    if (!open || connectionState !== "online") return undefined;
    let active = true;
    let pendingStatus = false;
    setListLoading(true);
    setListError(null);
    setActionError(null);
    const isActive = () => active;
    const refreshStatus = async () => {
      if (pendingStatus) return;
      pendingStatus = true;
      try {
        const response = await getRuntimeStatus(
          claim.workspace_id,
          claim.session_id,
          claim.claim_version,
          claim.reconnect_credential,
        );
        if (active) {
          setMainStatus(response.status);
          setMainStatusError(false);
        }
      } catch {
        if (active) setMainStatusError(true);
      } finally {
        pendingStatus = false;
      }
    };
    void fetchAllItems(isActive).then((allItems) => {
      if (!active || allItems === null) return;
      setListLoading(false);
      const selected = selectedAgentRef.current;
      if (selected !== null && !allItems.some((item) => item.agent_id === selected)) {
        setSelectedAgentId(null);
        setDetails((current) => {
          const next = { ...current };
          delete next[selected];
          return next;
        });
      }
    }).catch(() => {
      if (active) {
        setListError("subagents.listFailed");
        setListLoading(false);
      }
    });
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [claim, connectionState, fetchAllItems, listRefreshVersion, open, refreshVersion]);

  useEffect(() => {
    if (!open || selectedAgentId === null || connectionState !== "online") return undefined;
    let active = true;
    void getSubAgent(claim.workspace_id, claim.session_id, selectedAgentId).then((detail) => {
      if (!active || detail.agent_id !== selectedAgentId
        || detail.workspace_id !== claim.workspace_id || detail.session_id !== claim.session_id) return;
      applySnapshot(detail);
    }).catch(() => {
      if (active) setActionError("subagents.detailFailed");
    });
    return () => { active = false; };
  }, [applySnapshot, claim.session_id, claim.workspace_id, connectionState, detailRefreshVersion, open, selectedAgentId]);

  const handleServiceEvent = useCallback((event: ServiceEvent) => {
    if (event.type === "snapshot.required") {
      setListRefreshVersion((version) => version + 1);
      setDetailRefreshVersion((version) => version + 1);
      return;
    }
    if (event.workspace_id !== claim.workspace_id || event.session_id !== claim.session_id) return;
    const parsed = eventData(event);
    if (parsed === null) return;
    const lastRevision = latestEventRevision.current[parsed.agentId] ?? -1;
    if (parsed.revision <= lastRevision) return;
    latestEventRevision.current[parsed.agentId] = parsed.revision;
    setLiveStates((current) => {
      const previous = current[parsed.agentId] ?? { revision: -1, parts: [], events: [] };
      return {
        ...current,
        [parsed.agentId]: updateLiveState(previous, parsed),
      };
    });
    if (parsed.kind === "subagent.status") {
      const status = statusFrom(parsed.data.status);
      if (status !== null && isTerminal(status)) setDetailRefreshVersion((version) => version + 1);
      setListRefreshVersion((version) => version + 1);
    }
  }, [claim.session_id, claim.workspace_id]);

  useEffect(() => subscribeServiceEvents(handleServiceEvent), [handleServiceEvent, subscribeServiceEvents]);

  const selectedDetail = selectedAgentId === null ? undefined : details[selectedAgentId];
  const selectedLive = selectedAgentId === null ? undefined : liveStates[selectedAgentId];
  useEffect(() => {
    if (!open || selectedAgentId === null) return undefined;
    const narrowScreen = window.matchMedia("(max-width: 600px)");
    const focusDetails = () => {
      if (narrowScreen.matches) detailBackButton.current?.focus();
    };
    focusDetails();
    narrowScreen.addEventListener("change", focusDetails);
    return () => narrowScreen.removeEventListener("change", focusDetails);
  }, [open, selectedAgentId, selectedDetail?.agent_id]);
  const effectiveStatus = (item: SubAgentListItem): SubAgentStatus => (
    liveStates[item.agent_id]?.status ?? item.status
  );
  const visibleItems = items.slice(0, visibleCount);
  const subAgentUsage = useMemo(() => items.reduce((total, item) => addUsage(
    total,
    liveStates[item.agent_id]?.usage ?? item.usage,
  ), emptyUsage()), [items, liveStates]);
  const mainUsage = usageFrom(mainStatus?.cumulative_usage) ?? emptyUsage();
  const totalUsage = addUsage(mainUsage, subAgentUsage);
  const numberFormat = new Intl.NumberFormat(i18n.language);
  const formatTokens = (value: number | undefined) => value == null ? "—" : numberFormat.format(value);
  const usageLabels = {
    calls: t("subagents.calls"),
    input: t("subagents.input"),
    output: t("subagents.output"),
    total: t("subagents.total"),
  };

  async function cancelTask(agentId: string) {
    if (canceling[agentId]) return;
    setCanceling((current) => ({ ...current, [agentId]: true }));
    setActionError(null);
    try {
      const response = await cancelSubAgent(claim.workspace_id, claim.session_id, agentId);
      applySnapshot(response.agent);
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : t("subagents.cancelFailed");
      setActionError(message || t("subagents.cancelFailed"));
    } finally {
      setCanceling((current) => ({ ...current, [agentId]: false }));
    }
  }

  function selectTask(agentId: string) {
    setSelectedAgentId(agentId);
    setActionError(null);
  }

  function closeDetails() {
    const previousId = selectedAgentId;
    setSelectedAgentId(null);
    setActionError(null);
    if (previousId !== null) window.requestAnimationFrame(() => taskButtons.current[previousId]?.focus());
  }

  function taskMessages(detail: SubAgentDetail): Record<string, unknown>[] {
    return liveMessages(detail, selectedLive);
  }

  function TaskStatus({ status }: { status: SubAgentStatus }) {
    return (
      <span className={`${styles.status} ${styles[`status${status}`]}`} data-status={status}>
        {statusIcon(status)}
        {t(`subagents.status.${status}`)}
      </span>
    );
  }

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <button className={styles.trigger} type="button" aria-label={t("subagents.open")} title={t("subagents.open")}>
          <ListTodo size={16} aria-hidden="true" />
          <span>{t("subagents.open")}</span>
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.overlay} />
        <Dialog.Content className={styles.sheet} aria-describedby="subagent-panel-description">
          <div>
          <div className={styles.header}>
            <div className={styles.headerTitle}>
              <Dialog.Title className={styles.title}>{t("subagents.title")}</Dialog.Title>
              <Dialog.Description id="subagent-panel-description" className={styles.visuallyHidden}>
                {t("subagents.description")}
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button className={styles.iconButton} type="button" aria-label={t("controls.close")} title={t("controls.close")}>
                <X size={17} aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>
          {actionError !== null ? (
            <div className={styles.error} role="alert">
              <CircleAlert size={15} aria-hidden="true" />
              <span>{t(actionError)}</span>
            </div>
          ) : null}
          </div>
          <div className={styles.body} data-selected={selectedAgentId !== null}>
            <section className={styles.listPane} aria-label={t("subagents.listLabel")}>
              <div className={styles.listHeader}>
                <h2>{t("subagents.tasks")}</h2>
                <span>{items.length}</span>
              </div>
              {listError !== null ? (
                <div className={styles.error} role="alert">
                  <CircleAlert size={15} aria-hidden="true" />
                  <span>{t(listError)}</span>
                  <button type="button" onClick={() => setListRefreshVersion((version) => version + 1)}>
                    {t("controls.retry")}
                  </button>
                </div>
              ) : null}
              {listLoading && items.length === 0 ? (
                <div className={styles.loading} role="status" aria-live="polite">
                  <RefreshCw className={styles.spinner} size={15} aria-hidden="true" />
                  {t("subagents.loading")}
                </div>
              ) : null}
              {!listLoading && listError === null && items.length === 0 ? (
                <p className={styles.empty}>{t("subagents.empty")}</p>
              ) : null}
              <ul className={styles.taskList}>
                {visibleItems.map((item) => {
                  const status = effectiveStatus(item);
                  const error = liveStates[item.agent_id]?.error ?? item.error;
                  const preview = (liveStates[item.agent_id]?.result ?? item.result_preview)?.slice(0, 240);
                  const isCanceling = canceling[item.agent_id] === true;
                  return (
                    <li className={styles.taskItem} key={item.agent_id} data-agent-id={item.agent_id}>
                      <button
                        className={styles.taskSelect}
                        ref={(node) => { taskButtons.current[item.agent_id] = node; }}
                        type="button"
                        aria-pressed={selectedAgentId === item.agent_id}
                        onClick={() => selectTask(item.agent_id)}
                      >
                        <span className={styles.taskHeading}>
                          <strong>{item.title}</strong>
                          <TaskStatus status={status} />
                        </span>
                        {error !== null ? <span className={styles.taskError}>{error.message}</span> : null}
                        {preview ? <span className={styles.preview}>{preview}</span> : null}
                      </button>
                      {!isTerminal(status) ? (
                        <button
                          className={styles.cancelButton}
                          type="button"
                          aria-label={isCanceling ? t("subagents.canceling") : t("subagents.cancel")}
                          title={isCanceling ? t("subagents.canceling") : t("subagents.cancel")}
                          disabled={connectionState !== "online" || isCanceling}
                          onClick={() => void cancelTask(item.agent_id)}
                        >
                          {isCanceling
                            ? <LoaderCircle className={styles.spinner} size={14} aria-hidden="true" />
                            : <Square size={13} aria-hidden="true" />}
                        </button>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
              {visibleCount < items.length ? (
                <button className={styles.loadMore} type="button" onClick={() => setVisibleCount((count) => count + VISIBLE_PAGE_SIZE)}>
                  {t("subagents.loadMore")}
                </button>
              ) : null}
            </section>

            <section className={styles.detailPane} aria-label={t("subagents.detailLabel")}>
              {selectedAgentId === null ? (
                <div className={styles.detailEmpty}>
                  <p>{t("subagents.selectTask")}</p>
                </div>
              ) : selectedDetail === undefined ? (
                <div className={styles.loading} role="status" aria-live="polite">
                  <RefreshCw className={styles.spinner} size={15} aria-hidden="true" />
                  {t("subagents.loadingDetail")}
                </div>
              ) : (
                <>
                  <div className={styles.detailHeader}>
                    <button ref={detailBackButton} className={styles.backButton} type="button" onClick={closeDetails}>
                      <ArrowLeft size={15} aria-hidden="true" />
                      {t("subagents.backToList")}
                    </button>
                    <Dialog.Close asChild>
                      <button className={styles.iconButton} type="button" aria-label={t("controls.close")} title={t("controls.close")}>
                        <X size={16} aria-hidden="true" />
                      </button>
                    </Dialog.Close>
                  </div>
                  <div
                    className={styles.detailScroll}
                    role="region"
                    tabIndex={0}
                    aria-label={t("subagents.scrollDetails")}
                  >
                    <div className={styles.detailTitle}>
                      <div>
                        <h2>{selectedDetail.title}</h2>
                        <TaskStatus status={selectedLive?.status ?? selectedDetail.status} />
                      </div>
                      {!isTerminal(selectedLive?.status ?? selectedDetail.status) ? (
                        <button
                          className={styles.cancelDetailButton}
                          type="button"
                          disabled={connectionState !== "online" || canceling[selectedAgentId] === true}
                          onClick={() => void cancelTask(selectedAgentId)}
                        >
                          {canceling[selectedAgentId] ? <LoaderCircle className={styles.spinner} size={14} aria-hidden="true" /> : <Square size={13} aria-hidden="true" />}
                          {canceling[selectedAgentId] ? t("subagents.canceling") : t("subagents.cancel")}
                        </button>
                      ) : null}
                    </div>
                    <section className={styles.taskInput}>
                      <h3>{t("subagents.taskInput")}</h3>
                      <p>{selectedDetail.task}</p>
                    </section>
                    <div className={styles.history} role="log" aria-label={t("subagents.outputLabel")} aria-live="off">
                      {renderConversation(taskMessages(selectedDetail))}
                    </div>
                    {selectedLive?.result ?? selectedDetail.result ? (
                      <section className={styles.result}>
                        <h3>{t("subagents.result")}</h3>
                        <p>{selectedLive?.result ?? selectedDetail.result}</p>
                      </section>
                    ) : null}
                    {selectedLive?.error ?? selectedDetail.error ? (
                      <section className={styles.detailError} role="alert">
                        <h3>{t("subagents.error")}</h3>
                        <p>{(selectedLive?.error ?? selectedDetail.error)?.message}</p>
                      </section>
                    ) : null}
                    <section className={styles.taskUsage} aria-label={t("subagents.taskUsage")}>
                      <h3>{t("subagents.taskUsage")}</h3>
                      <p>{usageLine(selectedLive?.usage ?? selectedDetail.usage, formatTokens, usageLabels)}</p>
                    </section>
                  </div>
                </>
              )}
            </section>
          </div>
          <section className={styles.sessionUsage} role="region" aria-label={t("subagents.sessionUsage")}>
            <h2>{t("subagents.sessionUsage")}</h2>
            <dl>
              <div><dt>{t("subagents.main")}</dt><dd>{usageLine(mainUsage, formatTokens, usageLabels)}</dd></div>
              <div><dt>{t("subagents.subagents")}</dt><dd>{usageLine(subAgentUsage, formatTokens, usageLabels)}</dd></div>
              <div><dt>{t("subagents.combined")}</dt><dd>{usageLine(totalUsage, formatTokens, usageLabels)}</dd></div>
            </dl>
            <p>{t("subagents.mainContext")}: {formatTokens(mainStatus?.projected_next_request_tokens)} / {formatTokens(mainStatus?.available_context)}</p>
            {mainStatusError ? <span role="status">{t("subagents.usageUnavailable")}</span> : null}
          </section>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
