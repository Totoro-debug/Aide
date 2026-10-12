import { CircleAlert, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useParams, useSearchParams } from "react-router-dom";

import {
  ApiError,
  getTraceRecord,
  getTraceRecords,
} from "../../shared/service/api";
import type {
  TraceKind,
  TraceListQueryDTO,
  TracePurpose,
  TraceRecordDTO,
  TraceRecordDetailDTO,
  TraceStatus,
} from "../../shared/service/protocol";
import type { AuthState } from "../../shared/service/types.ts";
import commonStyles from "../../shared/styles/controls.module.css";
import moduleStyles from "./TraceSessionView.module.css";
import { TraceRecordDetailView } from "./TraceRecordDetailView";
import { formatDateTime, formatDuration, mergeTraceRecords, traceLoadErrorState, usageLabel } from "./presentation";

const styles = { ...commonStyles, ...moduleStyles };
const RECORD_PAGE_SIZE = 50;
const TRACE_PURPOSES: TracePurpose[] = [
  "agent_run", "subagent_run", "schedule_run", "title_generation", "task_framing",
  "conversation_summary", "action_summary", "dream", "mcp_keywords",
];

type LoadState = "idle" | "loading" | "ready" | "error" | "unavailable";
type TraceFilters = {
  kind: TraceKind | "";
  model: string;
  tool_name: string;
  status: TraceStatus | "";
  purpose: TracePurpose | "";
};

const emptyFilters: TraceFilters = {
  kind: "",
  model: "",
  tool_name: "",
  status: "",
  purpose: "",
};

interface TraceSessionViewProps {
  authState: AuthState;
}

export function TraceSessionView({ authState }: TraceSessionViewProps) {
  const { t, i18n } = useTranslation();
  const { workspaceKey = "", sessionId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const [filters, setFilters] = useState<TraceFilters>(emptyFilters);
  const [records, setRecords] = useState<TraceRecordDTO[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [listState, setListState] = useState<LoadState>("idle");
  const [loadingMore, setLoadingMore] = useState(false);
  const [detail, setDetail] = useState<TraceRecordDetailDTO | null>(null);
  const [detailState, setDetailState] = useState<LoadState | "not-found">("idle");
  const [mobilePane, setMobilePane] = useState<"records" | "details">(() => (
    new URLSearchParams(window.location.search).has("recordId") ? "details" : "records"
  ));
  const [rawSourceName, setRawSourceName] = useState("");
  const [refreshVersion, setRefreshVersion] = useState(0);
  const listRequestVersion = useRef(0);
  const detailRequestVersion = useRef(0);

  const query = useMemo<TraceListQueryDTO>(() => ({
    workspace_key: workspaceKey,
    owner_session_id: sessionId,
    ...(filters.kind ? { kind: filters.kind } : {}),
    ...(filters.model ? { model: filters.model } : {}),
    ...(filters.tool_name ? { tool_name: filters.tool_name } : {}),
    ...(filters.status ? { status: filters.status } : {}),
    ...(filters.purpose ? { purpose: filters.purpose } : {}),
    limit: RECORD_PAGE_SIZE,
  }), [filters, sessionId, workspaceKey]);

  const loadFirstPage = useCallback(async () => {
    const requestVersion = ++listRequestVersion.current;
    setListState("loading");
    setRecords([]);
    setCursor(null);
    setLoadingMore(false);
    try {
      const response = await getTraceRecords(query);
      if (requestVersion !== listRequestVersion.current) return;
      setRecords(response.items);
      setCursor(response.next_cursor);
      setListState("ready");
    } catch (error) {
      if (requestVersion === listRequestVersion.current) setListState(traceLoadErrorState(error));
    }
  }, [query]);

  useEffect(() => {
    if (authState === "ready") void loadFirstPage();
    return () => { listRequestVersion.current += 1; };
  }, [authState, loadFirstPage]);

  const selectedFromUrl = searchParams.get("recordId");
  const selectedRecordId = selectedFromUrl ?? records[0]?.record_id ?? null;

  useEffect(() => {
    const requestVersion = ++detailRequestVersion.current;
    if (authState !== "ready" || selectedRecordId === null) {
      setDetail(null);
      setDetailState("idle");
      return;
    }
    setDetail(null);
    setDetailState("loading");
    void getTraceRecord(selectedRecordId).then((response) => {
      if (requestVersion !== detailRequestVersion.current) return;
      if (response.record.record_id !== selectedRecordId
        || response.record.workspace_key !== workspaceKey
        || response.record.owner_session_id !== sessionId) {
        setDetailState("not-found");
        return;
      }
      setDetail(response);
      setDetailState("ready");
    }).catch((error: unknown) => {
      if (requestVersion === detailRequestVersion.current) {
        setDetailState(error instanceof ApiError && error.status === 404 ? "not-found" : traceLoadErrorState(error));
      }
    });
    return () => {
      if (detailRequestVersion.current === requestVersion) detailRequestVersion.current += 1;
    };
  }, [authState, refreshVersion, selectedRecordId, sessionId, workspaceKey]);

  const groups = useMemo(() => groupTraceRecords(records), [records]);
  const models = useMemo(() => uniqueValues(records.map((record) => record.model)), [records]);
  const tools = useMemo(() => uniqueValues(records.map((record) => record.tool_name)), [records]);

  function setFilter<K extends keyof TraceFilters>(key: K, value: TraceFilters[K]) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  function selectRecord(recordId: string) {
    const next = new URLSearchParams(searchParams);
    next.set("recordId", recordId);
    setSearchParams(next);
    setMobilePane("details");
  }

  async function loadMoreRecords() {
    if (cursor === null || loadingMore) return;
    const requestVersion = ++listRequestVersion.current;
    setLoadingMore(true);
    try {
      const response = await getTraceRecords({ ...query, cursor });
      if (requestVersion !== listRequestVersion.current) return;
      setRecords((current) => mergeTraceRecords(current, response.items));
      setCursor(response.next_cursor);
      setListState("ready");
    } catch (error) {
      if (requestVersion === listRequestVersion.current) setListState(traceLoadErrorState(error));
    } finally {
      if (requestVersion === listRequestVersion.current) setLoadingMore(false);
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div>
          <p className={styles.eyebrow}>{t("traces.eyebrow")}</p>
          <h1 id="trace-session-heading">{t("traces.title")}</h1>
          <p className={styles.sessionIdentity}>
            <span>{t("traces.session")}</span>
            <code>{sessionId}</code>
          </p>
        </div>
        <button
          className={commonStyles.secondaryButton}
          type="button"
          disabled={authState !== "ready" || listState === "loading"}
          onClick={() => {
            setRefreshVersion((current) => current + 1);
            void loadFirstPage();
          }}
        >
          <RefreshCw size={15} aria-hidden="true" />
          {t("traces.refresh")}
        </button>
      </header>

      <section className={styles.filters} aria-label={t("traces.filters")}>
        <label className={styles.filterGroup}>
          <span>{t("traces.kind")}</span>
          <select aria-label={t("traces.kind")} value={filters.kind} onChange={(event) => setFilter("kind", event.target.value as TraceFilters["kind"])}>
            <option value="">{t("traces.allKinds")}</option>
            <option value="model">{t("traces.kindValue.model")}</option>
            <option value="tool">{t("traces.kindValue.tool")}</option>
          </select>
        </label>
        <label className={styles.filterGroup}>
          <span>{t("traces.model")}</span>
          <input aria-label={t("traces.model")} list="trace-model-options" value={filters.model}
            placeholder={t("traces.allModels")} onChange={(event) => setFilter("model", event.target.value)} />
          <datalist id="trace-model-options">
            {models.map((model) => <option key={model} value={model} />)}
          </datalist>
        </label>
        <label className={styles.filterGroup}>
          <span>{t("traces.tool")}</span>
          <input aria-label={t("traces.tool")} list="trace-tool-options" value={filters.tool_name}
            placeholder={t("traces.allTools")} onChange={(event) => setFilter("tool_name", event.target.value)} />
          <datalist id="trace-tool-options">
            {tools.map((tool) => <option key={tool} value={tool} />)}
          </datalist>
        </label>
        <label className={styles.filterGroup}>
          <span>{t("traces.status")}</span>
          <select aria-label={t("traces.status")} value={filters.status} onChange={(event) => setFilter("status", event.target.value as TraceFilters["status"])}>
            <option value="">{t("traces.allStatuses")}</option>
            <option value="success">{t("traces.statusValue.success")}</option>
            <option value="failed">{t("traces.statusValue.failed")}</option>
          </select>
        </label>
        <label className={styles.filterGroup}>
          <span>{t("traces.purpose")}</span>
          <select aria-label={t("traces.purpose")} value={filters.purpose} onChange={(event) => setFilter("purpose", event.target.value as TraceFilters["purpose"])}>
            <option value="">{t("traces.allPurposes")}</option>
            {TRACE_PURPOSES.map((purpose) => (
              <option key={purpose} value={purpose}>{t(`traces.purposeValue.${purpose}`, { defaultValue: purpose })}</option>
            ))}
          </select>
        </label>
        <button className={styles.textButton} type="button" onClick={() => setFilters(emptyFilters)}>
          {t("traces.clearFilters")}
        </button>
      </section>

      {authState !== "ready" ? (
        <p className={styles.stateMessage} role="status">{t(authState === "required" ? "traces.authenticationRequired" : "traces.loading")}</p>
      ) : listState === "loading" || listState === "idle" ? (
        <p className={styles.stateMessage} role="status">{t("traces.loading")}</p>
      ) : listState === "unavailable" ? (
        <div className={styles.errorState} role="alert"><CircleAlert size={17} aria-hidden="true" /><span>{t("traces.storageUnavailable")}</span></div>
      ) : listState === "error" ? (
        <div className={styles.errorState} role="alert"><CircleAlert size={17} aria-hidden="true" /><span>{t("traces.loadError")}</span></div>
      ) : records.length === 0 && selectedFromUrl === null ? (
        <p className={styles.emptyState} role="status">{t("traces.empty")}</p>
      ) : (
        <>
          <div className={styles.mobileTabs} role="tablist" aria-label={t("traces.views")}>
            <button role="tab" aria-selected={mobilePane === "records"} type="button" onClick={() => setMobilePane("records")}>
              {t("traces.records")}
            </button>
            <button role="tab" aria-selected={mobilePane === "details"} type="button" onClick={() => setMobilePane("details")}>
              {t("traces.details")}
            </button>
          </div>
          <div className={`${styles.traceColumns} ${styles[`mobile-${mobilePane}`]}`}>
            <section className={styles.recordPanel} aria-label={t("traces.records")}>
              <div className={styles.panelHeading}>
                <h2>{t("traces.records")}</h2>
                <span>{t("traces.savedCount", { count: records.length })}</span>
              </div>
              {records.length === 0 ? (
                <p className={styles.emptyState} role="status">
                  {selectedFromUrl === null ? t("traces.empty") : t("traces.noFilteredRecords")}
                </p>
              ) : groups.length === 0 ? (
                <p className={styles.emptyState} role="status">{t("traces.noFilteredRecords")}</p>
              ) : (
                <div className={styles.groupList}>
                  {groups.map((group) => (
                    <RecordGroup
                      key={group.rootId}
                      group={group}
                      t={t}
                      language={i18n.language}
                      selectedRecordId={selectedRecordId}
                      onSelect={selectRecord}
                      depth={0}
                    />
                  ))}
                </div>
              )}
              {cursor !== null ? (
                <button className={styles.loadMore} type="button" disabled={loadingMore} onClick={() => void loadMoreRecords()}>
                  {t(loadingMore ? "traces.loadingMore" : "traces.loadMore")}
                </button>
              ) : null}
            </section>

            <section className={styles.detailPanel} aria-label={t("traces.details")}>
              {selectedRecordId === null ? (
                <p className={styles.emptyState}>{t("traces.selectRecord")}</p>
              ) : detailState === "loading" || detailState === "idle" ? (
                <p className={styles.stateMessage} role="status">{t("traces.loadingDetail")}</p>
              ) : detailState === "unavailable" ? (
                <div className={styles.errorState} role="alert"><CircleAlert size={17} aria-hidden="true" /><span>{t("traces.storageUnavailable")}</span></div>
              ) : detailState === "not-found" ? (
                <p className={styles.emptyState} role="status">{t("traces.recordExpired")}</p>
              ) : detailState === "error" || detail === null ? (
                <div className={styles.errorState} role="alert"><CircleAlert size={17} aria-hidden="true" /><span>{t("traces.detailError")}</span></div>
              ) : (
                <TraceRecordDetailView
                  key={`${detail.record.record_id}:${refreshVersion}`}
                  detail={detail}
                  language={i18n.language}
                  rawSourceName={rawSourceName}
                  onRawSourceChange={setRawSourceName}
                  t={t}
                />
              )}
            </section>
          </div>
        </>
      )}
    </div>
  );
}

interface TraceRunGroup {
  rootId: string;
  runId: string | null;
  parentRootId: string | null;
  agentId: string | null;
  records: TraceRecordDTO[];
  children: TraceRunGroup[];
}

function groupTraceRecords(records: TraceRecordDTO[]): TraceRunGroup[] {
  const groups = new Map<string, TraceRunGroup>();
  for (const record of records) {
    let group = groups.get(record.root_id);
    if (group === undefined) {
      group = {
        rootId: record.root_id,
        runId: record.run_id,
        parentRootId: record.parent_root_id,
        agentId: record.agent_id,
        records: [],
        children: [],
      };
      groups.set(group.rootId, group);
    }
    group.records.push(record);
  }
  const roots: TraceRunGroup[] = [];
  for (const group of groups.values()) {
    group.records.sort((left, right) => right.finished_at.localeCompare(left.finished_at));
    const parent = group.parentRootId === null ? undefined : groups.get(group.parentRootId);
    if (parent === undefined || parent === group) roots.push(group);
    else parent.children.push(group);
  }
  const sort = (items: TraceRunGroup[]) => {
    items.sort((left, right) => right.records[0]!.finished_at.localeCompare(left.records[0]!.finished_at));
    for (const item of items) sort(item.children);
  };
  sort(roots);
  return roots;
}

function RecordGroup({
  group,
  t,
  language,
  selectedRecordId,
  onSelect,
  depth,
}: {
  group: TraceRunGroup;
  t: (key: string, options?: Record<string, unknown>) => string;
  language: string;
  selectedRecordId: string | null;
  onSelect: (recordId: string) => void;
  depth: number;
}) {
  const title = group.agentId !== null
    ? t("traces.subagentGroup", { agent: group.agentId })
    : group.runId !== null
      ? t("traces.runGroup", { run: group.runId })
      : t("traces.rootGroup", { root: group.rootId });
  return (
    <section className={styles.runGroup} aria-label={title} data-depth={depth}>
      <div className={styles.groupHeading}>
        <h3>{title}</h3>
        {group.parentRootId !== null && group.children.length === 0 ? (
          <span>{t("traces.parentRun", { root: group.parentRootId })}</span>
        ) : null}
      </div>
      <ul className={styles.recordList}>
        {group.records.map((record) => (
          <li key={record.record_id}>
            <button
              className={selectedRecordId === record.record_id ? styles.recordSelected : styles.recordButton}
              type="button"
              aria-current={selectedRecordId === record.record_id ? "true" : undefined}
              aria-label={t("traces.openRecord", { id: record.record_id })}
              onClick={() => onSelect(record.record_id)}
            >
              <span className={styles.recordPrimary}>
                <strong>{record.kind === "model" ? record.model ?? t("traces.unknownModel") : record.tool_name ?? t("traces.unknownTool")}</strong>
                <span className={record.status === "failed" ? styles.failedStatus : styles.successStatus}>
                  {t(`traces.statusValue.${record.status}`)}
                </span>
              </span>
              <span className={styles.recordSecondary}>
                <span>{formatDateTime(record.finished_at, language)}</span>
                <span>{t(`traces.purposeValue.${record.purpose}`, { defaultValue: record.purpose })}</span>
              </span>
              <span className={styles.recordSecondary}>
                <span>{formatDuration(record.duration_ms, language)}</span>
                <span>{t("traces.retryCount", { count: record.retry_count })}</span>
                <span>{usageLabel(record, t, language)}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
      {group.children.map((child) => (
        <RecordGroup
          key={child.rootId}
          group={child}
          t={t}
          language={language}
          selectedRecordId={selectedRecordId}
          onSelect={onSelect}
          depth={depth + 1}
        />
      ))}
    </section>
  );
}

function uniqueValues(values: Array<string | null>): string[] {
  return Array.from(new Set(values.filter((value): value is string => value !== null))).sort((left, right) => left.localeCompare(right));
}

