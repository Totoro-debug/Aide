import { CircleAlert, Download, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useParams, useSearchParams } from "react-router-dom";

import {
  ApiError,
  downloadTraceRaw,
  getTraceRawPage,
  getTraceRecord,
  getTraceRecords,
} from "../../shared/service/api";
import type {
  TraceKind,
  TraceListQueryDTO,
  TracePurpose,
  TraceRecordDTO,
  TraceRecordDetailDTO,
  TraceRawSourceDTO,
  TraceStatus,
} from "../../shared/service/protocol";
import type { AuthState } from "../../shared/service/types.ts";
import commonStyles from "../../shared/styles/controls.module.css";
import moduleStyles from "./TraceSessionView.module.css";

const styles = { ...commonStyles, ...moduleStyles };
const RECORD_PAGE_SIZE = 50;
const RAW_PAGE_SIZE = 256 * 1024;
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
      if (requestVersion === listRequestVersion.current) setListState(loadStateFor(error));
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
        setDetailState(error instanceof ApiError && error.status === 404 ? "not-found" : loadStateFor(error));
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
      setRecords((current) => mergeRecords(current, response.items));
      setCursor(response.next_cursor);
      setListState("ready");
    } catch (error) {
      if (requestVersion === listRequestVersion.current) setListState(loadStateFor(error));
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

function TraceRecordDetailView({
  detail,
  language,
  rawSourceName,
  onRawSourceChange,
  t,
}: {
  detail: TraceRecordDetailDTO;
  language: string;
  rawSourceName: string;
  onRawSourceChange: (source: string) => void;
  t: (key: string, options?: Record<string, unknown>) => string;
}) {
  const record = detail.record;
  const selectedRawSource = detail.raw_sources.find((source) => source.source === rawSourceName)
    ?? detail.raw_sources[0];
  return (
    <div className={styles.detailContent}>
      <header className={styles.detailHeading}>
        <div>
          <p className={styles.eyebrow}>{t("traces.record")}</p>
          <h2>{record.kind === "model" ? record.model ?? t("traces.unknownModel") : record.tool_name ?? t("traces.unknownTool")}</h2>
          <code>{record.record_id}</code>
        </div>
        <span className={record.status === "failed" ? styles.failedStatus : styles.successStatus}>
          {t(`traces.statusValue.${record.status}`)}
        </span>
      </header>

      <section className={styles.detailSection}>
        <h3>{t("traces.overview")}</h3>
        <dl className={styles.fieldGrid}>
          <DetailField label={t("traces.origin")} value={t(`traces.originValue.${record.origin}`, { defaultValue: record.origin })} />
          <DetailField label={t("traces.purpose")} value={t(`traces.purposeValue.${record.purpose}`, { defaultValue: record.purpose })} />
          <DetailField label={t("traces.kind")} value={t(`traces.kindValue.${record.kind}`)} />
          <DetailField label={t("traces.workspace")} value={record.workspace_path} />
          <DetailField label={t("traces.workspaceKey")} value={record.workspace_key} />
          <DetailField label={t("traces.root")} value={record.root_id} />
          <DetailField label={t("traces.session")} value={record.owner_session_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.subjectSession")} value={record.subject_session_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.run")} value={record.run_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.subagent")} value={record.agent_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.parentRecord")} value={record.parent_record_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.parentRoot")} value={record.parent_root_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.toolCall")} value={record.tool_call_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.scheduleRun")} value={record.schedule_run_id ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.restoreRunToken")} value={record.restore_run_token ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.model")} value={record.model ?? t(record.kind === "model" ? "traces.usageUnknown" : "traces.notApplicable")} />
          <DetailField label={t("traces.reasoningEffort")} value={record.reasoning_effort ?? t(record.kind === "model" ? "traces.usageUnknown" : "traces.notApplicable")} />
          <DetailField label={t("traces.endpoint")} value={record.endpoint ?? t("traces.notApplicable")} />
          <DetailField label={t("traces.status")} value={t(`traces.statusValue.${record.status}`)} />
          <DetailField label={t("traces.retryCountLabel")} value={String(record.retry_count)} />
          <DetailField label={t("traces.startedAt")} value={formatDateTime(record.started_at, language)} />
          <DetailField label={t("traces.finishedAt")} value={formatDateTime(record.finished_at, language)} />
          <DetailField label={t("traces.duration")} value={formatDuration(record.duration_ms, language)} />
          <DetailField label={t("traces.finalAttemptDuration")} value={record.final_attempt_duration_ms === null
            ? t("traces.notApplicable") : formatDuration(record.final_attempt_duration_ms, language)} />
          <DetailField label={t("traces.usage")} value={usageLabel(record, t, language)} />
          <DetailField label={t("traces.inputTokens")} value={tokenValue(record.usage.input_tokens, record.usage.state, t)} />
          <DetailField label={t("traces.outputTokens")} value={tokenValue(record.usage.output_tokens, record.usage.state, t)} />
          <DetailField label={t("traces.totalTokens")} value={tokenValue(record.usage.total_tokens, record.usage.state, t)} />
          <DetailField label={t("traces.cachedInputTokens")} value={tokenValue(record.usage.cached_input_tokens, record.usage.state, t)} />
          <DetailField label={t("traces.cacheCreationInputTokens")} value={tokenValue(record.usage.cache_creation_input_tokens, record.usage.state, t)} />
          <DetailField label={t("traces.captureState")} value={t(`traces.captureValue.${record.capture_state}`)} />
          <DetailField label={t("traces.restoreState")} value={t(record.restored ? "traces.restored" : "traces.notRestored")} />
        </dl>
        {record.error !== null ? (
          <div className={styles.errorDetail}>
            <strong>{t(`traces.errorCategory.${record.error.category}`, { defaultValue: record.error.category })}</strong>
            {record.error.code !== null ? <code>{record.error.code}</code> : null}
            <p>{record.error.message}</p>
          </div>
        ) : null}
      </section>

      <section className={styles.detailSection}>
        <h3>{t("traces.content")}</h3>
        <div className={styles.payloadBlock}>
          <h4>{t("traces.request")}</h4>
          <pre>{displayStructured(detail.content.request)}</pre>
        </div>
        <div className={styles.payloadBlock}>
          <h4>{record.status === "failed" ? t("traces.resultOrError") : t("traces.result")}</h4>
          <pre>{displayStructured(detail.content.result)}</pre>
        </div>
      </section>

      <section className={styles.detailSection}>
        <div className={styles.sectionHeading}>
          <h3>{t("traces.rawData")}</h3>
          {selectedRawSource !== undefined ? (
            <label className={styles.rawSourceSelect}>
              <span>{t("traces.rawSource")}</span>
              <select aria-label={t("traces.rawSource")} value={selectedRawSource.source}
                onChange={(event) => onRawSourceChange(event.target.value)}>
                {detail.raw_sources.map((source) => (
                  <option key={source.source} value={source.source}>{source.source}</option>
                ))}
              </select>
            </label>
          ) : null}
        </div>
        {detail.capture_issues.length > 0 ? (
          <ul className={styles.captureIssues}>
            {detail.capture_issues.map((issue, index) => (
              <li key={`${issue.source ?? "capture"}-${index}`}>
                <strong>{t(`traces.captureValue.${issue.state}`)}</strong>
                <span>{issue.message}</span>
                {issue.source !== null ? <code>{issue.source}</code> : null}
              </li>
            ))}
          </ul>
        ) : null}
        {selectedRawSource === undefined ? (
          <p className={styles.emptyState}>{t("traces.noRawData")}</p>
        ) : (
          <TraceRawContent
            key={`${record.record_id}:${selectedRawSource.source}`}
            recordId={record.record_id}
            source={selectedRawSource}
            t={t}
          />
        )}
      </section>
    </div>
  );
}

function DetailField({ label, value }: { label: string; value: string }) {
  return <div className={styles.detailField}><dt>{label}</dt><dd>{value}</dd></div>;
}

function TraceRawContent({
  recordId,
  source,
  t,
}: {
  recordId: string;
  source: TraceRawSourceDTO;
  t: (key: string, options?: Record<string, unknown>) => string;
}) {
  const [bytes, setBytes] = useState<Uint8Array>(new Uint8Array());
  const [nextOffset, setNextOffset] = useState<number | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [loadingMore, setLoadingMore] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState(false);
  const requestVersion = useRef(0);

  const readPage = useCallback(async (offset: number, append: boolean) => {
    const version = ++requestVersion.current;
    if (append) setLoadingMore(true);
    else setState("loading");
    try {
      const page = await getTraceRawPage(recordId, {
        source: source.source,
        offset,
        limit_bytes: RAW_PAGE_SIZE,
      });
      if (version !== requestVersion.current) return;
      const chunk = decodeBase64(page.data_base64);
      setBytes((current) => append ? concatenateBytes(current, chunk) : chunk);
      setNextOffset(page.next_offset);
      setState("ready");
    } catch (error) {
      if (version === requestVersion.current) setState(loadStateFor(error));
    } finally {
      if (version === requestVersion.current) setLoadingMore(false);
    }
  }, [recordId, source.source]);

  useEffect(() => {
    setBytes(new Uint8Array());
    setNextOffset(null);
    setState("loading");
    void readPage(0, false);
    return () => { requestVersion.current += 1; };
  }, [readPage]);

  const textContent = useMemo(() => {
    if (!isTextContent(source.content_type) || source.content_encoding !== null) return null;
    return new TextDecoder("utf-8").decode(bytes);
  }, [bytes, source.content_encoding, source.content_type]);

  async function download() {
    setDownloading(true);
    setDownloadError(false);
    try {
      const blob = await downloadTraceRaw(recordId, source.source);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${safeFilePart(recordId)}-${safeFilePart(source.source)}.bin`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch {
      setDownloadError(true);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className={styles.rawContent}>
      <div className={styles.rawMetadata}>
        <span>{source.direction === "request" ? t("traces.request") : t("traces.response")}</span>
        <span>{t(`traces.captureLayer.${source.layer}`, { defaultValue: source.layer })}</span>
        <span>{t("traces.byteCount", { count: source.total_bytes.toLocaleString() })}</span>
        {source.content_encoding !== null ? <span>{t("traces.encoding", { value: source.content_encoding })}</span> : null}
      </div>
      {state === "loading" ? <p className={styles.stateMessage} role="status">{t("traces.loadingRaw")}</p>
        : state === "unavailable" ? <p className={styles.inlineError} role="alert">{t("traces.storageUnavailable")}</p>
          : state === "error" ? <p className={styles.inlineError} role="alert">{t("traces.rawError")}</p>
            : textContent !== null ? <pre className={styles.rawText}>{textContent || t("traces.emptyRaw")}</pre>
              : <p className={styles.binaryNotice}>{t("traces.binaryRaw")}</p>}
      {nextOffset !== null ? (
        <button className={styles.textButton} type="button" disabled={loadingMore}
          onClick={() => void readPage(nextOffset, true)}>
          {t(loadingMore ? "traces.loadingRaw" : "traces.loadMoreRaw")}
        </button>
      ) : null}
      <button className={commonStyles.secondaryButton} type="button" disabled={downloading} onClick={() => void download()}>
        <Download size={15} aria-hidden="true" />
        {t(downloading ? "traces.downloading" : "traces.downloadRaw")}
      </button>
      {downloadError ? <p className={styles.inlineError} role="alert">{t("traces.downloadError")}</p> : null}
    </div>
  );
}

function loadStateFor(error: unknown): LoadState {
  return error instanceof ApiError && error.status === 503 ? "unavailable" : "error";
}

function mergeRecords(current: TraceRecordDTO[], incoming: TraceRecordDTO[]): TraceRecordDTO[] {
  const byId = new Map(current.map((record) => [record.record_id, record]));
  for (const record of incoming) byId.set(record.record_id, record);
  return Array.from(byId.values());
}

function uniqueValues(values: Array<string | null>): string[] {
  return Array.from(new Set(values.filter((value): value is string => value !== null))).sort((left, right) => left.localeCompare(right));
}

function usageLabel(record: TraceRecordDTO, t: (key: string, options?: Record<string, unknown>) => string, language: string): string {
  if (record.usage.state === "not_applicable") return t("traces.notApplicable");
  if (record.usage.state === "unknown" || record.usage.total_tokens === null) return t("traces.usageUnknown");
  return `${record.usage.total_tokens.toLocaleString(language)} ${t("traces.tokensUnit")}`;
}

function tokenValue(value: number | null, state: TraceRecordDTO["usage"]["state"], t: (key: string) => string): string {
  if (state === "not_applicable") return t("traces.notApplicable");
  if (state === "unknown" || value === null) return t("traces.usageUnknown");
  return value.toLocaleString();
}

function displayStructured(value: unknown): string {
  if (value === null) return "-";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

function decodeBase64(value: string): Uint8Array {
  const binary = window.atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function concatenateBytes(left: Uint8Array, right: Uint8Array): Uint8Array {
  const joined = new Uint8Array(left.length + right.length);
  joined.set(left);
  joined.set(right, left.length);
  return joined;
}

function isTextContent(contentType: string | null): boolean {
  if (contentType === null) return false;
  const mediaType = contentType.split(";", 1)[0]!.trim().toLowerCase();
  return mediaType.startsWith("text/") || mediaType.includes("json") || mediaType.includes("xml")
    || mediaType.includes("javascript") || mediaType.includes("x-www-form-urlencoded");
}

function safeFilePart(value: string): string {
  return value.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 100) || "trace";
}

function formatDateTime(value: string, language: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString(language, { dateStyle: "medium", timeStyle: "medium" });
}

function formatDuration(milliseconds: number, language: string): string {
  if (milliseconds >= 1000) return `${(milliseconds / 1000).toLocaleString(language, { maximumFractionDigits: 1 })} s`;
  return `${Math.round(milliseconds).toLocaleString(language)} ms`;
}
