import { CircleAlert, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

import { traceRecordRoute, traceSessionRoute } from "../traces/routing";
import { ApiError, getTraceRecords, getTraceStats } from "../../shared/service/api";
import type {
  TraceBreakdownDTO,
  TraceListQueryDTO,
  TraceRecordDTO,
  TraceStatsQueryDTO,
  TraceStatsResponseDTO,
  TraceTimelinePointDTO,
  TracePurpose,
} from "../../shared/service/protocol";
import type { AuthState } from "../../shared/service/types.ts";
import commonStyles from "../../shared/styles/controls.module.css";
import moduleStyles from "./StatisticsView.module.css";

const styles = { ...commonStyles, ...moduleStyles };
const RECORD_PAGE_SIZE = 50;
const WORKSPACE_PAGE_SIZE = 200;

type RangePreset = "today" | "7days" | "30days" | "custom";
type Dimension = "model" | "tool_name" | "purpose";
type LoadState = "idle" | "loading" | "ready" | "error" | "unavailable";

interface DateRange {
  preset: RangePreset;
  fromDate: string;
  toDate: string;
  fromAt: string;
  toAt: string;
  utcDay?: boolean;
}

interface StatisticsViewProps {
  authState: AuthState;
}

interface WorkspaceOption {
  workspaceKey: string;
  workspacePath: string;
}

export function StatisticsView({ authState }: StatisticsViewProps) {
  const { t, i18n } = useTranslation();
  const initialRange = useMemo(() => presetDateRange("7days"), []);
  const [dateRange, setDateRange] = useState<DateRange>(initialRange);
  const [customFrom, setCustomFrom] = useState(initialRange.fromDate);
  const [customTo, setCustomTo] = useState(initialRange.toDate);
  const [query, setQuery] = useState<TraceStatsQueryDTO>(() => queryForRange(initialRange));
  const [dateError, setDateError] = useState(false);
  const [statistics, setStatistics] = useState<TraceStatsResponseDTO | null>(null);
  const [statisticsState, setStatisticsState] = useState<LoadState>("idle");
  const [records, setRecords] = useState<TraceRecordDTO[]>([]);
  const [recordCursor, setRecordCursor] = useState<string | null>(null);
  const [recordsState, setRecordsState] = useState<LoadState>("idle");
  const [loadingMoreRecords, setLoadingMoreRecords] = useState(false);
  const [workspaces, setWorkspaces] = useState<WorkspaceOption[]>([]);
  const [workspaceCursor, setWorkspaceCursor] = useState<string | null>(null);
  const [workspacesState, setWorkspacesState] = useState<LoadState>("idle");
  const [loadingMoreWorkspaces, setLoadingMoreWorkspaces] = useState(false);
  const [workspaceError, setWorkspaceError] = useState(false);
  const queryRef = useRef(query);
  const requestVersionRef = useRef(0);
  const workspaceRequestVersionRef = useRef(0);
  const initialQueryRef = useRef(query);

  const loadQuery = useCallback(async (nextQuery: TraceStatsQueryDTO) => {
    const requestVersion = ++requestVersionRef.current;
    queryRef.current = nextQuery;
    setQuery(nextQuery);
    setStatistics(null);
    setStatisticsState("loading");
    setRecords([]);
    setRecordCursor(null);
    setLoadingMoreRecords(false);
    setRecordsState("loading");
    const recordQuery: TraceListQueryDTO = { ...nextQuery, limit: RECORD_PAGE_SIZE };
    const [statsResult, recordsResult] = await Promise.allSettled([
      getTraceStats(nextQuery),
      getTraceRecords(recordQuery),
    ]);
    if (requestVersion !== requestVersionRef.current) return;
    if (statsResult.status === "fulfilled") {
      setStatistics(statsResult.value);
      setStatisticsState("ready");
    } else {
      setStatisticsState(errorLoadState(statsResult.reason));
    }
    if (recordsResult.status === "fulfilled") {
      setRecords(recordsResult.value.items);
      setRecordCursor(recordsResult.value.next_cursor);
      setRecordsState("ready");
    } else {
      setRecordsState(errorLoadState(recordsResult.reason));
    }
  }, []);

  const loadWorkspacePage = useCallback(async (cursor: string | null) => {
    const requestVersion = ++workspaceRequestVersionRef.current;
    setWorkspaceError(false);
    if (cursor === null) setWorkspacesState("loading");
    else setLoadingMoreWorkspaces(true);
    try {
      const response = await getTraceRecords({
        ...(cursor === null ? {} : { cursor }),
        limit: WORKSPACE_PAGE_SIZE,
      });
      if (requestVersion !== workspaceRequestVersionRef.current) return;
      setWorkspaces((current) => mergeWorkspaces(current, response.items));
      setWorkspaceCursor(response.next_cursor);
      setWorkspacesState("ready");
    } catch {
      if (requestVersion !== workspaceRequestVersionRef.current) return;
      setWorkspaceError(true);
      setWorkspacesState("error");
    } finally {
      if (requestVersion === workspaceRequestVersionRef.current) setLoadingMoreWorkspaces(false);
    }
  }, []);

  useEffect(() => {
    if (authState !== "ready") return;
    void loadQuery(initialQueryRef.current);
    void loadWorkspacePage(null);
  }, [authState, loadQuery, loadWorkspacePage]);

  const sessionLinks = useMemo(() => collectSessionLinks(records), [records]);
  const auxiliarySources = useMemo(() => collectAuxiliarySources(records), [records]);
  const activeDimension = useMemo(() => selectedDimension(query), [query]);
  const hasCalls = statistics !== null
    && statistics.totals.model_calls + statistics.totals.tool_calls > 0;

  function applyPreset(preset: Exclude<RangePreset, "custom">) {
    const nextRange = presetDateRange(preset);
    setDateRange(nextRange);
    setCustomFrom(nextRange.fromDate);
    setCustomTo(nextRange.toDate);
    setDateError(false);
    void loadQuery({ ...queryRef.current, ...queryForRange(nextRange) });
  }

  function refresh() {
    if (dateRange.preset === "custom") void loadQuery(queryRef.current);
    else applyPreset(dateRange.preset);
    void loadWorkspacePage(null);
  }

  function applyCustomRange() {
    const start = localDayStart(customFrom);
    const end = localDayEnd(customTo);
    if (start === null || end === null || start > end) {
      setDateError(true);
      return;
    }
    const nextRange: DateRange = {
      preset: "custom",
      fromDate: customFrom,
      toDate: customTo,
      fromAt: start.toISOString(),
      toAt: end.toISOString(),
    };
    setDateRange(nextRange);
    setDateError(false);
    void loadQuery({ ...queryRef.current, ...queryForRange(nextRange) });
  }

  function changeWorkspace(workspaceKey: string) {
    const nextQuery = { ...queryRef.current };
    if (workspaceKey) nextQuery.workspace_key = workspaceKey;
    else delete nextQuery.workspace_key;
    void loadQuery(nextQuery);
  }

  function filterDimension(dimension: Dimension, key: string) {
    const nextQuery = { ...queryRef.current };
    const field: Dimension = dimension;
    const alreadySelected = nextQuery[field] === key;
    delete nextQuery.model;
    delete nextQuery.tool_name;
    delete nextQuery.purpose;
    if (!alreadySelected) {
      if (field === "model") nextQuery.model = key;
      else if (field === "tool_name") nextQuery.tool_name = key;
      else nextQuery.purpose = key as TracePurpose;
    }
    void loadQuery(nextQuery);
  }

  function selectDay(date: string) {
    // T11 groups daily totals by the UTC completion date.
    const start = new Date(`${date}T00:00:00.000Z`);
    const end = new Date(`${date}T23:59:59.999Z`);
    if (Number.isNaN(start.valueOf()) || Number.isNaN(end.valueOf())) return;
    const nextRange: DateRange = {
      preset: "custom",
      fromDate: date,
      toDate: date,
      fromAt: start.toISOString(),
      toAt: end.toISOString(),
      utcDay: true,
    };
    setDateRange(nextRange);
    setCustomFrom(date);
    setCustomTo(date);
    setDateError(false);
    void loadQuery({ ...queryRef.current, ...queryForRange(nextRange) });
  }

  async function loadMoreRecords() {
    if (recordCursor === null || loadingMoreRecords) return;
    const requestVersion = requestVersionRef.current;
    const cursor = recordCursor;
    setLoadingMoreRecords(true);
    try {
      const nextQuery: TraceListQueryDTO = {
        ...queryRef.current,
        cursor,
        limit: RECORD_PAGE_SIZE,
      };
      const response = await getTraceRecords(nextQuery);
      if (requestVersion !== requestVersionRef.current) return;
      setRecords((current) => mergeRecords(current, response.items));
      setRecordCursor(response.next_cursor);
      setRecordsState("ready");
    } catch (error) {
      if (requestVersion === requestVersionRef.current) setRecordsState(errorLoadState(error));
    } finally {
      if (requestVersion === requestVersionRef.current) setLoadingMoreRecords(false);
    }
  }

  const numberFormat = useMemo(() => new Intl.NumberFormat(i18n.language), [i18n.language]);
  const percentFormat = useMemo(() => new Intl.NumberFormat(i18n.language, {
    style: "percent",
    maximumFractionDigits: 1,
  }), [i18n.language]);

  return (
    <div className={styles.statisticsPage}>
      <header className={styles.pageHeading}>
        <div>
          <p className={styles.eyebrow}>{t("statistics.eyebrow")}</p>
          <h1 id="statistics-heading">{t("statistics.title")}</h1>
        </div>
        <button
          className={commonStyles.secondaryButton}
          type="button"
          disabled={authState !== "ready" || statisticsState === "loading"}
          onClick={refresh}
        >
          <RefreshCw size={15} aria-hidden="true" />
          {t("statistics.refresh")}
        </button>
      </header>

      <section className={styles.filters} aria-label={t("statistics.filters")}>
        <div className={styles.filterGroup}>
          <span className={styles.filterLabel}>{t("statistics.period")}</span>
          <div className={styles.rangeOptions} role="group" aria-label={t("statistics.period")}>
            {(["today", "7days", "30days", "custom"] as const).map((preset) => (
              <button
                className={dateRange.preset === preset ? styles.rangeActive : styles.rangeButton}
                key={preset}
                type="button"
                aria-pressed={dateRange.preset === preset}
                onClick={() => {
                  if (preset === "custom") setDateRange((current) => ({ ...current, preset }));
                  else applyPreset(preset);
                }}
              >
                {t(`statistics.range.${preset}`)}
              </button>
            ))}
          </div>
        </div>

        {dateRange.preset === "custom" ? (
          <form className={styles.customRange} onSubmit={(event) => { event.preventDefault(); applyCustomRange(); }}>
            <label>
              <span>{t("statistics.from")}</span>
              <input type="date" value={customFrom} onChange={(event) => setCustomFrom(event.target.value)} />
            </label>
            <label>
              <span>{t("statistics.to")}</span>
              <input type="date" value={customTo} onChange={(event) => setCustomTo(event.target.value)} />
            </label>
            <button className={commonStyles.secondaryButton} type="submit">{t("statistics.applyRange")}</button>
            {dateRange.utcDay ? <span className={styles.filterLabel}>{t("statistics.selectedUtcDay")}</span> : null}
            {dateError ? <span className={styles.fieldError} role="alert">{t("statistics.invalidRange")}</span> : null}
          </form>
        ) : null}

        <div className={styles.filterGroup}>
          <label className={styles.filterLabel} htmlFor="statistics-workspace">{t("statistics.workspace")}</label>
          <select
            id="statistics-workspace"
            value={query.workspace_key ?? ""}
            aria-busy={workspacesState === "loading"}
            onChange={(event) => changeWorkspace(event.target.value)}
          >
            <option value="">{t("statistics.allWorkspaces")}</option>
            {workspaces.map((workspace) => (
              <option key={workspace.workspaceKey} value={workspace.workspaceKey}>{workspace.workspacePath}</option>
            ))}
          </select>
          {workspaceCursor !== null ? (
            <button
              className={styles.textButton}
              type="button"
              disabled={loadingMoreWorkspaces}
              onClick={() => void loadWorkspacePage(workspaceCursor)}
            >
              {t(loadingMoreWorkspaces ? "statistics.loadingWorkspaces" : "statistics.loadMoreWorkspaces")}
            </button>
          ) : null}
          {workspaceError ? (
            <div className={styles.workspaceError} role="status">
              <span>{t("statistics.workspacesUnavailable")}</span>
              <button className={styles.textButton} type="button" onClick={() => void loadWorkspacePage(null)}>
                {t("statistics.retryWorkspaces")}
              </button>
            </div>
          ) : null}
        </div>
      </section>

      {activeDimension !== null ? (
        <div className={styles.activeFilter}>
          <span>{t("statistics.filteredBy", {
            value: activeDimension.field === "purpose"
              ? t(`statistics.purpose.${activeDimension.key}`, { defaultValue: activeDimension.label })
              : activeDimension.label,
          })}</span>
          <button className={styles.textButton} type="button" onClick={() => filterDimension(activeDimension.field, activeDimension.key)}>
            {t("statistics.clearFilter")}
          </button>
        </div>
      ) : null}

      <p className={styles.scopeNote}>{t("statistics.scopeNote")}</p>

      {authState !== "ready" || statisticsState === "loading" || statisticsState === "idle" ? (
        <p className={styles.stateMessage} role="status">{t("statistics.loading")}</p>
      ) : statisticsState === "unavailable" ? (
        <div className={styles.errorState} role="alert">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{t("statistics.storageUnavailable")}</span>
        </div>
      ) : statisticsState === "error" ? (
        <div className={styles.errorState} role="alert">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{t("statistics.loadError")}</span>
        </div>
      ) : statistics !== null ? (
        <>
          {!hasCalls ? <p className={styles.emptyState} role="status">{t("statistics.empty")}</p> : null}
          <section aria-label={t("statistics.metrics")}>
            <dl className={styles.metrics}>
              <Metric label={t("statistics.modelCalls")} value={numberFormat.format(statistics.totals.model_calls)} />
              <Metric label={t("statistics.toolCalls")} value={numberFormat.format(statistics.totals.tool_calls)} />
              <Metric label={t("statistics.succeeded")} value={numberFormat.format(statistics.totals.success_count)} />
              <Metric label={t("statistics.failed")} value={numberFormat.format(statistics.totals.failed_count)} />
              <Metric
                label={t("statistics.failureRate")}
                value={statistics.totals.failure_rate === null ? "-" : percentFormat.format(statistics.totals.failure_rate)}
              />
              <Metric
                label={t("statistics.retries")}
                value={numberFormat.format(statistics.totals.retry_count)}
                detail={t("statistics.retriedCalls", { count: statistics.totals.retried_calls })}
              />
              <Metric
                label={t("statistics.knownTokens")}
                value={numberFormat.format(statistics.tokens.total_tokens)}
                detail={t("statistics.inputOutput", {
                  input: numberFormat.format(statistics.tokens.input_tokens),
                  output: numberFormat.format(statistics.tokens.output_tokens),
                })}
              />
              <Metric label={t("statistics.unknownUsage")} value={numberFormat.format(statistics.totals.unknown_usage_count)} />
              <Metric
                label={t("statistics.cachedTokens")}
                value={numberFormat.format(statistics.tokens.cached_input_tokens)}
                detail={t("statistics.cacheCreation", { count: statistics.tokens.cache_creation_input_tokens })}
              />
              <Metric
                label={t("statistics.averageDuration")}
                value={statistics.totals.average_duration_ms === null ? "-" : formatDuration(statistics.totals.average_duration_ms, i18n.language)}
              />
            </dl>
          </section>

          {hasCalls ? (
            <>
              <section className={styles.section} aria-labelledby="statistics-trend-heading">
                <div className={styles.sectionHeading}>
                  <h2 id="statistics-trend-heading">{t("statistics.trends")}</h2>
                  <span>{t("statistics.selectDay")}</span>
                </div>
                <div className={styles.trendGrid}>
                  <TrendChart
                    title={t("statistics.calls")}
                    points={statistics.daily}
                    valueFor={(point) => point.model_calls + point.tool_calls}
                    labelFor={(point) => t("statistics.callTrendLabel", {
                      date: formatDate(point.date, i18n.language),
                      models: numberFormat.format(point.model_calls),
                      tools: numberFormat.format(point.tool_calls),
                    })}
                    onSelectDay={selectDay}
                    language={i18n.language}
                    tone="calls"
                  />
                  <TrendChart
                    title={t("statistics.tokens")}
                    points={statistics.daily}
                    valueFor={(point) => point.tokens.total_tokens}
                    labelFor={(point) => t("statistics.tokenTrendLabel", {
                      date: formatDate(point.date, i18n.language),
                      tokens: numberFormat.format(point.tokens.total_tokens),
                    })}
                    onSelectDay={selectDay}
                    language={i18n.language}
                    tone="tokens"
                  />
                </div>
              </section>

              <section className={styles.section} aria-labelledby="statistics-breakdown-heading">
                <h2 id="statistics-breakdown-heading">{t("statistics.distributions")}</h2>
                <div className={styles.breakdownGrid}>
                  <Breakdown
                    title={t("statistics.models")}
                    items={statistics.by_model}
                    selectedKey={query.model}
                    tone="model"
                    onSelect={(key) => filterDimension("model", key)}
                    formatNumber={numberFormat}
                    labelFor={(key) => key}
                    callsLabel={t("statistics.callsCount")}
                  />
                  <Breakdown
                    title={t("statistics.tools")}
                    items={statistics.by_tool}
                    selectedKey={query.tool_name}
                    tone="tool"
                    onSelect={(key) => filterDimension("tool_name", key)}
                    formatNumber={numberFormat}
                    labelFor={(key) => key}
                    callsLabel={t("statistics.callsCount")}
                  />
                  <Breakdown
                    title={t("statistics.purposes")}
                    items={statistics.by_purpose}
                    selectedKey={query.purpose}
                    tone="purpose"
                    onSelect={(key) => filterDimension("purpose", key)}
                    formatNumber={numberFormat}
                    labelFor={(key) => t(`statistics.purpose.${key}`, { defaultValue: key })}
                    callsLabel={t("statistics.callsCount")}
                  />
                </div>
              </section>
            </>
          ) : null}
        </>
      ) : null}

      {auxiliarySources.length > 0 ? (
        <section className={styles.section} aria-labelledby="statistics-auxiliary-heading">
          <h2 id="statistics-auxiliary-heading">{t("statistics.auxiliarySources")}</h2>
          <ul className={styles.sessionList}>
            {auxiliarySources.map((source) => (
              <li key={`${source.workspaceKey}:${source.purpose}`}>
                <div className={styles.auxiliarySource}>
                  <span className={styles.sessionTitle}>{t(`statistics.purpose.${source.purpose}`)}</span>
                  <span className={styles.sessionPath}>{source.workspacePath}</span>
                </div>
                <ul className={styles.sessionList}>
                  {source.records.map((record) => (
                    <li key={record.record_id}>
                      <Link to={traceRecordRoute(record.record_id)}>
                        <span className={styles.sessionTitle}>
                          {record.kind === "model" ? record.model : record.tool_name}
                        </span>
                        <span className={styles.sessionPath}>{record.record_id}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className={styles.section} aria-labelledby="statistics-sessions-heading">
        <div className={styles.sectionHeading}>
          <h2 id="statistics-sessions-heading">{t("statistics.relatedSessions")}</h2>
          {recordsState === "loading" ? <span role="status">{t("statistics.loading")}</span> : null}
        </div>
        {recordsState === "error" || recordsState === "unavailable" ? (
          <p className={styles.inlineError} role="alert">
            {t(recordsState === "unavailable" ? "statistics.storageUnavailable" : "statistics.sessionsError")}
          </p>
        ) : sessionLinks.length === 0 && recordsState === "ready" ? (
          <p className={styles.emptySessions}>{t("statistics.noSessions")}</p>
        ) : (
          <ul className={styles.sessionList}>
            {sessionLinks.map((session) => (
              <li key={`${session.workspaceKey}:${session.sessionId}`}>
                <Link to={traceSessionRoute(session.workspaceKey, session.sessionId)}>
                  <span className={styles.sessionTitle}>{session.sessionId}</span>
                  <span className={styles.sessionPath}>{session.workspacePath}</span>
                </Link>
              </li>
            ))}
          </ul>
        )}
        {recordCursor !== null ? (
          <button
            className={styles.textButton}
            type="button"
            disabled={loadingMoreRecords}
            onClick={() => void loadMoreRecords()}
          >
            {t(loadingMoreRecords ? "statistics.loadingSessions" : "statistics.loadMoreSessions")}
          </button>
        ) : null}
      </section>
    </div>
  );
}

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className={styles.metric}>
      <dt>{label}</dt>
      <dd>{value}</dd>
      {detail ? <span>{detail}</span> : null}
    </div>
  );
}

function TrendChart({
  title,
  points,
  valueFor,
  labelFor,
  onSelectDay,
  language,
  tone,
}: {
  title: string;
  points: TraceTimelinePointDTO[];
  valueFor: (point: TraceTimelinePointDTO) => number;
  labelFor: (point: TraceTimelinePointDTO) => string;
  onSelectDay: (date: string) => void;
  language: string;
  tone: "calls" | "tokens";
}) {
  const maxValue = Math.max(1, ...points.map(valueFor));
  return (
    <div className={styles.trendChart}>
      <h3>{title}</h3>
      {points.length === 0 ? <p className={styles.emptySessions}>-</p> : (
        <div className={styles.chartScroll}>
          <div className={styles.trendItems}>
            {points.map((point) => {
              const value = valueFor(point);
              const height = value === 0 ? 2 : Math.max(7, (value / maxValue) * 100);
              return (
                <button
                  className={styles.trendItem}
                  key={point.date}
                  type="button"
                  title={labelFor(point)}
                  aria-label={labelFor(point)}
                  onClick={() => onSelectDay(point.date)}
                >
                  <span className={styles.barValue}>{value.toLocaleString(language)}</span>
                  <span className={styles.barTrack}>
                    <span className={tone === "calls" ? styles.callBar : styles.tokenBar} style={{ height: `${height}%` }} />
                  </span>
                  <span className={styles.barDate}>{formatShortDate(point.date, language)}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function Breakdown({
  title,
  items,
  selectedKey,
  tone,
  onSelect,
  formatNumber,
  labelFor,
  callsLabel,
}: {
  title: string;
  items: TraceBreakdownDTO[];
  selectedKey: string | undefined;
  tone: "model" | "tool" | "purpose";
  onSelect: (key: string) => void;
  formatNumber: Intl.NumberFormat;
  labelFor: (key: string) => string;
  callsLabel: string;
}) {
  const maxCalls = Math.max(1, ...items.map((item) => item.calls));
  return (
    <div className={styles.breakdown}>
      <h3>{title}</h3>
      {items.length === 0 ? <p className={styles.emptySessions}>-</p> : (
        <ul>
          {items.map((item) => (
            <li key={item.key}>
              <button
                className={styles.breakdownItem}
                type="button"
                aria-pressed={selectedKey === item.key}
                aria-label={`${labelFor(item.key)}, ${formatNumber.format(item.calls)} ${callsLabel}`}
                onClick={() => onSelect(item.key)}
              >
                <span className={styles.breakdownName}>{labelFor(item.key)}</span>
                <span className={styles.breakdownCount}>{formatNumber.format(item.calls)}</span>
                <span className={styles.breakdownTrack}>
                  <span className={styles[`${tone}Bar`]} style={{ width: `${(item.calls / maxCalls) * 100}%` }} />
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function presetDateRange(preset: Exclude<RangePreset, "custom">): DateRange {
  const now = new Date();
  const today = toDateInput(now);
  const end = localDayStart(today) ?? now;
  const start = new Date(end);
  if (preset === "7days") start.setDate(start.getDate() - 6);
  if (preset === "30days") start.setDate(start.getDate() - 29);
  const fromDate = toDateInput(start);
  const toDate = today;
  const fromAt = (localDayStart(fromDate) ?? start).toISOString();
  const toAt = now.toISOString();
  return { preset, fromDate, toDate, fromAt, toAt };
}

function queryForRange(range: DateRange): TraceStatsQueryDTO {
  return { from_at: range.fromAt, to_at: range.toAt };
}

function localDayStart(value: string): Date | null {
  const parts = value.split("-").map(Number);
  if (parts.length !== 3 || parts.some((part) => !Number.isInteger(part))) return null;
  const [year, month, day] = parts;
  const date = new Date(year, month - 1, day);
  if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day) return null;
  return date;
}

function localDayEnd(value: string): Date | null {
  const start = localDayStart(value);
  if (start === null) return null;
  const end = new Date(start);
  end.setDate(end.getDate() + 1);
  end.setMilliseconds(end.getMilliseconds() - 1);
  return end;
}

function toDateInput(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function errorLoadState(error: unknown): LoadState {
  return error instanceof ApiError && error.status === 503 ? "unavailable" : "error";
}

function mergeWorkspaces(current: WorkspaceOption[], next: TraceRecordDTO[]) {
  const byKey = new Map(current.map((workspace) => [workspace.workspaceKey, workspace]));
  for (const record of next) {
    byKey.set(record.workspace_key, { workspaceKey: record.workspace_key, workspacePath: record.workspace_path });
  }
  return Array.from(byKey.values());
}

function mergeRecords(current: TraceRecordDTO[], next: TraceRecordDTO[]) {
  const byId = new Map(current.map((record) => [record.record_id, record]));
  for (const record of next) byId.set(record.record_id, record);
  return Array.from(byId.values());
}

function collectSessionLinks(records: TraceRecordDTO[]) {
  const bySession = new Map<string, {
    workspaceKey: string;
    workspacePath: string;
    sessionId: string;
  }>();
  for (const record of records) {
    if (record.owner_session_id === null) continue;
    const key = `${record.workspace_key}:${record.owner_session_id}`;
    const session = bySession.get(key);
    if (session === undefined) {
      bySession.set(key, {
        workspaceKey: record.workspace_key,
        workspacePath: record.workspace_path,
        sessionId: record.owner_session_id,
      });
    }
  }
  return Array.from(bySession.values());
}

function collectAuxiliarySources(records: TraceRecordDTO[]) {
  const sources = new Map<string, {
    workspaceKey: string; workspacePath: string; purpose: TracePurpose; records: TraceRecordDTO[];
  }>();
  for (const record of records) {
    if (record.owner_session_id !== null) continue;
    const key = `${record.workspace_key}:${record.purpose}`;
    const source = sources.get(key) ?? {
      workspaceKey: record.workspace_key,
      workspacePath: record.workspace_path,
      purpose: record.purpose,
      records: [],
    };
    source.records.push(record);
    sources.set(key, source);
  }
  return Array.from(sources.values());
}

function selectedDimension(query: TraceStatsQueryDTO): { field: Dimension; key: string; label: string } | null {
  if (query.model !== undefined) return { field: "model", key: query.model, label: query.model };
  if (query.tool_name !== undefined) return { field: "tool_name", key: query.tool_name, label: query.tool_name };
  if (query.purpose !== undefined) return { field: "purpose", key: query.purpose, label: query.purpose };
  return null;
}

function formatDuration(milliseconds: number, language: string): string {
  if (milliseconds >= 1000) return `${(milliseconds / 1000).toLocaleString(language, { maximumFractionDigits: 1 })} s`;
  return `${Math.round(milliseconds).toLocaleString(language)} ms`;
}

function formatDate(value: string, language: string): string {
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleDateString(language, { dateStyle: "medium" });
}

function formatShortDate(value: string, language: string): string {
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleDateString(language, { month: "numeric", day: "numeric" });
}
