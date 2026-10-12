import { Download } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, downloadTraceRaw, getTraceRawPage } from "../../shared/service/api";
import type { TraceRecordDTO, TraceRecordDetailDTO, TraceRawSourceDTO } from "../../shared/service/protocol";
import commonStyles from "../../shared/styles/controls.module.css";
import moduleStyles from "./TraceSessionView.module.css";
import { formatDateTime, formatDuration, usageLabel } from "./presentation";

const styles = { ...commonStyles, ...moduleStyles };
const RAW_PAGE_SIZE = 256 * 1024;
type LoadState = "idle" | "loading" | "ready" | "error" | "unavailable";

export function TraceRecordDetailView({
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

