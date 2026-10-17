import { Link } from "react-router-dom";
import type { TraceRecordDTO, TraceRecordDetailDTO } from "../../shared/service/protocol";
import commonStyles from "../../shared/styles/controls.module.css";
import moduleStyles from "./TraceSessionView.module.css";
import { formatDateTime, formatDuration } from "./presentation";

const styles = { ...commonStyles, ...moduleStyles };

export function TraceRecordDetailView({ detail, language, t }: {
  detail: TraceRecordDetailDTO;
  language: string;
  t: (key: string, options?: Record<string, unknown>) => string;
}) {
  const record = detail.record;
  const model = record.kind === "model";
  const output = objectValue(detail.content.output);
  const requested = Array.isArray(output.tool_calls) ? output.tool_calls.map(objectValue) : [];
  const tools = requested.map((call) => ({
    id: String(call.id), name: String(call.name),
    trace: detail.related_tools.find((tool) => tool.tool_call_id === call.id),
  }));
  for (const tool of detail.related_tools) {
    if (!tools.some((item) => item.trace?.record_id === tool.record_id)) {
      tools.push({ id: tool.tool_call_id ?? tool.record_id,
        name: tool.tool_name ?? t("traces.unknownTool"), trace: tool });
    }
  }
  return (
    <div className={styles.detailContent}>
      <header className={styles.detailHeading}>
        <h2>{model ? record.model ?? t("traces.unknownModel") : record.tool_name ?? t("traces.unknownTool")}</h2>
        <span className={record.status === "failed" ? styles.failedStatus : styles.successStatus}>
          {t(`traces.statusValue.${record.status}`)}
        </span>
      </header>
      <section className={styles.detailSection}>
        <dl className={styles.fieldGrid}>
          <DetailField label={t("traces.startedAt")} value={formatDateTime(record.started_at, language)} />
          <DetailField label={t("traces.duration")} value={formatDuration(record.duration_ms, language)} />
          {model ? <>
            <DetailField label={t("traces.inputTokens")} value={tokenValue(record.usage.input_tokens, record, t)} />
            <DetailField label={t("traces.outputTokens")} value={tokenValue(record.usage.output_tokens, record, t)} />
          </> : null}
        </dl>
        {record.error !== null ? <div className={styles.errorDetail} role="status">
          <strong>{t(`traces.errorCategory.${record.error.category}`, { defaultValue: record.error.category })}</strong>
          <p>{record.error.message}</p>
        </div> : null}
        {model && !record.output_complete ? <p role="status">{t("traces.incompleteOutput")}</p> : null}
        {record.restored ? <p>{t("traces.restored")}</p> : null}
        {detail.capture_issues.length > 0 ? <ul className={styles.captureIssues}>
          {detail.capture_issues.map((issue, index) => <li key={index}>
            <strong>{t(`traces.captureValue.${issue.state}`)}</strong><span>{issue.message}</span>
          </li>)}
        </ul> : null}
      </section>
      <section className={styles.detailSection}>
        {model ? <>
          <h3>{t("traces.modelOutput")}</h3>
          <pre className={styles.payloadBlock}>{typeof output.text === "string" && output.text ? output.text : t("traces.noOutput")}</pre>
          {typeof output.reasoning === "string" && output.reasoning ? <details>
            <summary>{t("traces.reasoningOutput")}</summary><pre className={styles.payloadBlock}>{output.reasoning}</pre>
          </details> : null}
          {tools.length > 0 ? <>
            <h3>{t("traces.relatedTools")}</h3>
            <ul>{tools.map((tool) => <li key={tool.id}>
              {tool.trace ? <Link to={`/traces/records/${encodeURIComponent(tool.trace.record_id)}`}>
                {tool.name} · {t(`traces.statusValue.${tool.trace.status}`)}
              </Link> : <span>{tool.name} · {t("traces.relatedTraceUnavailable")}</span>}
            </li>)}</ul>
          </> : null}
        </> : <>
          {record.parent_record_id !== null ? <Link to={`/traces/records/${encodeURIComponent(record.parent_record_id)}`}>
            {t("traces.parentModel")}
          </Link> : null}
          <h3>{t("traces.arguments")}</h3><pre className={styles.payloadBlock}>{displayStructured(detail.content.arguments)}</pre>
          <h3>{t("traces.result")}</h3><pre className={styles.payloadBlock}>{displayStructured(objectValue(detail.content.result).content ?? detail.content.result)}</pre>
          {objectValue(detail.content.result).artifact ? <>
            <h4>{t("traces.artifact")}</h4><pre className={styles.payloadBlock}>{displayStructured(objectValue(detail.content.result).artifact)}</pre>
          </> : null}
        </>}
      </section>
      <details className={styles.detailSection}>
        <summary>{t("traces.moreInformation")}</summary>
        <dl className={styles.fieldGrid}>
          {record.protocol !== null ? <DetailField label={t("traces.protocol")} value={record.protocol} /> : null}
          {record.endpoint !== null ? <DetailField label={t("traces.endpoint")} value={record.endpoint} /> : null}
          <DetailField label={t("traces.purpose")} value={t(`traces.purposeValue.${record.purpose}`, { defaultValue: record.purpose })} />
          {model ? <>
            <DetailField label={t("traces.retryCountLabel")} value={String(record.retry_count)} />
            {record.reasoning_effort !== null ? <DetailField label={t("traces.reasoningEffort")} value={record.reasoning_effort} /> : null}
            {record.usage.cached_input_tokens !== null ? <DetailField label={t("traces.cachedInputTokens")} value={String(record.usage.cached_input_tokens)} /> : null}
            {record.usage.cache_creation_input_tokens !== null ? <DetailField label={t("traces.cacheCreationInputTokens")} value={String(record.usage.cache_creation_input_tokens)} /> : null}
          </> : null}
        </dl>
      </details>
    </div>
  );
}

function DetailField({ label, value }: { label: string; value: string }) {
  return <div className={styles.detailField}><dt>{label}</dt><dd>{value}</dd></div>;
}

function objectValue(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" ? value as Record<string, unknown> : {};
}

function displayStructured(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value ?? null, null, 2);
}

function tokenValue(value: number | null, record: TraceRecordDTO,
  t: (key: string, options?: Record<string, unknown>) => string): string {
  return record.usage.state === "reported" && value !== null ? String(value) : t("traces.usageUnknown");
}
