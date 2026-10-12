import { CircleAlert, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useParams } from "react-router-dom";
import { ApiError, getTraceRecord } from "../../shared/service/api";
import type { TraceRecordDetailDTO } from "../../shared/service/protocol";
import type { AuthState } from "../../shared/service/types";
import commonStyles from "../../shared/styles/controls.module.css";
import { traceLoadErrorState } from "./presentation";
import moduleStyles from "./TraceSessionView.module.css";
import { TraceRecordDetailView } from "./TraceRecordDetailView";

const styles = { ...commonStyles, ...moduleStyles };
type LoadState = "loading" | "ready" | "not-found" | "unavailable" | "unauthorized" | "error";

export function TraceRecordView({ authState }: { authState: AuthState }) {
  const { t, i18n } = useTranslation();
  const { recordId = "" } = useParams();
  const [detail, setDetail] = useState<TraceRecordDetailDTO | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [rawSourceName, setRawSourceName] = useState("");
  const requestVersion = useRef(0);

  useEffect(() => {
    const version = ++requestVersion.current;
    setDetail(null);
    setState("loading");
    if (authState !== "ready") return;
    void getTraceRecord(recordId).then((response) => {
      if (version !== requestVersion.current) return;
      if (response.record.record_id !== recordId) {
        setState("not-found");
        return;
      }
      setDetail(response);
      setState("ready");
    }).catch((error: unknown) => {
      if (version !== requestVersion.current) return;
      setState(error instanceof ApiError
        ? error.status === 404 ? "not-found" : error.status === 401 ? "unauthorized" : traceLoadErrorState(error)
        : traceLoadErrorState(error));
    });
    return () => { requestVersion.current += 1; };
  }, [authState, recordId, refreshVersion]);

  return (
    <div className={styles.page}>
      <header className={styles.pageHeading}>
        <div>
          <p className={styles.eyebrow}>{t("traces.eyebrow")}</p>
          <h1>{t("traces.recordDetails")}</h1>
        </div>
        <button className={commonStyles.secondaryButton} type="button"
          disabled={authState !== "ready" || state === "loading"}
          onClick={() => setRefreshVersion((version) => version + 1)}>
          <RefreshCw size={15} aria-hidden="true" />
          {t("traces.refresh")}
        </button>
      </header>
      <section className={styles.detailPanel} aria-label={t("traces.details")}>
        {authState === "required" || state === "unauthorized" ? (
          <p className={styles.stateMessage} role="status">{t("traces.authenticationRequired")}</p>
        ) : authState !== "ready" || state === "loading" ? (
          <p className={styles.stateMessage} role="status">{t("traces.loadingDetail")}</p>
        ) : state === "not-found" ? (
          <p className={styles.emptyState} role="status">{t("traces.recordExpired")}</p>
        ) : state === "unavailable" || state === "error" || detail === null ? (
          <div className={styles.errorState} role="alert">
            <CircleAlert size={17} aria-hidden="true" />
            <span>{t(state === "unavailable" ? "traces.storageUnavailable" : "traces.detailError")}</span>
          </div>
        ) : (
          <TraceRecordDetailView key={`${recordId}:${refreshVersion}`} detail={detail}
            language={i18n.language} rawSourceName={rawSourceName}
            onRawSourceChange={setRawSourceName} t={t} />
        )}
      </section>
    </div>
  );
}
