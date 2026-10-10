import * as Dialog from "@radix-ui/react-dialog";
import { BookOpen, Brain, RefreshCw, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { getProjectMemory, getRuntimeMemory, triggerProjectDream, triggerRuntimeDream } from "../../shared/service/api";
import type { DreamResult, SessionClaim } from "../../shared/service/protocol";
import commonStyles from "../../shared/styles/controls.module.css";
import moduleStyles from "./Runtime.module.css";

const styles = { ...commonStyles, ...moduleStyles };

export type WorkspaceMemoryTarget = {
  title: string;
  trigger: HTMLElement;
} & ({ projectId: string; claim?: never } | { projectId: null; claim: SessionClaim });

export default function WorkspaceMemoryDialog({ target, open, online, onClose }: {
  target: WorkspaceMemoryTarget;
  open: boolean;
  online: boolean;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const [memory, setMemory] = useState<string | null>(null);
  const [result, setResult] = useState<DreamResult | null>(null);
  const [busy, setBusy] = useState<"read" | "dream" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const epoch = useRef(0);
  const pending = useRef(false);
  useEffect(() => () => { epoch.current += 1; }, []);
  useEffect(() => {
    epoch.current += 1;
    pending.current = false;
    setBusy(null);
  }, [online]);
  useEffect(() => {
    if (open || busy !== "read") return;
    epoch.current += 1;
    pending.current = false;
    setBusy(null);
  }, [open, busy]);

  async function operate(action: "read" | "dream") {
    if (!online || pending.current) return;
    pending.current = true;
    const version = epoch.current;
    setBusy(action);
    setError(null);
    setNotice(null);
    try {
      if (action === "read") {
        const response = target.projectId !== null
          ? await getProjectMemory(target.projectId)
          : await getRuntimeMemory(target.claim.workspace_id, target.claim.session_id,
            target.claim.claim_version, target.claim.reconnect_credential);
        if (version === epoch.current) {
          setMemory(response.content);
          setNotice("management.memoryLoaded");
        }
      } else {
        setResult(null);
        const response = target.projectId !== null
          ? await triggerProjectDream(target.projectId)
          : await triggerRuntimeDream(target.claim.workspace_id, target.claim.session_id,
            target.claim.claim_version, target.claim.reconnect_credential);
        if (version === epoch.current) setResult(response.result);
      }
    } catch {
      if (version === epoch.current) setError(t(action === "read" ? "management.memoryError" : "management.dreamError"));
    } finally {
      if (version === epoch.current) {
        pending.current = false;
        setBusy(null);
      }
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={(next) => { if (!next) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.dialogOverlay} />
        <Dialog.Content className={`${styles.dialogContent} ${styles.managementDialog}`}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            if (target.trigger.isConnected) {
              const collapsed = target.trigger.closest("#app-sidebar")?.getAttribute("data-open") === "false"
                && window.matchMedia("(max-width: 1023px)").matches;
              (collapsed ? document.getElementById("app-sidebar-toggle") : target.trigger)?.focus();
            }
          }}>
          <div className={styles.dialogHeader}>
            <div>
              <Dialog.Title className={styles.dialogTitle}>{t("management.memoryPanelTitle")}</Dialog.Title>
              <Dialog.Description className={styles.dialogDescription}>{target.title}</Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                <X size={17} aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>
          {error !== null ? <div className={styles.errorBanner} role="alert">{error}</div> : null}
          {notice !== null ? <div className={styles.notice} role="status">{t(notice)}</div> : null}
          <div className={styles.managementTools}>
            <section className={styles.managementTool} aria-labelledby="workspace-memory-title">
              <div className={styles.managementToolHeader}>
                <h3 id="workspace-memory-title">{t("management.memoryTitle")}</h3>
                <button className={styles.secondaryButton} type="button" disabled={busy !== null || !online}
                  onClick={() => void operate("read")}>
                  {busy === "read" ? <RefreshCw size={15} className={styles.spin} aria-hidden="true" />
                    : <BookOpen size={15} aria-hidden="true" />}
                  {t(busy === "read" ? "management.loading" : "management.viewMemory")}
                </button>
              </div>
              {memory !== null ? (
                <div className={styles.managementMemory} role="region" aria-label={t("management.memoryRegion")}>
                  <h4>{t("management.memoryRegion")}</h4>
                  <pre>{memory || t("management.memoryEmpty")}</pre>
                </div>
              ) : null}
            </section>
            <section className={styles.managementTool} aria-labelledby="workspace-dream-title">
              <div className={styles.managementToolHeader}>
                <h3 id="workspace-dream-title">{t("management.dreamTitle")}</h3>
                <button className={styles.secondaryButton} type="button" disabled={busy !== null || !online}
                  onClick={() => void operate("dream")}>
                  <Brain size={15} aria-hidden="true" />
                  {t(busy === "dream" ? "management.dreamRunning" : "management.runDream")}
                </button>
              </div>
              {result !== null ? (
                <div className={styles.managementOperationStatus} role="status" aria-live="polite">
                  <strong>{t(result.error !== null
                    ? result.error.code === "memory_task_running" ? "management.dreamAlreadyRunning" : "management.dreamFailed"
                    : result.status === "No pending summaries" ? "management.dreamNoPending" : "management.dreamCompleted")}</strong>
                  <span>{t("management.dreamProcessed", { count: result.processed_count })}</span>
                  <span>{t(result.memory_updated ? "management.dreamUpdated" : "management.dreamUnchanged")}</span>
                  <span>{t("management.dreamCursor", { cursor: result.cursor })}</span>
                  {result.error !== null ? <span>{result.error.code}: {result.error.message}</span> : null}
                </div>
              ) : null}
            </section>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
