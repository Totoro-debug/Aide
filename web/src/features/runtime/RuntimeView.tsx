import * as Dialog from "@radix-ui/react-dialog";
import {
  BookOpen,
  Brain,
  Check,
  CircleAlert,
  Info,
  RefreshCw,
  X
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  getRuntimeMemory,
  getRuntimeStatus,
  triggerRuntimeDream,
  updateRuntimeEffort,
  updateRuntimePermission
} from "../../shared/service/api";
import { PERMISSION_LEVELS } from "../../shared/service/permissions.ts";
import { serviceStateLabel } from "../../shared/service/presentation.ts";
import type {
  DreamResult,
  ReasoningEffort,
  RuntimeStatus,
  ServiceStatus,
  SessionClaim,
  SessionModelConfiguration,
  ToolPermissionLevel
} from "../../shared/service/protocol";
import type { AuthState, ConnectionState } from "../../shared/service/types.ts";
import commonStyles from "../../shared/styles/controls.module.css";
import { REASONING_EFFORTS } from "../conversations/reasoningEffort.ts";
import type { LiveRun } from "../conversations/run.ts";
import { isLiveRunActive } from "../conversations/run.ts";
import { managementErrorKey, operationManagementErrorKey } from "./errors.ts";
import moduleStyles from "./Runtime.module.css";

const styles = { ...commonStyles, ...moduleStyles };

interface StatusViewProps {
  authState: AuthState;
  connectionLabel: string;
  connectionState: ConnectionState;
  detailsOpen: boolean;
  onDetailsOpenChange: (open: boolean) => void;
  serviceStatus: ServiceStatus | null;
}

export function StatusView({
  authState,
  connectionLabel,
  connectionState,
  detailsOpen,
  onDetailsOpenChange,
  serviceStatus,
}: StatusViewProps) {
  const { t } = useTranslation();
  const serviceState = serviceStatus?.state ?? "starting";
  const serviceLabel = serviceStatus === null ? t("status.checking") : serviceStateLabel(serviceState, t);
  const authMessage = authState === "required" ? t("status.authenticationRequired") : t("status.unavailable");

  return (
    <section className={styles.statusPage} aria-labelledby="status-heading">
      <div className={styles.pageHeading}>
        <div>
          <p className={styles.eyebrow}>{t("nav.status")}</p>
          <h1 id="status-heading">{t("status.title")}</h1>
        </div>
        <div className={styles.connectionBadge} data-state={connectionState} role="status" aria-live="polite">
          <span className={styles.statusDot} aria-hidden="true" />
          {connectionLabel}
        </div>
      </div>

      {authState === "ready" ? (
        <>
          <div className={styles.statusGrid}>
            <StatusMetric label={t("status.connection")} value={connectionLabel} state={connectionState} />
            <StatusMetric label={t("status.title")} value={serviceLabel} state={serviceState} />
            <StatusMetric
              label={t("status.workspaces")}
              value={serviceStatus?.active_workspace_count.toString() ?? "-"}
            />
            <StatusMetric
              label={t("status.protocol")}
              value={`v${serviceStatus?.protocol_version ?? "-"}`}
            />
          </div>
          <div className={styles.actionRow}>
            <Dialog.Root open={detailsOpen} onOpenChange={onDetailsOpenChange}>
              <Dialog.Trigger asChild>
                <button className={styles.secondaryButton} type="button">
                  <Info size={16} aria-hidden="true" />
                  {t("controls.details")}
                </button>
              </Dialog.Trigger>
              <Dialog.Portal>
                <Dialog.Overlay className={styles.dialogOverlay} />
                <Dialog.Content className={styles.dialogContent}>
                  <div className={styles.dialogHeader}>
                    <div>
                      <Dialog.Title className={styles.dialogTitle}>{t("status.detailsTitle")}</Dialog.Title>
                      <Dialog.Description className={styles.dialogDescription}>
                        {t("status.detailsDescription")}
                      </Dialog.Description>
                    </div>
                    <Dialog.Close asChild>
                      <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                        <X size={17} aria-hidden="true" />
                      </button>
                    </Dialog.Close>
                  </div>
                  <dl className={styles.detailList}>
                    <div>
                      <dt>{t("status.instance")}</dt>
                      <dd>{serviceStatus?.service_instance_id ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>{t("status.protocol")}</dt>
                      <dd>v{serviceStatus?.protocol_version ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>{t("status.connection")}</dt>
                      <dd>{connectionLabel}</dd>
                    </div>
                  </dl>
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>
          </div>
        </>
      ) : (
        <div className={styles.emptyState} role="status">
          <div className={styles.emptyIcon} aria-hidden="true">
            {authState === "required" ? <Info size={22} /> : <CircleAlert size={22} />}
          </div>
          <div>
            <h2>{authMessage}</h2>
            <p>{connectionLabel}</p>
          </div>
        </div>
      )}
    </section>
  );
}

function StatusMetric({
  label,
  value,
  state,
}: {
  label: string;
  value: string;
  state?: string;
}) {
  return (
    <div className={styles.metric}>
      <div className={styles.metricLabel}>{label}</div>
      <div className={styles.metricValue} data-state={state}>
        {state === "ready" || state === "online" ? <Check size={15} aria-hidden="true" /> : null}
        {value}
      </div>
    </div>
  );
}

export function ConversationRuntimeStatus({ claim, connectionState, refreshVersion, activeRuns }: {
  claim: SessionClaim;
  connectionState: ConnectionState;
  refreshVersion: number;
  activeRuns: LiveRun[];
}) {
  const { i18n, t } = useTranslation();
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [failed, setFailed] = useState(false);
  const activeCount = activeRuns.filter(isLiveRunActive).length;
  useEffect(() => {
    if (connectionState !== "online") return;
    let active = true;
    let pending = false;
    const refresh = async () => {
      if (pending) return;
      pending = true;
      try {
        const result = await getRuntimeStatus(
          claim.workspace_id, claim.session_id, claim.claim_version, claim.reconnect_credential,
        );
        if (active) {
          setStatus(result.status);
          setFailed(false);
        }
      } catch {
        if (active) setFailed(true);
      } finally {
        pending = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, [claim.workspace_id, claim.session_id, claim.claim_version, claim.reconnect_credential,
    connectionState, refreshVersion, activeCount]);
  const formatTokens = (value: number | null | undefined) => value == null ? "—" : value.toLocaleString(i18n.language);
  return (
    <div className={styles.runtimeStatusSummary} aria-label={t("management.tokenUsage")}>
      <span>{t("management.contextUsed")} {formatTokens(status?.projected_next_request_tokens)}</span>
      <span>{t("management.contextWindow")} {formatTokens(status?.context_window)}</span>
      <span>{t("management.historicalInput")} {formatTokens(status?.cumulative_usage.input_tokens)}</span>
      <span>{t("management.inputTokens")} {formatTokens(status?.last_request_usage?.input_tokens)}</span>
      <span>{t("management.cachedInput", { tokens: formatTokens(status?.last_request_usage?.cached_input_tokens) })}</span>
      {failed ? <span role="status">{t("management.actionError")}</span> : null}
    </div>
  );
}

interface RuntimeManagementDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  panel: "runtime" | "memory";
  onPermissionChanged: (permission: ToolPermissionLevel) => void;
  claim: SessionClaim;
  sessionTitle: string;
  sessionModelConfiguration: SessionModelConfiguration | null;
  activeRuns: LiveRun[];
  connectionState: ConnectionState;
  triggerRef: { current: HTMLElement | null };
}

export function RuntimeManagementDialog({
  open,
  onOpenChange,
  panel,
  onPermissionChanged,
  claim,
  sessionTitle,
  sessionModelConfiguration,
  activeRuns,
  connectionState,
  triggerRef,
}: RuntimeManagementDialogProps) {
  const { i18n, t } = useTranslation();
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [permission, setPermission] = useState<ToolPermissionLevel>("workspace-write");
  const [effort, setEffort] = useState<ReasoningEffort>("mid");
  const [loadState, setLoadState] = useState<"idle" | "loading" | "ready">("idle");
  const [saving, setSaving] = useState<"permission" | "effort" | null>(null);
  const [operation, setOperation] = useState<"memory" | "dream" | null>(null);
  const [memoryContent, setMemoryContent] = useState<string | null>(null);
  const [dreamResult, setDreamResult] = useState<DreamResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const requestEpoch = useRef(0);
  const dreamEpoch = useRef(0);
  const dreamInFlight = useRef(false);

  useEffect(() => {
    dreamEpoch.current += 1;
    dreamInFlight.current = false;
    setDreamResult(null);
    return () => { dreamEpoch.current += 1; };
  }, [claim, connectionState]);

  useEffect(() => {
    requestEpoch.current += 1;
    setSaving(null);
    setOperation(dreamInFlight.current ? "dream" : null);
    setMemoryContent(null);
    if (!open || connectionState !== "online") return;
    if (dreamInFlight.current) return;
    let active = true;
    setLoadState("loading");
    setStatus(null);
    setError(null);
    setNotice(null);
    void getRuntimeStatus(
      claim.workspace_id,
      claim.session_id,
      claim.claim_version,
      claim.reconnect_credential,
    ).then((result) => {
      if (!active) return;
      if (result.status === undefined) {
        setError("management.invalidStatus");
        setLoadState("ready");
        return;
      }
      setStatus(result.status);
      setPermission(result.status.current_permission_level);
      onPermissionChanged(result.status.current_permission_level);
      setEffort(result.status.chat_reasoning_effort);
      setLoadState("ready");
    }).catch((reason: unknown) => {
      if (!active) return;
      setError(managementErrorKey(reason));
      setLoadState("ready");
    });
    return () => { active = false; requestEpoch.current += 1; };
  }, [claim, onPermissionChanged, open, connectionState]);

  async function savePermission(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (status === null || saving !== null || connectionState !== "online") return;
    const epoch = requestEpoch.current;
    setSaving("permission");
    setError(null);
    setNotice(null);
    try {
      const result = await updateRuntimePermission(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
        permission,
      );
      if (epoch !== requestEpoch.current) return;
      const published = result.published_permission_level;
      if (published == null || !PERMISSION_LEVELS.includes(published)) {
        setError("management.invalidSelection");
        return;
      }
      setPermission(published);
      onPermissionChanged(published);
      setStatus((current) => current === null ? current : {
        ...current,
        current_permission_level: published,
      });
      setNotice("management.permissionSaved");
    } catch (reason: unknown) {
      if (epoch !== requestEpoch.current) return;
      setError(managementErrorKey(reason));
    } finally {
      if (epoch === requestEpoch.current) setSaving(null);
    }
  }

  async function saveEffort(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (status === null || saving !== null || connectionState !== "online") return;
    const epoch = requestEpoch.current;
    setSaving("effort");
    setError(null);
    setNotice(null);
    try {
      const result = await updateRuntimeEffort(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
        effort,
      );
      if (epoch !== requestEpoch.current) return;
      const published = result.published_effort;
      if (published == null || !REASONING_EFFORTS.includes(published)) {
        setError("management.invalidSelection");
        return;
      }
      setEffort(published);
      setStatus((current) => current === null ? current : {
        ...current,
        chat_reasoning_effort: published,
      });
      setNotice("management.effortSaved");
    } catch (reason: unknown) {
      if (epoch !== requestEpoch.current) return;
      setError(managementErrorKey(reason));
    } finally {
      if (epoch === requestEpoch.current) setSaving(null);
    }
  }

  async function viewMemory() {
    if (operation !== null || connectionState !== "online") return;
    const epoch = requestEpoch.current;
    setOperation("memory");
    setError(null);
    setNotice(null);
    try {
      const result = await getRuntimeMemory(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
      );
      if (epoch !== requestEpoch.current) return;
      if (typeof result.content !== "string") {
        setError("management.memoryError");
        return;
      }
      setMemoryContent(result.content);
      setNotice("management.memoryLoaded");
    } catch (reason: unknown) {
      if (epoch !== requestEpoch.current) return;
      setError(operationManagementErrorKey(reason, "management.memoryError"));
    } finally {
      if (epoch === requestEpoch.current) setOperation(null);
    }
  }

  async function runDream() {
    if (operation !== null || connectionState !== "online") return;
    const epoch = dreamEpoch.current;
    dreamInFlight.current = true;
    setOperation("dream");
    setDreamResult(null);
    setError(null);
    setNotice(null);
    try {
      const result = await triggerRuntimeDream(
        claim.workspace_id,
        claim.session_id,
        claim.claim_version,
        claim.reconnect_credential,
      );
      if (epoch !== dreamEpoch.current) return;
      if (result.result === undefined) {
        setError("management.dreamError");
        return;
      }
      const nextDream = result.result;
      setDreamResult(nextDream);
      setNotice(nextDream.error === null
        ? nextDream.status === "No pending summaries"
          ? "management.dreamNoPending"
          : "management.dreamCompleted"
        : nextDream.error.code === "memory_task_running"
          ? "management.dreamAlreadyRunning"
          : "management.dreamFailed");
    } catch (reason: unknown) {
      if (epoch !== dreamEpoch.current) return;
      setError(operationManagementErrorKey(reason, "management.dreamError"));
    } finally {
      if (epoch === dreamEpoch.current) {
        dreamInFlight.current = false;
        setOperation(null);
      }
    }
  }

  const activeWorkCount = activeRuns.filter(isLiveRunActive).length;
  const numberFormat = new Intl.NumberFormat(i18n.language);
  const usedTokens = status?.projected_next_request_tokens ?? 0;
  const availableTokens = status?.available_context ?? 0;
  const usedPercent = status === null ? 0 : Math.max(0, status.input_budget_used_percent);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={styles.dialogOverlay} />
        <Dialog.Content
          className={`${styles.dialogContent} ${styles.managementDialog}`}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            const trigger = triggerRef.current;
            if (trigger?.isConnected && (!(trigger instanceof HTMLButtonElement) || !trigger.disabled)) trigger.focus();
          }}
        >
          <div className={styles.dialogHeader}>
            <div>
              <Dialog.Title className={styles.dialogTitle}>
                {t(panel === "runtime" ? "management.title" : "management.memoryPanelTitle")}
              </Dialog.Title>
              <Dialog.Description className={styles.dialogDescription}>
                {t("management.scope", { session: sessionTitle })}
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                <X size={17} aria-hidden="true" />
              </button>
            </Dialog.Close>
          </div>

          {loadState === "loading" ? (
            <div className={styles.managementLoading} role="status" aria-live="polite">
              <RefreshCw size={16} className={styles.spin} aria-hidden="true" />
              {t("management.loading")}
            </div>
          ) : null}
          {error !== null ? (
            <div className={styles.errorBanner} role="alert">
              <CircleAlert size={16} aria-hidden="true" />
              <span>{t(error)}</span>
            </div>
          ) : null}
          {notice !== null ? (
            <div className={styles.notice} role="status" aria-live="polite">
              <Check size={16} aria-hidden="true" />
              {t(notice)}
            </div>
          ) : null}

          {status !== null ? (
            <>
              {panel === "runtime" ? (
                <>
              {status.model_configuration_available === false ? (
                <p className={styles.notice} role="status">{t("conversation.modelUnavailable")}</p>
              ) : null}
              <dl className={styles.managementStatusGrid} aria-label={t("management.statusTitle")}>
                {status.active_model_configuration !== undefined ? (
                  <div className={styles.managementMetric}>
                    <dt>{t("conversation.activeModel")}</dt>
                    <dd>
                      {`${status.active_model_configuration.provider_id}/${status.active_model_configuration.model} · ${status.active_model_configuration.reasoning_effort}`}
                    </dd>
                  </div>
                ) : null}
                <div className={styles.managementMetric}>
                  <dt>{t("conversation.sessionModel")}</dt>
                  <dd>{sessionModelConfiguration === null
                    ? t("management.inheritedSessionModel", { model: status.chat_model })
                    : `${sessionModelConfiguration.provider_id}/${sessionModelConfiguration.model} · ${sessionModelConfiguration.reasoning_effort}`}
                  </dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.model")}</dt>
                  <dd>{status.chat_model || "-"}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.context")}</dt>
                  <dd>
                    {numberFormat.format(usedTokens)} / {numberFormat.format(availableTokens)}
                    <span>{Math.round(usedPercent)}%</span>
                  </dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.activeWork")}</dt>
                  <dd>{activeWorkCount > 0 ? t("management.activeWorkCount", { count: activeWorkCount }) : t("management.idle")}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.permission")}</dt>
                  <dd>{t(`management.permissionLevels.${status.current_permission_level}`)}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.effort")}</dt>
                  <dd>{status.chat_reasoning_effort}</dd>
                </div>
                <div className={styles.managementMetric}>
                  <dt>{t("management.messages")}</dt>
                  <dd>{numberFormat.format(status.session_message_count)}</dd>
                </div>
              </dl>

              <div className={styles.managementControls}>
                <form className={styles.managementControl} onSubmit={(event) => void savePermission(event)}>
                  <label className={styles.fieldLabel} htmlFor="runtime-permission">
                    {t("management.permissionLabel")}
                  </label>
                  <select
                    id="runtime-permission"
                    className={styles.textInput}
                    value={permission}
                    disabled={saving !== null || connectionState !== "online"}
                    onChange={(event) => setPermission(event.target.value as ToolPermissionLevel)}
                  >
                    {PERMISSION_LEVELS.map((level) => (
                      <option key={level} value={level}>{t(`management.permissionLevels.${level}`)}</option>
                    ))}
                  </select>
                  <button className={styles.secondaryButton} type="submit" disabled={saving !== null || connectionState !== "online"}>
                    <Check size={15} aria-hidden="true" />
                    {saving === "permission" ? t("management.saving") : t("controls.save")}
                  </button>
                </form>
                <form className={styles.managementControl} onSubmit={(event) => void saveEffort(event)}>
                  <label className={styles.fieldLabel} htmlFor="runtime-effort">
                    {t("management.effortLabel")}
                  </label>
                  <select
                    id="runtime-effort"
                    className={styles.textInput}
                    value={effort}
                    disabled={saving !== null || connectionState !== "online"}
                    onChange={(event) => setEffort(event.target.value as ReasoningEffort)}
                  >
                    {REASONING_EFFORTS.map((level) => (
                      <option key={level} value={level}>{level}</option>
                    ))}
                  </select>
                  <button className={styles.secondaryButton} type="submit" disabled={saving !== null || connectionState !== "online"}>
                    <Check size={15} aria-hidden="true" />
                    {saving === "effort" ? t("management.saving") : t("controls.save")}
                  </button>
                </form>
              </div>
                </>
              ) : null}

              {panel === "memory" ? <div className={styles.managementTools}>
                <section className={styles.managementTool} aria-labelledby="management-memory-title">
                  <div className={styles.managementToolHeader}>
                    <div>
                      <h3 id="management-memory-title">{t("management.memoryTitle")}</h3>
                    </div>
                    <button
                      className={styles.secondaryButton}
                      type="button"
                      disabled={operation !== null || connectionState !== "online"}
                      onClick={() => void viewMemory()}
                    >
                      <BookOpen size={15} aria-hidden="true" />
                      {operation === "memory" ? t("management.loading") : t("management.viewMemory")}
                    </button>
                  </div>
                  {memoryContent !== null ? (
                    <div className={styles.managementMemory} role="region" aria-label={t("management.memoryRegion")}>
                      <h4>{t("management.memoryRegion")}</h4>
                      <pre>{memoryContent || t("management.memoryEmpty")}</pre>
                    </div>
                  ) : null}
                </section>

                <section className={styles.managementTool} aria-labelledby="management-dream-title">
                  <div className={styles.managementToolHeader}>
                    <div>
                      <h3 id="management-dream-title">{t("management.dreamTitle")}</h3>
                    </div>
                    <button
                      className={styles.secondaryButton}
                      type="button"
                      disabled={operation !== null || connectionState !== "online"}
                      onClick={() => void runDream()}
                    >
                      <Brain size={15} aria-hidden="true" />
                      {operation === "dream" ? t("management.dreamRunning") : t("management.runDream")}
                    </button>
                  </div>
                  {dreamResult !== null ? (
                    <div className={styles.managementOperationStatus} role="status" aria-live="polite">
                      <strong>
                        {dreamResult.error !== null
                          ? t(dreamResult.error.code === "memory_task_running"
                            ? "management.dreamAlreadyRunning" : "management.dreamFailed")
                          : dreamResult.status === "No pending summaries"
                            ? t("management.dreamNoPending")
                            : t("management.dreamCompleted")}
                      </strong>
                      <span>{t("management.dreamProcessed", { count: dreamResult.processed_count })}</span>
                      <span>{dreamResult.memory_updated ? t("management.dreamUpdated") : t("management.dreamUnchanged")}</span>
                      <span>{t("management.dreamCursor", { cursor: dreamResult.cursor })}</span>
                      {dreamResult.error !== null ? (
                        <span>{dreamResult.error.code}: {dreamResult.error.message}</span>
                      ) : null}
                    </div>
                  ) : null}
                </section>
              </div> : null}
            </>
          ) : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
