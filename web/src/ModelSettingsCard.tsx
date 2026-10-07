import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, Trash2 } from "lucide-react";
import { REASONING_EFFORTS } from "./reasoningEffort.ts";
import type { ReasoningEffort } from "./protocol";
import { modelSettingsFieldId } from "./modelSettings";
import type { ModelForm } from "./modelSettings";
import styles from "./App.module.css";

interface Props {
  providerId: string;
  row: string;
  model: ModelForm;
  persisted: boolean;
  referencedBy: string[];
  disabled: boolean;
  errorFor: (path: string) => string | undefined;
  onChange: (change: Partial<ModelForm>) => void;
  onRemove: () => void;
}

export function ModelSettingsCard({ providerId, row, model, persisted, referencedBy, disabled, errorFor, onChange, onRemove }: Props) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(!persisted || model.migration_candidates !== undefined);
  const prefix = `models.providers.${providerId}.models.${model.id || row}`;
  const identityError = errorFor(`models.providers.${providerId}.models.${row}.id`) ?? errorFor(`${prefix}.id`);
  const fields = ["context_window", "max_output", "temperature", "timeout"] as const;
  const labels = { context_window: "contextWindow", max_output: "maxOutput", temperature: "temperature", timeout: "timeout" };
  const hasConflict = model.migration_candidates !== undefined && (fields.some((name) => model[name] === "") || model.reasoning_effort === "");
  return (
    <div className={styles.modelCard} id={modelSettingsFieldId(providerId, row)}>
      <details open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
        <summary className={styles.modelCardSummary}>
          <span>
            <strong>{model.id || t("settings.newModel")}</strong>
            <small>{providerId} · {t("settings.contextWindow")} {model.context_window || "—"} · {t("settings.maxOutput")} {model.max_output || "—"}</small>
          </span>
          <ChevronDown size={15} aria-hidden="true" />
        </summary>
        <div className={styles.modelCardFields}>
          {model.migration_candidates !== undefined && hasConflict ? (
            <div className={styles.modelMigration}>
              <p>{t("settings.modelMigrationConflict")}</p>
              {model.migration_candidates.map((candidate) => (
                <button className={styles.secondaryButton} key={candidate.route} type="button" disabled={disabled} onClick={() => onChange({
                  context_window: String(candidate.context_window), max_output: String(candidate.max_output),
                  temperature: String(candidate.temperature), reasoning_effort: candidate.reasoning_effort, timeout: String(candidate.timeout),
                })}>
                  {t("settings.useRouteParameters", { route: candidate.route })}
                  <small>{t("settings.modelParameterSummary", { ...candidate })}</small>
                </button>
              ))}
            </div>
          ) : null}
          <label className={styles.settingsField} htmlFor={modelSettingsFieldId(providerId, row, "id")}>
            <span className={styles.fieldLabel} id={`${modelSettingsFieldId(providerId, row, "id")}-label`}>{t("settings.model")}</span>
            <input className={styles.textInput} id={modelSettingsFieldId(providerId, row, "id")} value={model.id} readOnly={persisted} disabled={disabled}
              aria-labelledby={`${modelSettingsFieldId(providerId, row, "id")}-label`}
              aria-invalid={identityError !== undefined} aria-describedby={`${modelSettingsFieldId(providerId, row, "id")}-error`}
              onChange={(event) => onChange({ id: event.currentTarget.value })} />
            <span className={styles.fieldError} id={`${modelSettingsFieldId(providerId, row, "id")}-error`}>{identityError ?? ""}</span>
          </label>
          {fields.map((field) => {
            const id = modelSettingsFieldId(providerId, row, field);
            return <label className={styles.settingsField} htmlFor={id} key={field}>
              <span className={styles.fieldLabel} id={`${id}-label`}>{t(`settings.${labels[field]}`)}</span>
              <input className={styles.textInput} type="number" id={id} value={model[field]} step={field === "temperature" ? "0.1" : "1"} disabled={disabled}
                aria-labelledby={`${id}-label`}
                aria-invalid={errorFor(`${prefix}.${field}`) !== undefined} aria-describedby={`${id}-error`}
                onChange={(event) => onChange({ [field]: event.currentTarget.value })} />
              <span className={styles.fieldError} id={`${id}-error`}>{errorFor(`${prefix}.${field}`) ?? ""}</span>
            </label>;
          })}
          <label className={styles.settingsField} htmlFor={modelSettingsFieldId(providerId, row, "reasoning_effort")}>
            <span className={styles.fieldLabel} id={`${modelSettingsFieldId(providerId, row, "reasoning_effort")}-label`}>{t("settings.reasoningEffort")}</span>
            <select className={styles.selectInput} id={modelSettingsFieldId(providerId, row, "reasoning_effort")} value={model.reasoning_effort} disabled={disabled}
              aria-labelledby={`${modelSettingsFieldId(providerId, row, "reasoning_effort")}-label`}
              aria-invalid={errorFor(`${prefix}.reasoning_effort`) !== undefined} aria-describedby={`${modelSettingsFieldId(providerId, row, "reasoning_effort")}-error`}
              onChange={(event) => onChange({ reasoning_effort: event.currentTarget.value as ReasoningEffort })}>
              <option value="" disabled>{t("settings.required")}</option>
              {REASONING_EFFORTS.map((effort) => <option key={effort} value={effort}>{effort}</option>)}
            </select>
            <span className={styles.fieldError} id={`${modelSettingsFieldId(providerId, row, "reasoning_effort")}-error`}>{errorFor(`${prefix}.reasoning_effort`) ?? ""}</span>
          </label>
        </div>
      </details>
      <button className={`${styles.iconButton} ${styles.modelCardRemove}`} type="button" disabled={disabled || referencedBy.length > 0}
        aria-label={t("settings.removeModel")} title={referencedBy.length > 0 ? t("settings.modelReferenced", { routes: referencedBy.join(", ") }) : t("settings.removeModel")}
        onClick={onRemove}><Trash2 size={15} aria-hidden="true" /></button>
    </div>
  );
}
