import {
  Save,
  Trash2
} from "lucide-react";
import { useTranslation } from "react-i18next";
import commonStyles from "../../shared/styles/controls.module.css";
import settingsStyles from "./Settings.module.css";
import type { SecretAction, SecretDraft } from "./forms.ts";

const styles = { ...commonStyles, ...settingsStyles };

interface ApiKeyInputProps {
  id: string;
  configured: boolean;
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSave: () => void;
  error?: string;
}

export function ApiKeyInput({ id, configured, value, disabled, onChange, onSave, error }: ApiKeyInputProps) {
  const { t } = useTranslation();
  return (
    <div className={styles.settingsField} id={id} tabIndex={-1} data-api-key-editor>
      <label className={styles.fieldLabel} htmlFor={`${id}-value`}>{t("settings.apiKey")}</label>
      <div className={styles.apiKeyRow}>
        {value.length > 0 ? (
          <button
            className={styles.apiKeySave}
            id={`${id}-save`}
            type="button"
            aria-label={t("settings.saveApiKey")}
            title={t("settings.saveApiKey")}
            disabled={disabled}
            onClick={onSave}
          ><Save size={16} aria-hidden="true" /></button>
        ) : null}
        <input
          className={styles.textInput}
          id={`${id}-value`}
          type="password"
          autoComplete="new-password"
          aria-invalid={error !== undefined}
          aria-describedby={error !== undefined ? `${id}-error` : undefined}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.currentTarget.value)}
        />
      </div>
      {error !== undefined ? <span className={styles.fieldError} id={`${id}-error`}>{error}</span> : null}
      <small className={styles.settingsSecretState}>
        {configured ? t("settings.secretConfigured") : t("settings.secretNotConfigured")}
      </small>
    </div>
  );
}

interface SecretInputProps {
  id: string;
  label: string;
  secret: SecretDraft;
  disabled: boolean;
  onChange: (update: Partial<SecretDraft>) => void;
  error?: string;
}

export function SecretInput({ id, label, secret, disabled, onChange, error }: SecretInputProps) {
  const { t } = useTranslation();
  return (
    <div className={styles.settingsField} id={id} tabIndex={-1}>
      <span className={styles.fieldLabel} id={`${id}-label`}>{label}</span>
      <div className={styles.settingsSecretRow}>
        <select
          className={styles.selectInput}
          id={`${id}-action`}
          aria-labelledby={`${id}-label`}
          aria-invalid={error !== undefined}
          aria-describedby={error !== undefined ? `${id}-error` : undefined}
          value={secret.action}
          disabled={disabled}
          onChange={(event) => onChange({ action: event.currentTarget.value as SecretAction, value: "" })}
        >
          <option value="keep">{t("settings.secretKeep")}</option>
          <option value="replace">{t("settings.secretReplace")}</option>
          <option value="clear">{t("settings.secretClear")}</option>
        </select>
        {secret.action === "replace" ? (
          <input
            className={styles.textInput}
            id={`${id}-value`}
            type="password"
            aria-invalid={error !== undefined}
            aria-describedby={error !== undefined ? `${id}-error` : undefined}
            autoComplete="new-password"
            aria-label={t("settings.secretValue")}
            value={secret.value}
            disabled={disabled}
            onChange={(event) => onChange({ value: event.currentTarget.value })}
          />
        ) : null}
      </div>
      {error !== undefined ? <span className={styles.fieldError} id={`${id}-error`}>{error}</span> : null}
      <small className={styles.settingsSecretState}>
        {secret.configured ? t("settings.secretConfigured") : t("settings.secretNotConfigured")}
      </small>
    </div>
  );
}

interface SettingsListFieldProps {
  error?: string;
  id: string;
  label: string;
  values: string[];
  disabled: boolean;
  onChange: (values: string[]) => void;
}

export function SettingsListField({ id, label, values, disabled, onChange, error }: SettingsListFieldProps) {
  const { t } = useTranslation();
  return (
    <div className={styles.settingsField} id={id} tabIndex={-1}>
      <span className={styles.fieldLabel}>{label}</span>
      {values.map((value, index) => (
        <div className={styles.settingsListRow} key={index}>
          <textarea className={styles.textArea} aria-label={`${label} ${index + 1}`} rows={2} aria-invalid={error !== undefined} aria-describedby={error !== undefined ? `${id}-error` : undefined} value={value} disabled={disabled} onChange={(event) => onChange(values.map((entry, position) => position === index ? event.currentTarget.value : entry))} />
          <button type="button" className={styles.iconButton} aria-label={`${t("settings.removeItem")} ${label} ${index + 1}`} disabled={disabled} onClick={() => { onChange(values.filter((_, position) => position !== index)); document.getElementById(id)?.focus(); }}><Trash2 size={15} aria-hidden="true" /></button>
        </div>
      ))}
      {error !== undefined ? <span className={styles.fieldError} id={`${id}-error`}>{error}</span> : null}
      <button type="button" className={styles.secondaryButton} disabled={disabled} onClick={() => onChange([...values, ""])}>{t("settings.addItem")}</button>
    </div>
  );
}

interface SettingsNumberFieldProps {
  id: string;
  label: string;
  value: string;
  error: string | undefined;
  disabled: boolean;
  step?: string;
  onChange: (value: string) => void;
  onBlur: () => void;
}

export function SettingsNumberField({ id, label, value, error, disabled, step, onChange, onBlur }: SettingsNumberFieldProps) {
  return (
    <label className={styles.settingsField} htmlFor={id}>
      <span className={styles.fieldLabel} id={`${id}-label`}>{label}</span>
      <input
        className={styles.textInput}
        id={id}
        aria-labelledby={`${id}-label`}
        type="number"
        inputMode="decimal"
        step={step}
        value={value}
        disabled={disabled}
        aria-invalid={error !== undefined}
        aria-describedby={error !== undefined ? `${id}-error` : undefined}
        onChange={(event) => onChange(event.currentTarget.value)}
        onBlur={onBlur}
      />
      <span className={styles.fieldError} id={`${id}-error`}>{error ?? ""}</span>
    </label>
  );
}
