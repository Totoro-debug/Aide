import * as Dialog from "@radix-ui/react-dialog";
import {
  Activity,
  ArrowLeft,
  BookOpen,
  Brain,
  CircleAlert,
  CircleCheck,
  Clock3,
  Gauge,
  Plus,
  RefreshCw,
  RotateCcw,
  Settings2,
  ShieldX,
  Trash2,
  TriangleAlert,
  X
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, useLocation } from "react-router-dom";
import type { Theme } from "../../app/theme.ts";
import {
  ApiError,
  createRequestId,
  getConfig,
  patchConfig,
  reloadRuntimeSkills,
  repairConfig,
  restartService
} from "../../shared/service/api";
import { serviceStateLabel } from "../../shared/service/presentation.ts";
import type {
  ConfigFields,
  ConfigResponse,
  ModelRouteName,
  ServiceStatus,
  SessionClaim,
  SkillMetadata,
  ToolPermissionLevel
} from "../../shared/service/protocol";
import type { AuthState, ConnectionState } from "../../shared/service/types.ts";
import commonStyles from "../../shared/styles/controls.module.css";
import { operationManagementErrorKey } from "../runtime/errors.ts";
import { ApiKeyInput, SecretInput, SettingsListField, SettingsNumberField } from "./fields.tsx";
import type { ConfigSettingsSection, McpForm, PendingSettingsSave, ProviderForm, RouteForm, SettingsFieldError, SettingsForm, SettingsSection } from "./forms.ts";
import { changedConfigFieldPaths, configFieldsForSections, configFromForm, configSecretsForSections, formFromConfig, isCompleteModelForm, isSelectableProvider, preserveSettingsInput, sameSettingsValue, secretDraft, settingsSectionsForChanges } from "./forms.ts";
import type { ModelForm } from "./modelSettings";
import { modelSettingsFieldId } from "./modelSettings";
import { ModelSettingsCard } from "./ModelSettingsCard";
import settingsStyles from "./Settings.module.css";

const styles = { ...commonStyles, ...settingsStyles };

interface SettingsViewProps {
  authState: AuthState;
  connectionState: ConnectionState;
  language: "en" | "zh-CN";
  onBack: () => void;
  onLanguageChange: (language: "en" | "zh-CN") => void;
  onThemeChange: (theme: Theme) => void;
  onRestartExpectation: (expected: boolean) => void;
  serviceStatus: ServiceStatus | null;
  runtimeClaim: SessionClaim | null;
  theme: Theme;
}

export function SettingsView({
  authState,
  connectionState,
  language,
  onBack,
  onLanguageChange,
  onThemeChange,
  onRestartExpectation,
  serviceStatus,
  runtimeClaim,
  theme,
}: SettingsViewProps) {
  const { t } = useTranslation();
  const location = useLocation();
  const settingsContentRef = useRef<HTMLDivElement | null>(null);
  const [skillReloadBusy, setSkillReloadBusy] = useState(false);
  const [skillReloadError, setSkillReloadError] = useState<string | null>(null);
  const [skillMetadata, setSkillMetadata] = useState<SkillMetadata[] | null>(null);
  const skillReloadVersionRef = useRef(0);
  const [activeSection, setActiveSection] = useState<SettingsSection>("general");
  useEffect(() => {
    skillReloadVersionRef.current += 1;
    setSkillReloadBusy(false);
    setSkillReloadError(null);
    setSkillMetadata(null);
  }, [runtimeClaim]);

  const reloadSkills = useCallback(async () => {
    if (runtimeClaim === null || connectionState !== "online" || skillReloadBusy) return;
    const version = skillReloadVersionRef.current + 1;
    skillReloadVersionRef.current = version;
    setSkillReloadBusy(true);
    setSkillReloadError(null);
    try {
      const result = await reloadRuntimeSkills(
        runtimeClaim.workspace_id,
        runtimeClaim.session_id,
        runtimeClaim.claim_version,
        runtimeClaim.reconnect_credential,
      );
      if (skillReloadVersionRef.current !== version) return;
      if (!Array.isArray(result.skills)) {
        setSkillReloadError("management.skillsError");
        return;
      }
      setSkillMetadata(result.skills);
    } catch (reason: unknown) {
      if (skillReloadVersionRef.current === version) {
        setSkillReloadError(operationManagementErrorKey(reason, "management.skillsError"));
      }
    } finally {
      if (skillReloadVersionRef.current === version) setSkillReloadBusy(false);
    }
  }, [connectionState, runtimeClaim, skillReloadBusy]);
  useEffect(() => {
    if (location.hash === "#models") setActiveSection("models");
  }, [location.hash]);
  const [response, setResponse] = useState<ConfigResponse | null>(null);
  const [draft, setDraft] = useState<SettingsForm | null>(null);
  const [apiKeyInputs, setApiKeyInputs] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<SettingsFieldError>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [saveFailed, setSaveFailed] = useState(false);
  const [conflictedPaths, setConflictedPaths] = useState<string[]>([]);
  const [restartOpen, setRestartOpen] = useState(false);
  const [microCompressionOpen, setMicroCompressionOpen] = useState(false);
  const microCompressionTriggerRef = useRef<HTMLInputElement | null>(null);
  const [restartBusy, setRestartBusy] = useState(false);
  const [restartError, setRestartError] = useState<string | null>(null);
  const restartingInstanceRef = useRef<string | null>(null);
  const restartRequestRef = useRef<string | null>(null);
  const restartTriggerRef = useRef<HTMLButtonElement | null>(null);
  const dirtyRef = useRef(false);
  const draftRef = useRef<SettingsForm | null>(null);
  const draftRevisionRef = useRef<string | null>(null);
  const baselineFieldsRef = useRef<ConfigFields | null>(null);
  const baselineSecretRevisionsRef = useRef<Record<string, string | null> | null>(null);
  const requestSequence = useRef(0);
  const editorIdRef = useRef(createRequestId());
  const mutationSequence = useRef(0);
  const retryOperationRef = useRef<PendingSettingsSave | null>(null);
  const dirtySectionsRef = useRef(new Set<ConfigSettingsSection>());
  const conflictPathsRef = useRef<string[]>([]);
  const mutationInFlight = useRef(false);
  const focusErrorSummaryRef = useRef(false);
  const errorSummaryRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (notice !== "settings.restarted") return;
    const timer = window.setTimeout(() => setNotice(null), 10_000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (submitError === null || !saveFailed || !focusErrorSummaryRef.current) return;
    focusErrorSummaryRef.current = false;
    const timer = window.setTimeout(() => errorSummaryRef.current?.focus(), 0);
    return () => window.clearTimeout(timer);
  }, [saveFailed, submitError]);

  const applyResponse = useCallback((next: ConfigResponse) => {
    setResponse(next);
    if (!dirtyRef.current || draftRef.current === null) {
      const nextDraft = formFromConfig(next.fields, draftRef.current);
      setDraft(nextDraft);
      draftRef.current = nextDraft;
      draftRevisionRef.current = next.revision;
      baselineFieldsRef.current = next.fields;
      baselineSecretRevisionsRef.current = next.secret_revisions;
      dirtySectionsRef.current.clear();
      conflictPathsRef.current = [];
      setConflictedPaths([]);
      setDirty(false);
      dirtyRef.current = false;
    }
    setLoading(false);
  }, []);

  const loadSettings = useCallback(async () => {
    if (mutationInFlight.current || connectionState !== "online") return;
    const sequence = requestSequence.current + 1;
    requestSequence.current = sequence;
    try {
      const next = await getConfig();
      if (requestSequence.current !== sequence) return;
      applyResponse(next);
      setLoadError(null);
    } catch (error) {
      if (requestSequence.current !== sequence) return;
      if (!dirtyRef.current) setLoading(false);
      setLoadError(error instanceof ApiError ? error.message : t("settings.unavailable"));
    }
  }, [applyResponse, connectionState, t]);

  useEffect(() => {
    if (authState !== "ready") {
      setLoading(false);
      return;
    }
    let active = true;
    void loadSettings();
    const timer = window.setInterval(() => {
      if (active) void loadSettings();
    }, 2500);
    return () => {
      active = false;
      requestSequence.current += 1;
      window.clearInterval(timer);
    };
  }, [authState, loadSettings]);

  useEffect(() => () => {
    mutationSequence.current += 1;
  }, []);

  const updateField = useCallback((path: string, value: string | boolean) => {
    const current = draftRef.current;
    if (current === null) return;
    const [section, field] = path.split(".");
    if (section !== "runtime" && section !== "memory" && section !== "web") return;
    if (section === "web" && typeof value !== "string") return;
    const next = {
      ...current,
      [section]: { ...current[section], [field]: value },
    } as SettingsForm;
    dirtySectionsRef.current.add(section);
    draftRef.current = next;
    dirtyRef.current = true;
    setDraft(next);
    setDirty(true);
    setSaveFailed(false);
    setNotice(null);
    const hasErrorsOutsideSection = Object.keys(fieldErrors).some((errorPath) => errorPath.split(".")[0] !== section);
    if (conflictPathsRef.current.length === 0 && !hasErrorsOutsideSection) setSubmitError(null);
  }, [fieldErrors]);

  const updateDraft = useCallback((update: (current: SettingsForm) => SettingsForm) => {
    const current = draftRef.current;
    if (current === null) return;
    const next = update(current);
    const changedSections: ConfigSettingsSection[] = [];
    for (const section of ["runtime", "memory", "web", "models", "mcp"] as const) {
      if (!sameSettingsValue(current[section], next[section])) {
        dirtySectionsRef.current.add(section);
        changedSections.push(section);
      }
    }
    draftRef.current = next;
    dirtyRef.current = true;
    setDraft(next);
    setDirty(true);
    setSaveFailed(false);
    setNotice(null);
    const hasErrorsOutsideChangedSections = Object.keys(fieldErrors).some((errorPath) => (
      !changedSections.includes(errorPath.split(".")[0] as ConfigSettingsSection)
    ));
    if (conflictPathsRef.current.length === 0 && !hasErrorsOutsideChangedSections) setSubmitError(null);
  }, [fieldErrors]);

  const updateProvider = useCallback((id: string, update: Partial<ProviderForm>) => {
    updateDraft((current) => ({
      ...current,
      models: {
        ...current.models,
        providers: {
          ...current.models.providers,
          [id]: { ...current.models.providers[id], ...update },
        },
      },
    }));
  }, [updateDraft]);

  const updateRoute = useCallback((id: ModelRouteName, update: Partial<RouteForm>) => {
    updateDraft((current) => ({
      ...current,
      models: {
        ...current.models,
        routes: { ...current.models.routes, [id]: { ...current.models.routes[id], ...update } },
      },
    }));
  }, [updateDraft]);

  const updateMcp = useCallback((id: string, update: Partial<McpForm>) => {
    updateDraft((current) => ({
      ...current,
      mcp: { ...current.mcp, [id]: { ...current.mcp[id], ...update } },
    }));
  }, [updateDraft]);

  const addProvider = useCallback(() => {
    const current = draftRef.current;
    if (current === null) return;
    let id = "new-provider";
    let index = 2;
    while (current.models.providers[id] !== undefined) id = `new-provider-${index++}`;
    setApiKeyInputs((inputs) => ({ ...inputs, [id]: "" }));
    updateDraft((form) => ({
      ...form,
      models: {
        ...form.models,
        providers: {
          ...form.models.providers,
          [id]: { id, protocol: "openai-compatible", base_url: "", models: {}, api_key: secretDraft(false) },
        },
      },
    }));
  }, [updateDraft]);

  const removeProvider = useCallback((id: string) => {
    setApiKeyInputs((inputs) => {
      const next = { ...inputs };
      delete next[id];
      return next;
    });
    updateDraft((current) => {
      const providers = { ...current.models.providers };
      delete providers[id];
      return { ...current, models: { ...current.models, providers } };
    });
  }, [updateDraft]);

  const addModel = useCallback((providerRow: string) => {
    updateDraft((current) => {
      const provider = current.models.providers[providerRow];
      const row = createRequestId();
      return { ...current, models: { ...current.models, providers: {
        ...current.models.providers, [providerRow]: { ...provider, models: {
          ...provider.models, [row]: { id: "", saved: false, context_window: "", max_output: "8192", temperature: "0.2", reasoning_effort: "mid", timeout: "120" },
        } },
      } } };
    });
  }, [updateDraft]);

  const updateModel = useCallback((providerRow: string, modelRow: string, update: Partial<ModelForm>) => {
    updateDraft((current) => {
      const provider = current.models.providers[providerRow];
      return { ...current, models: { ...current.models, providers: {
        ...current.models.providers, [providerRow]: { ...provider, models: {
          ...provider.models, [modelRow]: { ...provider.models[modelRow], ...update },
        } },
      } } };
    });
  }, [updateDraft]);

  const removeModel = useCallback((providerRow: string, modelRow: string) => {
    updateDraft((current) => {
      const provider = current.models.providers[providerRow];
      const models = { ...provider.models };
      delete models[modelRow];
      return { ...current, models: { ...current.models, providers: {
        ...current.models.providers, [providerRow]: { ...provider, models },
      } } };
    });
  }, [updateDraft]);

  const addMcp = useCallback(() => {
    const current = draftRef.current;
    if (current === null) return;
    let name = "new-mcp";
    let index = 2;
    while (current.mcp[name] !== undefined) name = `new-mcp-${index++}`;
    updateDraft((form) => ({
      ...form,
      mcp: {
        ...form.mcp,
        [name]: {
          name,
          enabled: false,
          transport: "stdio",
          command: "python",
          args: [],
          cwd: "",
          url: "",
          headers: {},
          connect_timeout: "30",
          call_timeout: "60",
          tool_keywords: [],
        },
      },
    }));
  }, [updateDraft]);

  const removeMcp = useCallback((id: string) => {
    updateDraft((current) => {
      const mcp = { ...current.mcp };
      delete mcp[id];
      return { ...current, mcp };
    });
  }, [updateDraft]);

  const addHeader = useCallback((serverId: string) => {
    const current = draftRef.current?.mcp[serverId];
    if (current === undefined) return;
    let header = "Authorization";
    let index = 2;
    while (current.headers[header] !== undefined) header = `X-Header-${index++}`;
    updateMcp(serverId, { headers: { ...current.headers, [header]: { ...secretDraft(false), name: header } } });
  }, [updateMcp]);

  const removeHeader = useCallback((serverId: string, header: string) => {
    const current = draftRef.current?.mcp[serverId];
    if (current === undefined) return;
    const headers = { ...current.headers };
    delete headers[header];
    updateMcp(serverId, { headers });
  }, [updateMcp]);

  const saveDraft = useCallback(async (
    overwriteConflicts = false,
    focusError = false,
    autoSection?: ConfigSettingsSection | null,
    confirmedApiKey?: { providerRow: string; value: string },
  ) => {
    if (authState !== "ready" || connectionState !== "online") {
      return;
    }
    const keyProvider = confirmedApiKey === undefined ? undefined : draftRef.current?.models.providers[confirmedApiKey.providerRow];
    const previousKeyChange = keyProvider === undefined ? undefined
      : retryOperationRef.current?.secrets[`models.providers.${keyProvider.id}.api_key`];
    const retryConfirmedKey = confirmedApiKey !== undefined && previousKeyChange?.action === "replace"
      && previousKeyChange.value === confirmedApiKey.value;
    const pending = saveFailed && (focusError || retryConfirmedKey) ? retryOperationRef.current : null;
    if (saveFailed && !focusError && confirmedApiKey === undefined) {
      return;
    }
    if (confirmedApiKey !== undefined) {
      if (keyProvider === undefined || confirmedApiKey.value.length === 0) return;
      updateProvider(confirmedApiKey.providerRow, {
        api_key: { ...keyProvider.api_key, action: "replace", value: confirmedApiKey.value },
      });
    }
    focusErrorSummaryRef.current = focusError;
    const snapshot = pending?.snapshot ?? draftRef.current;
    const baseline = pending?.baseline ?? baselineFieldsRef.current;
    const baselineSecrets = pending?.baselineSecrets ?? baselineSecretRevisionsRef.current;
    const revision = pending?.revision ?? draftRevisionRef.current;
    if (snapshot === null || baseline === null || baselineSecrets === null || revision === null) return;
    const fullCandidate = pending === null ? configFromForm(snapshot) : null;
    const allChangedSections = fullCandidate === null
      ? []
      : settingsSectionsForChanges(baseline, fullCandidate.fields, fullCandidate.secrets);
    const repairing = pending?.repairing ?? response?.configuration.repair_required === true;
    const sections = pending?.sections ?? (repairing
      ? ["runtime", "memory", "web", "models", "mcp"]
      : focusError
        ? allChangedSections
        : autoSection === undefined
          ? [...dirtySectionsRef.current]
          : autoSection === null
            ? []
            : allChangedSections.includes(autoSection) || dirtySectionsRef.current.has(autoSection)
              ? [autoSection] : []);
    const candidate = pending === null
      ? {
        fields: configFieldsForSections(fullCandidate?.fields ?? {}, sections),
        secrets: configSecretsForSections(fullCandidate?.secrets ?? {}, sections),
      }
      : { fields: pending.fields, secrets: pending.secrets };
    const incompleteReplacement = (sections.includes("models") && Object.values(snapshot.models.providers).some((provider) => (
      provider.api_key.action === "replace" && provider.api_key.value.length === 0
    ))) || (sections.includes("mcp") && Object.values(snapshot.mcp).some((server) => Object.values(server.headers).some((header) => (
      header.action === "replace" && header.value.length === 0
    ))));
    if (incompleteReplacement && !focusError) return;
    if (sections.includes("models")) {
      const modelErrors: SettingsFieldError = {};
      let incompleteModels = false;
      for (const provider of Object.values(snapshot.models.providers)) {
        const names = Object.values(provider.models).map((model) => model.id);
        for (const [row, model] of Object.entries(provider.models)) {
          if (model.id && names.filter((name) => name === model.id).length > 1) {
            if (!model.saved || !Object.values(provider.models).some((candidate) => candidate.id === model.id && !candidate.saved)) {
              modelErrors[`models.providers.${provider.id}.models.${row}.id`] = t("settings.duplicateModel");
            }
          }
          if ([model.id, model.context_window, model.max_output, model.temperature, model.reasoning_effort, model.timeout].some((value) => value.trim() === "")) incompleteModels = true;
        }
      }
      if (Object.keys(modelErrors).length > 0) {
        setFieldErrors(modelErrors);
        setSubmitError(t("settings.duplicateModel"));
        return;
      }
      const incompleteRoute = Object.values(snapshot.models.routes).some((route) => (
        route.name === "chat" ? route.provider_id === "" || route.model === ""
          : (route.provider_id === "") !== (route.model === "")
      ));
      if ((incompleteModels || incompleteRoute) && !focusError) return;
    }
    const changedPaths = fullCandidate === null ? [] : changedConfigFieldPaths(baseline, fullCandidate.fields);
    const hasFieldChanges = pending === null
      ? changedPaths.some(([section]) => sections.includes(section as ConfigSettingsSection))
      : Object.keys(candidate.fields).length > 0;
    const hasSecretChanges = Object.values(candidate.secrets).some(({ action }) => action !== "keep");
    if (!repairing && !hasFieldChanges && !hasSecretChanges) {
      for (const section of sections) dirtySectionsRef.current.delete(section);
      const remainingErrors = Object.fromEntries(Object.entries(fieldErrors).filter(([path]) => (
        !sections.includes(path.split(".")[0] as ConfigSettingsSection)
      )));
      setFieldErrors(remainingErrors);
      dirtyRef.current = dirtySectionsRef.current.size > 0;
      setDirty(dirtyRef.current);
      setSaveFailed(false);
      if (Object.keys(remainingErrors).length === 0 && conflictPathsRef.current.length === 0) setSubmitError(null);
      return;
    }
    const inFlight = retryOperationRef.current;
    if (mutationInFlight.current && inFlight !== null
      && sameSettingsValue(inFlight.snapshot, snapshot)
      && sameSettingsValue(inFlight.sections, sections)) return;

    requestSequence.current += 1;
    const sequence = ++mutationSequence.current;
    const operation = pending ?? {
      requestId: createRequestId(),
      editSequence: requestSequence.current,
      revision,
      snapshot,
      baseline,
      baselineSecrets,
      fields: candidate.fields,
      secrets: candidate.secrets,
      sections,
      overwriteConflicts,
      repairing,
    };
    retryOperationRef.current = operation;
    mutationInFlight.current = true;
    setSaving(true);
    setSaveFailed(false);
    setNotice(null);
    const hasUnsubmittedFieldErrors = Object.keys(fieldErrors).some((path) => (
      !sections.includes(path.split(".")[0] as ConfigSettingsSection)
    ));
    if (!overwriteConflicts && !hasUnsubmittedFieldErrors && conflictPathsRef.current.length === 0) {
      setSubmitError(null);
    }

    try {
      const options = {
        baseline: operation.baseline,
        baselineSecrets: operation.baselineSecrets,
        overwriteConflicts: operation.overwriteConflicts,
        editorId: editorIdRef.current,
        editSequence: operation.editSequence,
      };
      const next = operation.repairing
        ? await repairConfig(operation.revision, operation.fields, operation.secrets, operation.requestId, options)
        : await patchConfig(operation.revision, operation.fields, operation.secrets, operation.requestId, options);
      if (mutationSequence.current !== sequence) return;

      retryOperationRef.current = null;
      setResponse(next);
      draftRevisionRef.current = next.revision;
      const latestDraft = draftRef.current ?? snapshot;
      const savedDraft = formFromConfig(next.fields, snapshot, latestDraft);
      for (const section of ["runtime", "memory", "web", "models", "mcp"] as const) {
        if (!operation.sections.includes(section)) {
          Object.assign(savedDraft, { [section]: latestDraft[section] });
        }
      }
      const preservedDraft = preserveSettingsInput(snapshot, latestDraft, savedDraft) as SettingsForm;
      draftRef.current = preservedDraft;
      setDraft(preservedDraft);
      setApiKeyInputs((inputs) => {
        const nextInputs = { ...inputs };
        for (const [row, provider] of Object.entries(operation.snapshot.models.providers)) {
          const change = operation.secrets[`models.providers.${provider.id}.api_key`];
          if (change?.action === "replace" && nextInputs[row] === change.value) delete nextInputs[row];
        }
        return nextInputs;
      });
      baselineFieldsRef.current = next.fields;
      baselineSecretRevisionsRef.current = next.secret_revisions;
      const remainingCandidate = configFromForm(preservedDraft);
      dirtySectionsRef.current = new Set(settingsSectionsForChanges(
        next.fields,
        remainingCandidate.fields,
        remainingCandidate.secrets,
      ));
      const isSubmittedPath = (path: string) => operation.sections.includes(path.split(".")[0] as ConfigSettingsSection);
      setFieldErrors((current) => Object.fromEntries(Object.entries(current).filter(([path]) => !isSubmittedPath(path))));
      conflictPathsRef.current = conflictPathsRef.current.filter((path) => !isSubmittedPath(path));
      setConflictedPaths(conflictPathsRef.current);
      const hasRemainingConflicts = conflictPathsRef.current.length > 0;
      const hasRemainingFieldErrors = Object.keys(fieldErrors).some((path) => !isSubmittedPath(path));
      if (!hasRemainingConflicts && !hasRemainingFieldErrors) setSubmitError(null);
      setSaveFailed(hasRemainingConflicts);
      const hasRemainingChanges = dirtySectionsRef.current.size > 0;
      dirtyRef.current = hasRemainingChanges;
      setDirty(hasRemainingChanges);
      setNotice(next.application.status === "next-run-required" ? "settings.nextRunRequired" : "settings.saved");
    } catch (error) {
      if (mutationSequence.current !== sequence) return;
      if (error instanceof ApiError && error.body !== null) {
        retryOperationRef.current = null;
        const conflict = error.body.code === "config_revision_conflict";
        const serverPaths = Object.keys(error.body.field_errors);
        const errors = Object.fromEntries(serverPaths.map((path) => [
          path,
          conflict
            ? t("settings.conflictField")
            : t(path === "memory.schedule" ? "settings.invalidSchedule"
              : path === "runtime.compact_ratio" ? "settings.invalidRatio" : "settings.invalidValue"),
        ]));
        const isSubmittedPath = (path: string) => operation.sections.includes(path.split(".")[0] as ConfigSettingsSection);
        setFieldErrors((current) => ({
          ...Object.fromEntries(Object.entries(current).filter(([path]) => !isSubmittedPath(path))),
          ...errors,
        }));
        const retainedConflicts = conflictPathsRef.current.filter((path) => !isSubmittedPath(path));
        const paths = conflict
          ? [...new Set([...retainedConflicts, ...(serverPaths.length > 0 ? serverPaths : ["configuration"])])]
          : retainedConflicts;
        conflictPathsRef.current = paths;
        setConflictedPaths(paths);
        if (conflict) {
          try {
            setResponse(await getConfig());
          } catch {
            // Keep the draft and original conflict if the refresh is unavailable.
          }
        }
        setSaveFailed(true);
        setSubmitError(conflict
          ? t("settings.conflict")
          : error.body.code === "config_invalid" ? t("settings.validationSummary")
            : error.body.code === "persistence_error" ? t("settings.persistenceFailed") : error.body.message);
      } else {
        setSaveFailed(true);
        setSubmitError(error instanceof ApiError ? error.message : t("settings.unavailable"));
      }
      setNotice(null);
    } finally {
      if (mutationSequence.current === sequence) {
        mutationInFlight.current = false;
        setSaving(false);
      }
    }
  }, [authState, connectionState, fieldErrors, response?.configuration.repair_required, saveFailed, t, updateProvider]);

  const blurField = useCallback((path: string, value: string | boolean) => {
    void value;
    setFieldErrors((current) => {
      const next = { ...current };
      delete next[path];
      return next;
    });
    const section = path.split(".")[0] as ConfigSettingsSection;
    void saveDraft(false, false, section);
  }, [saveDraft]);

  const reloadSaved = async () => {
    if (saving || mutationInFlight.current || connectionState !== "online") return;
    const sequence = ++mutationSequence.current;
    setSaving(true);
    mutationInFlight.current = true;
    try {
      const next = await getConfig();
      if (mutationSequence.current !== sequence) return;
      retryOperationRef.current = null;
      dirtyRef.current = false;
      setDirty(false);
      dirtySectionsRef.current.clear();
      applyResponse(next);
      setApiKeyInputs({});
      setFieldErrors({});
      setSubmitError(null);
      setSaveFailed(false);
      setConflictedPaths([]);
      conflictPathsRef.current = [];
      setNotice(null);
    } catch (error) {
      if (mutationSequence.current === sequence) {
        setSaveFailed(true);
        setSubmitError(error instanceof ApiError ? error.message : t("settings.unavailable"));
      }
    } finally {
      if (mutationSequence.current === sequence) {
        mutationInFlight.current = false;
        setSaving(false);
      }
    }
  };

  const keepLocalChanges = () => {
    conflictPathsRef.current = [];
    setConflictedPaths([]);
    setFieldErrors({});
    setSubmitError(null);
    setSaveFailed(false);
    void saveDraft(true, true);
  };

  const retryPendingChanges = () => {
    setSaveFailed(false);
    setSubmitError(null);
    void saveDraft(conflictPathsRef.current.length > 0, true);
  };

  const errorEntries = Object.entries(fieldErrors);
  const controlDisabled = draft === null || loading || connectionState !== "online" || restartBusy;
  useEffect(() => {
    if (!restartBusy) return;
    const timer = window.setTimeout(() => {
      restartingInstanceRef.current = null;
      onRestartExpectation(false);
      setRestartBusy(false);
      setRestartError(t("settings.restartFailed"));
    }, 60000);
    return () => window.clearTimeout(timer);
  }, [restartBusy, onRestartExpectation, t]);
  useEffect(() => {
    if (restartingInstanceRef.current !== null && serviceStatus !== null
      && restartingInstanceRef.current !== serviceStatus.service_instance_id) {
      restartingInstanceRef.current = null;
      restartRequestRef.current = null;
      setRestartBusy(false);
      setRestartError(null);
      setNotice("settings.restarted");
    }
  }, [serviceStatus]);

  async function restartAide() {
    if (restartingInstanceRef.current !== null || dirtyRef.current || mutationInFlight.current
      || response === null || serviceStatus === null || connectionState !== "online") return;
    setRestartBusy(true);
    setRestartError(null);
    restartingInstanceRef.current = serviceStatus.service_instance_id;
    onRestartExpectation(true);
    restartRequestRef.current ??= createRequestId();
    try {
      await restartService(response.application.saved_revision, restartRequestRef.current);
      setRestartOpen(false);
    } catch (error) {
      if (restartingInstanceRef.current === null) return;
      if (!(error instanceof ApiError)) {
        setRestartOpen(false);
        return;
      }
      restartingInstanceRef.current = null;
      restartRequestRef.current = null;
      onRestartExpectation(false);
      setRestartBusy(false);
      setRestartError(error instanceof ApiError ? error.message : t("settings.restartFailed"));
    }
  }
  const captureSettingsBlur = (event?: React.FocusEvent<HTMLFormElement>) => {
    if (event?.target instanceof Element && event.target.closest("[data-api-key-editor]") !== null) return;
    if (event !== undefined && !(event.target instanceof HTMLInputElement
      || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement)) return;
    if (event?.target instanceof HTMLSelectElement
      && event.target.id.endsWith("-action") && event.target.value === "replace") {
      const value = document.getElementById(event.target.id.replace(/-action$/, "-value"));
      if (!(value instanceof HTMLInputElement) || value.value.length === 0) return;
    }
    void saveDraft(false, false, activeSection === "general" ? null : activeSection);
  };
  const captureSettingsChange = (event: React.FormEvent<HTMLFormElement>) => {
    const target = event.target;
    if (target instanceof HTMLSelectElement && target.id.endsWith("-action") && target.value === "replace") return;
    if (target instanceof HTMLSelectElement && target.id.endsWith("-transport") && target.value === "streamable-http") return;
    if (
      target instanceof HTMLSelectElement
      || (target instanceof HTMLInputElement && (target.type === "checkbox" || target.type === "radio"))
    ) window.setTimeout(() => void saveDraft(false, false, activeSection === "general" ? null : activeSection), 0);
  };
  const captureSettingsClick = (event: React.MouseEvent<HTMLFormElement>) => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest("button");
    if (button === null || button.closest("nav") !== null) return;
    if (button.closest("[data-api-key-editor]") !== null) return;
    const label = (button.getAttribute("aria-label") ?? button.textContent ?? "").trim();
    if (/^(?:add|添加)(?:\s|$)/i.test(label)) return;
    window.setTimeout(() => void saveDraft(false, false, activeSection === "general" ? null : activeSection), 0);
  };
  const labelFor = (path: string): string => {
    const labels: Record<string, string> = {
      "runtime.max_tool_result_chars": t("settings.maxToolResultChars"),
      "runtime.max_iterations": t("settings.maxIterations"),
      "runtime.enable_skill_always_load": t("settings.enableSkillAlwaysLoad"),
      "runtime.enable_tool_micro_compression": t("settings.enableToolMicroCompression"),
      "runtime.compact_ratio": t("settings.compactRatio"),
      "runtime.permission_level": t("settings.permissionLevel"),
      "runtime.exec_shell": t("settings.execShell"),
      "memory.batch_size": t("settings.batchSize"),
      "memory.schedule": t("settings.schedule"),
    };
    return labels[path] ?? path;
  };
  const fieldId = (path: string) => `settings-${path.replaceAll(".", "-")}`;
  const headerInputId = (server: string, header: string) => `settings-mcp-${server}-headers-${encodeURIComponent(header).replaceAll(".", "%2E")}`;
  const fieldError = (path: string) => fieldErrors[path];
  const groupError = (prefix: string) => Object.entries(fieldErrors).find(([path]) => path === prefix || path.startsWith(`${prefix}.`))?.[1];
  const focusError = (path: string, switchSection = true) => {
    const section = path.startsWith("web.")
      ? "general"
      : (["models", "runtime", "memory", "mcp"] as const).find((candidate) => path.startsWith(`${candidate}.`));
    if (switchSection && section !== undefined && section !== activeSection) {
      setActiveSection(section);
      window.setTimeout(() => focusError(path, false), 0);
      return;
    }
    let targetPath = path;
    for (const server of Object.values(draft?.mcp ?? {})) {
      const prefix = `mcp.${server.name}.headers.`;
      for (const [row, secret] of Object.entries(server.headers).sort((left, right) => right[1].name.length - left[1].name.length)) {
        const secretPath = `${prefix}${secret.name}`;
        const rowPath = `${prefix}${row}`;
        const matchedPath = path === secretPath || path.startsWith(`${secretPath}.`) ? secretPath
          : path === rowPath || path.startsWith(`${rowPath}.`) ? rowPath : null;
        if (matchedPath !== null) {
          const id = `${headerInputId(server.name, row)}${path.slice(matchedPath.length).replaceAll(".", "-")}`;
          const target = document.getElementById(id);
          (target?.matches("input, select") ? target : target?.querySelector<HTMLElement>("input, select") ?? target)?.focus();
          return;
        }
      }
      const keywordPrefix = `mcp.${server.name}.tool_keywords`;
      if (path.startsWith(`${keywordPrefix}.`)) {
        const index = server.tool_keywords.findIndex((tool) => path === `${keywordPrefix}.${tool.name}`);
        if (index >= 0) {
          document.getElementById(`${fieldId(keywordPrefix)}-${index}`)?.querySelector<HTMLTextAreaElement>("textarea")?.focus();
          return;
        }
      }
    }
    for (const provider of Object.values(draft?.models.providers ?? {})) {
      for (const [row, model] of Object.entries(provider.models)) {
        const prefixes = [row, model.id].filter(Boolean).map((name) => `models.providers.${provider.id}.models.${name}`);
        const prefix = prefixes.find((candidate) => path === candidate || path.startsWith(`${candidate}.`));
        if (prefix !== undefined) {
          const target = document.getElementById(modelSettingsFieldId(provider.id, row, path === prefix ? "id" : path.slice(prefix.length + 1)));
          if (target === null) continue;
          const details = target?.closest("details");
          if (details) details.open = true;
          target?.focus();
          return;
        }
      }
    }
    let target = document.getElementById(fieldId(targetPath));
    while (target === null && targetPath.includes(".")) {
      targetPath = targetPath.slice(0, targetPath.lastIndexOf("."));
      target = document.getElementById(fieldId(targetPath));
    }
    (target?.matches("input, select, textarea") ? target : target?.querySelector<HTMLElement>("input, select, textarea") ?? target)?.focus();
  };

  const settingsStatusState = saveFailed
    ? "error"
    : saving
      ? "saving"
      : dirty
        ? "unsaved"
        : response?.application.status;
  const settingsStatusLabel = saveFailed
    ? t("settings.saveFailed")
    : saving
      ? t("settings.saving")
      : dirty
        ? t("settings.unsaved")
        : response?.application.status === "next-run-required"
          ? t("settings.nextRunRequired")
          : response?.application.status === "pending-repair"
            ? t("settings.pendingRepair")
            : t("settings.active");

  const settingsSidebar = (
    <aside className={styles.settingsSidebar}>
      <button className={styles.settingsBack} type="button" onClick={onBack}>
        <ArrowLeft size={17} aria-hidden="true" />
        {t("settings.backToApp")}
      </button>
      <nav className={styles.settingsNavigation} aria-label={t("settings.sections")}>
        <span className={styles.settingsNavigationLabel}>{t("settings.title")}</span>
        {([
          ["general", "settings.generalAppearance", Settings2],
          ["models", "settings.models", Brain],
          ["runtime", "settings.runtime", Gauge],
          ["memory", "settings.memory", BookOpen],
          ["mcp", "settings.mcp", Activity],
        ] as const).map(([section, label, Icon]) => (
          <button
            className={styles.settingsNavigationItem}
            type="button"
            key={section}
            disabled={draft === null}
            aria-current={activeSection === section ? "page" : undefined}
            onClick={() => {
              setActiveSection(section);
              if (settingsContentRef.current !== null) settingsContentRef.current.scrollTop = 0;
            }}
          >
            <Icon size={17} aria-hidden="true" />
            {t(label)}
          </button>
        ))}
      </nav>
    </aside>
  );

  if (authState !== "ready") {
    return (
      <section className={styles.settingsPage} aria-labelledby="settings-title">
        {settingsSidebar}
        <div className={styles.settingsContent}>
          <div className={styles.settingsContentInner}>
        <div className={styles.pageHeading}>
          <div>
            <p className={styles.eyebrow}>{t("nav.settings")}</p>
            <h1 id="settings-title">{t("settings.title")}</h1>
          </div>
        </div>
        <div className={styles.errorBanner} role="status">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{t("settings.authenticationRequired")}</span>
        </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className={styles.settingsPage} aria-labelledby="settings-title">
      {settingsSidebar}
      <div className={styles.settingsContent} ref={settingsContentRef}>
        <div className={styles.settingsContentInner}>
      <div className={styles.settingsHeader}>
        <div className={styles.pageHeading}>
          <div>
            <h1 id="settings-title">{t("settings.title")}</h1>
          </div>
          {dirty && !saving && !saveFailed ? (
            <button className={styles.secondaryButton} type="button" onClick={() => void saveDraft(false, true)} disabled={controlDisabled}>
              {t("settings.saveChanges")}
            </button>
          ) : null}
          {response !== null ? (
            <div className={styles.settingsStatus} data-state={settingsStatusState} role="status" aria-live="polite">
              {saveFailed
                ? <CircleAlert size={15} aria-hidden="true" />
                : saving
                  ? <RefreshCw size={15} className={styles.spin} aria-hidden="true" />
                  : response.application.status === "active"
                    ? <CircleCheck size={15} aria-hidden="true" />
                    : <Clock3 size={15} aria-hidden="true" />}
              <span>{settingsStatusLabel}</span>
            </div>
          ) : null}
        </div>
      </div>

      {loadError !== null && response === null ? (
        <div className={styles.errorBanner} role="alert">
          <CircleAlert size={17} aria-hidden="true" />
          <span>{loadError}</span>
          <button className={styles.secondaryButton} type="button" onClick={() => void loadSettings()}>
            <RefreshCw size={14} aria-hidden="true" />
            {t("controls.retry")}
          </button>
        </div>
      ) : null}

      {response?.configuration.repair_required ? (
        <div className={styles.errorBanner} role="alert">
          <TriangleAlert size={17} aria-hidden="true" />
          <span>
            {response.configuration.state === "malformed"
              ? t("settings.malformedBackup")
              : t(response.application.active_revision === null ? "settings.repairRequired" : "settings.savedRepairRequired")}
            {response.configuration.requires_secret_reentry ? ` ${t("settings.secretReentry")}` : ""}
          </span>
        </div>
      ) : null}

      {submitError !== null ? (
        <div
          className={styles.errorSummary}
          ref={errorSummaryRef}
          tabIndex={-1}
          role="alert"
          aria-labelledby="settings-error-title"
        >
          <strong id="settings-error-title">{t("settings.errorSummary")}</strong>
          {submitError !== null ? <p>{submitError}</p> : null}
          {errorEntries.length > 0 ? (
            <ul>
              {errorEntries.map(([path, message]) => (
                <li key={path}>
                  <a href={`#${fieldId(path)}`} onClick={(event) => {
                    event.preventDefault();
                    focusError(path);
                  }}>{labelFor(path)}: {message}</a>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      {conflictedPaths.length > 0 ? (
        <div className={styles.settingsErrorActions}>
          <button className={styles.secondaryButton} type="button" onClick={keepLocalChanges} disabled={controlDisabled}>
            {t("settings.keepChanges")}
          </button>
          <button className={styles.secondaryButton} type="button" onClick={() => void reloadSaved()} disabled={controlDisabled}>
            <RotateCcw size={14} aria-hidden="true" />
            {t("settings.reload")}
          </button>
        </div>
      ) : saveFailed ? (
        <div className={styles.settingsErrorActions}>
          <button className={styles.secondaryButton} type="button" onClick={retryPendingChanges} disabled={controlDisabled}>
            <RefreshCw size={14} aria-hidden="true" />
            {t("settings.retrySave")}
          </button>
        </div>
      ) : null}

      <Dialog.Root open={microCompressionOpen} onOpenChange={setMicroCompressionOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content className={styles.dialogContent} onCloseAutoFocus={(event) => {
            event.preventDefault();
            microCompressionTriggerRef.current?.focus();
          }}>
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{t("settings.microCompressionTitle")}</Dialog.Title>
                <Dialog.Description className={styles.dialogDescription}>{t("settings.microCompressionImpact")}</Dialog.Description>
              </div>
              <Dialog.Close asChild>
                <button className={styles.iconButton} type="button" aria-label={t("controls.close")}>
                  <X size={17} aria-hidden="true" />
                </button>
              </Dialog.Close>
            </div>
            <div className={styles.actionRow}>
              <Dialog.Close asChild>
                <button className={styles.secondaryButton} type="button">{t("controls.cancel")}</button>
              </Dialog.Close>
              <button className={styles.primaryButton} type="button" disabled={controlDisabled}
                onClick={() => {
                  updateField("runtime.enable_tool_micro_compression", true);
                  setMicroCompressionOpen(false);
                  void saveDraft(false, false, "runtime");
                }}>{t("settings.confirmMicroCompression")}</button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root open={restartOpen} onOpenChange={(open) => { if (!restartBusy) setRestartOpen(open); }}>
        <Dialog.Portal>
          <Dialog.Overlay className={styles.dialogOverlay} />
          <Dialog.Content className={styles.dialogContent} onCloseAutoFocus={(event) => {
            event.preventDefault();
            restartTriggerRef.current?.focus();
          }}>
            <div className={styles.dialogHeader}>
              <div>
                <Dialog.Title className={styles.dialogTitle}>{t("settings.restartAide")}</Dialog.Title>
                <Dialog.Description className={styles.dialogDescription}>{t("settings.restartConfirm")}</Dialog.Description>
              </div>
            </div>
            {restartError !== null ? <p role="alert">{restartError}</p> : null}
            <div className={styles.actionRow}>
              <Dialog.Close asChild>
                <button className={styles.secondaryButton} type="button" disabled={restartBusy}>{t("controls.cancel")}</button>
              </Dialog.Close>
              <button className={styles.primaryButton} type="button" disabled={restartBusy || dirty || saving || connectionState !== "online"}
                onClick={() => void restartAide()}>{t(restartBusy ? "settings.restarting" : "settings.restartAide")}</button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      {draft !== null ? (
        <form
          className={styles.settingsForm}
          onSubmit={(event) => event.preventDefault()}
          onBlurCapture={captureSettingsBlur}
          onChangeCapture={captureSettingsChange}
          onClickCapture={captureSettingsClick}
          noValidate
        >
            <div className={styles.settingsDetail}>
              {activeSection === "general" ? (
                <div className={styles.settingsSection}>
                  <div className={styles.settingsSectionHeader}>
                    <div>
                      <p className={styles.eyebrow}>{t("settings.generalAppearance")}</p>
                      <h2>{t("settings.generalAppearance")}</h2>
                    </div>
                  </div>
                  <div className={styles.settingsFieldGrid}>
                    <label className={styles.settingsField} htmlFor="settings-theme">
                      <span className={styles.fieldLabel}>{t("controls.theme")}</span>
                      <select
                        className={styles.selectInput}
                        id="settings-theme"
                        value={theme}
                        onChange={(event) => onThemeChange(event.currentTarget.value as Theme)}
                      >
                        {(["system", "light", "dark"] as const).map((value) => (
                          <option key={value} value={value}>{t(`controls.${value}`)}</option>
                        ))}
                      </select>
                    </label>
                    <label className={styles.settingsField} htmlFor="settings-language">
                      <span className={styles.fieldLabel}>{t("controls.language")}</span>
                      <select
                        className={styles.selectInput}
                        id="settings-language"
                        value={language}
                        onChange={(event) => onLanguageChange(event.currentTarget.value as "en" | "zh-CN")}
                      >
                        <option value="en">English</option>
                        <option value="zh-CN">简体中文</option>
                      </select>
                    </label>
                    <label className={styles.settingsField} htmlFor={fieldId("web.default_chat_workspace")}>
                      <span className={styles.fieldLabel}>{t("settings.defaultChatWorkspace")}</span>
                      <input
                        className={styles.textInput}
                        id={fieldId("web.default_chat_workspace")}
                        value={draft?.web.default_chat_workspace ?? "~/.aide/chat"}
                        disabled={controlDisabled}
                        aria-invalid={fieldError("web.default_chat_workspace") !== undefined}
                        aria-describedby={fieldError("web.default_chat_workspace") !== undefined
                          ? `${fieldId("web.default_chat_workspace")}-error` : undefined}
                        onChange={(event) => updateField("web.default_chat_workspace", event.currentTarget.value)}
                        onBlur={() => blurField("web.default_chat_workspace", draft?.web.default_chat_workspace ?? "~/.aide/chat")}
                      />
                      <span className={styles.fieldError} id={`${fieldId("web.default_chat_workspace")}-error`}>
                        {fieldError("web.default_chat_workspace") ?? ""}
                      </span>
                    </label>
                  </div>
                </div>
              ) : null}

              {activeSection === "general" ? (
                <div className={styles.managementTool}>
                  <div className={styles.managementToolHeader}>
                    <div>
                      <h3>{t("settings.restartAide")}</h3>
                      <p className={styles.managementHint}>{t("settings.restartDescription")}</p>
                    </div>
                    <button className={styles.secondaryButton} type="button" ref={restartTriggerRef}
                      disabled={controlDisabled || dirty || saving || saveFailed
                        || response?.configuration.repair_required === true}
                      onClick={() => { setRestartError(null); setRestartOpen(true); }}>
                      <RefreshCw size={15} className={restartBusy ? styles.spin : undefined} aria-hidden="true" />
                      {t(restartBusy ? "settings.restarting" : "settings.restartAide")}
                    </button>
                  </div>
                  {restartError !== null ? <p className={styles.composerError} role="alert">{restartError}</p> : null}
                </div>
              ) : null}
              {activeSection === "runtime" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.runtime")}</p>
                <h2>{t("settings.runtime")}</h2>
              </div>
              <Settings2 size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsFieldGrid}>
              <SettingsNumberField
                id={fieldId("runtime.max_tool_result_chars")}
                label={t("settings.maxToolResultChars")}
                value={draft.runtime.max_tool_result_chars}
                error={fieldError("runtime.max_tool_result_chars")}
                disabled={controlDisabled}
                onChange={(value) => updateField("runtime.max_tool_result_chars", value)}
                onBlur={() => blurField("runtime.max_tool_result_chars", draft.runtime.max_tool_result_chars)}
              />
              <SettingsNumberField
                id={fieldId("runtime.max_iterations")}
                label={t("settings.maxIterations")}
                value={draft.runtime.max_iterations}
                error={fieldError("runtime.max_iterations")}
                disabled={controlDisabled}
                onChange={(value) => updateField("runtime.max_iterations", value)}
                onBlur={() => blurField("runtime.max_iterations", draft.runtime.max_iterations)}
              />
              <SettingsNumberField
                id={fieldId("runtime.compact_ratio")}
                label={t("settings.compactRatio")}
                value={draft.runtime.compact_ratio}
                error={fieldError("runtime.compact_ratio")}
                disabled={controlDisabled}
                step="0.01"
                onChange={(value) => updateField("runtime.compact_ratio", value)}
                onBlur={() => blurField("runtime.compact_ratio", draft.runtime.compact_ratio)}
              />
              <label className={styles.settingsField} htmlFor={fieldId("runtime.permission_level")}>
                <span className={styles.fieldLabel}>{t("settings.permissionLevel")}</span>
                <select
                  className={styles.selectInput}
                  id={fieldId("runtime.permission_level")}
                  value={draft.runtime.permission_level}
                  disabled={controlDisabled}
                  onChange={(event) => updateField("runtime.permission_level", event.currentTarget.value)}
                >
                  {(["read-only", "workspace-write", "full-access"] as ToolPermissionLevel[]).map((level) => (
                    <option key={level} value={level}>{t(`settings.permissionLevels.${level}`)}</option>
                  ))}
                </select>
              </label>
              <label className={styles.settingsField} htmlFor={fieldId("runtime.exec_shell")}>
                <span className={styles.fieldLabel}>{t("settings.execShell")}</span>
                <select
                  className={styles.selectInput}
                  id={fieldId("runtime.exec_shell")}
                  value={draft.runtime.exec_shell}
                  disabled={controlDisabled}
                  onChange={(event) => updateField("runtime.exec_shell", event.currentTarget.value)}
                >
                  {(["auto", "powershell", "pwsh"] as const).map((shell) => (
                    <option key={shell} value={shell}>{t(`settings.shells.${shell}`)}</option>
                  ))}
                </select>
              </label>
              <label className={styles.settingsToggle} htmlFor={fieldId("runtime.enable_skill_always_load")}>
                <input
                  id={fieldId("runtime.enable_skill_always_load")}
                  type="checkbox"
                  checked={draft.runtime.enable_skill_always_load}
                  disabled={controlDisabled}
                  onChange={(event) => updateField("runtime.enable_skill_always_load", event.currentTarget.checked)}
                />
                <span>
                  <strong>{t("settings.enableSkillAlwaysLoad")}</strong>
                  <small>{draft.runtime.enable_skill_always_load ? t("settings.yes") : t("settings.no")}</small>
                </span>
              </label>
              <label className={styles.settingsToggle} htmlFor={fieldId("runtime.enable_tool_micro_compression")}>
                <input
                  ref={microCompressionTriggerRef}
                  id={fieldId("runtime.enable_tool_micro_compression")}
                  type="checkbox"
                  checked={draft.runtime.enable_tool_micro_compression}
                  disabled={controlDisabled}
                  aria-describedby="settings-micro-compression-hint"
                  onChange={(event) => {
                    if (event.currentTarget.checked) setMicroCompressionOpen(true);
                    else updateField("runtime.enable_tool_micro_compression", false);
                  }}
                />
                <span>
                  <strong>{t("settings.enableToolMicroCompression")}</strong>
                  <small>{draft.runtime.enable_tool_micro_compression ? t("settings.yes") : t("settings.no")}</small>
                  <small id="settings-micro-compression-hint">{t("settings.microCompressionHint")}</small>
                </span>
              </label>
            </div>
            <dl className={styles.settingsServiceStatus} aria-label={t("status.title")}>
              <div>
                <dt>{t("status.title")}</dt>
                <dd>{serviceStatus === null ? t("status.checking") : serviceStateLabel(serviceStatus.state, t)}</dd>
              </div>
              <div><dt>{t("status.connection")}</dt><dd>{t(`status.${connectionState}`)}</dd></div>
              <div><dt>{t("status.workspaces")}</dt><dd>{serviceStatus?.active_workspace_count ?? "-"}</dd></div>
              <div><dt>{t("status.protocol")}</dt><dd>v{serviceStatus?.protocol_version ?? "-"}</dd></div>
            </dl>
            <Link className={styles.secondaryButton} to="/status">{t("nav.status")}</Link>
            <section className={styles.managementTool} aria-labelledby="settings-skills-title">
              <div className={styles.managementToolHeader}>
                <h3 id="settings-skills-title">{t("management.skillsTitle")}</h3>
                <button
                  className={styles.secondaryButton}
                  type="button"
                  disabled={runtimeClaim === null || connectionState !== "online" || skillReloadBusy}
                  onClick={() => void reloadSkills()}
                >
                  <RefreshCw size={15} aria-hidden="true" />
                  {skillReloadBusy ? t("management.skillsReloading") : t("management.reloadSkills")}
                </button>
              </div>
              {runtimeClaim === null ? (
                <p className={styles.managementHint}>{t("settings.skillReloadRequiresSession")}</p>
              ) : null}
              {skillReloadError !== null ? (
                <div className={styles.errorBanner} role="alert"><CircleAlert size={16} aria-hidden="true" />{t(skillReloadError)}</div>
              ) : null}
              {skillMetadata !== null ? (
                <div className={styles.managementOperationStatus} role="status" aria-live="polite">
                  <strong>{t("management.skillsReloaded", { count: skillMetadata.length })}</strong>
                  {skillMetadata.length > 0 ? (
                    <ul aria-label={t("management.skillsList")}>
                      {skillMetadata.map((skill) => (
                        <li key={skill.name}><strong>{skill.name}</strong><span>{skill.description}</span></li>
                      ))}
                    </ul>
                  ) : <span>{t("management.skillsEmpty")}</span>}
                </div>
              ) : null}
            </section>
          </div>

              : null}

              {activeSection === "memory" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.memory")}</p>
                <h2>{t("settings.memory")}</h2>
              </div>
              <Brain size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsFieldGrid}>
              <SettingsNumberField
                id={fieldId("memory.batch_size")}
                label={t("settings.batchSize")}
                value={draft.memory.batch_size}
                error={fieldError("memory.batch_size")}
                disabled={controlDisabled}
                onChange={(value) => updateField("memory.batch_size", value)}
                onBlur={() => blurField("memory.batch_size", draft.memory.batch_size)}
              />
              <label className={styles.settingsField} htmlFor={fieldId("memory.schedule")}>
                <span className={styles.fieldLabel} id={`${fieldId("memory.schedule")}-label`}>{t("settings.schedule")}</span>
                <input
                  className={styles.textInput}
                  id={fieldId("memory.schedule")}
                  aria-labelledby={`${fieldId("memory.schedule")}-label`}
                  value={draft.memory.schedule}
                  disabled={controlDisabled}
                  aria-invalid={fieldError("memory.schedule") !== undefined}
                  aria-describedby={fieldError("memory.schedule") !== undefined ? `${fieldId("memory.schedule")}-error` : undefined}
                  onChange={(event) => updateField("memory.schedule", event.currentTarget.value)}
                  onBlur={() => blurField("memory.schedule", draft.memory.schedule)}
                />
                <span className={styles.fieldError} id={`${fieldId("memory.schedule")}-error`}>{fieldError("memory.schedule") ?? ""}</span>
              </label>
            </div>
          </div>

              : null}

              {activeSection === "models" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.models")}</p>
                <h2>{t("settings.models")}</h2>
              </div>
              <Gauge size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsSubsection}>
              <div className={styles.settingsCollectionHeader}>
                <h3>{t("settings.providers")}</h3>
                <button className={styles.secondaryButton} type="button" onClick={addProvider} disabled={controlDisabled}>
                  <Plus size={14} aria-hidden="true" />{t("settings.addProvider")}
                </button>
              </div>
              <div className={styles.settingsCollection}>
                {Object.entries(draft.models.providers).map(([providerRow, provider]) => (
                  <div className={styles.settingsCollectionItem} key={providerRow} id={fieldId(`models.providers.${provider.id}`)} tabIndex={-1}>
                    <div className={styles.settingsCollectionItemHeader}>
                      <h4>{provider.id}</h4>
                      <button
                        className={styles.iconButton}
                        type="button"
                        aria-label={t("settings.removeProvider")}
                        title={t("settings.removeProvider")}
                        disabled={controlDisabled || Object.values(draft.models.routes).some((route) => route.provider_id === provider.id)}
                        onClick={() => removeProvider(providerRow)}
                      ><Trash2 size={15} aria-hidden="true" /></button>
                    </div>
                    <div className={styles.settingsFieldGrid}>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.providers.${providerRow}.id`)}>
                        <span className={styles.fieldLabel}>{t("settings.providerId")}</span>
                        <input
                          className={styles.textInput}
                          id={fieldId(`models.providers.${providerRow}.id`)}
                          value={provider.id}
                          readOnly={response?.fields.models.providers[providerRow] !== undefined}
                          disabled={controlDisabled}
                          onChange={(event) => updateProvider(providerRow, { id: event.currentTarget.value })}
                          aria-invalid={fieldError(`models.providers.${providerRow}.id`) !== undefined}
                        />
                        <span className={styles.fieldError}>{fieldError(`models.providers.${providerRow}.id`) ?? ""}</span>
                      </label>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.providers.${provider.id}.protocol`)}>
                        <span className={styles.fieldLabel}>{t("settings.protocol")}</span>
                        <select className={styles.selectInput} id={fieldId(`models.providers.${provider.id}.protocol`)} value={provider.protocol} disabled={controlDisabled} onChange={(event) => updateProvider(providerRow, { protocol: event.currentTarget.value })}>
                          {!['openai-compatible', 'anthropic'].includes(provider.protocol) ? <option value={provider.protocol}>{provider.protocol}</option> : null}
                          <option value="openai-compatible">openai-compatible</option><option value="anthropic">anthropic</option>
                        </select>
                      </label>
                      <label className={styles.settingsField} htmlFor={fieldId(`models.providers.${provider.id}.base_url`)}>
                        <span className={styles.fieldLabel}>{t("settings.baseUrl")}</span>
                        <input
                          className={styles.textInput}
                          id={fieldId(`models.providers.${provider.id}.base_url`)}
                          value={provider.base_url}
                          aria-invalid={fieldError(`models.providers.${provider.id}.base_url`) !== undefined}
                          aria-describedby={fieldError(`models.providers.${provider.id}.base_url`) !== undefined ? `${fieldId(`models.providers.${provider.id}.base_url`)}-error` : undefined}
                          disabled={controlDisabled}
                          onChange={(event) => updateProvider(providerRow, { base_url: event.currentTarget.value })}
                        />
                        <span className={styles.fieldError} id={`${fieldId(`models.providers.${provider.id}.base_url`)}-error`}>{fieldError(`models.providers.${provider.id}.base_url`) ?? ""}</span>
                      </label>
                      <ApiKeyInput
                        id={fieldId(`models.providers.${provider.id}.api_key`)}
                        configured={provider.api_key.configured}
                        value={apiKeyInputs[providerRow] ?? ""}
                        error={groupError(`models.providers.${provider.id}.api_key`)}
                        disabled={controlDisabled}
                        onChange={(value) => setApiKeyInputs((inputs) => ({ ...inputs, [providerRow]: value }))}
                        onSave={() => void saveDraft(false, false, "models", { providerRow, value: apiKeyInputs[providerRow] ?? "" })}
                      />
                    </div>
                    <div className={styles.settingsCollectionHeader}>
                      <h4>{t("settings.modelsList")}</h4>
                      <button className={styles.secondaryButton} type="button" aria-label={t("settings.addModel")} onClick={() => addModel(providerRow)} disabled={controlDisabled}>
                        <Plus size={14} aria-hidden="true" />{t("settings.addModel")}
                      </button>
                    </div>
                    <div className={styles.modelCards}>
                      {Object.entries(provider.models).map(([row, model]) => <ModelSettingsCard key={row} row={row} model={model} providerId={provider.id}
                        persisted={!response?.configuration.repair_required && model.saved} disabled={controlDisabled} errorFor={fieldError}
                        referencedBy={Object.values(draft.models.routes).filter((route) => route.provider_id === provider.id && route.model === model.id).map((route) => route.name)}
                        onChange={(change) => updateModel(providerRow, row, change)} onRemove={() => removeModel(providerRow, row)} />)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className={styles.settingsSubsection}>
              <div className={styles.settingsCollectionHeader} id="settings-models-routes" tabIndex={-1}>
                <h3>{t("settings.routes")}</h3>
              </div>
              <div className={styles.settingsCollection}>
                {Object.values(draft.models.routes).map((route) => {
                  const selectableProviders = Object.values(draft.models.providers).filter(isSelectableProvider);
                  const selectedProvider = selectableProviders.find((provider) => provider.id === route.provider_id);
                  const selectableModels = Object.values(selectedProvider?.models ?? {}).filter(isCompleteModelForm);
                  const unavailableProvider = route.provider_id !== "" && selectedProvider === undefined;
                  const unavailableModel = route.model !== "" && !selectableModels.some((model) => model.id === route.model);
                  return (
                  <div className={styles.modelRouteRow} key={route.name} id={fieldId(`models.routes.${route.name}`)} tabIndex={-1}>
                    <h4>{route.name === "chat" ? t("settings.chatRequired") : route.name}</h4>
                    <div className={styles.modelRouteField}>
                      <select
                        className={styles.selectInput}
                        id={fieldId(`models.routes.${route.name}.provider_id`)}
                        aria-label={t("settings.providerId")}
                        aria-describedby={fieldId(`models.routes.${route.name}.provider_id.error`)}
                        value={route.provider_id}
                        aria-invalid={unavailableProvider || fieldError(`models.routes.${route.name}.provider_id`) !== undefined}
                        disabled={controlDisabled}
                        onChange={(event) => updateRoute(route.name, { provider_id: event.currentTarget.value, model: "" })}
                      >
                        <option value="" disabled={route.name === "chat"}>{t(route.name === "chat" ? "settings.selectProvider" : "settings.unconfigured")}</option>
                        {unavailableProvider ? <option value={route.provider_id} disabled>{route.provider_id} · {t("settings.unavailable")}</option> : null}
                        {selectableProviders.map((provider) => <option key={provider.id} value={provider.id}>{provider.id}</option>)}
                      </select>
                      <span className={styles.fieldError} id={fieldId(`models.routes.${route.name}.provider_id.error`)}>{fieldError(`models.routes.${route.name}.provider_id`) ?? ""}</span>
                    </div>
                    <div className={styles.modelRouteField}>
                      <select
                        className={styles.selectInput}
                        id={fieldId(`models.routes.${route.name}.model`)}
                        aria-label={t("settings.model")}
                        aria-describedby={fieldId(`models.routes.${route.name}.model.error`)}
                        value={route.model}
                        aria-invalid={unavailableModel || fieldError(`models.routes.${route.name}.model`) !== undefined}
                        disabled={controlDisabled || selectedProvider === undefined}
                        onChange={(event) => updateRoute(route.name, { model: event.currentTarget.value })}
                      >
                        <option value="" disabled>{t("settings.selectModel")}</option>
                        {unavailableModel ? <option value={route.model} disabled>{route.model} · {t("settings.unavailable")}</option> : null}
                        {selectableModels.map((model) => <option key={model.id} value={model.id}>{model.id}</option>)}
                      </select>
                      <span className={styles.fieldError} id={fieldId(`models.routes.${route.name}.model.error`)}>{fieldError(`models.routes.${route.name}.model`) ?? ""}</span>
                    </div>
                  </div>
                ); })}
              </div>
            </div>
          </div>

              : null}

              {activeSection === "mcp" ? <div className={styles.settingsSection}>
            <div className={styles.settingsSectionHeader}>
              <div>
                <p className={styles.eyebrow}>{t("settings.mcp")}</p>
                <h2>{t("settings.mcp")}</h2>
              </div>
              <ShieldX size={20} aria-hidden="true" />
            </div>
            <div className={styles.settingsCollectionHeader}>
              <h3>{t("settings.mcpServers")}</h3>
              <button className={styles.secondaryButton} type="button" onClick={addMcp} disabled={controlDisabled}>
                <Plus size={14} aria-hidden="true" />{t("settings.addMcp")}
              </button>
            </div>
            <div className={styles.settingsCollection}>
              {Object.entries(draft.mcp).map(([serverRow, server]) => (
                <div className={styles.settingsCollectionItem} key={serverRow} id={fieldId(`mcp.${server.name}`)} tabIndex={-1}>
                  <div className={styles.settingsCollectionItemHeader}>
                    <h4>{server.name}</h4>
                    <button className={styles.iconButton} type="button" aria-label={t("settings.removeMcp")} title={t("settings.removeMcp")} disabled={controlDisabled} onClick={() => removeMcp(serverRow)}><Trash2 size={15} aria-hidden="true" /></button>
                  </div>
                  <div className={styles.settingsFieldGrid}>
                    <label className={styles.settingsField} htmlFor={fieldId(`mcp.${serverRow}.name`)}><span className={styles.fieldLabel}>{t("settings.serverName")}</span><input className={styles.textInput} id={fieldId(`mcp.${serverRow}.name`)} value={server.name} aria-invalid={fieldError(`mcp.${serverRow}.name`) !== undefined} readOnly={response?.fields.mcp[serverRow] !== undefined} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { name: event.currentTarget.value })} /></label>
                    <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.transport`)}>
                      <span className={styles.fieldLabel}>{t("settings.transport")}</span>
                      <select className={styles.selectInput} id={fieldId(`mcp.${server.name}.transport`)} value={server.transport} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { transport: event.currentTarget.value as McpForm["transport"] })}>
                        <option value="stdio">stdio</option>
                        <option value="streamable-http">streamable-http</option>
                      </select>
                    </label>
                    <label className={styles.settingsToggle} htmlFor={fieldId(`mcp.${server.name}.enabled`)}>
                      <input id={fieldId(`mcp.${server.name}.enabled`)} type="checkbox" checked={server.enabled} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { enabled: event.currentTarget.checked })} />
                      <span><strong>{t("settings.enabled")}</strong><small>{server.enabled ? t("settings.yes") : t("settings.no")}</small></span>
                    </label>
                    {server.transport === "stdio" ? (
                      <>
                        <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.command`)}><span className={styles.fieldLabel}>{t("settings.command")}</span><input className={styles.textInput} id={fieldId(`mcp.${server.name}.command`)} value={server.command} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { command: event.currentTarget.value })} /><span className={styles.fieldError}>{fieldError(`mcp.${server.name}.command`) ?? ""}</span></label>
                        <SettingsListField id={fieldId(`mcp.${server.name}.args`)} label={t("settings.args")} values={server.args} error={groupError(`mcp.${server.name}.args`)} disabled={controlDisabled} onChange={(args) => updateMcp(serverRow, { args })} />
                        <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.cwd`)}><span className={styles.fieldLabel}>{t("settings.cwd")}</span><input className={styles.textInput} id={fieldId(`mcp.${server.name}.cwd`)} value={server.cwd} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { cwd: event.currentTarget.value })} /></label>
                      </>
                    ) : (
                      <label className={styles.settingsField} htmlFor={fieldId(`mcp.${server.name}.url`)}><span className={styles.fieldLabel}>{t("settings.url")}</span><input className={styles.textInput} id={fieldId(`mcp.${server.name}.url`)} value={server.url} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { url: event.currentTarget.value })} onBlur={() => blurField(`mcp.${server.name}.url`, server.url)} /><span className={styles.fieldError}>{fieldError(`mcp.${server.name}.url`) ?? ""}</span></label>
                    )}
                    <SettingsNumberField id={fieldId(`mcp.${server.name}.connect_timeout`)} label={t("settings.connectTimeout")} value={server.connect_timeout} error={fieldError(`mcp.${server.name}.connect_timeout`)} disabled={controlDisabled} onChange={(value) => updateMcp(serverRow, { connect_timeout: value })} onBlur={() => blurField(`mcp.${server.name}.connect_timeout`, server.connect_timeout)} />
                    <SettingsNumberField id={fieldId(`mcp.${server.name}.call_timeout`)} label={t("settings.callTimeout")} value={server.call_timeout} error={fieldError(`mcp.${server.name}.call_timeout`)} disabled={controlDisabled} onChange={(value) => updateMcp(serverRow, { call_timeout: value })} onBlur={() => blurField(`mcp.${server.name}.call_timeout`, server.call_timeout)} />
                    <div className={styles.settingsField} id={fieldId(`mcp.${server.name}.tool_keywords`)} tabIndex={-1}>
                      <span className={styles.fieldLabel}>{t("settings.toolKeywords")}</span>
                      {server.tool_keywords.map((tool, index) => (
                        <div className={styles.settingsCollectionItem} key={tool.id}>
                          <label className={styles.settingsField}><span className={styles.fieldLabel}>{t("settings.toolName")}</span><input className={styles.textInput} value={tool.name} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { tool_keywords: server.tool_keywords.map((entry) => entry.id === tool.id ? { ...entry, name: event.currentTarget.value } : entry) })} /></label>
                          <SettingsListField id={`${fieldId(`mcp.${server.name}.tool_keywords`)}-${index}`} label={t("settings.toolKeywords")} values={tool.keywords} error={fieldError(`mcp.${server.name}.tool_keywords.${tool.name}`)} disabled={controlDisabled} onChange={(keywords) => updateMcp(serverRow, { tool_keywords: server.tool_keywords.map((entry) => entry.id === tool.id ? { ...entry, keywords } : entry) })} />
                          <button type="button" className={styles.secondaryButton} disabled={controlDisabled} onClick={() => updateMcp(serverRow, { tool_keywords: server.tool_keywords.filter((entry) => entry.id !== tool.id) })}>{t("settings.removeTool")}</button>
                        </div>
                      ))}
                      <span className={styles.fieldError}>{groupError(`mcp.${server.name}.tool_keywords`) ?? ""}</span>
                      <button type="button" className={styles.secondaryButton} disabled={controlDisabled} onClick={() => updateMcp(serverRow, { tool_keywords: [...server.tool_keywords, { id: createRequestId(), name: "", keywords: [] }] })}>{t("settings.addTool")}</button>
                    </div>
                  </div>
                  {server.transport === "streamable-http" ? (
                    <div className={styles.settingsSubsection} id={fieldId(`mcp.${server.name}.headers`)} tabIndex={-1}>
                      <span className={styles.fieldError}>{groupError(`mcp.${server.name}.headers`) ?? ""}</span>
                      <div className={styles.settingsCollectionHeader}><h4>{t("settings.headers")}</h4><button className={styles.secondaryButton} type="button" onClick={() => addHeader(serverRow)} disabled={controlDisabled}><Plus size={14} aria-hidden="true" />{t("settings.addHeader")}</button></div>
                      <div className={styles.settingsCollection}>
                        {Object.entries(server.headers).map(([header, secret]) => (
                          <div className={styles.settingsSecretItem} key={header}>
                            <label className={styles.settingsField}><span className={styles.fieldLabel}>{t("settings.headerName")}</span><input className={styles.textInput} value={secret.name} readOnly={response?.fields.mcp[serverRow]?.headers[header] !== undefined} disabled={controlDisabled} onChange={(event) => updateMcp(serverRow, { headers: { ...server.headers, [header]: { ...secret, name: event.currentTarget.value } } })} /></label>
                            <SecretInput id={headerInputId(server.name, header)} label={t("settings.headerValue")} secret={secret} error={groupError(`mcp.${server.name}.headers.${header}`) ?? groupError(`mcp.${server.name}.headers.${secret.name}`)} disabled={controlDisabled} onChange={(update) => updateMcp(serverRow, { headers: { ...server.headers, [header]: { ...secret, ...update } } })} />
                            <button className={styles.iconButton} type="button" aria-label={t("settings.removeHeader")} title={t("settings.removeHeader")} disabled={controlDisabled} onClick={() => removeHeader(serverRow, header)}><Trash2 size={15} aria-hidden="true" /></button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </div>

              : null}
            </div>
          {notice !== null ? (
            <div className={styles.notice} role="status" aria-live="polite">
              <CircleCheck size={16} aria-hidden="true" />
              <span>{t(notice)}</span>
            </div>
          ) : null}
        </form>
      ) : null}
        </div>
      </div>
    </section>
  );
}
