import {
  createRequestId
} from "../../shared/service/api";
import type {
  ConfigFields,
  ConfigPatchFields,
  ConfigSecretChange,
  ConfigSecrets,
  ToolPermissionLevel
} from "../../shared/service/protocol";
import type { ModelForm } from "./modelSettings";

export type SettingsSection = "general" | "models" | "runtime" | "memory" | "mcp";

export type ConfigSettingsSection = "models" | "runtime" | "memory" | "mcp" | "web";

export interface SettingsForm {
  runtime: {
    max_tool_result_chars: string;
    max_iterations: string;
    enable_skill_always_load: boolean;
    enable_tool_micro_compression: boolean;
    compact_ratio: string;
    permission_level: ToolPermissionLevel;
    exec_shell: "auto" | "powershell" | "pwsh";
  };
  memory: {
    batch_size: string;
    schedule: string;
  };
  web: {
    default_chat_workspace: string;
  };
  models: {
    providers: Record<string, ProviderForm>;
    routes: Record<string, RouteForm>;
  };
  mcp: Record<string, McpForm>;
}

export interface PendingSettingsSave {
  requestId: string;
  editSequence: number;
  revision: string;
  snapshot: SettingsForm;
  baseline: ConfigFields;
  baselineSecrets: Record<string, string | null>;
  fields: ConfigPatchFields;
  secrets: ConfigSecrets;
  sections: ConfigSettingsSection[];
  overwriteConflicts: boolean;
  repairing: boolean;
}

export type SecretAction = ConfigSecretChange["action"];

export interface SecretDraft {
  configured: boolean;
  action: SecretAction;
  value: string;
}

export interface ProviderForm {
  id: string;
  protocol: string;
  base_url: string;
  models: Record<string, ModelForm>;
  api_key: SecretDraft;
}

export interface RouteForm {
  name: string;
  provider_id: string;
  model: string;
}

export interface McpForm {
  name: string;
  enabled: boolean;
  transport: "stdio" | "streamable-http";
  command: string;
  args: string[];
  cwd: string;
  url: string;
  headers: Record<string, SecretDraft & { name: string }>;
  connect_timeout: string;
  call_timeout: string;
  tool_keywords: { id: string; name: string; keywords: string[] }[];
}

export function secretDraft(configured: boolean): SecretDraft {
  return { configured, action: "keep", value: "" };
}

function settingsRowKeys(names: string[], previous: [string, string][], reserved: string[]): Map<string, string> {
  const rows = new Map<string, string>();
  const used = new Set<string>();
  const unavailable = new Set([...previous.map(([row]) => row), ...reserved]);
  for (const name of names) {
    const match = previous.find(([row, value]) => value === name && !used.has(row));
    if (match === undefined) continue;
    rows.set(name, match[0]);
    used.add(match[0]);
  }
  for (const name of names) {
    if (rows.has(name)) continue;
    let row = name;
    while (used.has(row) || unavailable.has(row)) row = createRequestId();
    rows.set(name, row);
    used.add(row);
  }
  return rows;
}

export function formFromConfig(
  fields: ConfigFields,
  previous: SettingsForm | null = null,
  latest: SettingsForm | null = previous,
): SettingsForm {
  const providerRows = settingsRowKeys(
    Object.keys(fields.models.providers),
    Object.entries(previous?.models.providers ?? {}).map(([row, value]) => [row, value.id]),
    Object.keys(latest?.models.providers ?? {}),
  );
  const serverRows = settingsRowKeys(
    Object.keys(fields.mcp),
    Object.entries(previous?.mcp ?? {}).map(([row, value]) => [row, value.name]),
    Object.keys(latest?.mcp ?? {}),
  );
  return {
    runtime: {
      max_tool_result_chars: String(fields.runtime.max_tool_result_chars),
      max_iterations: String(fields.runtime.max_iterations),
      enable_skill_always_load: fields.runtime.enable_skill_always_load,
      enable_tool_micro_compression: fields.runtime.enable_tool_micro_compression ?? false,
      compact_ratio: String(fields.runtime.compact_ratio),
      permission_level: fields.runtime.permission_level,
      exec_shell: fields.runtime.exec_shell,
    },
    memory: {
      batch_size: String(fields.memory.batch_size),
      schedule: fields.memory.schedule,
    },
    web: {
      default_chat_workspace: fields.web.default_chat_workspace,
    },
    models: {
      providers: Object.fromEntries(Object.entries(fields.models.providers).map(([id, provider]) => [
        providerRows.get(id)!, {
        id,
        protocol: provider.protocol,
        base_url: provider.base_url,
        models: modelFormsFromConfig(provider.models, previous?.models.providers[providerRows.get(id)!]?.models, latest?.models.providers[providerRows.get(id)!]?.models),
        api_key: secretDraft(provider.api_key.configured),
      }])),
      routes: Object.fromEntries(Object.entries(fields.models.routes).map(([name, route]) => [name, {
        name,
        provider_id: route.provider_id,
        model: route.model,
      }])),
    },
    mcp: Object.fromEntries(Object.entries(fields.mcp).map(([name, server]) => {
      const row = serverRows.get(name)!;
      const previousServer = previous?.mcp[row];
      const headerRows = settingsRowKeys(
        Object.keys(server.headers),
        Object.entries(previousServer?.headers ?? {}).map(([key, value]) => [key, value.name]),
        Object.keys(latest?.mcp[row]?.headers ?? {}),
      );
      return [row, {
        name,
        enabled: server.enabled,
        transport: server.transport,
        command: server.command ?? "",
        args: server.args,
        cwd: server.cwd ?? "",
        url: server.url ?? "",
        headers: Object.fromEntries(Object.entries(server.headers).map(([header, value]) => {
          return [headerRows.get(header)!, { ...secretDraft(value.configured), name: header }];
        })),
        connect_timeout: String(server.connect_timeout),
        call_timeout: String(server.call_timeout),
        tool_keywords: Object.entries(server.tool_keywords).map(([toolName, keywords]) => ({
          id: previousServer?.tool_keywords.find((tool) => tool.name === toolName)?.id ?? createRequestId(),
          name: toolName,
          keywords,
        })),
      }];
    })),
  };
}

function modelFormsFromConfig(
  models: import("../../shared/service/protocol").ConfigProviderFields["models"],
  previous?: Record<string, ModelForm>,
  latest?: Record<string, ModelForm>,
): Record<string, ModelForm> {
  const rows = settingsRowKeys(Object.keys(models), Object.entries(previous ?? {}).map(([row, model]) => [row, model.id]), Object.keys(latest ?? {}));
  return Object.fromEntries(Object.entries(models).map(([id, model]) => [rows.get(id)!, {
    id,
    saved: true,
    context_window: model.context_window === null ? "" : String(model.context_window),
    max_output: model.max_output === null ? "" : String(model.max_output),
    temperature: model.temperature === null ? "" : String(model.temperature),
    reasoning_effort: model.reasoning_effort ?? "",
    timeout: model.timeout === null ? "" : String(model.timeout),
  }]));
}

export function isCompleteModelForm(model: ModelForm): boolean {
  const capacity = Number(model.context_window);
  const output = Number(model.max_output);
  const temperature = Number(model.temperature);
  const timeout = Number(model.timeout);
  return model.id.trim() !== "" && Number.isInteger(capacity) && capacity >= 1024 && capacity <= 10000000
    && Number.isInteger(output) && output >= 1 && output < capacity
    && model.temperature.trim() !== "" && Number.isFinite(temperature) && temperature >= 0 && temperature <= 2
    && model.reasoning_effort !== "" && Number.isInteger(timeout) && timeout >= 1 && timeout <= 600;
}

export function isSelectableProvider(provider: ProviderForm): boolean {
  let hasUrl = false;
  try {
    const url = new URL(provider.base_url);
    hasUrl = ["http:", "https:"].includes(url.protocol) && url.hostname !== "";
  } catch { /* An incomplete provider stays outside route choices. */ }
  const hasKey = provider.api_key.action === "replace" ? provider.api_key.value.trim() !== ""
    : provider.api_key.action === "keep" && provider.api_key.configured;
  return /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(provider.id) && ["anthropic", "openai-compatible"].includes(provider.protocol)
    && hasUrl && hasKey && Object.values(provider.models).some(isCompleteModelForm);
}

export function configFromForm(form: SettingsForm): { fields: ConfigPatchFields; secrets: ConfigSecrets } {
  const secrets: ConfigSecrets = {};
  const providers: NonNullable<NonNullable<ConfigPatchFields["models"]>["providers"]> = {};
  for (const [providerRow, provider] of Object.entries(form.models.providers)) {
    providers[providerRow] = {
      id: provider.id,
      protocol: provider.protocol,
      base_url: provider.base_url,
      models: Object.fromEntries(Object.values(provider.models).map((model) => [model.id, {
        context_window: model.context_window.trim() === "" ? null : Number(model.context_window),
        max_output: model.max_output.trim() === "" ? null : Number(model.max_output),
        temperature: model.temperature.trim() === "" ? null : Number(model.temperature),
        reasoning_effort: model.reasoning_effort || null,
        timeout: model.timeout.trim() === "" ? null : Number(model.timeout),
      }])),
    };
    secrets[`models.providers.${provider.id}.api_key`] = provider.api_key.action === "replace"
      ? { action: "replace", value: provider.api_key.value }
      : { action: provider.api_key.action };
  }
  const routes: Record<string, Record<string, string | number>> = {};
  for (const route of Object.values(form.models.routes)) {
    routes[route.name] = {
      provider_id: route.provider_id,
      model: route.model,
    };
  }
  const mcp: Record<string, Record<string, unknown>> = {};
  for (const [serverRow, server] of Object.entries(form.mcp)) {
    const headerRows: { name: string; secret: { configured: boolean } }[] = [];
    for (const draft of server.transport === "streamable-http" ? Object.values(server.headers) : []) {
      const header = draft.name;
      headerRows.push({ name: header, secret: { configured: draft.configured } });
      const path = `mcp.${server.name}.headers.${header}`;
      secrets[path] = draft.action === "replace"
        ? { action: "replace", value: draft.value }
        : { action: draft.action };
    }
    mcp[serverRow] = {
      name: server.name,
      enabled: server.enabled,
      transport: server.transport,
      command: server.transport === "stdio" ? server.command : null,
      args: server.transport === "stdio" ? server.args : [],
      cwd: server.transport === "stdio" && server.cwd.trim() ? server.cwd.trim() : null,
      url: server.transport === "streamable-http" ? server.url.trim() : null,
      header_rows: headerRows,
      connect_timeout: Number(server.connect_timeout),
      call_timeout: Number(server.call_timeout),
      tool_keyword_rows: server.tool_keywords.map((tool) => ({ name: tool.name, keywords: tool.keywords })),
    };
  }
  return {
    fields: {
    runtime: {
      max_tool_result_chars: Number(form.runtime.max_tool_result_chars),
      max_iterations: Number(form.runtime.max_iterations),
      enable_skill_always_load: form.runtime.enable_skill_always_load,
      enable_tool_micro_compression: form.runtime.enable_tool_micro_compression,
      compact_ratio: Number(form.runtime.compact_ratio),
      permission_level: form.runtime.permission_level,
      exec_shell: form.runtime.exec_shell,
    },
    memory: {
      batch_size: Number(form.memory.batch_size),
      schedule: form.memory.schedule.trim(),
    },
      web: {
        default_chat_workspace: form.web.default_chat_workspace.trim(),
      },
      models: { providers, routes },
      mcp,
    },
    secrets,
  };
}

export type SettingsFieldError = Record<string, string>;

export function sameSettingsValue(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (typeof left !== "object" || left === null || typeof right !== "object" || right === null) return false;
  return JSON.stringify(left) === JSON.stringify(right);
}

function isSettingsRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function changedSettingsPaths(
  baseline: unknown,
  candidate: unknown,
  path: string[] = [],
): string[][] {
  if (sameSettingsValue(baseline, candidate)) return [];
  if (baseline === undefined || candidate === undefined) return path.length === 0 ? [] : [path];
  if (!isSettingsRecord(baseline) || !isSettingsRecord(candidate)) return [path];
  return [...new Set([...Object.keys(baseline), ...Object.keys(candidate)])]
    .flatMap((key) => changedSettingsPaths(baseline[key], candidate[key], [...path, key]));
}

function settingsRowsForComparison<T extends { id?: string; name?: string }>(
  rows: [string, T][],
  nameField: "id" | "name",
) {
  return Object.fromEntries(rows.map(([row, value]) => {
    const fields = { ...value };
    const name = fields[nameField] ?? row;
    delete fields[nameField];
    return { name, fields };
  }).sort((left, right) => left.name.localeCompare(right.name))
    .map((entry, index) => [String(index), entry]));
}

export function changedConfigFieldPaths(baseline: ConfigFields, candidate: ConfigPatchFields): string[][] {
  const comparable = structuredClone(candidate);
  const comparableBaseline: ConfigPatchFields = structuredClone(baseline);
  for (const server of Object.values(comparableBaseline.mcp ?? {})) {
    server.header_rows = Object.entries(server.headers ?? {}).map(([name, secret]) => ({ name, secret }));
    server.tool_keyword_rows = Object.entries(server.tool_keywords ?? {}).map(([name, keywords]) => ({ name, keywords }));
    delete server.headers;
    delete server.tool_keywords;
  }
  const normalized = (fields: ConfigPatchFields) => ({
    ...fields,
    models: {
      ...fields.models,
      providers: settingsRowsForComparison(Object.entries(fields.models?.providers ?? {}), "id"),
    },
    mcp: settingsRowsForComparison(Object.entries(fields.mcp ?? {}).map(([row, server]) => [row, {
      ...server,
      header_rows: server.header_rows?.slice().sort((left, right) => left.name.localeCompare(right.name)),
      tool_keyword_rows: server.tool_keyword_rows?.slice().sort((left, right) => left.name.localeCompare(right.name)),
    }]), "name"),
  });
  return changedSettingsPaths(normalized(comparableBaseline), normalized(comparable)).filter((path) => (
    !(path[0] === "models" && path[1] === "providers" && (
      path[path.length - 1] === "api_key"
    ))
  ));
}

export function preserveSettingsInput(sent: unknown, latest: unknown, saved: unknown): unknown {
  if (sameSettingsValue(sent, latest)) return saved;
  if (!isSettingsRecord(sent) || !isSettingsRecord(latest) || !isSettingsRecord(saved)) return latest;
  const result = { ...saved };
  for (const key of new Set([...Object.keys(sent), ...Object.keys(latest)])) {
    if (sameSettingsValue(sent[key], latest[key])) continue;
    if (!(key in latest)) delete result[key];
    else result[key] = preserveSettingsInput(sent[key], latest[key], saved[key]);
  }
  return result;
}

export function settingsSectionsForChanges(
  baseline: ConfigFields,
  fields: ConfigPatchFields,
  secrets: ConfigSecrets,
): ConfigSettingsSection[] {
  const sections = new Set<ConfigSettingsSection>(
    changedConfigFieldPaths(baseline, fields).map(([section]) => section as ConfigSettingsSection),
  );
  for (const [path, change] of Object.entries(secrets)) {
    if (change.action !== "keep") sections.add(path.split(".")[0] as ConfigSettingsSection);
  }
  return [...sections];
}

export function configFieldsForSections(fields: ConfigPatchFields, sections: ConfigSettingsSection[]): ConfigPatchFields {
  const selected: ConfigPatchFields = {};
  if (sections.includes("runtime") && fields.runtime !== undefined) selected.runtime = fields.runtime;
  if (sections.includes("memory") && fields.memory !== undefined) selected.memory = fields.memory;
  if (sections.includes("web") && fields.web !== undefined) selected.web = fields.web;
  if (sections.includes("models") && fields.models !== undefined) selected.models = fields.models;
  if (sections.includes("mcp") && fields.mcp !== undefined) selected.mcp = fields.mcp;
  return selected;
}

export function configSecretsForSections(secrets: ConfigSecrets, sections: ConfigSettingsSection[]): ConfigSecrets {
  return Object.fromEntries(Object.entries(secrets).filter(([path]) => (
    sections.includes(path.split(".")[0] as ConfigSettingsSection)
  )));
}
