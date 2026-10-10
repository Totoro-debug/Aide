import type { ReasoningEffort } from "../../protocol";

export interface ModelForm {
  id: string;
  saved: boolean;
  context_window: string;
  max_output: string;
  temperature: string;
  reasoning_effort: ReasoningEffort | "";
  timeout: string;
}

export function modelSettingsFieldId(provider: string, model: string, field = "") {
  const encode = (value: string) => encodeURIComponent(value).replaceAll(".", "%2E");
  return `settings-model-${encode(provider)}-${encode(model)}${field ? `-${field}` : ""}`;
}
