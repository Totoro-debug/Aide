import type {
  ServiceState
} from "../../protocol";

export function serviceStateLabel(state: ServiceState, translate: (key: string) => string): string {
  return translate(`status.${state}`);
}
