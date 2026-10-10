import {
  ApiError
} from "../../shared/service/api";

export function managementErrorKey(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.body?.code) {
      case "stale_claim":
      case "session_claimed":
        return "sessions.staleClaimError";
      case "admission_closed":
        return "sessions.admissionClosedError";
      case "validation_error":
      case "config_invalid":
        return "management.invalidSelection";
      default:
        return "management.actionError";
    }
  }
  return "management.actionError";
}

export function operationManagementErrorKey(error: unknown, operationError: string): string {
  if (error instanceof ApiError && error.body?.code !== undefined) {
    const errorKey = managementErrorKey(error);
    return errorKey === "management.actionError" ? operationError : errorKey;
  }
  return managementErrorKey(error);
}
