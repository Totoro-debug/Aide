export async function traceWorkspaceKeyForPath(workspacePath: string): Promise<string> {
  const windowsPath = workspacePath.replaceAll("/", "\\");
  const driveRoot = /^([A-Za-z]:)\\/.exec(windowsPath);
  const uncRoot = /^\\\\([^\\]+)\\([^\\]+)(?:\\|$)/.exec(windowsPath);
  let identity: string;
  if (driveRoot !== null) {
    const rest = windowsPath.slice(driveRoot[0].length);
    identity = `${driveRoot[1]}\\${normalizePathSegments(rest, "\\").join("\\")}`.toLowerCase();
  } else if (uncRoot !== null) {
    const root = `\\\\${uncRoot[1]}\\${uncRoot[2]}`;
    const rest = windowsPath.slice(uncRoot[0].length);
    const segments = normalizePathSegments(rest, "\\");
    identity = `${root}\\${segments.join("\\")}`.toLowerCase();
  } else if (workspacePath.startsWith("/")) {
    identity = `/${normalizePathSegments(workspacePath, "/").join("/")}`;
  } else {
    throw new Error("Workspace path must be absolute.");
  }
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(identity));
  const hash = Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
  return `sha256:${hash}`;
}

function normalizePathSegments(path: string, separator: string): string[] {
  const segments: string[] = [];
  for (const segment of path.split(separator)) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") {
      segments.pop();
    } else {
      segments.push(segment);
    }
  }
  return segments;
}

export function traceSessionRoute(workspaceKey: string, sessionId: string): string {
  return `/traces/sessions/${encodeURIComponent(workspaceKey)}/${encodeURIComponent(sessionId)}`;
}
