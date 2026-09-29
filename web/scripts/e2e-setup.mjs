import { spawn } from "node:child_process";
import { resolve } from "node:path";

export default async function setup() {
  const child = spawn("python", ["-u", "-m", "web.scripts.e2e_service"], {
    cwd: resolve(process.cwd(), ".."),
    stdio: ["pipe", "pipe", "pipe"],
  });
  let errors = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => { errors += chunk; });

  let details;
  try {
    details = await new Promise((resolveDetails, reject) => {
      let output = "";
      const timeout = setTimeout(() => reject(new Error(`E2E service startup timed out: ${errors}`)), 30000);
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk) => {
        output += chunk;
        const line = output.split("\n", 1)[0];
        if (!output.includes("\n")) return;
        clearTimeout(timeout);
        try {
          resolveDetails(JSON.parse(line));
        } catch {
          reject(new Error(`E2E service startup failed: ${errors || line}`));
        }
      });
      child.once("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.once("exit", (code) => {
        clearTimeout(timeout);
        reject(new Error(`E2E service exited (${code}): ${errors}`));
      });
    });
  } catch (error) {
    child.kill();
    throw error;
  }

  if (
    typeof details.url !== "string" ||
    typeof details.home_root !== "string" ||
    typeof details.ticket !== "string"
  ) {
    child.kill();
    throw new Error("E2E service returned invalid startup details.");
  }
  process.env.MYCLAW_E2E_URL = details.url;
  process.env.MYCLAW_E2E_HOME_ROOT = details.home_root;
  process.env.MYCLAW_E2E_TICKET = details.ticket;
  return async () => {
    const code = await new Promise((resolveExit, reject) => {
      const timeout = setTimeout(() => {
        child.kill();
        reject(new Error(`E2E service shutdown timed out: ${errors}`));
      }, 15000);
      child.once("exit", (exitCode) => {
        clearTimeout(timeout);
        resolveExit(exitCode);
      });
      child.stdin.end("\n");
    });
    if (code !== 0) throw new Error(`E2E service shutdown failed (${code}): ${errors}`);
  };
}
