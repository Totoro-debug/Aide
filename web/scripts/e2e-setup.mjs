import { spawn } from "node:child_process";
import { resolve } from "node:path";

export default async function setup() {
  const child = spawn("python", ["-u", "-m", "web.scripts.e2e_service"], {
    cwd: resolve(process.cwd(), ".."),
    stdio: ["pipe", "pipe", "pipe"],
  });
  let errors = "";
  let stdoutBuffer = "";
  let stdoutClosed = false;
  const lineWaiters = [];

  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => { errors += chunk; });
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    stdoutBuffer += chunk;
    let newline;
    while ((newline = stdoutBuffer.indexOf("\n")) >= 0) {
      const line = stdoutBuffer.slice(0, newline).trim();
      stdoutBuffer = stdoutBuffer.slice(newline + 1);
      const waiter = lineWaiters.shift();
      if (waiter) waiter.resolve(line);
    }
  });
  child.once("error", (error) => {
    stdoutClosed = true;
    while (lineWaiters.length > 0) lineWaiters.shift().reject(error);
  });
  child.once("exit", (code) => {
    stdoutClosed = true;
    const error = new Error(`E2E service exited (${code}): ${errors}`);
    while (lineWaiters.length > 0) lineWaiters.shift().reject(error);
  });

  function readLine(timeoutMs = 30000) {
    if (stdoutClosed) return Promise.reject(new Error(`E2E service closed: ${errors}`));
    let waiter;
    const result = new Promise((resolveLine, rejectLine) => {
      const timeout = setTimeout(() => {
        const index = lineWaiters.findIndex((candidate) => candidate === waiter);
        if (index >= 0) lineWaiters.splice(index, 1);
        rejectLine(new Error(`E2E service response timed out: ${errors}`));
      }, timeoutMs);
      waiter = {
        resolve: (line) => {
          clearTimeout(timeout);
          resolveLine(line);
        },
        reject: (error) => {
          clearTimeout(timeout);
          rejectLine(error);
        },
      };
    });
    lineWaiters.push(waiter);
    return result;
  }

  function parseDetails(line) {
    let details;
    try {
      details = JSON.parse(line);
    } catch {
      throw new Error(`E2E service startup failed: ${errors || line}`);
    }
    if (
      typeof details.url !== "string" ||
      typeof details.home_root !== "string" ||
      typeof details.ticket !== "string" ||
      typeof details.cli_workspace !== "string" ||
      typeof details.first_project !== "string" ||
      typeof details.project_alias !== "string" ||
      typeof details.second_project !== "string" ||
      typeof details.second_ticket !== "string" ||
      typeof details.occupied_session_id !== "string" ||
      typeof details.available_session_id !== "string"
    ) {
      throw new Error("E2E service returned invalid startup details.");
    }
    return details;
  }

  function publish(details) {
    process.env.MYCLAW_E2E_URL = details.url;
    process.env.MYCLAW_E2E_HOME_ROOT = details.home_root;
    process.env.MYCLAW_E2E_TICKET = details.ticket;
    process.env.MYCLAW_E2E_CLI_WORKSPACE = details.cli_workspace;
    process.env.MYCLAW_E2E_FIRST_PROJECT = details.first_project;
    process.env.MYCLAW_E2E_PROJECT_ALIAS = details.project_alias;
    process.env.MYCLAW_E2E_SECOND_PROJECT = details.second_project;
    process.env.MYCLAW_E2E_SECOND_TICKET = details.second_ticket;
    process.env.MYCLAW_E2E_OCCUPIED_SESSION = details.occupied_session_id;
    process.env.MYCLAW_E2E_AVAILABLE_SESSION = details.available_session_id;
    return details;
  }

  let details;
  try {
    details = publish(parseDetails(await readLine()));
  } catch (error) {
    child.kill();
    throw error;
  }

  return {
    details,
    async restart() {
      child.stdin.write("restart\n");
      details = publish(parseDetails(await readLine()));
      return details;
    },
    async shutdown() {
      if (child.exitCode !== null) return;
      const exited = new Promise((resolveExit, rejectExit) => {
        const timeout = setTimeout(() => {
          child.kill();
          rejectExit(new Error(`E2E service shutdown timed out: ${errors}`));
        }, 15000);
        child.once("exit", (code) => {
          clearTimeout(timeout);
          if (code === 0) resolveExit();
          else rejectExit(new Error(`E2E service shutdown failed (${code}): ${errors}`));
        });
      });
      child.stdin.end("stop\n");
      await exited;
    },
  };
}
