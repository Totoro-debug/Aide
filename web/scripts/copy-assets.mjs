import { cp, mkdir, mkdtemp, readdir, rename, rm, access } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";

const source = resolve(process.cwd(), "dist");
const target = resolve(process.cwd(), "../myclaw/web_assets");
const projectRoot = resolve(process.cwd(), "..");
const python = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
const validator = join(projectRoot, "scripts", "validate_web_assets.py");
const staging = await mkdtemp(join(dirname(target), ".web-assets-staging-"));
const previous = `${staging}-previous`;
let previousMoved = false;
let installed = false;

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

try {
  await mkdir(staging, { recursive: true });
  if (await exists(join(target, "__init__.py"))) {
    await cp(join(target, "__init__.py"), join(staging, "__init__.py"));
  }
  for (const name of await readdir(source)) {
    await cp(join(source, name), join(staging, name), { recursive: true });
  }

  const validation = spawnSync(
    python,
    [validator, "--root", staging, "--write-manifest"],
    { cwd: projectRoot, encoding: "utf8" },
  );
  if (validation.error) {
    throw validation.error;
  }
  if (validation.status !== 0) {
    const output = `${validation.stdout || ""}${validation.stderr || ""}`.trim();
    throw new Error(output || "Web asset validation failed");
  }

  if (await exists(target)) {
    await rename(target, previous);
    previousMoved = true;
  }
  await rename(staging, target);
  installed = true;
  if (previousMoved) {
    try {
      await rm(previous, { recursive: true, force: true });
    } catch (error) {
      console.warn(`Web assets published; previous backup cleanup failed at ${previous}: ${error.message}`);
    }
  }
  console.log(`Copied and verified Web assets to ${target}`);
} catch (error) {
  try {
    if (previousMoved && !(await exists(target)) && (await exists(previous))) {
      await rename(previous, target);
    }
  } catch (rollbackError) {
    throw new AggregateError([error, rollbackError], `Web asset rollback failed; backup: ${previous}`);
  }
  throw error;
} finally {
  if (!installed) {
    try {
      await rm(staging, { recursive: true, force: true });
    } catch (cleanupError) {
      console.error(`Web asset staging cleanup also failed at ${staging}: ${cleanupError.message}`);
    }
  }
}
