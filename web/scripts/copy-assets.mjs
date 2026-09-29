import { cp, mkdir, readdir, rm } from "node:fs/promises";
import { join, resolve } from "node:path";

const source = resolve(process.cwd(), "dist");
const target = resolve(process.cwd(), "../myclaw/web_assets");
await mkdir(target, { recursive: true });
for (const name of await readdir(target)) {
  if (name !== "__init__.py") {
    await rm(join(target, name), { recursive: true, force: true });
  }
}
for (const name of await readdir(source)) {
  await cp(join(source, name), join(target, name), { recursive: true });
}
console.log(`Copied Web assets to ${target}`);
