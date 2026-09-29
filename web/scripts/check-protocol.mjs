import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import ts from "typescript";

const schemaPath = resolve(process.cwd(), "../myclaw/service/protocol/v1.schema.json");
const schema = JSON.parse(await readFile(schemaPath, "utf8"));
const definitions = schema.$defs;
const protocolSource = ts.createSourceFile(
  "protocol.ts",
  await readFile(resolve(process.cwd(), "src/protocol.ts"), "utf8"),
  ts.ScriptTarget.Latest,
  true,
);
const eventInterface = protocolSource.statements.find(
  (statement) => ts.isInterfaceDeclaration(statement) && statement.name.text === "ServiceEvent",
);
if (!eventInterface) throw new Error("The Web client is missing the ServiceEvent contract");
const eventFields = new Set(eventInterface.members.map((member) => member.name?.getText(protocolSource)));
const schemaEventFields = new Set(definitions.event.required);
if (
  eventFields.size !== schemaEventFields.size ||
  [...schemaEventFields].some((field) => !eventFields.has(field))
) {
  throw new Error("The Web event fields differ from the protocol schema");
}

if (schema.$schema !== "https://json-schema.org/draft/2020-12/schema") {
  throw new Error("Unsupported protocol schema draft");
}
if (definitions.event.properties.protocol_version.const !== 1) {
  throw new Error("The event protocol version changed");
}
const requiredEventFields = new Set(definitions.event.required);
for (const field of [
  "protocol_version",
  "service_instance_id",
  "stream_id",
  "seq",
  "type",
  "workspace_id",
  "project_id",
  "session_id",
  "run_id",
  "payload",
]) {
  if (!requiredEventFields.has(field)) {
    throw new Error(`The event contract is missing ${field}`);
  }
}
const commandTypes = new Set(definitions.client_command.properties.type.enum);
for (const command of ["claim", "release", "input", "cancel", "confirmation_decide", "subscribe"]) {
  if (!commandTypes.has(command)) {
    throw new Error(`The client command contract is missing ${command}`);
  }
}
if ("value" in definitions.redacted_secret.properties) {
  throw new Error("Redacted secrets must remain write-only");
}
console.log("Protocol schema compatibility: passed");
