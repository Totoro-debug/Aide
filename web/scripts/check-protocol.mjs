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

function interfaceDeclaration(name) {
  const declaration = protocolSource.statements.find(
    (statement) => ts.isInterfaceDeclaration(statement) && statement.name.text === name,
  );
  if (!declaration) throw new Error(`The Web client is missing ${name}`);
  return declaration;
}

const referenceTypes = {
  identifier: "string",
  request_id: "string",
  tool_permission_level: "ToolPermissionLevel",
  reasoning_effort: "ReasoningEffort",
  runtime_status: "RuntimeStatus",
  management_error: "ManagementError",
  dream_result: "DreamResult",
  skill_metadata: "SkillMetadata",
  management_result: "ManagementResult",
  config_fields: "ConfigFields",
  config_models_fields: "ConfigModelsFields",
  config_mcp_fields: "ConfigMcpFields",
  config_provider_fields: "ConfigProviderFields",
  config_route_fields: "ConfigRouteFields",
  config_redacted_secret: "ConfigRedactedSecret",
  config_application: "ConfigApplication",
  config_application_status: "ConfigApplicationStatus",
};

function schemaType(definition) {
  if (definition.$ref) {
    const name = definition.$ref.split("/").at(-1);
    const type = referenceTypes[name];
    if (!type) throw new Error(`No Web type mapping for ${name}`);
    return type;
  }
  if (definition.anyOf) return definition.anyOf.map(schemaType).sort().join("|");
  if (definition.enum) return definition.enum.map((value) => JSON.stringify(value)).sort().join("|");
  if (Array.isArray(definition.type)) {
    return definition.type.map((type) => schemaType({ type })).sort().join("|");
  }
  if (definition.type === "integer") return "number";
  if (definition.type === "array" && definition.items) return `${schemaType(definition.items)}[]`;
  if (definition.type === "object" && definition.additionalProperties) {
    return `Record<string,${schemaType(definition.additionalProperties)}>`;
  }
  return definition.type;
}

function webType(node) {
  if (ts.isUnionTypeNode(node)) return node.types.map(webType).sort().join("|");
  return node.getText(protocolSource).replace(/\s+/g, "");
}

function checkMembers(name, members, definition, exact = true) {
  if (definition.$ref) {
    const referenced = definitions[definition.$ref.split("/").at(-1)];
    definition = { ...referenced, ...definition };
  }
  const expected = Object.entries(definition.properties);
  const actual = new Map(members.map((member) => [member.name?.getText(protocolSource), member]));
  if (exact && actual.size !== expected.length) {
    throw new Error(`${name} fields differ from the protocol schema`);
  }
  for (const [field, property] of expected) {
    const member = actual.get(field);
    if (!member?.type) throw new Error(`${name} is missing ${field}`);
    if (exact && Boolean(member.questionToken) === (definition.required ?? []).includes(field)) {
      throw new Error(`${name}.${field} optionality differs from the protocol schema`);
    }
    if (property.$ref && ts.isTypeLiteralNode(member.type)) {
      checkMembers(`${name}.${field}`, member.type.members, property);
    } else if (property.anyOf && ts.isUnionTypeNode(member.type)
      && property.anyOf.some((option) => option.type === "object" && option.properties)) {
      const objectType = member.type.types.find(ts.isTypeLiteralNode);
      if (!objectType || !member.type.types.some((node) => webType(node) === "null")) {
        throw new Error(`${name}.${field} must be a nullable object`);
      }
      checkMembers(`${name}.${field}`, objectType.members, property.anyOf.find((option) => option.type === "object"));
    } else if (property.type === "object" && property.properties) {
      if (!ts.isTypeLiteralNode(member.type)) throw new Error(`${name}.${field} must be an object`);
      checkMembers(`${name}.${field}`, member.type.members, property);
    } else if (webType(member.type) !== schemaType(property)) {
      throw new Error(`${name}.${field} type differs from the protocol schema`);
    }
  }
}

for (const [name, definition] of [
  ["ToolPermissionLevel", "tool_permission_level"],
  ["ReasoningEffort", "reasoning_effort"],
  ["ConfigApplicationStatus", "config_application_status"],
]) {
  const declaration = protocolSource.statements.find(
    (statement) => ts.isTypeAliasDeclaration(statement) && statement.name.text === name,
  );
  if (!declaration || webType(declaration.type) !== schemaType(definitions[definition])) {
    throw new Error(`${name} values differ from the protocol schema`);
  }
}
for (const [name, definitionName] of [
  ["ManagementError", "management_error"],
  ["DreamResult", "dream_result"],
  ["SkillMetadata", "skill_metadata"],
]) {
  checkMembers(name, interfaceDeclaration(name).members, definitions[definitionName]);
}
checkMembers("RuntimeStatus", interfaceDeclaration("RuntimeStatus").members, definitions.runtime_status);
checkMembers("ManagementResponse", interfaceDeclaration("ManagementResponse").members, definitions.management_response);
checkMembers("ConfigFields", interfaceDeclaration("ConfigFields").members, definitions.config_fields);
checkMembers("ConfigApplication", interfaceDeclaration("ConfigApplication").members, definitions.config_application);
checkMembers("ConfigResponse", interfaceDeclaration("ConfigResponse").members, definitions.config_response);
const configPatch = interfaceDeclaration("ConfigPatchResponse");
if (configPatch.heritageClauses?.[0]?.types?.[0]?.expression.getText(protocolSource) !== "ConfigResponse") {
  throw new Error("ConfigPatchResponse must extend ConfigResponse");
}
checkMembers("ConfigPatchResponse", [
  ...interfaceDeclaration("ConfigResponse").members, ...configPatch.members,
], definitions.config_mutation_response);
const managementFields = [
  "handled", "output", "status_view", "effort_selection", "permission_selection",
  "published_effort", "published_permission_level", "memory_content", "dream_result",
  "management_error", "skill_metadata",
];
checkMembers("ManagementResult", interfaceDeclaration("ManagementResult").members, {
  properties: Object.fromEntries(managementFields.map((field) => [field, definitions.management_result.properties[field]])),
}, false);
console.log("Protocol schema compatibility: passed");
