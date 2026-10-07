[runtime]
max_tool_result_chars = 4096
max_iterations = 50
enable_skill_always_load = false
enable_tool_micro_compression = false
compact_ratio = 0.9
permission_level = "workspace-write"
exec_shell = "auto"

[memory]
batch_size = 10
schedule = "0 * * * *"

[web]
default_chat_workspace = "~/.aide/chat"

# Configure a trusted local MCP Server by uncommenting and editing one item.
# [mcp.servers.filesystem]
# enabled = true
# transport = "stdio"
# command = "uvx"
# args = ["mcp-server-filesystem", "."]
# cwd = "."
# connect_timeout = 30
# call_timeout = 60
# [mcp.servers.filesystem.tool_keywords]
# list_files = ["files", "directory", "list"]

# Configure a trusted Streamable HTTP MCP Server with optional static headers.
# [mcp.servers.search]
# enabled = true
# transport = "streamable-http"
# url = "https://example.com/mcp"
# connect_timeout = 30
# call_timeout = 60
# [mcp.servers.search.tool_keywords]
# search = ["search", "query"]
# [mcp.servers.search.headers]
# Authorization = "Bearer replace-with-a-token"

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = ""
api_key = ""
# Configure each model once; routes below select its Provider and model ID.
[models.providers.openai-local.models."replace-with-a-model-id"]
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "mid"
timeout = 120

# Remove any purpose-specific route to fall back to default.
[models.routes.default]
provider_id = "openai-local"
model = "replace-with-a-model-id"

[models.routes.chat]
provider_id = "openai-local"
model = "replace-with-a-model-id"

[models.routes.memory]
provider_id = "openai-local"
model = "replace-with-a-model-id"

[models.routes.schedule]
provider_id = "openai-local"
model = "replace-with-a-model-id"
