# Omni

## 项目简介

Omni 是面向单用户、本地优先的个人 Agent 运行时。通过全屏终端或本地 Web 界面调用模型与工具，支持并行会话、会话恢复、三层记忆、Skill、MCP 和定时任务。CLI 与 Web 共用一个按需启动的本地服务，运行状态以文件形式保存在本地。

## 项目安装

仅支持 Windows，需要 Python 3.12+ 和 Git。`auto` 选择可用的 PowerShell 7，否则使用 Windows PowerShell 5.1。

目前已验证 Windows x64。应用、服务和发布验证入口在初始化前拒绝其他操作系统。

```powershell
git clone https://github.com/Totoro-debug/OmniAgent.git omni
cd omni
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

## Web 界面与服务生命周期

安装 wheel 或 source distribution 后，运行时只需要 Python 及其依赖，不需要 Node.js、npm 或前端源码。直接执行 `omni web`；首次配置或配置需要修复时，会打开对应的设置页面。已有有效配置时，也可运行裸 `omni` 进入 CLI 对话。

```powershell
omni web
```

`omni web` 会启动或复用当前 Agent Home 的本地服务，并打开一次性的浏览器登录地址。服务只监听 `127.0.0.1:8765`，浏览器和 CLI 连接使用同一个服务与生命周期；关闭浏览器后服务会保留短暂的重连窗口。需要立即收尾时执行：

```powershell
omni service stop
```

刷新页面或在 30 秒宽限期内重连会恢复已接受的输入、进行中的输出、工具状态和待处理确认；当前运行仍可取消，确认使用原请求。未登记为 Project 的 CLI Workspace 仅在有在线使用者时接收新的 Schedule 执行；最后一个使用者断线后立即暂停新执行，宽限期到期后收尾并释放运行时，保留 Job 和用户文件。已登记且可用的 Project 在服务仍有在线客户端时继续调度。

Web 中可登记已有目录为 Project，按标题搜索、改名或删除持久化 Session，并查看独立的 Schedule 历史。同一 Workspace 的不同 Session 可以并行运行；同一 Session 只能由一个客户端加载。切换页面后已接受的运行继续执行，空白且未提交的 Session 不保存。移除 Project 会停止其工作并移除登记，目录与运行状态保留；重新登记含用户任务的目录时需显式恢复调度。

停止命令请求服务正常排空活动运行、关闭连接并清理发现文件；它不会删除 Workspace、会话或用户配置。源码开发者只有在重新生成发布资源时才需要 Node.js/npm：

```powershell
npm --prefix web ci
npm --prefix web run build
python -m build --sdist --wheel
```

构建会校验入口 HTML、引用资源、SHA-256 清单和分发包资源；资源缺失或清单不一致时直接失败，不会生成只有后端的安装包。

发布者可在仓库根目录执行 `python -m scripts.installed_web_validation --output D:/omni-installed-check`（使用新的源码外目录）。该入口以正常隔离构建生成 wheel/sdist，独立重建并安装两种 wheel，验收生产 Web 页面、深层路由、对话、Settings、同服务 CLI 连接和停止，保存报告及截图。浏览器控制器需要 `web/` 的 Playwright/Node 开发依赖；安装后的应用使用无 Node/npm 的独立环境。默认端口被占用时检查会失败，不停止已有服务。

## Windows 发布验证

发布验证仅在 GitHub Actions 的 Windows runner 上运行 `--phase all`。Windows 必须真实执行 Windows PowerShell 5.1 和 PowerShell 7 的 Host Exec 检查、执行、Full-Access 动态命令及文件系统能力检查，并运行完整测试、lint、类型检查、构建，在隔离环境中安装 wheel，从源码树外执行 entry point 与配置 smoke。`release-gate` 仅依赖 Windows 作业成功；主机缺失或必需测试跳过均不算通过。

PowerShell 选择器配置位于 `[runtime].exec_shell`；`auto` 优先选择 PowerShell 7，随后选择 Windows PowerShell 5.1，显式选择不会交叉回退。验证脚本只在 Windows 运行；默认 `--shell both` 覆盖两个 PowerShell 版本，也可用 `--shell powershell|pwsh` 单独检查对应 Host。GitHub Actions 仅验证 Windows，可通过推送到 `main`、面向 `main` 的 PR 或手动触发工作流：

```powershell
python scripts/release_validation.py --phase all
```

`Full-Access` 只移除普通权限提示，不提供 OS sandbox；参数校验、能力错误、业务拒绝、执行错误、灾难性 Exec 和不确定检查仍然有效。Windows 发布门禁会输出 skip 分类、Windows junction/reparse 证据、PowerShell 主机结果及量化覆盖计数。发布验收以实际 Windows 作业和安装包验证结果为准；Issue 关闭不代表完整发布门禁已经通过。

## 项目最小配置

执行 `omni config` 生成默认配置，再将 `~/.omni/config.toml` 的内容替换为以下配置。已有配置不会被该命令覆盖。`~` 表示当前用户主目录，Windows 下通常为 `C:\Users\<用户名>`。

选择支持工具调用的模型，替换服务地址、API Key 和两处模型 ID。按模型实际限制设置 `context_window` 与 `max_output`，单位均为 token，后者必须小于前者；`timeout` 单位为秒。

```toml
[models.providers.my-provider]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "replace-with-your-api-key"
models = ["your-model-id"]

[models.routes.default]
provider_id = "my-provider"
model = "your-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
timeout = 120
```

`protocol` 支持 `openai-compatible` 和 `anthropic`。只配置 `default` 路由即可供对话、记忆和定时任务使用，其余配置采用默认值。

API Key 直接保存在配置文件中，当前不支持环境变量引用；`omni config` 显示时会脱敏。更多可选配置及 MCP 示例见[配置模板](omni/templates/default-config.md)。

MCP Server 可在 `[mcp.servers.<name>.tool_keywords]` 下按远端 Tool 原名配置英文关键词；缺失或为空的关键词会在启动时通过现有 `chat` Model Route 生成并尽力保存。生成失败时，当前进程使用对应的远端 Tool 原名；生成成功但保存失败时，当前进程继续使用已生成的内存关键词。两类失败都不会阻止 Agent 启动。

`[runtime].permission_level` 接受 `read-only`、`workspace-write` 和 `full-access`，默认 `workspace-write`。`/permission` 或 Web 中的权限选择只影响当前客户端后续的前台运行；切换 Session、Workspace 或在宽限期内重连会保留选择，新客户端从生效配置开始。选择 `full-access` 前需通过默认聚焦 Cancel 的警告。

| 权限级别 | Workspace 内文件 | Workspace 外文件 | MCP 调用 |
| --- | --- | --- | --- |
| `read-only` | 读取直接执行，写入逐次确认 | 逐次确认 | 逐次确认 |
| `workspace-write` | 读取、写入直接执行 | 逐次确认 | 逐次确认 |
| `full-access` | 直接执行 | 直接执行 | 直接执行 |

每次 Agent Run 捕获固定权限快照，批准只作用于一次工具调用。Exec 在低权限级别只直接执行满足固定命令、参数、身份和路径规则的调用；灾难性操作或检查不确定时，所有级别都需确认。Web Search 始终直接执行，Web Fetch 在低权限级别访问非公网或无法确定的地址时需确认。Full-Access 不提供 OS sandbox，也不绕过参数校验、能力错误、业务拒绝或执行错误。详细规则见 [ADR-0026](docs/adr/0026-tool-permission-levels-and-foreground-snapshots.md)。

`[runtime].exec_shell` 接受 `auto`、`powershell` 和 `pwsh`，默认 `auto`。`auto` 优先选择 PowerShell 7，否则使用 Windows PowerShell 5.1；显式选择不会交叉回退。检查和执行均禁用 Profile。所选 Shell 缺失时显示安全诊断，Exec 调用返回能力错误。用户定时任务使用生效 Workspace generation 的配置权限和 Shell，不继承客户端的临时权限选择。

可选字段缺失时使用默认值，显式非法值回退到默认值并产生脱敏诊断；未知字段被忽略，加载不会自动改写原始 TOML。`compact_ratio` 默认 `0.9`，有效范围为 `0.5` 至 `0.95`。`omni config` 与 `/config` 显示有效值、诊断和脱敏配置；完整默认值合同见 [ADR-0025](docs/adr/0025-ignore-unknown-user-configuration-fields.md)。

Web Settings 提供 Runtime、Memory、模型、路由和 MCP 的结构化编辑。已有 API Key 和 MCP 凭据不会回传浏览器，修改时需明确替换或清空。无效或版本冲突的保存不会覆盖原文件；有效保存等待当前工作完成后生效，失败时可重试。缺失或损坏配置可在设置页面初始化或修复。

## 项目启动

在已激活虚拟环境的交互式终端中，进入希望 Agent 操作的目录后启动（将示例路径替换为实际路径）：

```powershell
Set-Location "D:\path\to\workspace"
omni
```

新开终端后需重新激活安装目录中的虚拟环境，或使用其中 `omni` 可执行文件的绝对路径。启动需要交互式输入、输出，不能通过管道运行。

启动目录即 CLI Workspace，不会自动切换到 Git 根目录。运行状态保存在该目录的 `.omni/` 中；定时任务的运行条件见前面的“Web 界面与服务生命周期”。

从使用旧名称的版本升级时，先用该版本的服务停止命令退出服务和客户端，再安装新版，并使用 `omni` 命令启动。全局配置目录和各 Workspace 的运行目录统一使用 `.omni/`；若需要迁移旧目录，应整体重命名并保留所有文件，目标已存在时先核对数据，避免覆盖。浏览器需通过 `omni web` 重新登录并重新选择主题和语言。

Schedule Tool 和 Web Schedule 页面可创建、查看及删除任务；已有任务不能直接编辑。任务可以指定 `title`，省略时从消息的第一条非空行派生，并规范化为最多 60 个 Unicode code points。每个用户任务有独立的 Schedule Session，Web 中的历史按 occurrence 分组。持久化格式见 [ADR-0001](docs/adr/0001-file-first-local-persistence.md)。

`Enter` 提交输入，`Ctrl+J` 换行；`Ctrl+C` 取消当前回复，输入 `exit` 或 `quit` 退出。

以下管理命令需单独输入，不附带参数：

| 命令 | 用途 |
| --- | --- |
| `/resume` | 从当前 Workspace 的会话列表选择并恢复历史会话 |
| `/restore` | 将当前前台 Conversation Session 截断到一个 Restore Anchor |
| `/status` | 查看运行状态与上下文用量 |
| `/config` | 查看脱敏后的配置 |
| `/permission` | 选择前台 Tool 权限级别 |
| `/effort` | 选择对话模型的推理强度 |
| `/memory` | 查看长期记忆 |
| `/dream` | 将待处理的会话摘要整理为长期记忆 |
| `/reload_skill` | 重新加载 `~/.omni/skills/` 中的 Skill |

## Session Restore

CLI 的 `/restore` 和 Web Restore 都只恢复当前前台 Conversation Session。界面列出已提交的用户消息作为 Restore Anchor；当前会话仍有活动运行或排队输入时不能开始。选择 anchor 后会删除该消息及其后的会话内容，并恢复输入前的会话状态，保留原 Session ID。

恢复范围可选为：

- `conversation-only`：只恢复会话内容，文件保持不变。
- `conversation-plus-files`：同时尝试恢复所选范围内由已授权前台 `write_file`、`edit_file` 实际修改的文件，可能包含 Workspace 外路径。

文件恢复按变更记录工作。没有 tracked write 时直接确认仅恢复会话；已知 Backup Gap 会禁用文件恢复。后续文件冲突仍会尝试覆盖，单个文件无法安全恢复时保持原样并继续处理其他文件，会话仍严格回退。部分文件失败会显示失败路径和已恢复的冲突。

CLI 交互恢复成功后会将 anchor 的完整原文回填为未提交草稿；部分文件失败但会话已回退时也会回填。取消、提交失败或启动时续作不创建草稿。Web 显示恢复结果 10 秒。

最终确认后事务不能取消，进程重启时会先续作未完成的事务。Conversation Summary、Long-term Memory、Schedule、Dream、Exec/MCP effects、手工编辑、Tool Artifacts 和 Session Log 不作为独立回滚目标；它们后来修改过的 tracked file 仍可能被文件恢复覆盖。备份机制无法保证在完全存储失败后识别所有遗漏。事务、备份及恢复边界见 [ADR-0028](docs/adr/0028-session-restore-architecture.md)。

## 项目架构

本地服务负责客户端身份、Project 注册、Session Claim、事件分发和确认协调。每个 Workspace 共用 Memory、Dream、Schedule、Model Router 和 MCP 连接；不同 Session 有独立 Agent Loop、消息队列和运行上下文。输入经服务到达 Session Loop，由 Agent Runner 调用模型与工具，输出经事件通道返回 CLI 或 Web。

全局配置、Skill、Project 注册和服务发现状态位于 `~/.omni/`；会话、记忆、定时任务、工具产物、日志和恢复记录归各 Workspace 的 `.omni/` 所有。详细所有权及生命周期见 [ADR-0029](docs/adr/0029-host-cli-and-web-through-one-local-service.md)，存储边界见 [ADR-0001](docs/adr/0001-file-first-local-persistence.md)。

## 文档与行为合同

| 文档 | 内容 |
| --- | --- |
| [CONTEXT.md](CONTEXT.md) | 领域术语 |
| [现行 ADR](docs/adr/) | 架构决策及约束 |
| [ADR-0023](docs/adr/0023-manage-agent-run-context-by-projected-token-budget.md) | 前台与用户定时任务共用的上下文预算、压缩和 `/status` 字段 |
| [ADR-0024](docs/adr/0024-use-one-shot-dream-model-request.md) | Dream 的一次逻辑模型请求与顺序编辑 |
| [ADR-0025](docs/adr/0025-ignore-unknown-user-configuration-fields.md) | 配置加载、严格编辑及延后生效 |
| [ADR-0026](docs/adr/0026-tool-permission-levels-and-foreground-snapshots.md) | 文件、Exec、MCP、Schedule 与 Web 权限 |
| [GitHub Issues](https://github.com/Totoro-debug/OmniAgent/issues) | 产品需求和讨论记录 |

`/status` 显示当前已提交会话的下一次请求预算基线；Session 累计 token 用量是观测数据，不等于当前上下文占用。前台与用户定时任务在运行前和每次 ReAct 请求前检查预算，Dream 使用独立的一次请求路径。

行为证据见[服务测试](tests/service/)、[终端测试](tests/terminal/)、[前台循环测试](tests/agent/test_loop.py)、[定时任务测试](tests/agent/test_schedule_loop.py)、[Restore 测试](tests/restore/)、[上下文控制器测试](tests/memory/test_agent_run_context_controller.py)、[Dream 测试](tests/memory/test_dream.py)和[发布合同测试](tests/test_release_contract.py)。
