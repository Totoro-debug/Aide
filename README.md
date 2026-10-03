# MyClaw

## 项目简介

MyClaw 是面向单用户、本地优先的个人 Agent 运行时。通过全屏终端对话调用模型与工具，支持会话恢复、三层记忆、Skill、MCP 和定时任务，运行状态以文件形式保存在本地。

## 项目安装

仅支持 Windows，需要 Python 3.12+ 和 Git。`auto` 选择可用的 PowerShell 7，否则使用 Windows PowerShell 5.1。

目前已验证 Windows x64。应用、服务和发布验证入口在初始化前拒绝其他操作系统。

```powershell
git clone https://github.com/Totoro-debug/myclaw.git
cd myclaw
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

## Web 界面与服务生命周期

安装 wheel 或 source distribution 后，运行时只需要 Python 及其依赖，不需要 Node.js、npm 或前端源码。直接执行 `myclaw web`；首次配置或配置需要修复时，会打开对应的设置页面。已有有效配置时，也可运行裸 `myclaw` 进入 CLI 对话。

```powershell
myclaw web
```

`myclaw web` 会启动或复用当前 Agent Home 的本地服务，并打开一次性的浏览器登录地址。服务只监听 `127.0.0.1:8765`，浏览器和 CLI 连接使用同一个服务与生命周期；关闭浏览器后服务会保留短暂的重连窗口。需要立即收尾时执行：

```powershell
myclaw service stop
```

刷新页面或在 30 秒宽限期内重连会恢复已接受的输入、进行中的输出、工具状态和待处理确认；当前运行仍可取消，确认使用原请求。未登记为 Project 的 CLI Workspace 仅在有在线使用者时接收新的 Schedule 执行；最后一个使用者断线后立即暂停新执行，宽限期到期后收尾并释放运行时，保留 Job 和用户文件。已登记且可用的 Project 在服务仍有在线客户端时继续调度。

停止命令请求服务正常排空活动运行、关闭连接并清理发现文件；它不会删除 Workspace、会话或用户配置。源码开发者只有在重新生成发布资源时才需要 Node.js/npm：

```powershell
npm --prefix web ci
npm --prefix web run build
python -m build --sdist --wheel
```

构建会校验入口 HTML、引用资源、SHA-256 清单和分发包资源；资源缺失或清单不一致时直接失败，不会生成只有后端的安装包。

发布者可在仓库根目录执行 `python -m scripts.installed_web_validation --output D:/myclaw-installed-check`（使用新的源码外目录）。该入口以正常隔离构建生成 wheel/sdist，独立重建并安装两种 wheel，验收生产 Web 页面、深层路由、对话、Settings、同服务 CLI 连接和停止，保存报告及截图。浏览器控制器需要 `web/` 的 Playwright/Node 开发依赖；安装后的应用使用无 Node/npm 的独立环境。默认端口被占用时检查会失败，不停止已有服务。

## Windows 发布验证

发布验证仅在 GitHub Actions 的 Windows runner 上运行 `--phase all`。Windows 必须真实执行 Windows PowerShell 5.1 和 PowerShell 7 的 Host Exec 检查、执行、Full-Access 动态命令及文件系统能力检查，并运行完整测试、lint、类型检查、构建，在隔离环境中安装 wheel，从源码树外执行 entry point 与配置 smoke。`release-gate` 仅依赖 Windows 作业成功；主机缺失或必需测试跳过均不算通过。

PowerShell 选择器配置位于 `[runtime].exec_shell`；`auto` 优先选择 PowerShell 7，随后选择 Windows PowerShell 5.1，显式选择不会交叉回退。验证脚本只在 Windows 运行；默认 `--shell both` 覆盖两个 PowerShell 版本，也可用 `--shell powershell|pwsh` 单独检查对应 Host。GitHub Actions 仅验证 Windows，可通过推送到 `main`、面向 `main` 的 PR 或手动触发工作流：

```powershell
python scripts/release_validation.py --phase all
```

`Full-Access` 只移除普通权限提示，不提供 OS sandbox；参数校验、能力错误、业务拒绝、执行错误、灾难性 Exec 和不确定检查仍然有效。Windows 发布门禁会输出 skip 分类、Windows junction/reparse 证据、PowerShell 主机结果及量化覆盖计数。合并工作流后，将 `release-gate` 设为 `main` 的必需检查；实际 Windows 作业通过前，不视为完成发布验证。

## 项目最小配置

执行 `myclaw config` 生成默认配置，再将 `~/.myclaw/config.toml` 的内容替换为以下配置。已有配置不会被该命令覆盖。`~` 表示当前用户主目录，Windows 下通常为 `C:\Users\<用户名>`。

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

API Key 直接保存在配置文件中，当前不支持环境变量引用；`myclaw config` 显示时会脱敏。更多可选配置及 MCP 示例见[配置模板](myclaw/templates/default-config.md)。

MCP Server 可在 `[mcp.servers.<name>.tool_keywords]` 下按远端 Tool 原名配置英文关键词；缺失或为空的关键词会在启动时通过现有 `chat` Model Route 生成并尽力保存。生成失败时，当前进程使用对应的远端 Tool 原名；生成成功但保存失败时，当前进程继续使用已生成的内存关键词。两类失败都不会阻止 Agent 启动。

`[runtime].permission_level` 接受 `read-only`、`workspace-write` 和 `full-access`，默认值为 `workspace-write`；`[runtime].exec_shell` 接受 `auto`、`powershell` 和 `pwsh`，默认值为 `auto`。Windows 下 `auto` 优先选择版本至少为 7 的 `pwsh`，否则选择 Windows PowerShell 5.1；显式选择不会交叉回退。Exec Host 的检查和执行都使用 `-NoLogo -NoProfile -NonInteractive` 禁用 Profile，并共享同一个可执行文件、工作目录、最小环境和超时/取消行为。选定 Shell 缺失不会阻止启动，只显示一次安全诊断；Exec 仍在工具目录中，但调用时返回稳定的能力错误。Shell 存在但检查器失败时按不确定结果继续沿用确认语义。
在 Read-Only 和 Workspace-Write 下，PowerShell Exec 只有固定候选命令、固定参数语法、唯一 canonical 身份、预期 Microsoft module 以及 Workspace 内静态 FileSystem 路径时才直通；别名、函数、脚本、shim、歧义身份、未知参数、变量、glob、表达式、pipeline-fed path、Provider 路径和外部路径都需要一次确认。候选只包括只读的 `Get-ChildItem`、`Get-Content`、`Get-Item`、`Get-Location`、`Get-FileHash`、`Measure-Object`、`Select-Object`、`Sort-Object`、`Select-String`、`Test-Path`、`Resolve-Path`、`Format-List`、`Format-Table`、`Out-String`，以及 Workspace-Write 新增的 `New-Item`、`Set-Content`、`Add-Content`、`Clear-Content`、`Copy-Item`、`Move-Item`、`Rename-Item`、`Remove-Item`、`Out-File`。Full-Access 允许可解析的非灾难性动态 Exec 直通，但灾难性操作和检查器不确定仍需确认；参数校验、能力错误、业务拒绝和执行错误不被绕过。Git 只直通固定只读 form；Host 固定禁用 pager、external diff/textconv、fsmonitor、hook 和 prompt。静态检查不会启动 Git；只有 canonical Git 已证明在 Workspace 外且仓库目标已规范化到 Workspace 内，才会在直通前审计 include 与 clean/process filter 配置。

前台 Agent Run 会在标题生成、Skill 解析和 Task Framing 之前捕获当前权限级别与已解析的 Exec Shell，并在本次运行内保持不变。Runtime Context 会显示该快照以及 Tool 调用可能需要确认；确认只绑定到一次规范化后的 Tool 调用，不会缓存到共享 Tool。每次 Gateway 调用都由 `BaseTool.prepare()` 产出规范化参数和结构化事实，再由唯一权限策略决定 direct 或 confirm；validation、business refusal、capability error 和 execution error 独立于权限决策并保持为 hard error。Tool 不再提供独立的字符串授权路径。File Tool 的权限矩阵和主机路径规范化见 [ADR-0026](docs/adr/0026-tool-permission-levels-and-foreground-snapshots.md)。模型发起的 `read_file` 不继承 Skill Root 的内部加载豁免；Dream 私有 Tool 和 Runtime 持久化写入不属于该模型 File 策略。

前台模型发起的 Schedule Tool 调用按当前 Agent Run 快照授权：`list` 在三种级别都 direct；`add/remove` 在 Read-Only 下请求一次确认，在 Workspace-Write 和 Full-Access 下 direct。当启动时配置的 Schedule 级别高于本次前台 Run 的 current 级别时，`add` 还会合并一条未来 Job 将使用更高 configured 级别的确认理由；`remove` 不触发这条升级理由。确认详情使用规范化后的 action、message、title、schedule 或 job_id，批准只作用于本次调用，拒绝不修改 Schedule Store。Schedule Job 和公开 JSON 不保存 permission、level 或 snapshot；`/permission` 只改变后续前台 Run 的 current，不改变启动时的 configured。每次 User Schedule occurrence admission 都捕获启动时 configured 权限级别和进程级 resolved Exec Shell，并将同一快照传入 Schedule Runtime Context 与 run-local Gateway；低权限等级复用 File、Exec、Web Fetch 和 MCP 的逐调用确认规则，Full-Access 取消普通权限提示但不绕过 hard error。后台确认使用共享 coordinator；Dream Schedule Job 和其他 System Schedule Job 仍保持内部 direct 行为。

Web Search 在所有权限级别始终 direct。Web Fetch 会在初始 URL 和每个 redirect hop 建立连接前进行一次受控 DNS 审计与网络授权：只有所有解析地址均 globally routable 的非空集合才是 public；private、loopback、link-local、reserved、unspecified、multicast、mixed、空结果和 DNS failure 在 Read-Only/Workspace-Write 下请求确认，Full-Access 只跳过权限提示。一次规范化调用最多一次确认，批准不产生 hostname、IP 或 network grant；普通 DNS、连接、TLS、HTTP、body/decode 和 timeout 错误在三种级别都仍返回 Tool Error。redirect 不由 HTTP client 自动跟随，connector 只使用当前 hop 刚审计的地址集合，禁用环境 proxy，且保留请求 Host、TLS SNI 和证书 hostname 校验。MyClaw 没有 OS 级 network sandbox。

`/permission` 只修改当前 Runtime Lifetime 后续前台 Agent Run 的权限级别，不写入 User Configuration、Conversation Session 或 Schedule。成功的 `/resume` replacement 会保留当前选择；失败的 replacement 不会改变它；新进程从配置值开始。选择 `full-access` 前必须通过默认聚焦 Cancel 的警告；当前该级别取消前台 File Tool、普通 Exec 和 Web Fetch 非公网目标的权限确认，但不是操作系统沙箱，也不绕过参数校验、能力错误、业务拒绝、灾难性 Exec 确认或 Tool 执行错误。前台 MCP Tool 在 Read-Only 和 Workspace-Write 下每次调用都对完整规范化的 server/tool identity 与 arguments 请求一次独立确认，批准不会缓存为 Server、Tool、参数或 Host grant；Full-Access 直接调用但仍保留 MCP 的 schema、参数、transport、timeout、server error 和 cancellation 错误。User Schedule Agent Run 的低权限 MCP 调用也逐调用进入共享后台 Tool Confirmation，Full-Access 直接调用。Web Search 始终 direct。进程启动时最多显示一次 Full-Access 安全提示；`/config` 显示 configured level，`/status` 同时显示 configured/current foreground level。Runtime Lifetime 只有一个共享 Tool Confirmation coordinator 和一个 Terminal presenter：同一时间最多显示一个 modal，前台队列优先于后台队列但不会打断当前 modal；请求按各自队列 FIFO 串行处理，等待期间没有超时。`/resume` replacement 会在旧 generation 的 destructive replacement 开始时取消其 confirmation，并在 Store 可写时排空 Schedule terminal outcome；关闭 Runtime 也先排空后台 confirmation abort。UI 文档见 [ADR-0027](docs/adr/0027-runtime-lifetime-tool-confirmation-coordinator.md)。

安全默认值字段包括 Runtime 的 Tool 结果大小、迭代上限、Always-load Skill 开关、`compact_ratio`、权限级别和 Exec Shell，Memory 的 batch size 与 schedule，以及每个 Model Route 的 `reasoning_effort`。缺失字段静默使用默认值；显式非法值使用有效默认值并产生且只产生一条不包含原始值的安全诊断。`compact_ratio` 的默认值为 `0.9`，只接受有限且非布尔的数值 `0.5` 至 `0.95`（含边界）。原始配置不会被自动改写；`myclaw config` 与 `/config` 显示相同的有效值和诊断，以及脱敏后的原始 TOML。已移除的 `compaction_message_threshold` 等未知字段会被忽略。

## 项目启动

在已激活虚拟环境的交互式终端中，进入希望 Agent 操作的目录后启动（将示例路径替换为实际路径）：

```powershell
Set-Location "D:\path\to\workspace"
myclaw
```

新开终端后需重新激活安装目录中的虚拟环境，或使用其中 `myclaw` 可执行文件的绝对路径。启动需要交互式输入、输出，不能通过管道运行。

启动目录即 Workspace，不会自动切换到 Git 根目录。运行状态保存在该目录的 `.myclaw/` 中；定时任务仅在 MyClaw 进程运行期间执行。

Schedule Tool 的 `add` 可选接收用户可见 `title`；它会按 Conversation Session 的空白和成对引号规则规范化，并确定性截断到 60 个 Unicode code points。只有省略该字段时才从 `message` 第一条非空行派生；显式 `null` 或规范化后为空的值会被拒绝。`add` 和 `list` 的每个公开 Job 都返回 `job_id`、`title`、`message` 和 `schedule`。新建用户 Schedule Conversation Session 的初始 title 与 Job title 相同，Job 始终是 title 的权威来源。

Workspace 的 `.myclaw/schedule.json` 使用严格 canonical Schedule Job schema，当前记录必须包含 `title`。升级时仅接受整份文档都使用完整旧字段集的状态，并在读取时派生 title；下一次成功的 Schedule Store 写入会把全部记录改写为新 schema。旧新记录混用、单条记录混合字段、未知字段和失败写入都不会被静默修复。

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
| `/reload_skill` | 重新加载 `~/.myclaw/skills/` 中的 Skill |

## Session Restore

`/restore` 只作用于当前前台 Conversation Session，不会选择其他 Session 或
Schedule Session。它只列出已经提交的前台 User message；每条消息都是一个
Restore Anchor，使用当前 Session 内单调递增且永不复用的数字 ID，并按最新优先显示
本地时间和一行预览。前台 Agent Run 或排队的 User input 尚未结束时，`/restore`
不会打开。产品需求以[父规格 #262](https://github.com/Totoro-debug/MyClaw/issues/262)
为准；当前架构与领域术语分别见
[ADR-0028](docs/adr/0028-session-restore-architecture.md)和 `CONTEXT.md`。

选择 anchor 后，Session Restore 会删除该 User message 及其后的所有 Session
message，并恢复该输入之前的 title、Blackboard、Action Summary、`last_compacted`、
token usage 和其他 Session-owned metadata。原 Session ID 保留，即使结果是空 Session
也会持久化；显示和前台输入历史会从持久化 Session 重建，Management Command history
保留。当前 `/restore` 交互成功后，所选 anchor 的完整 User message 原文会作为未提交草稿
回到输入框，保留换行及首尾空白；用户可以继续编辑、重新提交或清空。File Restore 部分失败
但 Session 已成功回退时仍会回填；取消、检查或提交失败不会回填，进程重启后由 startup
recovery 续作完成的恢复也不会创建该草稿。

恢复范围可选为：

- `conversation-only`：只恢复 Conversation Session，文件保持不变；被截断分支的
  File Backup 记录从 active journal branch 移除。
- `conversation-plus-files`：在恢复 Conversation Session 的同时，尝试恢复当前选定
  active range 中由已授权前台 `write_file` 和 `edit_file` 实际修改的文件，包含 Workspace
  外部路径。没有 tracked file write 时会直接进入 conversation-only confirmation。

File Restore 是按变更记录工作的，不是 filesystem snapshot。选定范围内的 known
Backup Gap 会禁用 File Restore，并说明 backup coverage 不完整；备份失败不会阻止原
Tool 继续执行，也不会立即弹出警告。如果 backup 和 Gap marker 都无法保存后进程崩溃，
缺失 coverage 可能无法知道。当前文件 bytes 或 existence 不同属于 conflict，但仍会尝试
恢复；成功恢复的 conflict 会列在普通结果中。单个文件无法安全恢复时会保持原样，其他
文件继续处理，Session 仍会严格截断；只有出现 partial file failure 才显示失败通知，列出
失败路径和成功恢复的 conflict，并持久化 acknowledgement。

最终确认后的 restore transaction 不能取消。进程重启时，startup recovery 会在加载全局
Memory、Schedule 和前台 Session 之前完成 pending transaction，并按 durable target list
幂等续作；Session persistence 失败会阻止普通对话继续。Conversation Summary、Long-term
Memory、Schedule、Dream、Exec/MCP effects、manual edits、Tool Artifacts、Session Log
以及 result files 不作为独立 rollback 目标而被保留；这些参与者后来改动过的 tracked file
仍可能被 File Restore 覆盖。

## 项目架构

CLI 负责组装运行时和管理组件生命周期。前台输入经终端与 Message Bus 进入 Agent Loop，由 Agent Runner 循环调用模型与工具，结果经 Message Bus 返回终端。

| 组件 | 职责 |
| --- | --- |
| Terminal / Message Bus | 基于 Textual 与 Rich 展示对话，通过输入、输出队列连接当前 Agent Loop |
| Agent Loop | 绑定一个会话，串行处理前台输入，管理上下文、任务目标和会话持久化 |
| Agent Runner | 执行有迭代上限的模型与工具循环，供前台和用户定时任务复用 |
| Model Router | 按用途选择模型，适配 OpenAI 兼容协议与 Anthropic，处理重试和回退 |
| Tool Gateway / MCP | 统一内置与 MCP 工具的调用入口；MCP 连接由 CLI 管理，每个 Agent Loop 使用固定工具快照，单次 Agent Run 通过 `tool_search` 按需暴露延迟 Tool schema；前台低权限 MCP 调用逐次确认且不缓存批准 |
| Memory / Dream | 管理短期记忆、会话摘要和长期记忆；Dream 通过一次受限的 `memory` 请求整理长期记忆 |
| Schedule Service | 持久化并调度任务；用户任务调用当前 Agent Loop，记忆整理任务直接调用 Dream |
| Skill Loader / Context Builder | 加载 Skill 快照，按需提供指令，统一构建 Agent Loop 的模型请求上下文 |

全局配置与 Skill 位于 `~/.myclaw/`；会话、记忆、定时任务、工具产物和日志归各 Workspace 的 `.myclaw/` 所有。

内置工具通过权限检查决定是否请求一次性确认；Exec 通过 Host 以当前用户权限执行，不提供操作系统沙箱。MCP 支持 stdio 和 Streamable HTTP，已启用的 Server 是用户配置的外部能力：前台 Read-Only/Workspace-Write 对每个 MCP Tool invocation 单独确认，Full-Access 直接调用；确认不缓存，且任何权限级别都保留既有 hard errors。

架构决策见[现行 ADR](docs/adr/)，领域术语见[CONTEXT.md](CONTEXT.md)。

## 运行时合同

### `/status`

前台和用户定时任务共用同一套上下文预算；Dream 不进入这套 Agent Run 预算。`/status` 只基于当前已提交的 Conversation Session 和当前 User Configuration，投影下一次独立 Foreground Agent Run 的初始模型请求基线，不读取未提交的 run-local staged state。Route 字段来自当前配置解析出的初始 `chat` Model Route；未配置 `chat` 时可使用配置层面的静态 `default`，上一逻辑调用的动态 fallback 不会成为下一次调用的初始 Route。上下文字段定义如下：

- `context_window`：当前配置解析出的初始 `chat` Model Route 的总上下文窗口。
- `max_output`：为模型输出预留的 token 上限。
- `available_context`：可用于输入的预算，等于 `context_window - max_output`。
- `compact_ratio`：实际生效的配置比例，默认 `0.9`。
- `compact_context_window`：软压缩阈值，等于 `ceil(available_context * compact_ratio)`。
- `projected_next_request_tokens`：上述下一次独立 Foreground Agent Run 初始模型请求的投影 token 数；仅当最新 main-Agent assistant 的 Provider 用量 provenance 合法且兼容时使用报告增量，缺失、非法或不兼容时使用完整本地估算，不跨过该 assistant 复用更早锚点。
- `projection_source`：投影来源，`reported_delta` 表示报告用量增量，`estimated` 表示本地估算。
- `input_budget_used_percent`：`projected_next_request_tokens / available_context * 100`，分母是可用输入预算。

`cumulative_usage` 是 Session 生命周期内的累计模型调用、输入、输出和总 token 用量，仅用于观测，不代表当前请求上下文占用；Schedule 状态不混入前台 Session 的上下文字段。

### Agent Run

Foreground 和 User Schedule Agent Run 在启动及每次 ReAct 请求前都执行同一套预算检查。历史压缩优先于 Provider-only Tool Result Micro-compression；达到软阈值时按 10%/50% 保留规则选择历史，达到硬输入上限则以 `model_context_overflow` 结束，且每个实际 Provider 尝试都重新检查自己的 Route。超过十个符合条件的 Tool Call 后，才会对请求投影中的旧 Tool Result 做微压缩；持久化消息、摘要和 Tool Artifact 不写入占位符。终止时由 Agent Loop 一次性提交本次 Session 增量及 Agent Run 状态，摘要流保持在 Session 事务之外。

### Dream

Dream 不是 Agent Run，不创建 Agent Runner 或 Agent Run Context Controller，也不执行上下文压缩。没有待处理 Conversation Summary 时，Dream 发起 `0` 次模型请求；领取到待处理 Summary 后，恰好发起一个逻辑上的 `memory` 模型请求，Router 自动重试只增加 Provider attempt，不增加逻辑请求数。该请求只暴露 `edit_file`，返回的编辑按顺序执行；前面的编辑成功后，即使后续编辑失败，已完成的修改仍保留。

### 权威合同与公共证据

Ticket #234 的统一要求由 [ADR-0023](docs/adr/0023-manage-agent-run-context-by-projected-token-budget.md)、[ADR-0024](docs/adr/0024-use-one-shot-dream-model-request.md) 和 [ADR-0025](docs/adr/0025-ignore-unknown-user-configuration-fields.md) 共同定义；[ADR-0022](docs/adr/0022-action-summary-and-tool-result-micro-compression.md) 已明确标记为被 ADR-0023 取代。公共行为证据集中在[控制器测试](tests/memory/test_agent_run_context_controller.py)、[前台循环测试](tests/agent/test_loop.py)、[定时任务测试](tests/agent/test_schedule_loop.py)、[Dream 测试](tests/memory/test_dream.py)、[Session 测试](tests/sessions/test_session.py)和[发布合同测试](tests/test_release_contract.py)。
