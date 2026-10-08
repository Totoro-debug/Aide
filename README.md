# Aide

## 简介

Aide 是面向个人、在本地运行的 Agent，通过你配置的模型服务和工具完成任务。提供本地 Web 和全屏终端界面，会话、记忆和定时任务以文件形式保存在本地。

## 主要功能

- **Web / CLI 对话**：在浏览器或终端中与 Agent 交互，查看工具执行过程。
- **会话管理**：继续历史对话、并行运行不同会话，或回退到某次输入之前。
- **工作区记忆**：整理对话摘要与长期记忆，同一工作目录中的会话共享记忆。
- **定时任务**：让 Agent 按计划执行任务，并查看执行历史。
- **Skill 与 MCP**：通过 Skill 复用任务指引，通过 MCP 接入外部工具。

## 快速开始

### 1. 安装

需要 **Windows、Python 3.12+ 和 Git**，目前已验证 Windows x64。安装和运行无需 Node.js 或 npm。在 PowerShell 中执行：

```powershell
git clone https://github.com/Totoro-debug/Aide.git aide
cd aide
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

### 2. 配置模型

生成默认配置，然后停止服务，以便下次启动加载你填写的模型设置：

```powershell
aide config
aide service stop
```

首次配置时可能提示模型尚未配置，继续填写即可。打开 `~/.aide/config.toml`，首次使用可将内容替换为以下最小配置；已有配置可按需修改。`~` 表示用户主目录，Windows 下通常为 `C:\Users\<用户名>`。

```toml
[models.providers.my-provider]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "replace-with-your-api-key"
[models.providers.my-provider.models."your-model-id"]
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "mid"
timeout = 120

[models.routes.default]
provider_id = "my-provider"
model = "your-model-id"
```

选择支持工具调用的模型，替换服务地址、API Key 和两处模型 ID。模型的上下文窗口、最大输出、温度、默认推理强度和请求超时集中配置在 Provider 下的模型中；路由仅引用 Provider 和模型。按模型实际限制设置 `context_window` 与 `max_output`，单位为 token，后者必须小于前者。

`protocol` 支持 `openai-compatible` 和 `anthropic`；只配置 `default` 路由即可用于对话、记忆和定时任务。更多选项见[配置模板](aide/templates/default-config.md)。

Web 模型设置以小卡片编辑参数，点击 Provider 下的加号添加模型；chat、memory、schedule 从已配置的 Provider 和模型中选择。保存设置后，重启服务生效。切换会话模型时采用所选模型自身的输出上限等参数，会话显式推理强度覆盖模型默认值。

旧模型列表与路由参数配置仍可读取。编辑模型设置时，参数一致的配置可以合并；同一模型存在不同路由参数时，卡片列出候选值，需要明确选择或补齐参数后再迁移保存。额外模型缺少参数时需要补齐，已引用的模型需先更换路由才能删除。

### 3. 开始对话

在已激活虚拟环境的终端中启动 Web：

```powershell
aide web
```

浏览器会自动打开对话页面，输入任务即可开始。模型和 MCP 等配置也可在 Web 设置中编辑。

使用终端界面时，先进入希望 Agent 操作的目录，再启动：

```powershell
Set-Location "D:\path\to\workspace"
aide
```

新开终端后需重新激活安装目录中的虚拟环境，或使用其中 `aide` 可执行文件的绝对路径。需要停止本地服务时执行：

```powershell
aide service stop
```

## 基本使用

**对话与项目**：Web 普通对话默认使用 `~/.aide/chat`，可在设置的「常规与外观」中更改。点击侧栏「项目」旁的「＋」选择已有目录，可让 Agent 在该目录中工作。CLI 使用启动时的当前目录作为工作区。

**历史会话**：Web 从侧栏打开历史对话；CLI 用 `/resume` 选择当前工作区的会话。同一工作区可同时运行多个会话，每个会话只能由一个客户端操作。同一服务支持一个 Web 客户端和多个 CLI 客户端。

**记忆与定时任务**：Web 项目菜单或普通对话菜单提供记忆、Dream 和定时任务入口。Dream 将待处理的对话摘要整理为长期记忆。也可以在对话中让 Agent 创建定时任务，例如「每天上午 9 点总结这个项目的待办事项」。

**Skill 与 MCP**：将 Skill 放在 `~/.aide/skills/<技能名>/SKILL.md`，文件使用包含 `name`、`description` 的 YAML frontmatter；输入 `/<技能名> 任务描述` 调用。修改后用 `/reload_skill` 重新加载。MCP 可在 Web 设置中配置，示例见[配置模板](aide/templates/default-config.md)。

**Tool 微压缩**：默认关闭，可在 Web「设置 → 运行时」中启用，或在配置文件的 `[runtime]` 中设置 `enable_tool_micro_compression = true`，保存后需重启 Aide。启用时，模型上下文中符合条件的 Tool 结果超过 10 条后，较早的长结果可能被省略，最新完整调用周期仍保留。这样可减少上下文占用，但模型可能遗漏细节或需要再次调用工具；原始结果和会话记录仍会保留。Web 启用前会弹窗说明影响，确认后才保存。

CLI 中，`Enter` 提交、`Ctrl+J` 换行、`Ctrl+C` 取消当前回复，输入 `exit` 或 `quit` 退出。以下管理命令需单独输入：

| 命令 | 用途 |
| --- | --- |
| `/resume` | 继续历史会话 |
| `/restore` | 回退当前会话，可选择同时恢复文件 |
| `/status` | 查看运行状态与上下文用量 |
| `/config` | 查看脱敏后的配置 |
| `/permission` | 调整工具权限级别 |
| `/effort` | 调整模型推理强度 |
| `/memory` | 查看长期记忆 |
| `/dream` | 将对话摘要整理为长期记忆 |
| `/reload_skill` | 重新加载 Skill |

## 使用须知

1. **数据位置**：全局配置与 Skill 位于 `~/.aide/`；会话、记忆和定时任务位于各工作区的 `.aide/`。模型请求会发送给你配置的服务商，API Key 直接保存在配置文件中，当前不支持环境变量引用。
2. **工具权限**：默认 `workspace-write`，允许直接读写工作区内文件，访问外部文件和调用 MCP 需逐次确认。可选择 `read-only` 或 `full-access`；`full-access` 不提供操作系统沙箱，可能造成大范围破坏或检查无法确定的命令仍需确认。详细规则见[权限说明](docs/adr/0026-tool-permission-levels-and-foreground-snapshots.md)。
3. **配置生效**：修改模型、MCP 等配置后需重启 Aide，可在 Web「常规与外观」中操作；重启会停止当前运行和连接的 CLI。权限选择与推理强度调整即时生效，用于后续运行。
4. **定时任务运行条件**：服务有在线客户端时，已登记的可用项目继续调度；未登记的 CLI 工作区仅在有在线使用者时接收新任务执行。最后一个客户端断线后暂停新执行，约 30 秒后清理并退出；定时任务定义仍保留。
5. **会话回退范围**：回退会删除所选输入及其后的对话。文件恢复仅覆盖当前会话通过内置 `write_file`、`edit_file` 修改且有可用备份的文件，不撤销命令执行、MCP 操作或记忆等状态；恢复失败的文件会单独报告。详细边界见[恢复说明](docs/adr/0028-session-restore-architecture.md)。

## 开发与发布验证

在安装了开发依赖的 Windows 工作区运行以下命令，验证覆盖、PowerShell 宿主、Python 回归与安装包。文件符号链接用例需要宿主启用开发者模式或具备相应权限；报告记录每项检查及跳过原因。

```powershell
python scripts/release_validation.py --phase all --report "$env:TEMP\aide-windows-release.json"
```

问题与建议请提交到 [GitHub Issues](https://github.com/Totoro-debug/Aide/issues)。
