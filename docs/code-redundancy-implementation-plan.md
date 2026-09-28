# 代码冗余清理实施方案（待评审）

状态：待方案评审。本文只规定拟实施的变更；评审确认前不修改生产代码或测试代码。各 Task 独立开发、测试和合并。全部完成后，依据 `docs/agents/domain.md` 的文档规则移除这份已完成的临时方案，历史由 Git 保留。

## 1. 范围与事实依据

本方案处理已通过生产调用链和测试引用核实的四项冗余：

| Task | 已确认的事实 | 涉及的问题类型 |
| --- | --- | --- |
| A：标题转调 | `Session._normalize_title()` 和 `Session._normalize_title_candidate()` 只转调 `myclaw.utils.text` 的现成函数；生产调用方在 `AgentLoop`。 | 本地代码复用、单调用链函数、接口冗余 |
| B：恢复预览 | `management/commands.py` 与 `terminal/conversation.py` 的 `_restore_preview()` 函数体完全相同，各有一个生产调用点。 | 跨模块重复 |
| C：路径条目判断 | `workspace_state.py` 与 `backup_store.py` 的 `_path_entry_exists()` 完全相同；`session/restore.py` 的 `_path_exists()` 在调用 `lstat()` 前额外执行主机路径转换。 | 跨模块重复、主机适配能力复用 |
| D：测试专用结果属性 | `RestoreResult.failed_files` 在仓库生产代码中无读取方，五处读取均在两个恢复测试文件；生产界面读取现有的 `RestoreResult.failures`。 | 测试专用生产接口 |

静态盘点覆盖 89 个生产 Python 模块。唯一无内部导入的 `terminal/process_entry.py` 是 `pyproject.toml` 声明的命令入口；没有确认可删除的整模块死代码。`has_pending_input`、`skipped_tool_count`、`unbind_confirmation_callback` 经 `getattr` 动态使用；Textual 事件处理方法由框架注册。单处实例化的 Screen 类、`ScheduleClock` 协议和 `ToolConfirmationCoordinator` 分别承担框架、注入或生命周期职责，本轮不按实例化次数删除。`croniter`、`tomlkit` 和 `fnmatch` 已在相应实现中复用，未确认可替换的重复库实现。

现有架构约束见 [ADR-0007](adr/0007-use-host-adapters.md)、[ADR-0017](adr/0017-use-cli-composition-root-and-session-scoped-agent-loop.md) 和 [ADR-0028](adr/0028-session-restore-architecture.md)。本方案不增加依赖，不修改配置格式、持久化格式或用户命令语义。

基线（Windows，方案编写前）：`ruff check myclaw tests` 通过；`mypy myclaw tests` 通过（212 个源文件）；`pytest -q` 为 3361 passed、23 skipped、退出码 0。23 项跳过主要受主机符号链接权限或平台能力限制。`pytest` 退出后发生过临时目录清理的 `PermissionError`，不计为测试失败。

## 2. 技术选型与总体约束

- 沿用现有 Python 3.12、`pathlib`、`HostFilesystem` 和 `myclaw.utils.text`；不引入新的通用工具类、配置或第三方库。
- 对字符串格式化只共享确定的规则，不把显示规则写入 `RestoreAnchor` 或恢复持久化层。
- 对路径存在性保留 `lstat()` 语义：悬空链接算作存在，仅 `FileNotFoundError` 表示不存在，其他 `OSError` 原样传播。不能用 `Path.exists()` 代替。
- 每项仅修改下述明确列出的文件及其直接测试；不顺手清理其他单调用函数或单实例类。
- 每个 Task 单独提交和运行其验收命令；后续 Task 可基于已合并分支继续，但验收不依赖其他 Task 的实现。Task C 与 D 都涉及 `session/restore.py`，只修改不同代码段，合并时重新检查上下文。

## 3. 阶段 0：评审与准备

1. 确认四项 Task 的范围，特别确认 Task D 移除 `RestoreResult.failed_files` 所带来的仓库外 Python 调用兼容风险。仓库内无生产读取方，不能据此证明仓库外也没有读取方。
2. 确认工作区无未归属变更：`git status --short`。若已有变更，先识别归属并在原状态上实施，不能覆盖。
3. 每项开始前用 CodeGraph 重新核对目标定义及调用方；索引提示过期时直接核对相应文件。若发现新增的生产调用方或已接受的接口契约，暂停该项并更新方案。

完成指标：方案获得评审确认；四项均有明确执行边界；工作区变更归属已核对。该阶段不写代码。

## 4. 阶段 1 / Task A：消除 Session 标题转调

**文件边界**：`myclaw/agent/session/session.py`、`myclaw/agent/loop.py`，以及直接验证标题行为的测试文件。

**接口与数据流**：`AgentLoop` 将原始内容直接传给 `normalize_title(value: str, *, fallback: str = "Untitled session") -> str`；模型生成的候选标题直接传给 `normalize_title_candidate(value: str) -> str`。返回值仍进入原有的 Session 元数据更新或标题模型请求。`Session.update_metadata()` 内部已有对 `normalize_title` 的调用，不改变该路径。

**执行步骤**：

1. 在 `AgentLoop` 导入上述两个现有工具函数，替换四处 `Session._normalize_title*()` 调用。
2. 删除 `Session` 中两个纯转调静态方法和由此闲置的导入；不改工具函数逻辑。
3. 核对空白标题、成对引号、60 字符截断和候选标题空值的行为；仅在现有覆盖不足时补最小化测试。

**上下游影响**：调用链由 `AgentLoop -> Session 静态方法 -> utils.text` 缩短为 `AgentLoop -> utils.text`。不改变标题文本、持久化内容、异常类型或全局状态；被删除的方法是私有接口。

**量化验收**：两个转调方法均不存在；四个生产调用点均直接使用现有工具函数；上述四类输入输出与基线一致；`pytest -q tests/sessions tests/agent/test_loop.py tests/scheduling/test_schedule_model.py`、`ruff check myclaw tests` 和 `mypy myclaw tests` 均退出码 0。通过后可独立合并。

## 5. 阶段 2 / Task B：统一恢复预览格式化

**文件边界**：`myclaw/management/commands.py`、`myclaw/terminal/conversation.py`，以及恢复命令和终端展示的直接测试。

**接口与数据流**：在 Management 现有模块保留一个 `format_restore_preview(content: str) -> str` 函数。输入是 `RestoreAnchor.content`；先用 `" ".join(content.split())` 折叠空白，长度不超过 96 时原样返回，否则返回前 93 个字符加 `...`。Management 输出与 Terminal 标签各自调用该函数。Terminal 已导入 `management.commands`，不会增加反向依赖或模块循环。

**执行步骤**：

1. 将 Management 中原私有实现命名为共享函数，并在原调用点使用它。
2. Terminal 从该模块导入函数，替换本地调用并删除重复函数；不改两处时间戳格式化规则。
3. 验证空字符串、连续空白、恰好 96 字符、97 字符和 Unicode 内容；保留既有命令和终端展示断言。

**上下游影响**：仅统一 `RestoreAnchor.content -> 预览文本 -> Management 结果/Terminal 标签` 的中间格式化步骤。Session Restore 计划、事务、持久化记录和失败提示均不变。新增的共享函数是两个展示模块间唯一的格式化接口。

**量化验收**：仓库生产代码中只剩一个恢复预览实现；两处展示对上述五类输入逐字符相同；`pytest -q tests/restore/test_cli_restore.py tests/terminal/test_conversation.py`、`ruff check myclaw tests` 和 `mypy myclaw tests` 均退出码 0。通过后可独立合并。

## 6. 阶段 3 / Task C：统一路径条目存在性判断

**文件边界**：`myclaw/utils/host_filesystem.py`、`myclaw/agent/workspace_state.py`、`myclaw/agent/session/backup_store.py`、`myclaw/agent/session/restore.py`，以及主机文件系统、工作区和恢复路径的直接测试。

**接口与数据流**：在现有 `HostFilesystem` 上增加 `entry_exists(path: Path) -> bool`，内部调用 `self.path_for_io(path).lstat()`；仅捕获 `FileNotFoundError` 并返回 `False`，其余成功返回 `True`。这是现有主机适配层内部能力，不修改 `FilesystemAdapter` 协议。三个调用模块通过 `HOST_FILESYSTEM.entry_exists()` 获取结果，后续目录创建、安全验证、恢复判断和异常处理顺序不变。

**执行步骤**：

1. 增加共享方法及针对存在、缺失、悬空链接和其他 `OSError` 的测试。Windows 无链接创建权限时，用受控 `lstat` 替身覆盖悬空链接语义，并保留已有平台测试。
2. 替换 `workspace_state.py` 和 `backup_store.py` 的 `_path_entry_exists()` 调用，删除两份定义；保留仍用于其他 I/O 的主机路径变量。
3. 替换 `restore.py` 的 `_path_exists()` 调用并删除定义；检查所有八处恢复调用保留相同的错误传播。
4. 复核 Windows 扩展路径转换的幂等性，确认传入已转换路径时结果不变。

**上下游影响**：调用链由各模块直接 `lstat()` 收敛为 `模块 -> HostFilesystem.entry_exists() -> path_for_io() -> lstat()`。该方法被工作区初始化、备份状态发布及恢复事务读取共享，是四项中影响面最大的修改；失败必须保留原调用方的安全异常路径。

**量化验收**：三个局部函数均删除；所有原调用点改用共享方法；存在/缺失/悬空链接及非 `FileNotFoundError` 四类结果与基线一致；`pytest -q tests/test_host_filesystem.py tests/test_windows_filesystem.py tests/test_workspace_state.py tests/restore/test_backup_store.py tests/restore/test_path_matrix.py tests/restore/test_transaction.py`、`ruff check myclaw tests` 和 `mypy myclaw tests` 均退出码 0。通过后可独立合并。

## 7. 阶段 4 / Task D：移除测试专用恢复结果属性

**前置决策**：评审明确接受移除未文档化的 `RestoreResult.failed_files`。它虽无仓库内生产读取方，但属于可从 Python 对象读取的属性；若项目承诺仓库外兼容，应取消此项而不是声称零 API 回归。

**文件边界**：`myclaw/agent/session/restore.py`、`tests/restore/test_transaction.py`、`tests/restore/test_path_matrix.py`。

**接口与数据流**：删除 `failed_files` 属性，不改变 `RestoreResult.file_results`、`failures` 或 `failure_notification_pending`。五处测试断言直接从 `result.failures` 中提取 `item.target`；这段测试专用投影留在测试代码中，不建立新的生产接口。

**执行步骤**：

1. 再次确认生产侧没有直接访问或字符串动态访问 `failed_files`。
2. 将五处测试断言改为读取 `failures` 的目标路径；保留失败顺序及逐路径比较。
3. 删除属性，运行恢复结果、终端失败通知和管理输出相关测试。

**上下游影响**：生产的失败数据仍由文件结果组成，Terminal 继续遍历 `failures`；事务和用户输出不变。唯一预期接口变化是移除 `failed_files` 这一 Python 属性。

**量化验收**：生产代码无 `failed_files` 定义或读取；五处断言仍逐路径通过；`pytest -q tests/restore/test_transaction.py tests/restore/test_path_matrix.py tests/restore/test_cli_restore.py tests/terminal/test_conversation.py`、`ruff check myclaw tests` 和 `mypy myclaw tests` 均退出码 0。通过后可独立合并。

## 8. 阶段 5：集成验收与交付

每个 Task 独立合并前运行其专属命令并检查 `git diff --check`、`git diff --stat` 和 `git status --short`，确认没有改动范围外的文件。最后一项合并后运行：

```powershell
ruff check myclaw tests
mypy myclaw tests
pytest -q
git diff --check
```

交付标准：三项检查命令均退出码 0；全量测试零失败，且既有正常行为与用户可见输出不变；仅有评审接受的 `failed_files` 属性移除这一接口差异；所有新跳过项必须说明原因。若主机权限造成符号链接测试跳过，受控 `lstat` 测试仍须覆盖对应语义。记录每项最终改动文件、测试结果和偏离本方案的决定。完成并合并后删除这份临时方案，避免把历史实施步骤留作当前架构契约。
