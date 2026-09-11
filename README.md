# AI 记忆协同管理工具

本项目从 ChatGPT、DeepSeek、豆包、Gemini（gemini.google.com）和 Kimi（kimi.com）
的分享页/会话页提取完整对话，并可选择调用
Gemini API、SiliconFlow 或 DeepSeek 兼容接口，对文字、图片和可用文档进行多模态
分层总结。

## 安装

    uv pip install -r requirements.txt
    playwright install chromium

GUI 通过右上角齿轮配置 Gemini、SiliconFlow 或 DeepSeek API KEY，密钥仅保存到当前
Windows 用户的凭据管理器，不写入源码、配置文件、命令参数、日志、输出文件
或提交记录。GUI 只使用用户在此配置的密钥，不会借用开发环境密钥；已配置
提供商按用户在设置页拖动排列的顺序回退，未配置项直接跳过。

“仅抓取对话”不要求配置密钥；选择任一总结模式但尚未配置时，GUI 会拒绝启动
并打开密钥设置页。命令行入口仍从 GEMINI_API_KEY、Silicon_API_KEY（兼容
SILICONFLOW_API_KEY）或 DEEPSEEK_API_KEY 读取密钥。

## 使用

启动图形界面：

    uv run python -m gui.app

## Windows 打包

发布版采用 PyInstaller 目录模式，并内置与当前 Playwright 版本匹配的 Chromium。
用户无需安装 Python、项目依赖或浏览器。构建机先准备依赖与浏览器：

    uv pip install --python .venv\Scripts\python.exe -r requirements-build.txt
    .venv\Scripts\python.exe -m playwright install --no-shell chromium

随后从项目根目录执行：

    .venv\Scripts\pyinstaller.exe --noconfirm --clean `
      --distpath release\dist --workpath release\build AI_Memory_Summary.spec

应将整个 `release\dist\AI记忆总结工具` 文件夹压缩后分发；不能只发送其中的
EXE。用户完整解压后双击 `AI记忆总结工具.exe` 即可。目录模式避免每次启动
重复解压数百 MiB 的内置浏览器，也更便于安全软件逐个扫描。当前发布包未使用
商业代码签名证书，因此 Windows SmartScreen 可能在首次运行时提示未知发布者。

打包版把浏览器登录会话和脱敏运行日志保存在当前用户的
`%LOCALAPPDATA%\AI Memory Summary`；API KEY 仍只保存在 Windows 凭据管理器。
完整网页调试快照默认关闭，只有开发者显式设置
`AI_MEMORY_SAVE_DEBUG_HTML=1` 时才会保存。

### Windows 轻量版

轻量版不内置 Chromium，优先调用系统 Edge，启动失败时自动回退到系统
Chrome；两者都不可用时提示用户安装浏览器或下载全量版。全量版 spec 与默认
`gui.app` 行为不变。源码启动轻量版：

    uv run python -m gui.lite_app

独立构建轻量版：

    .venv\Scripts\pyinstaller.exe --noconfirm --clean `
      --distpath release\lite-dist --workpath release\lite-build `
      AI_Memory_Summary_Lite.spec

分发时同样必须压缩整个
`release\lite-dist\AI记忆总结工具_轻量版` 文件夹，不能只发送 EXE。


在 GUI 中可粘贴公开分享链接（含 Gemini `/share/`、Kimi `/share/<id>` 分享页，
Kimi 链接可带 `/en/`、`/zh/` 语言段或 `?ra=1`、`?sharetype=link` 参数），
也可粘贴 ChatGPT `/c/`、DeepSeek `/a/chat/s/`、Gemini `/app/<id>` 或
Kimi `/chat/<id>` 私有会话地址。选择“不登录”时，
ChatGPT 私有会话直接使用始终最小化的浏览器，其他平台（含 Gemini）先在后台无头
复用登录态；只有确认当前登录态无法读取会话时，才恢复浏览器窗口要求登录。
选择“授权登录”时则直接显示浏览器，并在确认后重新打开原始会话。
勾选一个或多个模式，并选择保存位置和 Markdown 文件名。抓取到的图片与当前账号
有权下载的文档，会分别放入 Markdown 同目录下的专属 `_images` 与 `_files`
文件夹，并通过相对路径引用；分享或移动结果时，应把 Markdown 与这些同名资源
文件夹一起保留。平台未向当前会话提供下载权限的历史附件会明确标记为不可用，
程序不会绕过平台权限。普通版或详细版
完成主题综合后，会弹出“选择重点主题”窗口，直接列出本次分主题摘要中的
具体主题及其摘要。所有主题摘要始终完整显示；勾选只表示该主题更重要，并在
“重点主题详情”中展开其关键记忆和相关结构化记录。媒体开发检查与详细版细节
记忆不受影响。仅抓取和极简版不会弹出该窗口；短对话没有独立主题时也不弹窗。

GUI 会在进度栏显示当前总结模型，并按用户设置的提供商顺序依次尝试。每个
提供商内部仍保留自己的模型回退链；例如轮到 Gemini 时，默认模型失败后会
快速尝试 3.6 Flash 和 3.5 Flash Lite，再切换到下一个已配置提供商。未配置项
会从回退链中删除。

保持原有行为、只抓取对话：

    uv run python -m scripts.parser

抓取完成后继续调用已配置的总结后端：

    uv run python -m scripts.parser --summarize

临时覆盖集成流程使用的模型：

    uv run python -m scripts.parser --summarize --model gemini-3.6-flash

对已经导出的结果单独总结，不重新打开浏览器：

    uv run python -m scripts.summarize_memory "results/export/ChatGPT_含图片.md"

临时选择其他模型：

    uv run python -m scripts.summarize_memory "results/export/ChatGPT_含图片.md" --model gemini-3.6-flash

也可以通过环境变量长期调整，而不改代码：

    $env:SUMMARY_PROVIDER = "gemini"
    $env:SUMMARY_MODEL = "gemini-3.5-flash"
    $env:GEMINI_CHUNK_CHARS = "24000"
    $env:GEMINI_SHORT_CONVERSATION_CHARS = "18000"
    $env:GEMINI_MAX_OUTPUT_TOKENS = "16384"
    $env:GEMINI_THINKING_LEVEL = "medium"
    $env:GEMINI_API_RETRIES = "3"
    $env:GEMINI_RATE_LIMIT_WAIT_SECONDS = "65"
    $env:GEMINI_MAX_MEDIA_BYTES = "12582912"
    $env:GEMINI_MEDIA_BATCH_SIZE = "6"
    $env:GEMINI_BATCH_INTERVAL_SECONDS = "15"

SUMMARY_MODEL 未设置时默认使用稳定版 gemini-3.5-flash；仍兼容旧变量
GEMINI_MODEL。模型名只在统一配置入口设置，业务调用点不会写死；以后换模型
只需修改环境变量或 --model 参数。

2026-08-16 实测：gemini-3.5-flash 与 gemini-3.6-flash 均可返回结构化
JSON。本轮批量总结优先使用 3.5 Flash；其额度在超长对话中耗尽后，仅对该份
结果显式改用 3.6 Flash。模型可用性和额度会变化，运行时仍应以真实探测为准。

命令行核心不会静默切换模型，以免费用、能力和输出变化在用户不知情时发生；
上述 GUI 跨提供商回退会在进度栏明确显示每次切换。使用 SiliconFlow
时，GUI 会在当前模型失败后尝试 2026-08-22 完整总结流水线实测通过的免费
候选 Qwen/Qwen3-8B。连同当前模型最多尝试 2 个，不会为凑数量加入质量未
达标的模型。

显式使用 SiliconFlow 的 Qwen 备用模型：

    uv run python -m scripts.summarize_memory "results/export/DeepSeek_超长对话.md" --provider siliconflow --model Qwen/Qwen3.5-397B-A17B

显式使用 DeepSeek：

    uv run python -m scripts.summarize_memory "results/export/DeepSeek_超长对话.md" --provider deepseek --model deepseek-v4-pro

## 总结流水线

1. 程序解析统一消息结构，定位本地图片、PDF、文本附件及失效附件。
2. 总结模型先理解可访问媒体；程序记录附件所属消息、可否重新验证及随后 AI 结论。
3. 程序优先按完整消息边界分块；超长对话分块提取后再综合当前断点和历史主题。
4. 每条记忆显式记录 topic、source、status、message_ids 和 message_range。
5. 编程、语言学习、计算、决策等任务使用不同结构；短消息与上下文指代单独绑定。
6. 程序确定性保留全部用户原始查询和最近 20 条消息（最多最近 10 轮）。
7. 短对话不机械拆主题；长对话只在存在真实独立主题时生成主题摘要。
8. 只有明显影响理解、判断或执行的原文问题才单列；简单笔误直接忽略。
9. 程序二次校验来源、断点、问答配对、编程/自然语言分类与失效媒体表述。

模型完成分类后，单文件总结与“抓取后总结”会在终端列出当前实际存在的
可展开板块。直接回车表示全部不展开；输入一个或多个编号才会在“分主题摘要”
之后展示对应的编程、语言、计算、决策、上下文、递进或原文问题记录。
分主题摘要的所有主题始终原样完整显示；媒体与附件开发检查也始终显示，
不参加选择。完整结构化记录始终保存在 JSON 中，不因 Markdown 隐藏而丢失。
也可跳过交互：

    uv run python -m scripts.summarize_memory "results/export/ChatGPT_含图片.md" --expand learning,programming

使用 --expand all 展开全部可选板块，使用 --expand none 明确全部不选。
批量测试默认全部不选且不等待终端输入。该命令行兼容入口仍按结构化类别选择；
GUI 则按上方所述直接选择具体主题。

输出统一保存在 `results/summary` 文件夹，并使用源对话文件名区分：

- results/summary/ChatGPT_含图片_result.json：结构化结果，适合后续程序读取。
- results/summary/ChatGPT_含图片_summary.md：适合直接阅读和人工检查。

总结不同对话时会生成各自的文件，不再互相覆盖。仍可通过 --json-output 和
--markdown-output 显式指定其他保存路径。

Markdown 默认只输出精简总结，不重复展示分类记录，适合直接交给下一个 AI。需要提高历史细节
覆盖率时，可添加 --include-details，在同一文件末尾附加“细节记忆”。
该板块会合并原细粒度记忆与用户查询、聚合同类内容并过滤过程性信息，
最多保留 8 条关键细节；超长语言学习记录在 Markdown 中也最多展示 8 条
代表项并优先保留明确纠错。带细节模式的 Markdown、JSON 和批量报告统一
保存到 `results/summary_detailed`，普通版继续保存在 `results/summary`。JSON 始终保留
完整结构化数据。

分块结果会写入同目录的 *_progress.tmp 恢复缓存；API 中断后重跑会复用已经
完成的媒体说明和分块。成功生成最终结果后缓存自动删除。遇到 429 时程序
至少等待一个限额窗口，批处理也会在任务之间留出间隔。

对 `results/export` 中 `tests/tests.txt` 对应的全部现有结果直接批量总结，不重新抓取网页：

    uv run python -m tests.summarize_tests

批量生成带细节的版本：

    uv run python -m tests.summarize_tests --include-details

批量生成只含一个“总览”段落的极简版：

    uv run python -m tests.summarize_tests_simple

极简版复用 `results/summary` 中同 schema、同原文指纹的普通版 JSON，不重新抓取网页，
结果统一保存到 `results/summary_simple/源文件名_simple.md`。可通过 `--model` 临时指定
首选模型；批处理会先尝试该首选模型，再以 Gemini 3.6 Flash、3.5 Flash、
3.5 Flash Lite 补足回退链，配置了 SiliconFlow 密钥时再使用 Qwen 备用模型。只检查并确定性规范化现有
极简文件、不调用 API 时使用：

    uv run python -m tests.summarize_tests_simple --repair-existing

批量任务因限额或网络中断后续跑，并跳过已有同模型完整结果：

    uv run python -m tests.summarize_tests --include-details --model gemini-3.6-flash --resume

重新抓取 `tests/tests.txt` 中的全部链接，并在抓取完成后批量总结：

    uv run python -m tests.run_tests --summarize

只有显式使用 `scripts.parser --summarize`、`tests.run_tests --summarize`、
`scripts.summarize_memory`、`tests.summarize_tests` 或
`tests.summarize_tests_simple` 才会把对话和可用媒体发送给所选模型服务。
普通 `tests.run_tests` 仍只做网页抓取，不调用模型 API；
`tests.summarize_tests_simple --repair-existing` 也不调用 API。

## 目录结构

- `scripts/`：抓取、平台适配与三种总结的核心实现。
- `tests/`：单元测试、测试链接和三种批量验证入口。
- `results/export/`：当前抓取结果。
- `results/export_right/`：人工确认过的抓取黄金基线。
- `results/summary/`：普通版总结。
- `results/summary_detailed/`：详细版总结。
- `results/summary_simple/`：极简版总结。
- `工作文档/`：经验总结、非正式项目计划与临时指令。

## 当前媒体支持

- 本地图片：交给 Gemini 视觉理解。
- AI 回答中的 favicon、引用站点标识等装饰图片会在视觉分析前自动忽略；有明确内容的 AI 图片仍保留，并标注其来源角色。
- 本地 PDF：交给 Gemini 文档理解。
- 本地 TXT、CSV、Markdown、JSON、HTML：程序先提取文本。
- 只有文件名、下载失败或已失效的附件：明确写成“未能成功提取内容”。
- DOCX、XLSX、PPTX 当前若未被抓取到本地，不会猜测其内容。

## 测试

离线单元测试使用模拟网关，不读取任何 API 密钥，也不调用外部模型：

    python -m unittest discover -s tests -v

网页抓取完整回归（不调用模型 API）：

    uv run python -m tests.run_tests

批量总结现有测试结果（调用已配置模型）：

    uv run python -m tests.summarize_tests

批量生成详细版总结（调用已配置模型）：

    uv run python -m tests.summarize_tests --include-details

批量生成极简版总结（调用已配置模型）：

    uv run python -m tests.summarize_tests_simple

重新抓取并批量总结（调用已配置模型）：

    uv run python -m tests.run_tests --summarize

Gemini 官方资料：

- https://ai.google.dev/gemini-api/docs?hl=zh-cn
- https://ai.google.dev/gemini-api/docs/image-understanding?hl=zh-cn
- https://ai.google.dev/gemini-api/docs/document-processing?hl=zh-cn
- https://ai.google.dev/gemini-api/docs/structured-output?hl=zh-cn

SiliconFlow Chat Completions 资料：

- https://api-docs.siliconflow.cn/docs/api/chat-completions-post
