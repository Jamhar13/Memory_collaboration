"""Kimi（kimi.com）页面采集与消息解析。

Kimi 网页是 Vue 应用（scoped class）。据 2026-09 对 57 个真实公开分享链接
的 Playwright 无头校准：

- 公开分享页 ``/share/[<lang>/]<id>``（id 为 base36 短 token 或 UUID，可带
  ``?ra=1`` / ``?sharetype=link``）：无需登录，整段对话直接渲染在 DOM 中，
  **没有虚拟列表**（14 轮长对话滚动前 28 段即全部挂载），文档顺序严格
  用户/AI 交替；
- 账号内私有会话 ``/chat/<id>``：需登录，未登录会被重定向到首页，登录后
  与分享页是同一套组件；
- 旧域名 ``kimi.moonshot.cn`` 跳转到 ``www.kimi.com``，图片签名资产仍在
  ``kimi.moonshot.cn/api/sign-obj/...``。

真实 DOM 要点：

- 每条消息是 ``div.segment.segment-user`` / ``div.segment.segment-assistant``；
- 用户正文在 ``.user-content``（被 ``.toolcall-rollup__flat`` 包裹，即使纯
  文本也有这层包装，**不能把 rollup 整体当工具噪声删除**）；附件在
  ``.attachment-list``：图片是 ``.attachment-list-image img.image-main``，
  文件是 ``.attachment-list-file``（分享页不提供下载，按不可用占位处理）；
- AI 正文在 ``.markdown-container .markdown``，是语义化 HTML（真实 h/ul/ol/
  blockquote/hr/strong），代码卡片 ``.segment-code`` 内
  ``pre.language-xxx > code.language-xxx``（无行号，Prism token span 不影响
  get_text），表格是真实 ``<table>``（外包 ``div.table.markdown-table``，
  前置 ``.table-actions``/``.sticky-release`` 界面壳）；
- K3 智能体的思考/读文件/跑代码等中间轨迹在 ``.toolcall-flow`` /
  ``.toolcall-rollup-segment`` / ``.terminal-detail`` 内（其中也各有
  markdown-container），归档时只取**不被这些轨迹容器包裹**的
  markdown-container；若一个都没有（极端情况），降级取段内最后一个
  markdown-container，保证不丢内容；
- ``.okc-cards-container`` 是引用卡空插槽，按噪声剔除。

注意：旧版开源实现依赖的 ``window.HYDRATION_INIT_STATE`` 内联 JSON 在当前
版本已不存在（``requests`` 直连只返回官网 SEO 空壳），必须走浏览器渲染 DOM。

解析严格防御：页面不属于 Kimi（缺少 .segment-user/.segment-assistant）时
返回 ``None``，任何意外 DOM 都降级跳过，不抛异常。
"""

import re

import markdownify
from bs4 import NavigableString
from rich.console import Console


DISPLAY_NAME = "Kimi"
WAIT_SELECTOR = ".segment-user, .segment-assistant"
# kimi.moonshot.cn 为旧域名（跳转至 www）；签名图片也在该域名下。
HOSTS = ("kimi.com", "www.kimi.com", "kimi.moonshot.cn")

console = Console()

# 正文 markdown 之外的界面噪声（标签）。
_NOISE_TAGS = ("script", "style", "noscript", "button", "svg", "canvas", "input")
_NOISE_SELECTORS = (
    ".okc-cards-container",
    ".segment-code-header",
    ".table-actions",
    ".sticky-release",
    ".icon-button-tooltip",
    ".kimi-tooltip",
)
_NOISE_TEXT = {
    "复制", "重新生成", "点赞", "踩", "分享", "收起工具调用", "展开工具调用",
    "copy", "regenerate",
}
_FILE_SIZE_PATTERN = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|TB)$", re.IGNORECASE
)


async def collect_html(page):
    """采集 Kimi 会话片段；页面不属于 Kimi 时返回 None。

    Kimi 没有虚拟列表，所有 .segment 初始即挂载；滚动仅为触发
    ``loading=lazy`` 图片。两轮把每段滚入视口后，按文档顺序收集
    .segment 的 outerHTML，拼装成合成文档，排除侧栏/输入框。
    """
    turns = page.locator(WAIT_SELECTOR)
    try:
        count = await turns.count()
    except Exception:
        return None
    if not count:
        return None

    for _ in range(2):
        try:
            current = await turns.count()
        except Exception:
            current = 0
        for index in range(current):
            try:
                await turns.nth(index).scroll_into_view_if_needed(timeout=2500)
                await page.wait_for_timeout(100)
            except Exception:
                continue
        await page.wait_for_timeout(500)

    try:
        fragments = await page.evaluate(
            """() => Array.from(
                document.querySelectorAll('.segment-user, .segment-assistant')
            ).map(el => el.outerHTML)"""
        )
    except Exception:
        return None

    fragments = [fragment for fragment in (fragments or []) if fragment]
    if not fragments:
        return None
    console.print(
        f"[dim]已从 Kimi 会话中收集 {len(fragments)} 条消息。[/dim]"
    )
    return (
        "<!DOCTYPE html><html><body>"
        + "\n".join(fragments)
        + "</body></html>"
    )


def _is_trace_node(node, segment) -> bool:
    """node 是否为 K3 智能体中间轨迹容器的后代（不含正式答案）。"""
    current = node.parent
    while current is not None and current is not segment:
        classes = current.get("class") if hasattr(current, "get") else None
        if classes:
            class_text = (
                " ".join(classes)
                if isinstance(classes, list)
                else str(classes)
            )
            if re.search(
                r"toolcall-flow|toolcall-rollup-segment|terminal-detail",
                class_text,
            ):
                return True
        current = current.parent
    return False


def _answer_containers(segment):
    """返回 AI 段内属于正式答案的 markdown-container（文档顺序）。

    排除 K3 智能体中间轨迹；轨迹与答案并存时只留正式答案；若所有
    markdown-container 都在轨迹内（极端情况），降级取最后一个，避免
    整条回答丢失。
    """
    containers = segment.select(".markdown-container")
    official = [
        node for node in containers if not _is_trace_node(node, segment)
    ]
    return official or containers[-1:]


def _strip_noise(node) -> None:
    for selector in _NOISE_SELECTORS:
        for element in node.select(selector):
            element.decompose()
    for tag in _NOISE_TAGS:
        for element in node.find_all(tag):
            element.decompose()


def _localize_images(node, image_map) -> None:
    for img in node.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and src in image_map:
            img["src"] = image_map[src]
            if img.has_attr("data-src"):
                del img["data-src"]


def _normalize_code_cards(node) -> None:
    """代码卡片去工具栏；pre 缺 language- 时用头部语言标注补齐。"""
    for card in list(node.select(".segment-code")):
        pre = card.find("pre")
        if pre is None:
            card.decompose()
            continue
        code = pre.find("code")
        target = code if code is not None else pre

        has_lang = any(
            str(cls).startswith("language-")
            for cls in (target.get("class") or [])
        )
        if not has_lang:
            lang_node = card.select_one(".segment-code-lang")
            lang = (
                lang_node.get_text(strip=True)
                if lang_node is not None
                else ""
            ).strip().lower()
            if lang and re.fullmatch(r"[a-z0-9#+]{1,20}", lang):
                target["class"] = [
                    *(target.get("class") or []),
                    f"language-{lang}",
                ]
        for junk in card.select(
            ".segment-code-header, button, svg, .icon-button, .kimi-tooltip"
        ):
            junk.decompose()


def _cell_text(cell) -> str:
    text = cell.get_text(" ", strip=True) if cell is not None else ""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _table_to_gfm(table) -> str:
    """真实 <table> 转 GFM 管线表；无法识别表头时返回空串。"""
    rows = table.find_all("tr")
    if not rows:
        return ""

    def cells_of(row):
        return row.find_all(["th", "td"])

    thead = table.find("thead")
    if thead is not None and thead.find_all("th"):
        header_cells = thead.find_all("th")
        tbody = table.find("tbody")
        body_rows = tbody.find_all("tr") if tbody is not None else []
    else:
        header_cells = cells_of(rows[0])
        body_rows = rows[1:]

    if not header_cells:
        return ""

    def align_of(cell):
        align = (cell.get("align") or "").lower()
        if align == "center":
            return ":---:"
        if align == "right":
            return "---:"
        return ":---" if align == "left" else "---"

    aligns = [align_of(cell) for cell in header_cells]
    lines = [
        "| " + " | ".join(_cell_text(cell) for cell in header_cells) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in body_rows:
        values = [_cell_text(cell) for cell in cells_of(row)]
        if len(values) < len(header_cells):
            values += [""] * (len(header_cells) - len(values))
        lines.append(
            "| " + " | ".join(values[:len(header_cells)]) + " |"
        )
    return "\n".join(lines)


def _convert_tables(root) -> None:
    """把表格（含 div.table 包装）原地替换为 GFM 文本节点。"""
    converted = set()
    for wrapper in list(root.select("div.table, div.markdown-table")):
        table = wrapper.find("table")
        if table is None:
            continue
        converted.add(id(table))
        gfm = _table_to_gfm(table)
        if gfm:
            wrapper.replace_with(NavigableString("\n\n" + gfm + "\n\n"))
        else:
            wrapper.decompose()
    for table in list(root.find_all("table")):
        if id(table) in converted:
            continue
        gfm = _table_to_gfm(table)
        if gfm:
            table.replace_with(NavigableString("\n\n" + gfm + "\n\n"))


def _code_language(pre) -> str:
    """markdownify 回调：从 <pre><code class="language-xxx"> 提取语言。"""
    code = pre.find("code") if pre is not None else None
    classes = []
    if code is not None:
        classes = code.get("class", []) or []
    elif pre is not None:
        classes = pre.get("class", []) or []
    for cls in classes:
        if cls.startswith("language-"):
            return cls.split("language-", 1)[1]
    return ""


def _clean_markdown(text) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped.lower() in _NOISE_TEXT:
            continue
        lines.append(line.rstrip())
    markdown = "\n".join(lines)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def _render_assistant(segment, image_map) -> str:
    containers = _answer_containers(segment)
    parts = []
    for container in list(containers):
        _normalize_code_cards(container)
        _convert_tables(container)
        _strip_noise(container)
        _localize_images(container, image_map)
        markdown = markdownify.markdownify(
            str(container),
            heading_style="ATX",
            code_language_callback=_code_language,
        )
        markdown = _clean_markdown(markdown)
        if markdown:
            parts.append(markdown)
    return "\n\n".join(parts).strip()


def _parse_file_attachment(card):
    """从 .attachment-list-file 提取 (文件名, 大小文本)；失败返回 (None, None)。"""
    lines = [
        line.strip()
        for line in card.get_text("\n", strip=True).split("\n")
        if line.strip()
    ]
    size = ""
    ext = ""
    name_candidates = []
    for line in lines:
        if not size and _FILE_SIZE_PATTERN.match(line):
            size = line.replace(" ", "")
        elif re.fullmatch(r"[A-Za-z0-9]{1,8}", line) and line.isupper():
            ext = line.lower()
        else:
            name_candidates.append(line)
    name = name_candidates[0] if name_candidates else ""
    if not name:
        return None, None
    if ext and not re.search(
        rf"\.{re.escape(ext)}$", name, flags=re.IGNORECASE
    ):
        name = f"{name}.{ext}"
    return name, (size or None)


def _render_user(segment, image_map) -> str:
    parts = []

    # 图片附件（.file-card-icon 等 data:svg 图标天然不以 http 开头，会被跳过）。
    for img in segment.select(
        ".attachment-list-image img, .attachment-list img"
    ):
        src = img.get("src") or img.get("data-src") or ""
        if not src.startswith(("http://", "https://")):
            continue
        local_src = image_map.get(src, src)
        alt = (img.get("alt") or "用户图片").strip() or "用户图片"
        parts.append(f"![{alt}]({local_src})")

    # 文件附件：分享页不提供下载，按项目约定标记为不可用。
    for card in segment.select(".attachment-list-file"):
        name, size = _parse_file_attachment(card)
        if not name:
            continue
        suffix = f"（{size}）" if size else ""
        parts.append(f"📎 **[上传文件]** `{name}`{suffix}")

    # 正文（.user-content 在 rollup__flat 内；取最后一个非空文本，兼容多段）。
    text_nodes = segment.select(".user-content")
    texts = []
    for node in text_nodes:
        for clone_tag in node.select("button, svg, canvas, script, style"):
            clone_tag.decompose()
        text = node.get_text("\n", strip=True)
        clean_lines = [
            line.strip()
            for line in text.split("\n")
            if line.strip() and line.strip().lower() not in _NOISE_TEXT
        ]
        if clean_lines:
            texts.append("\n".join(clean_lines))
    if texts:
        parts.append("\n\n".join(texts))

    # 去重并保持顺序。
    seen = set()
    ordered = []
    for item in parts:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return "\n\n".join(ordered).strip()


def parse_messages(soup, image_map=None):
    """解析 Kimi 消息；页面不属于 Kimi 时返回 None。"""
    if image_map is None:
        image_map = {}
    if soup.select_one(WAIT_SELECTOR) is None:
        return None

    parsed_messages = []
    for node in soup.select(".segment-user, .segment-assistant"):
        classes = node.get("class") or []
        try:
            if "segment-assistant" in classes:
                content = _render_assistant(node, image_map)
                if content:
                    parsed_messages.append({"role": "AI", "content": content})
            elif "segment-user" in classes:
                content = _render_user(node, image_map)
                if content:
                    parsed_messages.append({"role": "User", "content": content})
        except Exception:
            # 单段解析失败不影响其他消息（DOM 可能随版本变化）。
            continue

    return parsed_messages or None
