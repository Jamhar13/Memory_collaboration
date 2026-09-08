"""Gemini（gemini.google.com）页面采集与消息解析。

Gemini 网页是 Angular 应用，会话用自定义元素承载：用户提问在
``<user-query>``，模型回答在 ``<message-content>``（旧版偶见
``<model-response>`` 包装）。这些标签名比混淆的 class 稳定，因此优先按
标签识别，class 仅作容器兜底。

- 公开分享页 ``/share/<id>``（短链 ``share.gemini.google/<id>`` 跳转而来）：
  无需登录，整段对话渲染在页面中。
- 账号内会话 ``/app/<id>``：需登录，旧消息滚动懒加载。

真实 DOM 要点（据 2026-09 真实分享页校准）：
- 用户正文在 ``.query-text`` 内，另有 ``.cdk-visually-hidden`` 的屏幕阅读器
  节点（含“你说”前缀），解析前必须剔除；
- 用户上传的文档在 ``<user-query-file-preview>`` 内（``.filename-label`` /
  ``.extension-label``），分享页按钮 aria-label 明确“无法查看或下载共享对话
  中的文件”，按不可用文档占位处理；图片型附件 aria 为“灯箱/图片”，保留其
  ``<img>`` 走通用图片本地化；
- AI 正文在 ``<message-content>`` 内的 ``.markdown.markdown-main-panel``，
  引用角标是大量空的 ``<response-element>`` / ``<source-footnote>`` /
  ``<sources-carousel-inline>``，属界面噪声；
- 代码块是 ``<code-block>``，语言在头部 ``.code-block-decoration`` 的
  ``<span>``，正文在内部 ``<pre><code>``；
- 公式节点带 ``data-math``（LaTeX），行内 class 含 ``math-inline``、块级含
  ``math-block``；
- 装饰图来自 ``gstatic.com`` / ``drive-thirdparty.../type/``，真实内容图在
  ``lh3.googleusercontent.com``。

解析严格防御：页面不属于 Gemini（缺少自定义元素）时返回 ``None``，
任何意外 DOM 都降级跳过，不抛异常。
"""

import re

import markdownify


DISPLAY_NAME = "Gemini"

# 用户提问与模型回答的自定义元素标签。
USER_TAG = "user-query"
RESPONSE_TAGS = ("message-content", "model-response")
TURN_TAGS = [USER_TAG, *RESPONSE_TAGS]
WAIT_SELECTOR = "user-query, message-content, model-response"
# share.gemini.google/<id> 短链会 302 跳转到 gemini.google.com/share/...；
# 两个域名都登记，跳转前后都能锁定本适配器。
HOSTS = ("gemini.google.com", "share.gemini.google")

# 会话容器候选（从具体到宽泛）；找不到则回退到整页。
_CONTAINER_SELECTORS = (
    "conversation-view",
    "chat-window",
    "[class*='conversation-view']",
    "[class*='conversation-container']",
    "main",
    "[role='main']",
)
# 不属于对话正文的界面节点（标签）。
# 注意：<response-element class="no-md"> 是富内容内联包装器，既包裹空的引用
# 角标，也包裹 <code-block> 等正文，不能整体删除（只在 _strip_noise 中
# unwrap 去标签留内容），故不列入此 decompose 名单。
_NOISE_TAGS = (
    "script", "style", "noscript", "button", "svg", "input", "textarea",
    # Gemini 自定义界面元素：引用角标/来源卡片/操作按钮/图标。
    "source-footnote", "sources-carousel-inline",
    "source-inline-chip", "message-actions", "gem-icon-button", "gem-icon",
    "mat-icon",
)
_NOISE_SELECTORS = (
    "input-area",
    "[class*='input-area']",
    "side-nav",
    "nav",
    "aside",
    "[class*='sidebar']",
    "[class*='side-nav']",
    "[class*='share-header']",
    "[class*='conversation-header']",
    "[class*='message-actions']",
    "[class*='feedback']",
    "[class*='disclaimer']",
    ".cdk-visually-hidden",
    "[class*='screen-reader']",
)
# 装饰图来源（logo、文件类型小图标、内联 SVG 公式占位），不进入正文。
_DECORATIVE_IMAGE_HOSTS = ("gstatic.com", "drive-thirdparty.googleusercontent.com")


async def collect_html(page):
    """采集 Gemini 会话片段；页面不属于 Gemini 时返回 None。

    逐轮滚入视口触发懒加载，再按文档顺序收集自定义元素的 outerHTML，
    拼装成合成文档，避免侧栏/输入框混入正文。
    """
    turns = page.locator(WAIT_SELECTOR)
    try:
        count = await turns.count()
    except Exception:
        return None
    if not count:
        return None

    # 两轮扫描：首轮滚入会挂载更早的懒加载消息，第二轮稳定顺序与数量。
    for _ in range(2):
        try:
            current = await turns.count()
        except Exception:
            current = 0
        for index in range(current):
            try:
                await turns.nth(index).scroll_into_view_if_needed(timeout=2500)
                await page.wait_for_timeout(120)
            except Exception:
                continue
        await page.wait_for_timeout(600)

    try:
        fragments = await page.evaluate(
            """() => {
                const nodes = Array.from(
                    document.querySelectorAll('user-query, message-content, model-response')
                );
                // 跳过外层响应包装（其内部还含 message-content/model-response），
                // 只保留最内层回答节点，避免同一条回答重复采集。
                const picked = nodes.filter(el => {
                    if (el.tagName.toLowerCase() === 'user-query') return true;
                    return !el.querySelector('message-content, model-response');
                });
                return picked.map((el, index) => {
                    const clone = el.cloneNode(true);
                    const isUser = el.tagName.toLowerCase() === 'user-query';
                    clone.setAttribute('data-gemini-role', isUser ? 'user' : 'assistant');
                    clone.setAttribute('data-gemini-order', String(index));
                    return clone.outerHTML;
                });
            }"""
        )
    except Exception:
        return None

    fragments = [fragment for fragment in (fragments or []) if fragment]
    if not fragments:
        return None
    return (
        "<!DOCTYPE html><html><body>"
        + "\n".join(fragments)
        + "</body></html>"
    )


def _conversation_container(soup):
    for selector in _CONTAINER_SELECTORS:
        node = soup.select_one(selector)
        if node is not None and node.select_one(WAIT_SELECTOR):
            return node
    return soup


def _strip_noise(node) -> None:
    # response-element 是富内容包装器：内部可能是 <code-block> 等正文，也可能
    # 是空的引用角标。先 unwrap 去标签保留内容，再 decompose 纯噪声标签。
    for element in node.find_all("response-element"):
        element.unwrap()
    for tag in _NOISE_TAGS:
        for element in node.find_all(tag):
            element.decompose()
    for selector in _NOISE_SELECTORS:
        for element in node.select(selector):
            element.decompose()


def _is_decorative_image(src) -> bool:
    src = str(src or "")
    if not src:
        return True
    if src.startswith("data:"):
        # 内联 base64/SVG（公式渲染、占位）体积大且无法本地引用，跳过。
        return True
    lowered = src.lower()
    return any(host in lowered for host in _DECORATIVE_IMAGE_HOSTS)


def _strip_decorative_images(node) -> None:
    for img in list(node.find_all("img")):
        src = img.get("src") or img.get("data-src")
        if _is_decorative_image(src):
            img.decompose()


def _clean_latex(value) -> str:
    latex = str(value or "").strip()
    for wrapper in (r"\(", r"\)", r"\[", r"\]"):
        latex = latex.replace(wrapper, "")
    return latex.strip().strip("$").strip()


def _extract_math(node):
    """把带 data-math 的公式节点替换为占位符，返回 {token: markdown}。"""
    replacements = {}
    counter = 0
    for element in list(node.find_all(attrs={"data-math": True})):
        latex = _clean_latex(element.get("data-math"))
        if not latex:
            continue
        classes = element.get("class") or []
        if "math-block" in classes:
            display = True
        elif "math-inline" in classes:
            display = False
        else:
            # 无明确 class 时：span 视为行内，其余视为块级。
            display = element.name != "span"
        token = f"AIMEMORYMATHPLACEHOLDER{counter:04d}Z"
        counter += 1
        replacements[token] = (
            f"\n\n$$\n{latex}\n$$\n\n" if display else f"${latex}$"
        )
        element.replace_with(token)
    return replacements


def _restore_math(text, replacements) -> str:
    for token, latex in replacements.items():
        text = text.replace(token, latex)
    return text


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


def _normalize_code_blocks(node) -> None:
    """把 <code-block> 归一为 <pre><code>，并把头部语言标注注入 code class。"""
    for code_block in list(node.find_all("code-block")):
        pre = code_block.find("pre")
        if pre is None:
            # 没有正文 pre 的 code-block 直接移除（纯界面壳）。
            code_block.decompose()
            continue
        lang_node = code_block.select_one(
            "[class*='code-block-decoration'] span, [class*='code-header'] span"
        )
        lang = (lang_node.get_text(strip=True) if lang_node else "").strip()
        code = pre.find("code")
        if lang and code is not None:
            existing = code.get("class", []) or []
            token = f"language-{lang}"
            if not any(cls.startswith("language-") for cls in existing):
                code["class"] = [*existing, token]
        code_block.replace_with(pre)


def _localize_images(node, image_map) -> None:
    """命中 image_map 的图片替换为本地相对路径。"""
    for img in node.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and src in image_map:
            img["src"] = image_map[src]
            if img.has_attr("data-src"):
                del img["data-src"]


def _extract_document_attachments(node):
    """提取用户上传的文档附件为不可用占位，返回 [markdown, ...]。

    分享页不提供文档下载（按钮 aria-label 为“无法查看或下载共享对话中的
    文件”），按项目约定标记为不可用；图片型附件（aria 含“图片/灯箱”）保留
    其 <img>，不在此处理。
    """
    placeholders = []
    for card in list(node.find_all("user-query-file-preview")):
        button = card.find("button")
        aria = str(
            (button.get("aria-label") if button else "")
            or card.get("aria-label") or ""
        )
        if "图片" in aria or "灯箱" in aria or "image" in aria.lower():
            # 图片型附件：<img> 嵌在灯箱 <button> 内，而 button 稍后会被当作
            # 界面噪声清理。先把 img 提升到卡片位置，避免被一并删除。
            images = card.find_all("img")
            for img in images:
                img.extract()
                card.insert_before(img)
            card.decompose()
            continue
        name_node = card.select_one(".filename-label")
        ext_node = card.select_one(".extension-label")
        name = name_node.get_text(strip=True) if name_node else ""
        ext = ext_node.get_text(strip=True) if ext_node else ""
        if name:
            filename = name
            if ext and not re.search(
                rf"\.{re.escape(ext)}$", name, flags=re.IGNORECASE
            ):
                filename = f"{name}.{ext.lower()}"
            placeholders.append(f"📎 **[上传文档]** `{filename}`")
        card.decompose()
    return placeholders


def _render_user(node, image_map) -> str:
    attachments = _extract_document_attachments(node)
    _strip_noise(node)
    _strip_decorative_images(node)
    _localize_images(node, image_map)

    parts = list(attachments)
    local_paths = set(image_map.values())
    for img in node.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if not src or _is_decorative_image(src):
            continue
        alt = (img.get("alt") or "用户图片").strip() or "用户图片"
        if src in local_paths:
            parts.append(f"![{alt}]({src})")
        elif src.startswith(("http://", "https://")):
            # 通用下载链路理论上已下载；兜底保留远程地址，避免丢图。
            parts.append(f"![{alt}]({src})")

    text = node.get_text(separator="\n", strip=True)
    clean_lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip() and line.strip().lower() not in _NOISE_TEXT
    ]
    if clean_lines:
        parts.append("\n".join(clean_lines))

    # 去重并保持顺序。
    seen = set()
    ordered = []
    for item in parts:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return "\n\n".join(ordered).strip()


def _render_assistant(node, image_map) -> str:
    _normalize_code_blocks(node)
    _strip_noise(node)
    math_replacements = _extract_math(node)
    _strip_decorative_images(node)
    _localize_images(node, image_map)

    markdown = markdownify.markdownify(
        str(node),
        heading_style="ATX",
        code_language_callback=_code_language,
    )
    markdown = _restore_math(markdown, math_replacements)

    cleaned_lines = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped and stripped.lower() in _NOISE_TEXT:
            continue
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


# 界面噪声关键词（Gemini 中文/英文按钮）。
_NOISE_TEXT = {
    "复制", "重新生成", "分享", "点赞", "好回答", "踩", "坏回答",
    "编辑", "朗读", "更多", "查看更多", "复制提示", "下载代码",
    "good response", "bad response", "copy", "share", "regenerate", "edit",
}


def parse_messages(soup, image_map=None):
    """解析 Gemini 消息；页面不属于 Gemini 时返回 None。"""
    if image_map is None:
        image_map = {}

    if soup.select_one(WAIT_SELECTOR) is None:
        return None

    # 注意：不在此处统一 _strip_noise——用户文档/图片附件的区分依赖
    # user-query-file-preview 内 <button> 的 aria-label，需在
    # _render_user 的附件提取之后再清理界面节点。
    container = _conversation_container(soup)

    parsed_messages = []
    for node in container.find_all(TURN_TAGS):
        tag = str(node.name or "").lower()
        if tag in RESPONSE_TAGS:
            # 跳过外层响应包装：内部还含回答节点时，交给内层节点处理，
            # 避免同一条 AI 回复被采集两次。
            if node.find(list(RESPONSE_TAGS)):
                continue
            content = _render_assistant(node, image_map)
            if content:
                parsed_messages.append({"role": "AI", "content": content})
        elif tag == USER_TAG:
            content = _render_user(node, image_map)
            if content:
                parsed_messages.append({"role": "User", "content": content})

    return parsed_messages or None
