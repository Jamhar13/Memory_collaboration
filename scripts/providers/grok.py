"""Grok（grok.com）页面采集与消息解析。

Grok 是 xAI 的对话产品，基于 Next.js SSR（React Server Components）
+ Tailwind CSS。据 2026-09 真机校准（有效分享链接实测）：

- 公开分享页 ``/share/<base64前缀>_<uuid>``：无需登录，但无头 Chromium
  裸启动时 hydration 失败（body 仅 "Skip to main content"，疑似 bot 检测）；
  使用持久化 profile + ~40s 等待后成功渲染。
- 私有会话 ``/c/<uuid>``：需登录。
- 每条消息是 ``div.message-bubble[role="article"]``；Tailwind class 区分：
  用户气泡含 ``bg-surface-l1`` + ``border-border-l1``，
  AI 气泡含 ``w-full max-w-none``（无背景边框）。
- AI 气泡正文在 ``.markdown`` 内，是语义化 HTML（真实 h/ul/ol/table 等）；
  气泡开头有 "工作了 Ns" 思考时间标注，需剥离。
- 表格是真实 ``<table class="table-card-row-dividers">``。
- 图片附件在分享页显示文件名文本而非 ``<img>``。
"""

import re
from collections.abc import Mapping

import markdownify
from bs4 import NavigableString, Tag
from rich.console import Console


DISPLAY_NAME = "Grok"
WAIT_SELECTOR = ".message-bubble[role='article']"
HOSTS = ("grok.com", "www.grok.com")

console = Console()

# 正文 markdown 之外的界面噪声（标签）。
_NOISE_TAGS = ("script", "style", "noscript", "button", "svg", "canvas", "input")
# "工作了 Ns" 思考时间标注前缀。
_THINK_TIME_RE = re.compile(r"^工作了\s*\d+\s*s\s*", re.IGNORECASE)
# 用户气泡 class 标志（有背景/边框 = 用户）。
_USER_CLASS_RE = re.compile(r"bg-surface|border-border")


def parse_api_messages(payload, image_map=None):
    """解析 Grok 接口返回的完整消息链。"""
    records = payload.get("responses", []) if isinstance(payload, Mapping) else []
    image_map = image_map or {}
    messages = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        role = "User" if record.get("sender") == "human" else "AI"
        chunks = record.get("inputChunks" if role == "User" else "outputChunks", [])
        texts = []
        for chunk in chunks or []:
            text = chunk.get("text") if isinstance(chunk, Mapping) else None
            if not isinstance(text, Mapping):
                continue
            if role == "AI" and text.get("channel") != "CHANNEL_ASSISTANT_RESPONSE":
                continue
            value = str(text.get("text") or "").strip()
            if value:
                texts.append(value)
        parts = []
        for item in record.get("fileAttachmentsMetadata", []) or []:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("fileName") or "").strip()
            uri = str(item.get("fileUri") or "").strip()
            remote = f"https://assets.grok.com/{uri}" if uri.startswith("users/") else ""
            local = (
                image_map.get(remote) or image_map.get(name.lower())
                or image_map.get(name, "")
            )
            mime_type = str(item.get("fileMimeType") or "").lower()
            if name and mime_type.startswith("image/"):
                parts.append(f"![{name}]({local or remote})" if local or remote else name)
            elif name:
                parts.append(f"📎 [{name}]({local})" if local else f"📎 **[上传文件]** `{name}`")
        parts.extend(texts)
        if not texts:
            message = str(record.get("message") or "").strip()
            if message and (role == "AI" or not parts):
                parts.append(message)
        if parts:
            messages.append({"role": role, "content": "\n\n".join(parts)})
    return messages or None


async def collect_html(page):
    """采集 Grok 会话 DOM，并补齐没有文字气泡的纯附件消息。"""
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
            """() => {
                const bubbles = Array.from(document.querySelectorAll(
                    '.message-bubble[role="article"]'
                ));
                const items = [];
                const seen = new Set();
                for (const bubble of bubbles) {
                    const container = bubble.closest('[id^="response-"]')
                        || bubble.parentElement;
                    seen.add(container);
                    items.push({
                        node: container,
                        html: container?.querySelector(
                            'button[aria-label="打开附件"]'
                        ) ? container.outerHTML : bubble.outerHTML,
                    });
                }
                for (const button of document.querySelectorAll(
                    'button[aria-label="打开附件"]'
                )) {
                    const container = button.closest('[id^="response-"]')
                        || button.parentElement;
                    if (seen.has(container)) continue;
                    seen.add(container);
                    const buttons = Array.from(container.querySelectorAll(
                        'button[aria-label="打开附件"]'
                    )).map(item => item.outerHTML).join('');
                    items.push({
                        node: container,
                        html: '<div class="message-bubble" role="article" '
                            + 'data-testid="user-message">'
                            + buttons + '</div>',
                    });
                }
                items.sort((a, b) => a.node.compareDocumentPosition(b.node) & 4 ? -1 : 1);
                return items.map(item => item.html);
            }"""
        )
    except Exception:
        return None

    fragments = [fragment for fragment in (fragments or []) if fragment]
    if not fragments:
        return None
    console.print(
        f"[dim]已从 Grok 会话中收集 {len(fragments)} 条消息。[/dim]"
    )
    return (
        "<!DOCTYPE html><html><body>"
        + "\n".join(fragments)
        + "</body></html>"
    )


def _is_user_bubble(bubble):
    """判断 .message-bubble 是用户还是 AI。"""
    test_id = bubble.get("data-testid")
    if test_id:
        return test_id == "user-message"
    classes = bubble.get("class") or []
    class_text = " ".join(classes) if isinstance(classes, list) else str(classes)
    return bool(_USER_CLASS_RE.search(class_text))


def _strip_noise(node) -> None:
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


def _clean_markdown(text):
    """清理 markdownify 输出的多余空行和噪声文本。"""
    # 剥离 "工作了 Ns" 前缀
    text = _THINK_TIME_RE.sub("", text)
    # 压缩 3+ 连续空行为 2 行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _render_user(bubble, image_map):
    """提取用户消息正文，并保留图片和文件附件文本。"""
    parts = []
    attachments = bubble.find_previous_sibling()
    if not attachments or not attachments.select_one('button[aria-label="打开附件"]'):
        attachments = bubble.parent
    if attachments:
        for button in attachments.select('button[aria-label="打开附件"]'):
            name = button.get_text(" ", strip=True)
            local = image_map.get(name.lower()) or image_map.get(name, "")
            img = button.find("img")
            if img:
                src = img.get("src") or ""
                original = src.rsplit("/", 1)[0] + "/content"
                href = image_map.get(original) or image_map.get(src, local)
                parts.append(f"![{name}]({href})" if href else name)
            elif local:
                parts.append(f"📎 [{name}]({local})")
            else:
                parts.append(f"📎 **[上传文件]** `{name}`")
    md = bubble.select_one(".markdown")
    if md is None:
        text = bubble.get_text("\n", strip=True).strip()
    else:
        _strip_noise(md)
        _localize_images(md, image_map)
        text = markdownify.markdownify(str(md), heading_style="ATX").strip()
    if text:
        parts.append(text)
    return "\n\n".join(parts) or None


def _render_ai(bubble, image_map):
    """提取 AI 回答正文，转为 markdown。"""
    md = bubble.select_one(".markdown")
    if md is None:
        return bubble.get_text("\n", strip=True).strip() or None
    _strip_noise(md)
    _localize_images(md, image_map)
    text = markdownify.markdownify(
        str(md),
        heading_style="ATX",
    )
    text = _clean_markdown(text)
    return text or None


def parse_messages(soup, image_map=None):
    """解析 Grok 消息；页面不属于 Grok 时返回 None。

    :param soup: BeautifulSoup 对象
    :param image_map: 可选的 ``{remote_url: local_path}`` 映射
    :return: ``[{"role": "User"|"AI", "content": "..."}]`` 或 None
    """
    bubbles = soup.select(".message-bubble[role='article']")
    if not bubbles:
        return None

    image_map = image_map or {}
    messages = []
    has_user_classes = any(_is_user_bubble(bubble) for bubble in bubbles)
    alternate_roles = not has_user_classes and len(bubbles) > 1
    for index, bubble in enumerate(bubbles):
        if not isinstance(bubble, Tag):
            continue
        try:
            if _is_user_bubble(bubble) or (alternate_roles and index % 2 == 0):
                content = _render_user(bubble, image_map)
                if content:
                    messages.append({"role": "User", "content": content})
            else:
                content = _render_ai(bubble, image_map)
                if content:
                    messages.append({"role": "AI", "content": content})
        except Exception:
            continue

    paired = []
    for message in messages:
        if paired and message["role"] == paired[-1]["role"]:
            paired[-1]["content"] += "\n\n" + message["content"]
        else:
            paired.append(message)
    return paired or None
