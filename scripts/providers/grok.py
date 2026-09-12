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


async def collect_html(page):
    """采集 Grok 会话片段；页面不属于 Grok 时返回 None。

    Grok 没有虚拟列表，消息初始即挂载；滚动仅为触发 lazy 图片。
    按文档顺序收集 ``.message-bubble[role="article"]`` 的 outerHTML。
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
                document.querySelectorAll('.message-bubble[role="article"]')
            ).map(el => el.outerHTML)"""
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
    """提取用户消息正文。"""
    md = bubble.select_one(".markdown")
    if md is None:
        return bubble.get_text("\n", strip=True).strip() or None
    for tag in _NOISE_TAGS:
        for el in md.find_all(tag):
            el.decompose()
    _localize_images(md, image_map)
    text = md.get_text("\n", strip=True).strip()
    return text or None


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
        strip=["img"],
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
    for bubble in bubbles:
        if not isinstance(bubble, Tag):
            continue
        try:
            if _is_user_bubble(bubble):
                content = _render_user(bubble, image_map)
                if content:
                    messages.append({"role": "User", "content": content})
            else:
                content = _render_ai(bubble, image_map)
                if content:
                    messages.append({"role": "AI", "content": content})
        except Exception:
            continue

    return messages or None
