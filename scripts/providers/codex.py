"""ChatGPT Codex 公有分享页与私有任务页解析。"""

import re

import markdownify
from bs4 import Tag
from rich.console import Console


DISPLAY_NAME = "Codex"
WAIT_SELECTOR = "section[data-turn-key] article[id^='message-']"
# Codex 与 ChatGPT 共用域名，按路径选择等待目标、按 DOM 识别正文。
HOSTS = ()
PUBLIC_PATH_PATTERN = re.compile(r"^/s/cx_[0-9A-Za-z_-]{8,}/?$")
PRIVATE_PATH_PATTERN = re.compile(r"^/codex/cloud/tasks/task_[0-9A-Za-z_-]{8,}/?$")


def is_codex_path(path):
    return bool(PUBLIC_PATH_PATTERN.match(path) or PRIVATE_PATH_PATTERN.match(path))

console = Console()
_NOISE_TAGS = ("script", "style", "noscript", "button", "svg", "canvas", "input")


async def collect_html(page):
    """只采集 Codex 消息区，避免页头和文件变更摘要混入正文。"""
    turns = page.locator(WAIT_SELECTOR)
    try:
        count = await turns.count()
    except Exception:
        return None
    if not count:
        return None

    for index in range(count):
        try:
            await turns.nth(index).scroll_into_view_if_needed(timeout=2500)
        except Exception:
            continue
    try:
        fragments = await turns.evaluate_all("elements => elements.map(el => el.outerHTML)")
    except Exception:
        return None
    fragments = [fragment for fragment in (fragments or []) if fragment]
    if not fragments:
        return None
    console.print(f"[dim]已从 Codex 会话中收集 {len(fragments)} 条消息。[/dim]")
    return "<!DOCTYPE html><html><body>" + "\n".join(fragments) + "</body></html>"


def _localize_assets(node, asset_map):
    for image in node.find_all("img"):
        src = image.get("src") or image.get("data-src")
        if src and src in asset_map:
            image["src"] = asset_map[src]
    for link in node.find_all("a", href=True):
        href = link["href"]
        name = link.get_text(" ", strip=True).lower()
        link["href"] = asset_map.get(href, asset_map.get(name, href))


def _render(article, asset_map):
    root = article.select_one("[data-selected-text-overlay-target]") or article
    _localize_assets(root, asset_map)
    for tag in _NOISE_TAGS:
        for element in root.find_all(tag):
            element.decompose()
    text = markdownify.markdownify(str(root), heading_style="ATX")
    return re.sub(r"\n{3,}", "\n\n", text).strip() or None


def parse_messages(soup, image_map=None):
    """解析 Codex 消息；页面不属于 Codex 时返回 None。"""
    articles = soup.select("section[data-turn-key] article[id^='message-']")
    if not articles:
        # collect_html 会去掉 section 包装，保留稳定的消息 id 与用户标志。
        articles = soup.select("article[id^='message-']")
    if not articles or not any(article.select_one("[data-user-message-bubble]") for article in articles):
        return None

    asset_map = image_map or {}
    messages = []
    for article in articles:
        if not isinstance(article, Tag):
            continue
        content = _render(article, asset_map)
        if not content:
            continue
        role = "User" if article.select_one("[data-user-message-bubble]") else "AI"
        messages.append({"role": role, "content": content})
    return messages or None
