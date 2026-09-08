"""豆包页面消息解析。"""

import re
from collections import Counter
from urllib.parse import unquote

import markdownify


DISPLAY_NAME = "豆包"
SHARE_MESSAGE_SELECTOR = ".message-item"
DIRECT_LIST_SELECTOR = "div[class*='message-list-']"
DIRECT_MESSAGE_SELECTOR = "div.my-0.w-full.mx-auto"
WAIT_SELECTOR = f"{SHARE_MESSAGE_SELECTOR}, {DIRECT_LIST_SELECTOR}"
HOSTS = ("doubao.com", "www.doubao.com")


async def _scroll_messages(page, messages, message_count):
    for index in range(message_count):
        try:
            await messages.nth(index).scroll_into_view_if_needed(timeout=3000)
            # 豆包在消息进入视野后异步挂载 picture/img；短暂停留可避免
            # 长对话滚动过快时图片节点尚未生成就读取页面快照。
            await page.wait_for_timeout(120)
        except Exception:
            continue
    await page.wait_for_timeout(700)


async def collect_html(page):
    """仅收集当前豆包会话，兼容分享页与账号内 `/chat/` 页面。"""
    share_messages = page.locator(SHARE_MESSAGE_SELECTOR)
    try:
        share_count = await share_messages.count()
    except Exception:
        return None
    if share_count:
        await _scroll_messages(page, share_messages, share_count)
        return await page.content()

    try:
        lists = page.locator(DIRECT_LIST_SELECTOR)
        list_stats = await lists.evaluate_all(
            """elements => elements.map((element, index) => {
                const messages = Array.from(
                    element.querySelectorAll('div.my-0.w-full.mx-auto')
                ).filter(wrapper => wrapper.querySelector(
                    'div.flex.flex-row.w-full'
                ));
                return {
                    index,
                    count: messages.length,
                    textLength: messages.reduce(
                        (total, message) => total + (message.innerText || '').length,
                        0
                    )
                };
            })"""
        )
    except Exception:
        return None
    usable_lists = [item for item in list_stats if item.get("count", 0) > 0]
    if not usable_lists:
        return None
    selected = max(
        usable_lists,
        key=lambda item: (item.get("count", 0), item.get("textLength", 0)),
    )
    direct_messages = lists.nth(selected["index"]).locator(
        DIRECT_MESSAGE_SELECTOR
    )
    direct_count = await direct_messages.count()
    await _scroll_messages(page, direct_messages, direct_count)

    message_fragments = await direct_messages.evaluate_all(
        """elements => elements.map((wrapper, index) => {
            const roleRows = Array.from(wrapper.querySelectorAll(
                'div.flex.flex-row.w-full'
            ));
            if (!roleRows.length) return null;
            const isUser = roleRows.some(row => row.classList.contains('justify-end'));
            const hasContent = Boolean(
                (wrapper.innerText || '').trim()
                || wrapper.querySelector('img, picture, [class*="file"], [class*="doc"]')
            );
            if (!hasContent) return null;
            const clone = wrapper.cloneNode(true);
            clone.classList.add('message-item');
            clone.setAttribute('data-doubao-role', isUser ? 'user' : 'assistant');
            clone.setAttribute('data-doubao-order', String(index));
            return clone.outerHTML;
        }).filter(Boolean)"""
    )
    if not message_fragments:
        return None
    return (
        "<!DOCTYPE html><html><body>"
        + "\n".join(message_fragments)
        + "</body></html>"
    )


def _is_empty_svg_placeholder(img) -> bool:
    """只识别豆包懒加载产生的空 SVG，不误删含实际图形的 SVG。"""
    src = str(img.get("src") or "")
    if not src.lower().startswith("data:image/svg+xml,"):
        return False
    try:
        svg = unquote(src.split(",", 1)[1]).strip()
    except (IndexError, ValueError):
        return False
    return bool(re.fullmatch(
        r"<svg\b[^>]*(?:/\s*>|>\s*</svg\s*>)",
        svg,
        flags=re.IGNORECASE | re.DOTALL,
    ))


def _remove_assistant_image_artifacts(msg) -> None:
    """移除豆包 AI 消息中的空占位图和重复界面图标。"""
    for img in list(msg.find_all("img")):
        src = str(img.get("src") or img.get("data-src") or "")
        alt = str(img.get("alt") or "").strip().lower()
        if (
            _is_empty_svg_placeholder(img)
            or alt == "asset cover"
            or "doc-canvas-card-fallback" in src.lower()
        ):
            img.decompose()

    images = list(msg.find_all("img"))
    reference_counts = Counter(
        str(img.get("src") or img.get("data-src") or "")
        for img in images
        if img.get("src") or img.get("data-src")
    )
    for img in images:
        classes = set(img.get("class") or ())
        reference = str(img.get("src") or img.get("data-src") or "")
        if "img-z0eKj1" in classes and reference_counts[reference] > 1:
            img.decompose()


def parse_messages(soup, image_map=None):
    """解析豆包消息；页面不属于豆包时返回 None。"""
    if image_map is None:
        image_map = {}

    for script_or_style in soup(['script', 'style', 'noscript', 'button']):
        script_or_style.decompose()

    message_items = soup.find_all('div', class_='message-item')
    if not message_items:
        return None

    parsed_messages = []
    for msg in message_items:
        classes = msg.get('class', [])
        declared_role = str(msg.get('data-doubao-role') or '').lower()
        is_user = declared_role == 'user' or (
            not declared_role and 'justify-end' in classes
        )

        # 只把成功下载的真实图片替换为本地路径。豆包文档卡片还会
        # 内嵌 Asset cover/base64/fallback 装饰图，不能当成用户上传图片。
        for img in msg.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src in image_map:
                img["src"] = image_map[src]

        if is_user:
            user_parts = []
            local_image_paths = set(image_map.values())
            for img in msg.find_all("img"):
                src = img.get("src") or img.get("data-src")
                alt = (img.get("alt") or "").strip().lower()
                is_document_cover = (
                    alt == "asset cover"
                    or "doc-canvas-card-fallback" in (src or "")
                    or (src or "").startswith("data:image/")
                )
                if (
                    src in local_image_paths
                    and not is_document_cover
                ):
                    user_parts.append(f"![用户图片]({src})")

            # 识别真实的 HTML 文件卡片（非全文正则）
            file_cards = msg.find_all(
                class_=re.compile(r'file|attachment|doc', re.I)
            )
            for card in file_cards:
                card_text = card.get_text(strip=True)
                match = re.search(
                    r'[\w\-"\u4e00-\u9fa5\“\”]+\.'
                    r'(?:docx|doc|pdf|txt|md|rtf|xlsx|xls|pptx|ppt|zip|rar|csv)',
                    card_text,
                    re.IGNORECASE
                )
                if match:
                    filename = match.group(0)
                    local_href = image_map.get(filename.lower())
                    if local_href:
                        user_parts.append(
                            f"[📄 {filename}]({local_href})"
                        )
                    else:
                        user_parts.append(
                            f"📎 **[上传文档]** `{filename}`"
                        )

            text = msg.get_text(separator='\n', strip=True)

            for match in re.finditer(
                r'[^\\/:*?"<>|\n]{1,180}\.'
                r'(?:docx?|pdf|txt|md|rtf|xlsx?|xls|pptx?|ppt|csv)',
                text,
                re.IGNORECASE,
            ):
                filename = match.group(0).strip()
                if any(filename.lower() in part.lower() for part in user_parts):
                    continue
                local_href = image_map.get(filename.lower())
                if local_href:
                    user_parts.append(f"[📄 {filename}]({local_href})")
                else:
                    user_parts.append(f"📎 **[上传文档]** `{filename}`")

            clean_lines = [
                line.strip()
                for line in text.split('\n')
                if len(line.strip()) > 0
                and line.strip() not in [
                    "复制", "重新生成", "点赞", "踩", "分享", "已采纳",
                    "查看更多", "编辑", "朗读", "Word", "PDF", "文档"
                ]
                and not any(
                    "上传文档" in part and line.strip().lower() in part.lower()
                    for part in user_parts
                )
            ]
            if clean_lines:
                user_parts.append('\n'.join(clean_lines))

            seen = set()
            ordered_parts = []
            for item in user_parts:
                if item not in seen:
                    seen.add(item)
                    ordered_parts.append(item)

            final_text = (
                "\n\n".join(ordered_parts)
                if ordered_parts
                else '\n'.join(clean_lines)
            )
            if final_text:
                parsed_messages.append({
                    'role': 'User',
                    'content': final_text
                })
        else:
            _remove_assistant_image_artifacts(msg)
            ai_document_links = []
            for card in msg.find_all(
                "div", class_=re.compile(r"^product-card-")
            ):
                if "创建时间" not in card.get_text(" ", strip=True):
                    continue
                title_node = card.find(
                    "div",
                    class_=re.compile(r"^card-content-info-title-text-"),
                )
                title = " ".join(
                    (
                        title_node.get_text(" ", strip=True)
                        if title_node else ""
                    ).split()
                )
                local_href = image_map.get(title.lower()) if title else None
                if local_href:
                    link = f"[📄 查看 AI 生成文档：{title}]({local_href})"
                    if link not in ai_document_links:
                        ai_document_links.append(link)
            md_text = markdownify.markdownify(
                str(msg),
                heading_style="ATX"
            ).strip()
            if ai_document_links:
                md_text = "\n\n".join([
                    md_text,
                    *ai_document_links,
                ]).strip()
            if md_text:
                parsed_messages.append({'role': 'AI', 'content': md_text})

    for key, local_href in image_map.items():
        if not isinstance(key, str) or not key.lower().startswith("doubao-ai-doc:"):
            continue
        title = key.split(":", 1)[1]
        if any(local_href in item["content"] for item in parsed_messages):
            continue
        target = next((
            item for item in reversed(parsed_messages)
            if item["role"] == "AI" and title.lower() in item["content"].lower()
        ), None)
        if target is None:
            target = next((
                item for item in reversed(parsed_messages) if item["role"] == "AI"
            ), None)
        if target is not None:
            target["content"] += (
                f"\n\n[📄 查看 AI 生成文档：{title}]({local_href})"
            )

    return parsed_messages
