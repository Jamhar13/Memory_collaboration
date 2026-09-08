"""DeepSeek 页面采集与消息解析。"""

import re

import markdownify
from bs4 import BeautifulSoup
from rich.console import Console


DISPLAY_NAME = "DeepSeek"
WAIT_SELECTOR = "[data-virtual-list-item-key] .ds-message"
HOSTS = ("chat.deepseek.com",)

console = Console()


async def collect_html(page):
    """逐屏收集 DeepSeek 虚拟列表中的消息，避免超长分享对话丢失。"""
    item_selector = "[data-virtual-list-item-key]"
    if await page.locator(f"{item_selector} .ds-message").count() == 0:
        return None

    scroll_container = None
    captured = {}
    discovery_index = 0

    async def capture_visible_items():
        """收集当前已挂载的消息；DeepSeek 的虚拟列表挂载有短暂延迟。"""
        nonlocal discovery_index
        visible_items = await page.locator(item_selector).evaluate_all(
            """elements => elements
                .filter(element => element.querySelector('.ds-message'))
                .map(element => {
                    const rawKey = element.getAttribute(
                        'data-virtual-list-item-key'
                    ) || '';
                    const numericKey = Number(rawKey);
                    const imageScore = Array.from(
                        element.querySelectorAll('img')
                    ).filter(image => {
                        const src = image.getAttribute('src')
                            || image.getAttribute('data-src') || '';
                        return src && !src.startsWith('data:image/svg');
                    }).length;
                    return {
                        key: rawKey || element.innerText || element.outerHTML,
                        order: Number.isFinite(numericKey) ? numericKey : null,
                        image_score: imageScore,
                        html: element.outerHTML
                    };
                })"""
        )

        for item in visible_items:
            key = item["key"]
            existing = captured.get(key)
            if existing is None:
                item["discovery_index"] = discovery_index
                captured[key] = item
                discovery_index += 1
            elif (
                item["image_score"],
                len(item["html"])
            ) > (
                existing["image_score"],
                len(existing["html"])
            ):
                item["discovery_index"] = existing["discovery_index"]
                captured[key] = item

    try:
        scroll_container = await page.locator(
            f"{item_selector} .ds-message"
        ).first.evaluate_handle(
            """element => {
                let current = element;
                while (current) {
                    const style = window.getComputedStyle(current);
                    const canScroll = current.scrollHeight > current.clientHeight + 1;
                    if (canScroll && /(auto|scroll)/.test(style.overflowY)) {
                        return current;
                    }
                    current = current.parentElement;
                }
                return document.scrollingElement || document.documentElement;
            }"""
        )

        scroll_top = 0
        max_scroll_top = 0
        for _ in range(500):
            await scroll_container.evaluate(
                "(element, top) => element.scrollTo(0, top)",
                scroll_top
            )
            await page.wait_for_timeout(700)

            await capture_visible_items()
            await page.wait_for_timeout(500)
            await capture_visible_items()

            metrics = await scroll_container.evaluate(
                """element => ({
                    scrollHeight: element.scrollHeight,
                    clientHeight: element.clientHeight
                })"""
            )
            max_scroll_top = max(
                0,
                metrics["scrollHeight"] - metrics["clientHeight"]
            )
            if scroll_top >= max_scroll_top:
                break

            step = max(int(metrics["clientHeight"] * 0.5), 800)
            next_scroll_top = min(scroll_top + step, max_scroll_top)
            if next_scroll_top <= scroll_top:
                break
            scroll_top = next_scroll_top

        # 人工登录使用系统窗口尺寸。此时首次滚到顶部后，DeepSeek
        # 偶尔要过一小段时间才会挂载第一条消息。回访首尾边界并固定
        # 复采数次，既补齐首条，也避免尾条遇到相同的竞态。
        for boundary in (max_scroll_top, 0):
            await scroll_container.evaluate(
                "(element, top) => element.scrollTo(0, top)",
                boundary
            )
            for _ in range(4):
                await page.wait_for_timeout(500)
                await capture_visible_items()

        ordered_items = sorted(
            captured.values(),
            key=lambda item: (
                item["order"] is None,
                item["order"]
                if item["order"] is not None
                else item["discovery_index"]
            )
        )

        if ordered_items:
            console.print(
                f"[dim]已从 DeepSeek 虚拟列表中完整收集 "
                f"{len(ordered_items)} 条消息。[/dim]"
            )
            item_html = "\n".join(item["html"] for item in ordered_items)
            return f"<!DOCTYPE html><html><body>{item_html}</body></html>"

    except Exception as e:
        console.print(
            f"[dim]提示: DeepSeek 虚拟列表收集失败，将使用当前页面快照: {e}[/dim]"
        )
    finally:
        if scroll_container is not None:
            await scroll_container.dispose()

    return None


def parse_messages(soup, image_map=None):
    """解析 DeepSeek 分享页；页面不属于 DeepSeek 时返回 None。"""
    if soup.select_one(WAIT_SELECTOR) is None:
        return None
    if image_map is None:
        image_map = {}

    parsed_messages = []
    ignored_ui_text = {
        "复制", "重新生成", "编辑", "分享", "下载", "点赞", "踩"
    }
    attachment_name_pattern = re.compile(
        r'^[^\\/:*?"<>|\n]+\.(?:png|jpe?g|webp|gif|bmp|svg|pdf|docx?|'
        r'xlsx?|pptx?|txt|csv|md|rtf|zip|rar)$',
        re.IGNORECASE
    )
    attachment_size_pattern = re.compile(
        r'^(?:PNG|JPE?G|WEBP|GIF|BMP|SVG|PDF|DOCX?|XLSX?|PPTX?|TXT|CSV|MD|RTF|ZIP|RAR)'
        r'\s+\d+(?:\.\d+)?\s*(?:B|KB|MB|GB)$',
        re.IGNORECASE
    )

    for item in soup.select('[data-virtual-list-item-key]'):
        message_node = item.select_one('.ds-message')
        if message_node is None:
            continue

        answer_node = item.select_one('.ds-assistant-message-main-content')
        if answer_node is not None:
            answer_soup = BeautifulSoup(str(answer_node), 'html.parser')
            answer = answer_soup.select_one(
                '.ds-assistant-message-main-content'
            )

            # 去掉代码块工具栏中的“复制/下载”等界面文字，保留代码本体。
            for removable in answer.select(
                'script, style, noscript, button, .md-code-block-banner-wrap'
            ):
                removable.decompose()

            for img in answer.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src in image_map:
                    img['src'] = image_map[src]

            md_text = markdownify.markdownify(
                str(answer),
                heading_style='ATX'
            ).strip()
            if md_text:
                parsed_messages.append({'role': 'AI', 'content': md_text})
            continue

        user_soup = BeautifulSoup(str(message_node), 'html.parser')
        user_message = user_soup.select_one('.ds-message')
        user_parts = []

        # DeepSeek 的历史分享图片可能已经从文件服务失效。此时前端只
        # 留下一个带宽高样式和 tabindex 的错误占位组件，没有 img/src。
        media_placeholder = user_message.find(attrs={'tabindex': '0'})
        placeholder_style = (
            media_placeholder.get('style', '')
            if media_placeholder is not None
            else ''
        )
        has_failed_image_placeholder = (
            user_message.find('img') is None
            and 'width:' in placeholder_style
            and 'height:' in placeholder_style
        )

        for img in user_message.find_all('img'):
            src = img.get('src') or img.get('data-src')
            alt = img.get('alt', '用户上传图片')
            if src and not src.startswith('data:image/svg'):
                local_src = image_map.get(src, src)
                user_parts.append(f'![{alt}]({local_src})')

        for removable in user_message.select(
            'script, style, noscript, button, svg'
        ):
            removable.decompose()

        raw_lines = [
            line.strip()
            for line in user_message.get_text(
                separator='\n',
                strip=True
            ).split('\n')
            if line.strip()
        ]
        clean_lines = []
        index = 0
        while index < len(raw_lines):
            line = raw_lines[index]
            next_line = (
                raw_lines[index + 1]
                if index + 1 < len(raw_lines)
                else ''
            )
            if (
                attachment_name_pattern.match(line)
                and attachment_size_pattern.match(next_line)
            ):
                local_href = image_map.get(line.lower())
                if local_href:
                    user_parts.append(
                        f'[📄 {line}]({local_href})（{next_line}）'
                    )
                else:
                    user_parts.append(
                        f'📎 **[上传文件]** `{line}`（{next_line}）'
                    )
                index += 2
                continue
            if line not in ignored_ui_text:
                clean_lines.append(line)
            index += 1

        if clean_lines:
            user_parts.append('\n'.join(clean_lines))

        if not user_parts and has_failed_image_placeholder:
            user_parts.append(
                '🖼️ **[用户上传图片]**'
                '（DeepSeek 分享页原图已失效或加载失败）'
            )

        seen = set()
        ordered_parts = []
        for part in user_parts:
            if part not in seen:
                seen.add(part)
                ordered_parts.append(part)

        final_text = '\n\n'.join(ordered_parts)
        if final_text:
            parsed_messages.append({'role': 'User', 'content': final_text})

    return parsed_messages
