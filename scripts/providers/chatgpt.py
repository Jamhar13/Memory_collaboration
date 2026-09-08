"""ChatGPT 页面采集与消息解析。"""

import re

import markdownify
from rich.console import Console


DISPLAY_NAME = "ChatGPT"
WAIT_SELECTOR = "[data-message-author-role]"
HOSTS = ("chatgpt.com", "chat.openai.com")

console = Console()

SCROLL_PRIMARY_SETTLE_MS = 400
SCROLL_SECONDARY_SETTLE_MS = 150
BOUNDARY_SETTLE_MS = 350
BOUNDARY_CAPTURE_ROUNDS = 3
TOP_PRELOAD_SETTLE_MS = 800
TOP_PRELOAD_MAX_ROUNDS = 12
TOP_PRELOAD_STABLE_ROUNDS = 4


def _prefer_snapshot(candidate, existing):
    """仅在文本或图片更完整且另一维不退步时替换消息快照。"""
    candidate_text = int(candidate.get("text_length") or 0)
    existing_text = int(existing.get("text_length") or 0)
    candidate_images = int(candidate.get("image_score") or 0)
    existing_images = int(existing.get("image_score") or 0)
    if existing_text == 0 and candidate_text > 0:
        return True
    return (
        candidate_text > existing_text
        and candidate_images >= existing_images
    ) or (
        candidate_images > existing_images
        and candidate_text >= existing_text
    )


def _collapse_nested_markdown_fences(text):
    """将 ChatGPT 偶发生成的双层 Markdown 代码围栏折叠为单层。"""
    fence = chr(96) * 3
    lines = text.splitlines(keepends=True)
    cleaned = []
    index = 0

    def is_fence(line):
        return line.strip().startswith(fence)

    while index < len(lines):
        # markdownify 偶尔产生“围栏 / 围栏 / 正文 / 围栏 / 围栏”。
        # 只折叠这一完整、对称的双层结构，不碰普通代码块。
        if (
            index + 1 < len(lines)
            and is_fence(lines[index])
            and is_fence(lines[index + 1])
        ):
            closing_index = index + 2
            while closing_index + 1 < len(lines):
                if (
                    is_fence(lines[closing_index])
                    and is_fence(lines[closing_index + 1])
                ):
                    cleaned.extend(lines[index + 1:closing_index + 1])
                    index = closing_index + 2
                    break
                closing_index += 1
            else:
                cleaned.append(lines[index])
                index += 1
            continue

        cleaned.append(lines[index])
        index += 1

    return "".join(cleaned)


def _replace_math_with_placeholders(message):
    """提取 ChatGPT 公式源并以占位符保护，避免 Markdown 转换破坏 LaTeX。"""
    replacements = {}
    counter = 0

    def replace(target, latex, display):
        nonlocal counter
        latex = str(latex or "").strip()
        if not latex:
            return
        token = f"AIMEMORYMATHPLACEHOLDER{counter:04d}Z"
        counter += 1
        replacements[token] = (
            f"\n\n$$\n{latex}\n$$\n\n" if display else f"${latex}$"
        )
        target.replace_with(token)

    # 当前 ChatGPT 把绝大多数公式源放在外层 role=math 节点中。
    for node in list(message.select('[role="math"][data-math-source]')):
        style = str(node.get("style") or "").replace(" ", "").lower()
        display = (
            node.select_one(".katex-display") is not None
            or "display:block" in style
        )
        replace(node, node.get("data-math-source"), display)

    # 兼容仍使用标准 KaTeX MathML annotation 的旧式公式节点。
    for annotation in list(
        message.select('annotation[encoding="application/x-tex"]')
    ):
        katex = annotation.find_parent(class_="katex")
        if katex is None:
            continue
        display_container = katex.find_parent(class_="katex-display")
        replace(
            display_container or katex,
            annotation.get_text(),
            display_container is not None,
        )

    return replacements


def _restore_math_placeholders(text, replacements):
    for token, latex in replacements.items():
        text = text.replace(token, latex)
    return text


async def collect_html(page):
    """逐屏收集 ChatGPT 虚拟列表中的消息，避免长对话首尾丢失。"""
    role_selector = WAIT_SELECTOR
    if await page.locator(role_selector).count() == 0:
        return None

    scroll_container = None
    captured = {}
    discovery_index = 0

    async def capture_visible_messages():
        """复采当前消息；保留文本和真实图片均不退步的较完整快照。"""
        nonlocal discovery_index
        visible_messages = await page.locator(role_selector).evaluate_all(
            """elements => elements.map(element => {
                const turn = element.closest(
                    '[data-testid^="conversation-turn-"]'
                );
                const testId = turn
                    ? turn.getAttribute('data-testid') || ''
                    : '';
                const turnMatch = testId.match(/(\d+)$/);
                const messageId =
                    element.getAttribute('data-message-id') || '';
                const role =
                    element.getAttribute('data-message-author-role') || '';
                const text =
                    element.innerText || element.textContent || '';
                const imageScore = Array.from(
                    element.querySelectorAll('img')
                ).filter(image => {
                    const src = image.getAttribute('src')
                        || image.getAttribute('data-src') || '';
                    return src && !src.startsWith('data:image/svg');
                }).length;
                return {
                    key: messageId || testId || role + ':' + text,
                    order: turnMatch ? Number(turnMatch[1]) : null,
                    text_length: text.length,
                    image_score: imageScore,
                    html: (turn || element).outerHTML
                };
            })"""
        )

        for message in visible_messages:
            key = message["key"]
            existing = captured.get(key)
            if existing is None:
                message["discovery_index"] = discovery_index
                captured[key] = message
                discovery_index += 1
            elif _prefer_snapshot(message, existing):
                observed_orders = [
                    order for order in (
                        existing.get("order"), message.get("order")
                    )
                    if order is not None
                ]
                if observed_orders:
                    message["order"] = max(observed_orders)
                message["discovery_index"] = existing["discovery_index"]
                captured[key] = message
            elif message.get("order") is not None:
                existing_order = existing.get("order")
                if existing_order is None or message["order"] > existing_order:
                    existing["order"] = message["order"]

    try:
        scroll_container = await page.locator(role_selector).first.evaluate_handle(
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

        # ChatGPT 到达顶部后会异步补挂更早消息，并让既有 turn 编号整体后移。
        # 先在顶部等到消息数和滚动高度连续稳定，再正式顺序采集。
        previous_state = None
        stable_rounds = 0
        for _ in range(TOP_PRELOAD_MAX_ROUNDS):
            # 从顶部轻微移开再返回，确保已在顶部时也能触发分页哨兵。
            await scroll_container.evaluate(
                "element => element.scrollTo(0, Math.min(240, element.scrollHeight))"
            )
            await page.wait_for_timeout(100)
            await scroll_container.evaluate(
                "element => element.scrollTo(0, 0)"
            )
            await page.wait_for_timeout(TOP_PRELOAD_SETTLE_MS)
            await capture_visible_messages()
            metrics = await scroll_container.evaluate(
                """element => ({
                    scrollHeight: element.scrollHeight,
                    clientHeight: element.clientHeight
                })"""
            )
            state = (len(captured), int(metrics["scrollHeight"]))
            if state == previous_state:
                stable_rounds += 1
            else:
                stable_rounds = 0
            previous_state = state
            if stable_rounds >= TOP_PRELOAD_STABLE_ROUNDS:
                break

        # 保留预热阶段已经出现过的历史消息。ChatGPT 的虚拟列表在顶部
        # 补挂旧消息后，正式向下遍历时不一定会再次挂载每一个中间节点。
        # capture_visible_messages 会持续更新同一 message-id 的最终 turn 顺序。
        scroll_top = 0
        max_scroll_top = 0
        for _ in range(100):
            await scroll_container.evaluate(
                "(element, top) => element.scrollTo(0, top)",
                scroll_top
            )
            await page.wait_for_timeout(SCROLL_PRIMARY_SETTLE_MS)

            await capture_visible_messages()
            await page.wait_for_timeout(SCROLL_SECONDARY_SETTLE_MS)
            await capture_visible_messages()

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

        # 首尾媒体均可能延迟挂载；回访边界并只接受图片更完整的快照。
        for boundary in (max_scroll_top, 0):
            await scroll_container.evaluate(
                "(element, top) => element.scrollTo(0, top)",
                boundary
            )
            for _ in range(BOUNDARY_CAPTURE_ROUNDS):
                await page.wait_for_timeout(BOUNDARY_SETTLE_MS)
                await capture_visible_messages()

        ordered_messages = sorted(
            captured.values(),
            key=lambda message: (
                message["order"] is None,
                message["order"]
                if message["order"] is not None
                else message["discovery_index"]
            )
        )

        if ordered_messages:
            console.print(
                f"[dim]已从 ChatGPT 虚拟列表中完整收集 "
                f"{len(ordered_messages)} 条消息。[/dim]"
            )
            message_html = "\n".join(
                message["html"] for message in ordered_messages
            )
            return f"<!DOCTYPE html><html><body>{message_html}</body></html>"

    except Exception as e:
        console.print(
            f"[dim]提示: ChatGPT 虚拟列表收集失败，将使用当前页面快照: {e}[/dim]"
        )
    finally:
        if scroll_container is not None:
            await scroll_container.dispose()

    return None


def parse_messages(soup, image_map=None):
    """解析 ChatGPT 消息；页面不属于 ChatGPT 时返回 None。"""
    if image_map is None:
        image_map = {}

    chatgpt_messages = soup.find_all(
        attrs={"data-message-author-role": True}
    )
    if not chatgpt_messages:
        return None

    parsed_messages = []
    for msg in chatgpt_messages:
        role = msg.get("data-message-author-role")
        math_replacements = _replace_math_with_placeholders(msg)
        if role == "user":
            content_parts = []
            message_container = msg.find_parent(
                attrs={"data-testid": re.compile(r"^conversation-turn-")}
            ) or msg
            seen_document_names = set()
            # 提取用户发送的所有图片 (精准映射本地路径)
            for img in msg.find_all("img"):
                src = img.get("src") or img.get("data-src")
                alt = img.get("alt", "用户上传图片")
                if src and not src.startswith("data:image/svg"):
                    local_src = image_map.get(src, src)
                    content_parts.append(f"![{alt}]({local_src})")

            # 精准检查显示的文件卡片或下载链接（排除纯文本里的误触发）
            for a in message_container.find_all("a", href=True):
                href = a["href"]
                link_text = a.get_text(strip=True) or "附件文件"
                if (
                    (href.startswith("http") or href.startswith("/"))
                    and any(
                        ext in link_text.lower()
                        for ext in [
                            '.doc', '.pdf', '.txt', '.xls', '.ppt',
                            '.zip', '.rar', '.csv', '.md', '.rtf'
                        ]
                    )
                ):
                    local_href = image_map.get(
                        href, image_map.get(link_text.lower(), href)
                    )
                    content_parts.append(
                        f"[📄 {link_text}]({local_href})"
                    )
                    seen_document_names.add(link_text.lower())

            # 识别真实的 HTML 文件卡片节点（非纯文本正则）
            file_cards = list(message_container.find_all(
                class_=re.compile(r'file|attachment|document', re.I)
            ))
            # ChatGPT 当前版文件卡片的标题节点不再带 file/attachment 类名。
            # 仅补充平台稳定使用的真实标题节点，避免扫描普通消息文本。
            for title in message_container.select("div.truncate.font-semibold"):
                if title not in file_cards:
                    file_cards.append(title)
            for card in file_cards:
                card_text = card.get_text(strip=True)
                if any(
                    ext in card_text.lower()
                    for ext in [
                        '.doc', '.pdf', '.txt', '.xls', '.ppt',
                        '.zip', '.rar', '.csv', '.md', '.rtf'
                    ]
                ):
                    # 提炼出真正文件名
                    match = re.search(
                        r'[\w\-()"\u4e00-\u9fa5\“\”]+\.'
                        r'(?:docx|doc|pdf|txt|md|rtf|xlsx|xls|pptx|ppt|zip|rar|csv)',
                        card_text,
                        re.IGNORECASE
                    )
                    if match:
                        filename = match.group(0)
                        if filename.lower() in seen_document_names:
                            continue
                        seen_document_names.add(filename.lower())
                        local_href = image_map.get(filename.lower())
                        if local_href:
                            content_parts.append(
                                f"[📄 {filename}]({local_href})"
                            )
                        else:
                            content_parts.append(
                                f"📎 **[上传文档]** `{filename}`"
                            )



            text = msg.get_text(separator='\n', strip=True)
            if seen_document_names:
                text = '\n'.join(
                    line for line in text.splitlines()
                    if line.strip() != '上传文件'
                    and line.strip().lower() not in seen_document_names
                ).strip()
            if text:
                content_parts.append(text)

            # 顺序去重但保留多个不同图片
            seen = set()
            ordered_parts = []
            for item in content_parts:
                if item not in seen:
                    seen.add(item)
                    ordered_parts.append(item)

            final_user_text = (
                "\n\n".join(ordered_parts) if ordered_parts else text
            )
            final_user_text = _restore_math_placeholders(
                final_user_text, math_replacements
            )
            parsed_messages.append({
                'role': 'User',
                'content': final_user_text
            })

        elif role == "assistant":
            # 替换本地图片路径
            for img in msg.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if src in image_map:
                    img["src"] = image_map[src]

            # 修复 ChatGPT 嵌套 <pre> 导致 markdownify 生成多重/错误代码块的问题
            for pre in msg.find_all("pre"):
                inner_pre = pre.find("pre")
                if inner_pre:
                    code_text = inner_pre.get_text()
                    # 构造标准 <pre><code> 避免嵌套冲突；不依赖易变化的类名
                    new_pre = soup.new_tag("pre")
                    new_code = soup.new_tag("code")
                    new_code.string = code_text
                    new_pre.append(new_code)
                    pre.replace_with(new_pre)

            md_text = markdownify.markdownify(
                str(msg),
                heading_style="ATX"
            ).strip()
            md_text = _collapse_nested_markdown_fences(md_text)
            md_text = _restore_math_placeholders(md_text, math_replacements)
            parsed_messages.append({'role': 'AI', 'content': md_text})

    return parsed_messages
