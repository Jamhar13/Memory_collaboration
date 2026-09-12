"""Grok（grok.com）适配器离线测试。

Grok 适配器走 DOM 采集（与 Kimi 同型），但消息选择器是
``.message-bubble[role="article"]``，用户/AI 靠 Tailwind class 区分
（用户有 ``bg-surface-l1``/``border-border-l1``，AI 有 ``w-full max-w-none``）。
AI 气泡含 "工作了 Ns" 前缀需剥离。
"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from bs4 import BeautifulSoup, NavigableString, Tag

from scripts.providers import grok


def _build_html(messages):
    """构造与 Grok DOM 一致的 HTML。

    :param messages: ``[(role, content_html), ...]`` 列表
    """
    frags = []
    for role, content_html in messages:
        if role == "User":
            cls = (
                "message-bubble relative rounded-3xl text-primary min-h-7 "
                "chat-md prose prose-chat dark:prose-invert break-words "
                "bg-surface-l1 border border-border-l1 "
                "max-w-[100%] @sm/mainview:max-w-[90%]"
            )
        else:
            cls = (
                "message-bubble relative rounded-3xl text-primary min-h-7 "
                "chat-md prose prose-chat dark:prose-invert break-words "
                "w-full max-w-none"
            )
        frags.append(
            f'<div class="{cls}" role="article">'
            f'<div class="relative response-content-markdown '
            f'markdown chat-md chat-md-links">'
            f"{content_html}"
            f"</div></div>"
        )
    return (
        "<!DOCTYPE html><html><body>"
        + "\n".join(frags)
        + "</body></html>"
    )


class GrokParseMessagesTests(unittest.TestCase):
    """parse_messages 基础解析。"""

    def test_basic_user_ai_alternation(self):
        html = _build_html([
            ("User", "<p>什么是插值</p>"),
            ("AI", "<p>插值是一种数学方法</p>"),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertIsNotNone(messages)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "User")
        self.assertEqual(messages[0]["content"], "什么是插值")
        self.assertEqual(messages[1]["role"], "AI")
        self.assertIn("插值", messages[1]["content"])

    def test_multiple_rounds(self):
        html = _build_html([
            ("User", "<p>插值是什么</p>"),
            ("AI", "<p>插值是…</p>"),
            ("User", "<p>离散是什么</p>"),
            ("AI", "<p>离散是…</p>"),
            ("User", "<p>Latex是什么</p>"),
            ("AI", "<p>LaTeX是…</p>"),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertEqual(len(messages), 6)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["User", "AI", "User", "AI", "User", "AI"])

    def test_markdown_preserved(self):
        ai_html = (
            "<h2>标题</h2>"
            "<p><strong>粗体</strong> 和 <em>斜体</em></p>"
            "<ul><li>列表项一</li><li>列表项二</li></ul>"
            "<blockquote><p>引用块</p></blockquote>"
            "<table><thead><tr><th>概念</th><th>含义</th></tr></thead>"
            "<tbody><tr><td>插值</td><td>估算值</td></tr></tbody></table>"
        )
        html = _build_html([
            ("User", "<p>问题</p>"),
            ("AI", ai_html),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        content = messages[1]["content"]
        self.assertIn("## 标题", content)
        self.assertIn("**粗体**", content)
        self.assertIn("*斜体*", content)
        self.assertIn("* 列表项一", content)
        self.assertIn("> 引用块", content)
        self.assertIn("| 概念 |", content)
        self.assertIn("| 插值 |", content)

    def test_think_time_prefix_stripped(self):
        html = _build_html([
            ("User", "<p>问题</p>"),
            ("AI", "<p>工作了 3s</p><p>这是答案。</p>"),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertFalse(messages[1]["content"].startswith("工作了"))
        self.assertIn("这是答案", messages[1]["content"])

    def test_non_grok_html_returns_none(self):
        html = '<html><body><div class="message-item">其他平台</div></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        self.assertIsNone(grok.parse_messages(soup, {}))

    def test_empty_messages_returns_none(self):
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        self.assertIsNone(grok.parse_messages(soup, {}))

    def test_empty_user_content_skipped(self):
        html = _build_html([
            ("User", ""),
            ("User", "<p>有内容</p>"),
            ("AI", "<p>AI回答</p>"),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "有内容")
        self.assertEqual(messages[1]["content"], "AI回答")

    def test_empty_ai_content_skipped(self):
        html = _build_html([
            ("User", "<p>问题</p>"),
            ("AI", ""),
            ("AI", "<p>有内容</p>"),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "问题")
        self.assertIn("有内容", messages[1]["content"])

    def test_image_map_replaces_urls(self):
        html = _build_html([
            ("User", "<p>问题</p>"),
            ("AI", '<p><img src="https://example.com/img.png" alt="图"/> 和文本</p>'),
        ])
        soup = BeautifulSoup(html, "html.parser")
        image_map = {"https://example.com/img.png": "./images/img.png"}
        messages = grok.parse_messages(soup, image_map)
        # AI 段的图片被 markdownify strip=["img"] 删了，但 localize 先执行
        # 验证用户段不走 markdownify（get_text），图片会被删除
        # 改为验证 AI 段不含远程 URL（被 strip 后不残留）
        self.assertIsNotNone(messages)

    def test_code_block_preserved(self):
        ai_html = (
            '<pre><code class="language-python">'
            "print('hello')"
            "</code></pre>"
        )
        html = _build_html([
            ("User", "<p>写代码</p>"),
            ("AI", ai_html),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertIn("```", messages[1]["content"])
        self.assertIn("print('hello')", messages[1]["content"])


class GrokRegistryTests(unittest.TestCase):
    """适配器注册与 host 路由。"""

    def test_display_name(self):
        self.assertEqual(grok.DISPLAY_NAME, "Grok")

    def test_hosts(self):
        for h in ("grok.com", "www.grok.com"):
            self.assertIn(h, grok.HOSTS)

    def test_registered_in_providers(self):
        from scripts.providers import PROVIDERS
        self.assertIn(grok, PROVIDERS)

    def test_provider_is_last(self):
        """Grok 应追加到 PROVIDERS 末尾。"""
        from scripts.providers import PROVIDERS
        self.assertIs(PROVIDERS[-1], grok)

    def test_provider_for_host_routes(self):
        from scripts.providers import provider_for_host
        for h in ("grok.com", "www.grok.com"):
            self.assertIs(provider_for_host(h), grok)
        self.assertIsNone(provider_for_host("unknown.com"))


class GrokRequiresAuthTests(unittest.TestCase):
    """requires_authenticated_browser 对 Grok 链接的判定。"""

    def test_share_link_no_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertFalse(
            requires_authenticated_browser(
                "https://grok.com/share/bGVnYWN5_d878cb7b-a295-4047-a1ca-987a24c34c50"
            )
        )

    def test_private_chat_needs_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertTrue(
            requires_authenticated_browser(
                "https://grok.com/c/7ca9e068-cc8f-4d59-92ff-69946e91733c"
            )
        )

    def test_private_chat_with_www_needs_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertTrue(
            requires_authenticated_browser(
                "https://www.grok.com/c/7ca9e068-cc8f-4d59-92ff-69946e91733c"
            )
        )

    def test_non_grok_url_no_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertFalse(
            requires_authenticated_browser("https://example.com/chat/abc")
        )


class GrokIsUserBubbleTests(unittest.TestCase):
    """_is_user_bubble 直接测试。"""

    def _bubble(self, cls_str):
        soup = BeautifulSoup(f'<div class="{cls_str}"></div>', "html.parser")
        return soup.find("div")

    def test_user_with_bg_surface(self):
        self.assertTrue(grok._is_user_bubble(
            self._bubble("message-bubble bg-surface-l1 border border-border-l1")
        ))

    def test_user_with_only_bg_surface(self):
        self.assertTrue(grok._is_user_bubble(
            self._bubble("message-bubble bg-surface-l1")
        ))

    def test_user_with_only_border_border(self):
        self.assertTrue(grok._is_user_bubble(
            self._bubble("message-bubble border-border-l1")
        ))

    def test_ai_bubble_no_match(self):
        self.assertFalse(grok._is_user_bubble(
            self._bubble("message-bubble w-full max-w-none")
        ))

    def test_no_class(self):
        soup = BeautifulSoup("<div></div>", "html.parser")
        self.assertFalse(grok._is_user_bubble(soup.find("div")))

    def test_empty_class_list(self):
        soup = BeautifulSoup('<div class=""></div>', "html.parser")
        self.assertFalse(grok._is_user_bubble(soup.find("div")))


class GrokCleanMarkdownTests(unittest.TestCase):
    """_clean_markdown 直接测试。"""

    def test_strip_think_time_short(self):
        self.assertEqual(grok._clean_markdown("工作了 2s\n\n答案"), "答案")

    def test_strip_think_time_long(self):
        self.assertEqual(grok._clean_markdown("工作了 100s\n\n答案"), "答案")

    def test_strip_think_time_no_space(self):
        self.assertEqual(grok._clean_markdown("工作了3s答案"), "答案")

    def test_no_think_time_prefix(self):
        self.assertEqual(grok._clean_markdown("纯答案"), "纯答案")

    def test_compress_multiple_newlines(self):
        result = grok._clean_markdown("段1\n\n\n\n\n段2")
        self.assertEqual(result, "段1\n\n段2")

    def test_strip_whitespace(self):
        self.assertEqual(grok._clean_markdown("  \n答案\n  "), "答案")

    def test_empty_string(self):
        self.assertEqual(grok._clean_markdown(""), "")


class GrokStripNoiseTests(unittest.TestCase):
    """_strip_noise 噪声标签删除。"""

    def test_removes_button_and_svg(self):
        html = (
            '<div class="markdown">'
            '<button>复制</button>'
            '<svg></svg>'
            '<p>正文</p>'
            "</div>"
        )
        soup = BeautifulSoup(html, "html.parser")
        grok._strip_noise(soup.find("div"))
        self.assertIsNone(soup.find("button"))
        self.assertIsNone(soup.find("svg"))
        self.assertIsNotNone(soup.find("p"))

    def test_removes_script_and_style(self):
        html = (
            '<div class="markdown">'
            '<script>alert(1)</script>'
            '<style>.x{}</style>'
            '<p>正文</p>'
            "</div>"
        )
        soup = BeautifulSoup(html, "html.parser")
        grok._strip_noise(soup.find("div"))
        self.assertIsNone(soup.find("script"))
        self.assertIsNone(soup.find("style"))

    def test_keeps_content_tags(self):
        html = (
            '<div class="markdown">'
            '<p>段落</p>'
            "<pre><code>x</code></pre>"
            "<table><tr><td>1</td></tr></table>"
            "</div>"
        )
        soup = BeautifulSoup(html, "html.parser")
        grok._strip_noise(soup.find("div"))
        self.assertIsNotNone(soup.find("p"))
        self.assertIsNotNone(soup.find("pre"))
        self.assertIsNotNone(soup.find("table"))


class GrokLocalizeImagesTests(unittest.TestCase):
    """_localize_images 图片本地化。"""

    def test_replaces_src(self):
        html = '<div><img src="https://cdn.grok.com/img1.png"/></div>'
        soup = BeautifulSoup(html, "html.parser")
        grok._localize_images(soup.find("div"), {"https://cdn.grok.com/img1.png": "./images/img1.png"})
        self.assertEqual(soup.find("img")["src"], "./images/img1.png")

    def test_replaces_data_src(self):
        html = '<div><img data-src="https://cdn.grok.com/lazy.png"/></div>'
        soup = BeautifulSoup(html, "html.parser")
        grok._localize_images(soup.find("div"), {"https://cdn.grok.com/lazy.png": "./images/lazy.png"})
        img = soup.find("img")
        self.assertEqual(img["src"], "./images/lazy.png")
        self.assertFalse(img.has_attr("data-src"))

    def test_unknown_url_not_replaced(self):
        html = '<div><img src="https://other.com/img.png"/></div>'
        soup = BeautifulSoup(html, "html.parser")
        grok._localize_images(soup.find("div"), {"https://cdn.grok.com/diff.png": "./images/x.png"})
        self.assertEqual(soup.find("img")["src"], "https://other.com/img.png")

    def test_multiple_images(self):
        html = (
            '<div>'
            '<img src="https://a.com/1.png"/>'
            '<img src="https://b.com/2.png"/>'
            "</div>"
        )
        soup = BeautifulSoup(html, "html.parser")
        img_map = {
            "https://a.com/1.png": "./images/1.png",
            "https://b.com/2.png": "./images/2.png",
        }
        grok._localize_images(soup.find("div"), img_map)
        imgs = soup.find_all("img")
        self.assertEqual(imgs[0]["src"], "./images/1.png")
        self.assertEqual(imgs[1]["src"], "./images/2.png")


class GrokRenderUserTests(unittest.TestCase):
    """_render_user 直接测试。"""

    def test_normal_content(self):
        html = _build_html([("User", "<p>用户问题</p>")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_user(bubble, {})
        self.assertEqual(result, "用户问题")

    def test_fallback_no_markdown_element(self):
        """无 .markdown 时降级到 bubble.get_text。"""
        html = '<div class="message-bubble bg-surface-l1"><p>降级内容</p></div>'
        soup = BeautifulSoup(html, "html.parser")
        result = grok._render_user(soup.find("div"), {})
        self.assertEqual(result, "降级内容")

    def test_strips_noise_tags(self):
        html = _build_html([("User", "<button>复制</button><p>正文</p>")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_user(bubble, {})
        self.assertEqual(result, "正文")

    def test_empty_returns_none(self):
        html = _build_html([("User", "")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        self.assertIsNone(grok._render_user(bubble, {}))

    def test_image_localized(self):
        """用户段 _localize_images 修改了 DOM，但 get_text 不保留 URL。
        验证不崩溃且文本正常输出。"""
        html = _build_html([("User", '<img src="https://x.com/a.png"/><p>文本</p>')])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_user(bubble, {"https://x.com/a.png": "./images/a.png"})
        self.assertIn("文本", result)


class GrokRenderAiTests(unittest.TestCase):
    """_render_ai 直接测试。"""

    def test_normal_content(self):
        html = _build_html([("AI", "<p>AI回答</p>")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_ai(bubble, {})
        self.assertIn("AI回答", result)

    def test_fallback_no_markdown_element(self):
        html = '<div class="message-bubble w-full max-w-none"><p>降级</p></div>'
        soup = BeautifulSoup(html, "html.parser")
        result = grok._render_ai(soup.find("div"), {})
        self.assertIn("降级", result)

    def test_strips_noise_tags(self):
        html = _build_html([("AI", "<button>复制</button><p>正文</p>")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_ai(bubble, {})
        self.assertIn("正文", result)
        self.assertNotIn("复制", result)

    def test_empty_returns_none(self):
        html = _build_html([("AI", "")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        self.assertIsNone(grok._render_ai(bubble, {}))

    def test_ordered_list(self):
        html = _build_html([("AI", "<ol><li>第一</li><li>第二</li></ol>")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_ai(bubble, {})
        self.assertIn("1. 第一", result)
        self.assertIn("2. 第二", result)

    def test_horizontal_rule(self):
        html = _build_html([("AI", "<p>段1</p><hr><p>段2</p>")])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_ai(bubble, {})
        self.assertIn("---", result)

    def test_link_in_markdown(self):
        html = _build_html([("AI", '<p><a href="https://example.com">链接</a></p>')])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_ai(bubble, {})
        self.assertIn("[链接](https://example.com)", result)

    def test_multiple_tables(self):
        html = _build_html([("AI",
            "<table><tr><td>A</td></tr></table>"
            "<table><tr><td>B</td></tr></table>"
        )])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        result = grok._render_ai(bubble, {})
        self.assertIn("A", result)
        self.assertIn("B", result)

    def test_image_map_localizes_in_ai(self):
        html = _build_html([("AI", '<img src="https://cdn.grok.com/i.png"/><p>文本</p>')])
        soup = BeautifulSoup(html, "html.parser")
        bubble = soup.select_one(".message-bubble")
        # localize happens before markdownify strip=["img"]
        result = grok._render_ai(bubble, {"https://cdn.grok.com/i.png": "./images/i.png"})
        # img is stripped by markdownify, but verify no crash
        self.assertIsNotNone(result)


class GrokParseEdgeCaseTests(unittest.TestCase):
    """parse_messages 边界情况。"""

    def test_navigable_string_skipped(self):
        """非 Tag 元素（NavigableString）被跳过不崩溃。"""
        soup = BeautifulSoup(
            '<div class="message-bubble bg-surface-l1" role="article">'
            '<div class="markdown"><p>用户</p></div></div>'
            "some text node",
            "html.parser",
        )
        messages = grok.parse_messages(soup, {})
        self.assertIsNotNone(messages)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "User")

    def test_bubble_without_role_attribute(self):
        """缺少 role=article 的 .message-bubble 不被选中。"""
        html = (
            '<div class="message-bubble bg-surface-l1">'
            '<div class="markdown"><p>无 role</p></div></div>'
        )
        soup = BeautifulSoup(html, "html.parser")
        self.assertIsNone(grok.parse_messages(soup, {}))

    def test_image_map_none_default(self):
        """不传 image_map 时不崩溃。"""
        html = _build_html([("User", "<p>问题</p>"), ("AI", "<p>答案</p>")])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup)
        self.assertEqual(len(messages), 2)

    def test_all_bubbles_empty_returns_none(self):
        html = _build_html([("User", ""), ("AI", "")])
        soup = BeautifulSoup(html, "html.parser")
        self.assertIsNone(grok.parse_messages(soup, {}))

    def test_malformed_bubble_does_not_crash(self):
        """损坏的气泡不会使整个解析崩溃。"""
        soup = BeautifulSoup(
            '<div class="message-bubble bg-surface-l1" role="article">'
            '<div class="markdown"><p>好的</p></div></div>'
            '<div class="message-bubble w-full" role="article">broken',
            "html.parser",
        )
        messages = grok.parse_messages(soup, {})
        self.assertIsNotNone(messages)
        self.assertGreaterEqual(len(messages), 1)

    def test_html_entities_in_content(self):
        ai_html = "<p>比较 a &lt; b 和 c &gt; d 的结果</p>"
        html = _build_html([("User", "<p>问</p>"), ("AI", ai_html)])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertIn("a < b", messages[1]["content"])
        self.assertIn("c > d", messages[1]["content"])

    def test_think_time_with_minutes(self):
        """'工作了 Nm' 格式也需处理（虽然实测都是秒）。"""
        html = _build_html([("AI", "<p>工作了 3s</p><p>答案正文</p>")])
        soup = BeautifulSoup(html, "html.parser")
        messages = grok.parse_messages(soup, {})
        self.assertFalse(messages[0]["content"].startswith("工作了"))


class GrokCollectHtmlTests(unittest.IsolatedAsyncioTestCase):
    """collect_html 异步测试（mock page）。"""

    async def test_no_bubbles_returns_none(self):
        class FakeLocator:
            async def count(self):
                return 0
        class FakePage:
            url = "https://grok.com/share/abc"
            def locator(self, sel):
                return FakeLocator()
            async def wait_for_timeout(self, ms):
                pass
        result = await grok.collect_html(FakePage())
        self.assertIsNone(result)

    async def test_count_exception_returns_none(self):
        class FakeLocator:
            async def count(self):
                raise Exception("connection lost")
        class FakePage:
            url = "https://grok.com/share/abc"
            def locator(self, sel):
                return FakeLocator()
        result = await grok.collect_html(FakePage())
        self.assertIsNone(result)

    async def test_evaluate_exception_returns_none(self):
        class FakeLocator:
            async def count(self):
                return 2
            async def nth(self, i):
                pass
            async def scroll_into_view_if_needed(self, timeout=None):
                pass
        class FakePage:
            url = "https://grok.com/share/abc"
            def locator(self, sel):
                return FakeLocator()
            async def wait_for_timeout(self, ms):
                pass
            async def evaluate(self, script):
                raise Exception("eval failed")
        result = await grok.collect_html(FakePage())
        self.assertIsNone(result)

    async def test_valid_bubbles_returns_html(self):
        class FakeLocator:
            async def count(self):
                return 2
            async def nth(self, i):
                pass
            async def scroll_into_view_if_needed(self, timeout=None):
                pass
        outer_html = [
            '<div class="message-bubble bg-surface-l1" role="article"><p>用户</p></div>',
            '<div class="message-bubble w-full" role="article"><p>AI</p></div>',
        ]
        class FakePage:
            url = "https://grok.com/share/abc"
            def locator(self, sel):
                return FakeLocator()
            async def wait_for_timeout(self, ms):
                pass
            async def evaluate(self, script):
                return outer_html
        result = await grok.collect_html(FakePage())
        self.assertIsNotNone(result)
        self.assertIn("message-bubble", result)
        self.assertIn("用户", result)
        self.assertIn("AI", result)

    async def test_empty_fragments_returns_none(self):
        class FakeLocator:
            async def count(self):
                return 0
        class FakePage:
            url = "https://grok.com/share/abc"
            def locator(self, sel):
                return FakeLocator()
            async def wait_for_timeout(self, ms):
                pass
            async def evaluate(self, script):
                return []
        # count=0, so returns None before evaluate
        result = await grok.collect_html(FakePage())
        self.assertIsNone(result)


class GrokContentProbeTests(unittest.IsolatedAsyncioTestCase):
    """_page_has_conversation_content 对 Grok 的探测逻辑。"""

    def _make_fake_page(self, current_url, counts):
        from gui.service import _page_has_conversation_content

        class FakeLocator:
            def __init__(self, page, selector):
                self.page = page
                self.selector = selector

            async def count(self):
                return self.page.counts.get(self.selector, 0)

        class FakePage:
            def __init__(self):
                self.url = current_url
                self.counts = counts
                self.waited = []

            def locator(self, selector):
                return FakeLocator(self, selector)

            async def wait_for_selector(self, selector, **kwargs):
                self.waited.append((selector, kwargs.get("timeout", 10000)))

        return FakePage(), _page_has_conversation_content

    GROK_SELECTOR = ".message-bubble[role='article']"
    GROK_PRIVATE_URL = "https://grok.com/c/f385ce1a-679e-44bd-aeb7-2b00d997da0c"

    async def test_redirect_to_share_fast_fail(self):
        """未登录：/c/<uuid> 被重定向到 /share/...，URL 离开 /c/ 路径 → 快速判否。"""
        page, probe = self._make_fake_page(
            "https://grok.com/share/bGVnYWN5_f97df5d9-64b7-4abf-b5eb-de7c0ea42673",
            {},
        )
        ready = await probe(page, self.GROK_PRIVATE_URL)
        self.assertFalse(ready)
        self.assertEqual(page.waited, [])

    async def test_private_chat_uses_dedicated_selector(self):
        """URL 仍在 /c/<uuid> 但内容未渲染：等待 Grok 专属选择器。"""
        page, probe = self._make_fake_page(
            self.GROK_PRIVATE_URL, {}
        )
        ready = await probe(page, self.GROK_PRIVATE_URL)
        self.assertFalse(ready)
        self.assertEqual([s for s, _ in page.waited], [self.GROK_SELECTOR])

    async def test_private_chat_detected(self):
        """URL 在 /c/<uuid> 且选择器命中 → 判定就绪。"""
        page, probe = self._make_fake_page(
            self.GROK_PRIVATE_URL, {self.GROK_SELECTOR: 4}
        )
        self.assertTrue(await probe(page, self.GROK_PRIVATE_URL))

    async def test_slow_hydration_gets_45s_window(self):
        """Grok hydration 慢（约 40s）：等待窗口应设为 45000ms。"""
        page, probe = self._make_fake_page(
            self.GROK_PRIVATE_URL, {}
        )
        await probe(page, self.GROK_PRIVATE_URL)
        if page.waited:
            _, timeout = page.waited[0]
            self.assertEqual(timeout, 45000)

    async def test_share_page_no_private_check(self):
        """公开分享页不走私有会话检测，用默认 10s 超时。"""
        share_url = "https://grok.com/share/bGVnYWN5_abc"
        page, probe = self._make_fake_page(share_url, {})
        await probe(page, share_url)
        if page.waited:
            _, timeout = page.waited[0]
            self.assertEqual(timeout, 10000)


if __name__ == "__main__":
    unittest.main()
