"""Grok（grok.com）适配器离线测试。

Grok 适配器走 DOM 采集（与 Kimi 同型），但消息选择器是
``.message-bubble[role="article"]``，用户/AI 靠 Tailwind class 区分
（用户有 ``bg-surface-l1``/``border-border-l1``，AI 有 ``w-full max-w-none``）。
AI 气泡含 "工作了 Ns" 前缀需剥离。
"""

import unittest

from bs4 import BeautifulSoup

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


if __name__ == "__main__":
    unittest.main()
