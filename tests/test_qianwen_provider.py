"""通义千问（qianwen.com）适配器离线测试。

千问适配器与其他平台不同——不走 DOM 采集，而是直接调 API 获取
完整对话 JSON（AI 回答为 markdown），合成简单 HTML 供 parse_messages 解析。
"""

import unittest
from bs4 import BeautifulSoup

from scripts.providers import qianwen


def _build_html(messages):
    """构造与 collect_html 输出格式一致的 HTML。

    :param messages: ``[(role, content), ...]`` 列表
    """
    import html as _html

    frags = []
    for role, content in messages:
        cls = "qk-user" if role == "User" else "qk-ai"
        frags.append(
            f'<div class="{cls}" data-content="'
            f"{_html.escape(content, quote=True)}"
            f'"></div>'
        )
    return (
        "<!DOCTYPE html><html><body>"
        + "\n".join(frags)
        + "</body></html>"
    )


class QianwenParseMessagesTests(unittest.TestCase):
    """parse_messages 基础解析。"""

    def test_basic_user_ai_alternation(self):
        html = _build_html([
            ("User", "什么是插值"),
            ("AI", "## 什么是插值\n\n插值是一种数学方法..."),
        ])
        soup = BeautifulSoup(html, "html.parser")
        provider, messages = qianwen, qianwen.parse_messages(soup, {})
        self.assertIsNotNone(messages)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "User")
        self.assertEqual(messages[0]["content"], "什么是插值")
        self.assertEqual(messages[1]["role"], "AI")
        self.assertIn("插值", messages[1]["content"])

    def test_multiple_rounds(self):
        html = _build_html([
            ("User", "插值是什么"),
            ("AI", "## 什么是插值\n\n插值是..."),
            ("User", "离散是什么"),
            ("AI", "## 什么是离散\n\n离散是..."),
            ("User", "Latex是什么"),
            ("AI", "## LaTeX\n\nLaTeX是..."),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = qianwen.parse_messages(soup, {})
        self.assertEqual(len(messages), 6)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["User", "AI", "User", "AI", "User", "AI"])

    def test_markdown_preserved(self):
        md = (
            "## 标题\n\n"
            "**粗体** 和 *斜体*\n\n"
            "- 列表项一\n- 列表项二\n\n"
            "> 引用块\n\n"
            "```\ncode block\n```\n"
        )
        html = _build_html([("User", "问题"), ("AI", md)])
        soup = BeautifulSoup(html, "html.parser")
        messages = qianwen.parse_messages(soup, {})
        self.assertEqual(messages[1]["content"], md)

    def test_html_in_content_escaped(self):
        """markdown 中的 < > 被 HTML-escape，解析后还原。"""
        md = "比较 `a < b` 和 `c > d` 的结果"
        html = _build_html([("User", "问题"), ("AI", md)])
        soup = BeautifulSoup(html, "html.parser")
        messages = qianwen.parse_messages(soup, {})
        self.assertEqual(messages[1]["content"], md)

    def test_non_qianwen_html_returns_none(self):
        html = '<html><body><div class="message-item">其他平台</div></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        self.assertIsNone(qianwen.parse_messages(soup, {}))

    def test_empty_messages_returns_none(self):
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        self.assertIsNone(qianwen.parse_messages(soup, {}))

    def test_empty_user_content_skipped(self):
        html = _build_html([
            ("User", ""),
            ("User", "有内容"),
            ("AI", "AI回答"),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = qianwen.parse_messages(soup, {})
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "有内容")
        self.assertEqual(messages[1]["content"], "AI回答")

    def test_empty_ai_content_skipped(self):
        html = _build_html([
            ("User", "问题"),
            ("AI", ""),
            ("AI", "有内容"),
        ])
        soup = BeautifulSoup(html, "html.parser")
        messages = qianwen.parse_messages(soup, {})
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "问题")
        self.assertEqual(messages[1]["content"], "有内容")

    def test_image_map_replaces_urls(self):
        md = "![图片](https://example.com/img.png) 和文本"
        html = _build_html([("User", "问题"), ("AI", md)])
        soup = BeautifulSoup(html, "html.parser")
        image_map = {"https://example.com/img.png": "./images/img.png"}
        messages = qianwen.parse_messages(soup, image_map)
        self.assertIn("./images/img.png", messages[1]["content"])
        self.assertNotIn("https://example.com/img.png", messages[1]["content"])


class QianwenShareIdPatternTests(unittest.TestCase):
    """share_id 正则匹配。"""

    def test_match_qianwen_com(self):
        url = "https://www.qianwen.com/share/chat/e60fcf3d455a4e1dbb8106cba69513d6"
        m = qianwen._SHARE_ID_PATTERN.search(url)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "e60fcf3d455a4e1dbb8106cba69513d6")

    def test_match_qianwen_my_cn(self):
        url = "https://qianwen.my.cn/share/chat/e60fcf3d455a4e1dbb8106cba69513d6"
        m = qianwen._SHARE_ID_PATTERN.search(url)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "e60fcf3d455a4e1dbb8106cba69513d6")

    def test_match_with_query_params(self):
        url = (
            "https://www.qianwen.com/share/chat/27a0ba3131a54421906f8d070c646a35"
            "?biz_id=ai_qwen&env=prod&qwcontainer=qk"
        )
        m = qianwen._SHARE_ID_PATTERN.search(url)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "27a0ba3131a54421906f8d070c646a35")

    def test_no_match_chat_url(self):
        url = "https://www.qianwen.com/chat"
        self.assertIsNone(qianwen._SHARE_ID_PATTERN.search(url))

    def test_session_id_pattern_matches_private_chat(self):
        url = "https://www.qianwen.com/chat/da38250d7a4848dd9657977a6ad98faf"
        m = qianwen._SESSION_ID_PATTERN.search(url)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "da38250d7a4848dd9657977a6ad98faf")

    def test_session_id_pattern_matches_private_chat_with_query(self):
        url = ("https://www.qianwen.com/chat/da38250d7a4848dd9657977a6ad98faf"
               "?source=tongyigw&bizPassParams=foo")
        m = qianwen._SESSION_ID_PATTERN.search(url)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "da38250d7a4848dd9657977a6ad98faf")

    def test_session_id_pattern_no_match_share(self):
        url = "https://www.qianwen.com/share/chat/e60fcf3d455a4e1dbb8106cba69513d6"
        self.assertIsNone(qianwen._SESSION_ID_PATTERN.search(url))

    def test_no_match_other_platform(self):
        url = "https://www.kimi.com/share/abc123"
        self.assertIsNone(qianwen._SHARE_ID_PATTERN.search(url))


class QianwenRegistryTests(unittest.TestCase):
    """适配器注册与 host 路由。"""

    def test_display_name(self):
        self.assertEqual(qianwen.DISPLAY_NAME, "千问")

    def test_hosts(self):
        for h in ("qianwen.com", "www.qianwen.com", "qianwen.my.cn"):
            self.assertIn(h, qianwen.HOSTS)

    def test_registered_in_providers(self):
        from scripts.providers import PROVIDERS
        self.assertIn(qianwen, PROVIDERS)

    def test_provider_registered(self):
        """千问应已注册（不要求末位）。"""
        from scripts.providers import PROVIDERS
        self.assertIn(qianwen, PROVIDERS)

    def test_provider_for_host_routes(self):
        from scripts.providers import provider_for_host
        for h in ("qianwen.com", "www.qianwen.com", "qianwen.my.cn"):
            self.assertIs(provider_for_host(h), qianwen)
        self.assertIsNone(provider_for_host("unknown.com"))


class QianwenRequiresAuthTests(unittest.TestCase):
    """requires_authenticated_browser 对千问链接的判定。"""

    def test_share_link_no_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertFalse(
            requires_authenticated_browser(
                "https://www.qianwen.com/share/chat/"
                "e60fcf3d455a4e1dbb8106cba69513d6"
            )
        )

    def test_share_link_my_cn_no_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertFalse(
            requires_authenticated_browser(
                "https://qianwen.my.cn/share/chat/"
                "e60fcf3d455a4e1dbb8106cba69513d6"
            )
        )

    def test_private_chat_needs_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertTrue(
            requires_authenticated_browser(
                "https://www.qianwen.com/chat"
            )
        )

    def test_private_chat_with_params_needs_auth(self):
        from gui.service import requires_authenticated_browser
        self.assertTrue(
            requires_authenticated_browser(
                "https://www.qianwen.com/chat?session_id=abc"
            )
        )


if __name__ == "__main__":
    unittest.main()
