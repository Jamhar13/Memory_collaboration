"""适配器注册表的 host 路由与抓取前等待逻辑测试。"""
import asyncio
import unittest

from scripts.parser import _wait_for_conversation_content
from scripts.providers import (
    PROVIDERS,
    WAIT_SELECTOR,
    chatgpt,
    deepseek,
    doubao,
    gemini,
    kimi,
    qianwen,
    provider_for_host,
)


class ProviderForHostTests(unittest.TestCase):
    def test_known_hosts_map_to_expected_provider(self):
        cases = {
            "chatgpt.com": chatgpt,
            "chat.openai.com": chatgpt,
            "chat.deepseek.com": deepseek,
            "doubao.com": doubao,
            "www.doubao.com": doubao,
            "gemini.google.com": gemini,
            "share.gemini.google": gemini,
            "kimi.com": kimi,
            "www.kimi.com": kimi,
            "kimi.moonshot.cn": kimi,
            "qianwen.com": qianwen,
            "www.qianwen.com": qianwen,
            "qianwen.my.cn": qianwen,
        }
        for host, expected in cases.items():
            with self.subTest(host=host):
                self.assertIs(provider_for_host(host), expected)

    def test_unknown_host_returns_none(self):
        self.assertIsNone(provider_for_host("example.com"))
        # 近似域名（多一段）不得误命中。
        self.assertIsNone(provider_for_host("share.gemini.google.com"))
        self.assertIsNone(provider_for_host(""))
        self.assertIsNone(provider_for_host(None))

    def test_host_normalized_case_and_port(self):
        self.assertIs(provider_for_host("GEMINI.GOOGLE.COM"), gemini)
        self.assertIs(provider_for_host("gemini.google.com:443"), gemini)
        self.assertIs(provider_for_host("Chat.DeepSeek.com:8443"), deepseek)

    def test_every_provider_declares_distinct_nonempty_hosts(self):
        seen = {}
        for provider in PROVIDERS:
            hosts = getattr(provider, "HOSTS", ())
            self.assertTrue(
                hosts, f"{provider.DISPLAY_NAME} 缺少 HOSTS 声明"
            )
            for host in hosts:
                self.assertNotIn(host, seen, f"HOSTS 冲突: {host}")
                seen[host] = provider


class WaitForConversationContentTests(unittest.TestCase):
    """用伪页面对象验证等待目标选择与降级路径（不启动浏览器）。"""

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def test_known_host_waits_on_provider_selector(self):
        calls = []

        class FakePage:
            url = "https://gemini.google.com/share/abc?skid=x"

            async def wait_for_selector(self, selector, state=None, timeout=None):
                calls.append((selector, state, timeout))

        self._run(_wait_for_conversation_content(FakePage()))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], gemini.WAIT_SELECTOR)
        self.assertEqual(calls[0][1], "attached")

    def test_short_share_domain_locks_gemini_before_redirect(self):
        calls = []

        class FakePage:
            url = "https://share.gemini.google/qER3ntHn8kOP"

            async def wait_for_selector(self, selector, state=None, timeout=None):
                calls.append(selector)

        self._run(_wait_for_conversation_content(FakePage()))
        self.assertEqual(calls, [gemini.WAIT_SELECTOR])

    def test_unknown_host_falls_back_to_combined_selector(self):
        calls = []

        class FakePage:
            url = "https://example.com/chat/123"

            async def wait_for_selector(self, selector, state=None, timeout=None):
                calls.append(selector)

        self._run(_wait_for_conversation_content(FakePage()))
        self.assertEqual(calls, [WAIT_SELECTOR])

    def test_provider_timeout_then_combined_fallback(self):
        calls = []

        class FakePage:
            url = "https://gemini.google.com/share/abc"

            async def wait_for_selector(self, selector, state=None, timeout=None):
                calls.append(selector)
                if selector == gemini.WAIT_SELECTOR:
                    raise TimeoutError("timeout")

        self._run(_wait_for_conversation_content(FakePage()))
        self.assertEqual(calls, [gemini.WAIT_SELECTOR, WAIT_SELECTOR])

    def test_all_timeouts_do_not_raise(self):
        class FakePage:
            url = "https://example.com/x"

            async def wait_for_selector(self, selector, state=None, timeout=None):
                raise TimeoutError("timeout")

        # 两次等待均超时也不应抛出异常中断抓取。
        self._run(_wait_for_conversation_content(FakePage()))


if __name__ == "__main__":
    unittest.main()
