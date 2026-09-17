"""Codex 适配器与登录路由测试。"""
import tempfile
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from gui.service import (
    _codex_payload_messages,
    _collect_response_assets,
    _rewrite_codex_local_links,
    _page_has_conversation_content,
    _parse_page_messages,
    _save_codex_file_changes,
    requires_authenticated_browser,
)
from scripts.providers import PROVIDERS, chatgpt, codex, provider_for_host, provider_for_url


CODEX_HTML = """
<section data-turn-key="one">
  <article id="message-user">
    <div data-user-message-bubble="true">
      <div data-selected-text-overlay-target><p>检查 <code>app.py</code></p></div>
    </div>
  </article>
</section>
<section data-turn-key="two">
  <article id="message-assistant">
    <div data-selected-text-overlay-target>
      <h2>结果</h2><p><img src="https://example.com/a.png"></p>
      <p><a href="https://example.com/report.pdf">report.pdf</a></p>
      <button>复制</button><pre><code>print("ok")</code></pre>
    </div>
  </article>
</section>
"""


class CodexProviderTests(unittest.TestCase):
    def test_parses_roles_markdown_and_local_assets(self):
        messages = codex.parse_messages(
            BeautifulSoup(CODEX_HTML, "html.parser"),
            {
                "https://example.com/a.png": "./chat_images/a.png",
                "report.pdf": "./chat_files/report.pdf",
            },
        )
        self.assertEqual([item["role"] for item in messages], ["User", "AI"])
        self.assertIn("`app.py`", messages[0]["content"])
        self.assertIn("## 结果", messages[1]["content"])
        self.assertIn("./chat_images/a.png", messages[1]["content"])
        self.assertIn("./chat_files/report.pdf", messages[1]["content"])
        self.assertIn('print("ok")', messages[1]["content"])
        self.assertNotIn("复制", messages[1]["content"])

    def test_non_codex_html_returns_none(self):
        soup = BeautifulSoup("<article id='message-x'>普通页面</article>", "html.parser")
        self.assertIsNone(codex.parse_messages(soup))

    def test_verification_page_is_not_fallback_content(self):
        soup = BeautifulSoup(
            "<main>请稍候… 验证成功。正在等待 chatgpt.com 响应 "
            "Enable JavaScript and cookies to continue</main>",
            "html.parser",
        )
        provider, messages = _parse_page_messages(
            "https://chatgpt.com/s/cx_6aab6a1018148191854778b04ecc6865",
            soup,
            {},
        )
        self.assertIsNone(provider)
        self.assertIsNone(messages)

    def test_registered_as_dom_only_provider(self):
        self.assertIs(PROVIDERS[-1], codex)
        self.assertEqual(codex.HOSTS, ())
        self.assertIs(provider_for_host("chatgpt.com"), chatgpt)
        self.assertIs(
            provider_for_url("https://chatgpt.com/s/cx_6aab6a1018148191854778b04ecc6865"),
            codex,
        )


class CodexFileChangeTests(unittest.TestCase):
    def test_payload_removes_source_computer_local_links(self):
        payload = {"turns": [{"items": [{
            "type": "agentMessage",
            "text": (
                "见 [README.md](C:\\Users\\old\\project\\README.md:3) "
                "和 [官网](https://example.com)"
            ),
        }]}]}
        content = _codex_payload_messages(payload)[0]["content"]
        self.assertEqual(
            _rewrite_codex_local_links(content),
            "见 README.md 和 [官网](https://example.com)",
        )

    def test_rewrites_matching_source_path_to_downloaded_file(self):
        text = "见 [README.md](C:\\Users\\old\\project\\README.md:3)"
        self.assertEqual(
            _rewrite_codex_local_links(
                text, {"README.md": "./AI_memory_files/README.md"}
            ),
            "见 [README.md](./AI_memory_files/README.md)",
        )

    def test_rewrites_code_file_references_to_downloaded_files(self):
        text = (
            "新增 `scripts/providers/gemini.py`，修改 `README.md:3`，"
            "保留 `parse_messages()`。"
        )
        self.assertEqual(
            _rewrite_codex_local_links(text, {
                "scripts/providers/gemini.py": (
                    "./AI_memory_files/scripts__providers__gemini.py"
                ),
                "README.md": "./AI_memory_files/README.md",
            }),
            "新增 [scripts/providers/gemini.py]"
            "(./AI_memory_files/scripts__providers__gemini.py)，"
            "修改 [README.md](./AI_memory_files/README.md)，"
            "保留 `parse_messages()`。",
        )

    def test_payload_uses_latest_agent_message_when_final_is_absent(self):
        payload = {"turns": [{"items": [
            {"type": "userMessage", "content": [{"type": "text", "text": "全量版"}]},
            {"type": "agentMessage", "text": "正在构建", "phase": "commentary"},
        ]}]}
        self.assertEqual(_codex_payload_messages(payload), [
            {"role": "User", "content": "全量版"},
            {"role": "AI", "content": "正在构建"},
        ])

    def test_codex_source_attachment_is_a_download_candidate(self):
        candidates = []
        _collect_response_assets(
            {"filename": "example.py", "file_id": "file_123456789012"},
            "https://chatgpt.com/s/cx_12345678",
            candidates,
            set(),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].filename, "example.py")

    def test_saves_each_change_with_original_extension_and_file_uri(self):
        changes = [{
            "path": "../gui/service.py",
            "diff": "@@ -1 +1 @@\n-old\n+new",
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "result_files"
            mapping = _save_codex_file_changes(
                changes, output_dir, "./result_files"
            )
            saved = list(output_dir.iterdir())
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].suffix, ".py")
            self.assertEqual(saved[0].parent, output_dir)
            content = saved[0].read_text(encoding="utf-8")
            self.assertIn("文件：../gui/service.py", content)
            self.assertIn("并非完整文件", content)
            self.assertIn("@@ -1 +1 @@", content)
            self.assertEqual(
                mapping["../gui/service.py"],
                f"./result_files/{saved[0].name}",
            )

    def test_keeps_chinese_filename_unencoded_for_local_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping = _save_codex_file_changes(
                [{"path": "工作文档/说明.md", "diff": "+内容"}],
                Path(temp_dir),
                "./结果 文件",
            )
        self.assertEqual(
            mapping["工作文档/说明.md"],
            "./结果 文件/工作文档__说明.md",
        )


class CodexAuthTests(unittest.TestCase):
    def test_public_share_does_not_require_login(self):
        self.assertFalse(requires_authenticated_browser(
            "https://chatgpt.com/s/cx_6aab6a1018148191854778b04ecc6865"
        ))

    def test_private_task_requires_login(self):
        self.assertTrue(requires_authenticated_browser(
            "https://chatgpt.com/codex/cloud/tasks/task_e_12345678"
        ))


class CodexContentProbeTests(unittest.IsolatedAsyncioTestCase):
    class Page:
        def __init__(self, url, count=0):
            self.url = url
            self.node_count = count
            self.waited = []

        async def wait_for_selector(self, selector, **kwargs):
            self.waited.append(selector)

        def locator(self, selector):
            return self

        async def count(self):
            return self.node_count

    async def test_public_share_uses_codex_selector(self):
        url = "https://chatgpt.com/s/cx_6aab6a1018148191854778b04ecc6865"
        page = self.Page(url, 2)
        self.assertTrue(await _page_has_conversation_content(page, url))
        self.assertEqual(page.waited, [codex.WAIT_SELECTOR])

    async def test_private_redirect_fails_without_waiting(self):
        requested = "https://chatgpt.com/codex/cloud/tasks/task_e_12345678"
        page = self.Page("https://chatgpt.com/codex")
        self.assertFalse(await _page_has_conversation_content(page, requested))
        self.assertEqual(page.waited, [])


if __name__ == "__main__":
    unittest.main()
