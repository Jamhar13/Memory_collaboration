"""Kimi（kimi.com）抓取适配器离线回归测试。

不联网、不起浏览器：用按 2026-09 真实分享页 DOM 构造的 HTML（Vue scoped
class：.segment-user/.segment-assistant、.markdown-container、
.segment-code、div.table、.attachment-list、.toolcall-flow 等）验证解析
逻辑，并验证注册表 host 路由与 GUI 登录判定。
"""

import unittest

from bs4 import BeautifulSoup

from scripts.providers import kimi
from scripts.providers import PROVIDERS, WAIT_SELECTOR
from scripts.providers import (
    parse_messages as registry_parse_messages,
    provider_for_host,
)
from gui.service import requires_authenticated_browser


def soup(html):
    return BeautifulSoup(html, "html.parser")


def user_segment(body):
    return f"""
    <div class="segment segment-user">
      <div class="segment-container" dir="auto">
        <div class="segment-content"><div class="segment-content-box">
          {body}
        </div></div>
      </div>
    </div>"""


def assistant_segment(body):
    return f"""
    <div class="segment segment-assistant">
      <div class="segment-avatar"><canvas class="rive-container"></canvas></div>
      <div class="segment-container" dir="auto">
        <div class="segment-content"><div class="segment-content-box">
          <div class="toolcall-rollup">
            <div class="toolcall-rollup__part toolcall-rollup__flat">
              <div class="markdown-container"><div class="markdown">
                {body}
              </div></div>
            </div>
          </div>
          <div class="okc-cards-container"><!----></div>
        </div></div>
      </div>
    </div>"""


# 典型两轮会话：含侧栏/输入框噪声、代码块、行内 code、表格、标题/列表/引用。
KIMI_HTML = f"""
<!DOCTYPE html><html><body>
<aside class="sidebar"><div>新建会话</div><div>登录</div></aside>
<div class="message-list">
  {user_segment('''
    <div class="toolcall-rollup"><div class="toolcall-rollup__part toolcall-rollup__flat">
      <div class="user-content"><span>请写一个 Python 加法函数</span></div>
    </div></div>
  ''')}
  {assistant_segment('''
    <div class="paragraph" dir="auto">好的，行内命令 <code class="segment-code-inline language-bash">
      pip install click</code> 如下。</div>
    <h3>方案</h3>
    <ul><li><div class="paragraph">先实现，再测试</div></li></ul>
    <blockquote><div class="paragraph">注意边界条件</div></blockquote>
    <hr>
    <div class="segment-code">
      <div class="segment-code-header">
        <span class="segment-code-lang">Python</span>
        <div class="segment-code-header-content"><button>复制</button></div>
      </div>
      <div class="syntax-highlighter light segment-code-content">
        <pre class="language-python"><code class="language-python"><span class="token keyword">def</span> add(a, b):
    <span class="token keyword">return</span> a + b</code></pre>
      </div>
    </div>
    <div class="table markdown-table">
      <div class="sticky-release">
        <div class="sticky-release-header">
          <header class="table-actions"><button>下载表格</button></header>
        </div>
      </div>
      <table>
        <thead><tr><th align="left">语言</th><th align="left">用途</th></tr></thead>
        <tbody>
          <tr><td align="left">Python</td><td align="left">脚本</td></tr>
          <tr><td align="left">Bash</td><td align="left">命令</td></tr>
        </tbody>
      </table>
    </div>
  ''')}
  {user_segment('<div class="toolcall-rollup"><div class="toolcall-rollup__part '
                'toolcall-rollup__flat"><div class="user-content"><span>第二轮问题</span>'
                '</div></div></div>')}
  {assistant_segment('<div class="paragraph">这是第二轮的回答。</div>')}
</div>
<div class="input-box"><textarea></textarea></div>
</body></html>
"""

# K3 智能体会话：思考/工具轨迹与正式答案并存。
KIMI_AGENT_HTML = f"""
<html><body><div class="message-list">
  {user_segment('<div class="toolcall-rollup"><div class="toolcall-rollup__part '
                 'toolcall-rollup__flat"><div class="user-content"><span>分析一下</span>'
                 '</div></div></div>')}
  <div class="segment segment-assistant">
    <div class="segment-container"><div class="segment-content"><div class="segment-content-box">
      <div class="toolcall-flow is-open">
        <div class="toolcall-flow__body"><div class="toolcall-flow__item">
          <div class="toolcall-container thinking-container"><div class="toolcall-content">
            <div class="markdown-container"><div class="markdown">
              <div class="paragraph">THINKING TRACE 中间思考</div>
            </div></div>
          </div></div>
        </div></div>
      </div>
      <div class="toolcall-rollup">
        <div class="toolcall-rollup__part toolcall-rollup__flat">
          <div class="markdown-container"><div class="markdown">
            <div class="paragraph">FINAL ANSWER 正式答案</div>
          </div></div>
        </div>
      </div>
    </div></div></div>
  </div>
</div></body></html>
"""

# 极端情况：只有轨迹 markdown，没有 flat 正式答案（降级取最后一个）。
KIMI_TRACE_ONLY_HTML = f"""
<html><body><div class="message-list">
  <div class="segment segment-assistant">
    <div class="segment-container"><div class="segment-content"><div class="segment-content-box">
      <div class="toolcall-flow"><div class="toolcall-flow__body"><div class="toolcall-flow__item">
        <div class="markdown-container"><div class="markdown">
          <div class="paragraph">TRACE ONLY LAST RESORT</div>
        </div></div>
      </div></div></div>
    </div></div></div>
  </div>
</div></body></html>
"""

# 用户图片附件（kimi.moonshot.cn 签名 URL）+ 文件附件（data-svg 图标）。
KIMI_ATTACHMENT_HTML = f"""
<html><body>
  {user_segment('''
    <div class="attachment-list">
      <div class="attachment-list-image single">
        <div class="image-thumbnail large success">
          <span class="image-wrapper image-detail">
            <img class="image-main is-cover" loading="lazy" alt="作业截图"
                 src="https://kimi.moonshot.cn/api/sign-obj/kfs%2F1%2Fx.jpeg?sig=abc&amp;t=t">
          </span>
        </div>
      </div>
      <div class="attachment-list-file">
        <div class="file-card-container normal success">
          <img class="file-card-icon" src="data:image/svg+xml,%3csvg/%3e">
          <div class="file-card-name">食品安全法第一章总则</div>
          <div class="file-card-meta"><span>DOC</span><span>22.5 KB</span></div>
        </div>
      </div>
    </div>
    <div class="toolcall-rollup"><div class="toolcall-rollup__part toolcall-rollup__flat">
      <div class="user-content"><span>Answer</span></div>
    </div></div>
  ''')}
</body></html>
"""

IMAGE_MAP = {
    "https://kimi.moonshot.cn/api/sign-obj/kfs%2F1%2Fx.jpeg?sig=abc&t=t":
        "./images/img_1_kimi.jpg",
}


class KimiParseMessagesTests(unittest.TestCase):
    def test_non_kimi_pages_return_none(self):
        self.assertIsNone(kimi.parse_messages(soup("<html><body></body></html>")))
        self.assertIsNone(kimi.parse_messages(soup(
            '<div data-message-author-role="user">ChatGPT</div>'
        )))
        self.assertIsNone(kimi.parse_messages(soup(
            '<div data-virtual-list-item-key="1"><div class="ds-message">x</div></div>'
        )))
        self.assertIsNone(kimi.parse_messages(soup(
            '<div class="message-item">豆包</div>'
        )))
        self.assertIsNone(kimi.parse_messages(soup(
            "<user-query><div class='query-text'>Gemini</div></user-query>"
        )))

    def test_roles_and_strict_order(self):
        messages = kimi.parse_messages(soup(KIMI_HTML), IMAGE_MAP)
        self.assertEqual(
            [m["role"] for m in messages],
            ["User", "AI", "User", "AI"],
        )
        self.assertIn("请写一个 Python 加法函数", messages[0]["content"])
        self.assertIn("第二轮问题", messages[2]["content"])
        self.assertIn("这是第二轮的回答", messages[3]["content"])

    def test_sidebar_and_chrome_not_in_content(self):
        messages = kimi.parse_messages(soup(KIMI_HTML), IMAGE_MAP)
        joined = "\n".join(m["content"] for m in messages)
        for noise in ["新建会话", "下载表格", "复制", "segment-avatar"]:
            self.assertNotIn(noise, joined)

    def test_code_block_preserved_with_language(self):
        messages = kimi.parse_messages(soup(KIMI_HTML), IMAGE_MAP)
        content = messages[1]["content"]
        self.assertIn("```python", content)
        self.assertIn("def add(a, b):", content)
        self.assertIn("return a + b", content)
        # 工具栏语言名不作为正文重复出现。
        self.assertNotIn("segment-code", content)

    def test_inline_code_preserved(self):
        messages = kimi.parse_messages(soup(KIMI_HTML), IMAGE_MAP)
        self.assertIn("`pip install click`", messages[1]["content"])

    def test_headings_lists_blockquote_converted(self):
        content = kimi.parse_messages(soup(KIMI_HTML), IMAGE_MAP)[1]["content"]
        self.assertIn("### 方案", content)
        self.assertIn("* 先实现，再测试", content)
        self.assertIn("> 注意边界条件", content)
        self.assertIn("---", content)

    def test_table_converted_to_gfm(self):
        content = kimi.parse_messages(soup(KIMI_HTML), IMAGE_MAP)[1]["content"]
        self.assertIn("| 语言 | 用途 |", content)
        self.assertIn("| :--- | :--- |", content)
        self.assertIn("| Python | 脚本 |", content)
        self.assertIn("| Bash | 命令 |", content)
        # 表格界面壳被剔除。
        self.assertNotIn("table-actions", content)
        self.assertNotIn("<table", content)

    def test_agent_trace_excluded_but_final_answer_kept(self):
        messages = kimi.parse_messages(soup(KIMI_AGENT_HTML), IMAGE_MAP)
        self.assertEqual(len(messages), 2)
        self.assertNotIn("THINKING TRACE", messages[1]["content"])
        self.assertIn("FINAL ANSWER 正式答案", messages[1]["content"])

    def test_trace_only_falls_back_to_last_container(self):
        messages = kimi.parse_messages(soup(KIMI_TRACE_ONLY_HTML), IMAGE_MAP)
        self.assertEqual(len(messages), 1)
        self.assertIn("TRACE ONLY LAST RESORT", messages[0]["content"])

    def test_user_image_localized(self):
        messages = kimi.parse_messages(soup(KIMI_ATTACHMENT_HTML), IMAGE_MAP)
        content = messages[0]["content"]
        self.assertIn("![作业截图](./images/img_1_kimi.jpg)", content)
        # data:svg 图标不进入正文。
        self.assertNotIn("data:image/svg", content)

    def test_user_file_attachment_placeholder(self):
        messages = kimi.parse_messages(soup(KIMI_ATTACHMENT_HTML), IMAGE_MAP)
        content = messages[0]["content"]
        self.assertIn("📎 **[上传文件]**", content)
        self.assertIn("`食品安全法第一章总则.doc`", content)
        self.assertIn("（22.5KB）", content)
        # 正文仍保留。
        self.assertIn("Answer", content)

    def test_empty_segments_produce_none(self):
        html = (
            '<div class="segment segment-user"></div>'
            '<div class="segment segment-assistant"></div>'
        )
        self.assertIsNone(kimi.parse_messages(soup(html), IMAGE_MAP))


class KimiRegistryTests(unittest.TestCase):
    def test_kimi_registered(self):
        self.assertIn(kimi, PROVIDERS)

    def test_host_routing(self):
        for host in ("kimi.com", "www.kimi.com", "kimi.moonshot.cn"):
            with self.subTest(host=host):
                self.assertIs(provider_for_host(host), kimi)
        self.assertIs(provider_for_host("KIMI.COM:443"), kimi)
        self.assertIsNone(provider_for_host("notkimi.com"))

    def test_combined_wait_selector_includes_kimi(self):
        self.assertIn(".segment-user", WAIT_SELECTOR)

    def test_registry_parse_detects_kimi(self):
        provider, messages = registry_parse_messages(
            soup(KIMI_HTML), IMAGE_MAP
        )
        self.assertIs(provider, kimi)
        self.assertEqual(len(messages), 4)


class KimiLoginPolicyTests(unittest.TestCase):
    def test_share_pages_never_require_login(self):
        for url in (
            "https://www.kimi.com/share/d2nhdgle09netp24ee9g?ra=1",
            "https://kimi.com/share/en/d3vt9q6ahlm8pgjb09eg",
            "https://www.kimi.com/share/zh/19a99232-7452-86a8-8000-x",
            "https://kimi.moonshot.cn/share/cthokgvhvltpvm2n4pbg",
        ):
            with self.subTest(url=url):
                self.assertFalse(requires_authenticated_browser(url))

    def test_private_chat_requires_login(self):
        self.assertTrue(requires_authenticated_browser(
            "https://www.kimi.com/chat/d2l7j01sfuv85si8hns0"
            "?chat_enter_method=history"
        ))
        self.assertTrue(requires_authenticated_browser(
            "https://kimi.com/chat/19a99232-7452-86a8-8000-00006932e5ff"
        ))

    def test_non_conversation_chat_paths_do_not_require_login(self):
        self.assertFalse(
            requires_authenticated_browser("https://www.kimi.com/chat")
        )
        self.assertFalse(
            requires_authenticated_browser("https://www.kimi.com/chat/history")
        )
        self.assertFalse(
            requires_authenticated_browser("https://www.kimi.com/")
        )


KIMI_PRIVATE_URL = "https://www.kimi.com/chat/19a99232-7452-86a8-8000-x"
GEMINI_PRIVATE_URL = "https://gemini.google.com/app/c94a4c03179dd07a"
KIMI_SELECTOR = ".segment-user, .segment-assistant"
GEMINI_SELECTOR = "user-query, message-content, model-response"


class ContentProbeTests(unittest.IsolatedAsyncioTestCase):
    """私有会话探测：未登录被重定向到首页时快速判否；仍留在会话页时
    等待该平台专属选择器（首页营销壳含 .message-item，不能走联合选择器）。"""

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

    async def test_kimi_redirect_home_fast_fail(self):
        # 未登录：/chat/<id> 被重定向到首页，即使首页壳含 .message-item
        # 也必须立即判否，且不等待选择器。
        page, probe = self._make_fake_page(
            "https://www.kimi.com/", {".message-item": 1}
        )
        ready = await probe(page, KIMI_PRIVATE_URL)
        self.assertFalse(ready)
        self.assertEqual(page.waited, [])

    async def test_kimi_on_chat_url_uses_dedicated_selector(self):
        # URL 仍在会话页但内容未渲染完（只有首页式噪声节点）。
        page, probe = self._make_fake_page(
            KIMI_PRIVATE_URL, {".message-item": 1}
        )
        ready = await probe(page, KIMI_PRIVATE_URL)
        self.assertFalse(ready)
        self.assertEqual([s for s, _ in page.waited], [KIMI_SELECTOR])

    async def test_kimi_private_conversation_detected(self):
        page, probe = self._make_fake_page(
            KIMI_PRIVATE_URL, {KIMI_SELECTOR: 2}
        )
        self.assertTrue(await probe(page, KIMI_PRIVATE_URL))

    async def test_gemini_redirect_to_app_landing_fast_fail(self):
        # 未登录：/app/<id> 被重定向到无 id 的 /app 登录墙。
        page, probe = self._make_fake_page(
            "https://gemini.google.com/app", {}
        )
        ready = await probe(page, GEMINI_PRIVATE_URL)
        self.assertFalse(ready)
        self.assertEqual(page.waited, [])

    async def test_gemini_slow_hydration_gets_long_wait(self):
        # 已登录但无头冷启动 hydration 慢（实测约 25 秒）：
        # 仍在 /app/<id> 时给 30 秒等待，不能 10 秒就误判需要登录。
        page, probe = self._make_fake_page(
            GEMINI_PRIVATE_URL, {GEMINI_SELECTOR: 4}
        )
        ready = await probe(page, GEMINI_PRIVATE_URL)
        self.assertTrue(ready)
        self.assertEqual(page.waited, [(GEMINI_SELECTOR, 30000)])


if __name__ == "__main__":
    unittest.main()
