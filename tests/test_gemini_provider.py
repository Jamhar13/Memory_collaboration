"""Gemini（gemini.google.com）抓取适配器离线回归测试。

不联网、不起浏览器：用构造的 Gemini 网页 HTML（自定义元素
<user-query>/<message-content>/<model-response>）验证解析逻辑，
并验证 GUI 服务层对 Gemini 私有/公开链接的登录判定。
"""

import unittest

from bs4 import BeautifulSoup

from scripts.providers import gemini
from scripts.providers import PROVIDERS, WAIT_SELECTOR
from gui.service import requires_authenticated_browser


def soup(html):
    return BeautifulSoup(html, "html.parser")


# 一段典型的 Gemini 会话：含用户图片、代码块、行内/块级公式、反馈按钮噪声，
# 且第一条回答用 <model-response> 包裹 <message-content>（外层包装）。
GEMINI_HTML = """
<!DOCTYPE html><html><body>
<side-nav><a>新建对话</a></side-nav>
<conversation-view>
  <user-query>
    <div class="query-text">请写一个 Python 加法函数，并解释公式</div>
    <img src="https://lh3.googleusercontent.com/d/abc123" alt="我上传的截图">
  </user-query>
  <model-response>
    <message-content>
      <div class="markdown-main-panel">
        <p>好的，行内公式 <span data-math="x^2+y^2">x²+y²</span> 如下。</p>
        <div data-math="\\int_0^1 x\\,dx">∫ 0 1 x dx</div>
        <code-block>
          <div class="code-header"><span>Python</span><button>复制</button></div>
          <pre><code class="language-python">def add(a, b):
    return a + b</code></pre>
        </code-block>
      </div>
      <div class="message-actions">
        <button>好回答</button><button>坏回答</button><button>分享</button>
      </div>
    </message-content>
  </model-response>
  <user-query>
    <div class="query-text">第二轮：再问一个问题</div>
  </user-query>
  <message-content>
    <div class="model-response-text"><p>这是第二轮的回答。</p></div>
  </message-content>
</conversation-view>
<input-area><textarea></textarea></input-area>
</body></html>
"""

# collect_html 采集后拼装出的合成文档形态（仅片段，无侧栏/输入框）。
GEMINI_FRAGMENT_HTML = """
<!DOCTYPE html><html><body>
<user-query data-gemini-role="user" data-gemini-order="0">
  <div class="query-text">片段里的提问</div>
</user-query>
<message-content data-gemini-role="assistant" data-gemini-order="1">
  <div class="model-response-text"><p>片段里的回答</p></div>
</message-content>
</body></html>
"""

IMAGE_MAP = {
    "https://lh3.googleusercontent.com/d/abc123": "./images/gemini_001.jpg",
}


class GeminiParseMessagesTests(unittest.TestCase):
    def test_non_gemini_pages_return_none(self):
        self.assertIsNone(gemini.parse_messages(soup("<html><body></body></html>")))
        self.assertIsNone(gemini.parse_messages(soup(
            '<div data-message-author-role="user">ChatGPT</div>'
        )))
        self.assertIsNone(gemini.parse_messages(soup(
            '<div data-virtual-list-item-key="1"><div class="ds-message">x</div></div>'
        )))
        self.assertIsNone(gemini.parse_messages(soup(
            '<div class="message-item">豆包</div>'
        )))

    def test_roles_and_strict_order(self):
        messages = gemini.parse_messages(soup(GEMINI_HTML), IMAGE_MAP)
        self.assertEqual(
            [m["role"] for m in messages],
            ["User", "AI", "User", "AI"],
        )
        self.assertIn("请写一个 Python 加法函数", messages[0]["content"])
        self.assertIn("第二轮：再问一个问题", messages[2]["content"])
        self.assertIn("这是第二轮的回答", messages[3]["content"])

    def test_nested_response_wrapper_not_duplicated(self):
        messages = gemini.parse_messages(soup(GEMINI_HTML), IMAGE_MAP)
        # 第一条 AI 回答同时命中外层 model-response 与内层 message-content，
        # 只能产生一条 AI 消息。
        self.assertEqual(len(messages), 4)
        self.assertIn("好的，行内公式", messages[1]["content"])

    def test_code_block_preserved(self):
        messages = gemini.parse_messages(soup(GEMINI_HTML), IMAGE_MAP)
        content = messages[1]["content"]
        self.assertIn("```python", content)
        self.assertIn("def add(a, b):", content)

    def test_math_restored_to_latex(self):
        messages = gemini.parse_messages(soup(GEMINI_HTML), IMAGE_MAP)
        content = messages[1]["content"]
        self.assertIn("$x^2+y^2$", content)
        self.assertIn("$$", content)
        self.assertIn(r"\int_0^1 x\,dx", content)
        # 渲染后的视觉残留不应进入正文。
        self.assertNotIn("∫", content)

    def test_user_image_localized(self):
        messages = gemini.parse_messages(soup(GEMINI_HTML), IMAGE_MAP)
        self.assertIn("./images/gemini_001.jpg", messages[0]["content"])
        self.assertIn("![", messages[0]["content"])
        # 远程地址不应泄漏到最终正文。
        self.assertNotIn("lh3.googleusercontent.com", messages[0]["content"])

    def test_interface_noise_removed(self):
        messages = gemini.parse_messages(soup(GEMINI_HTML), IMAGE_MAP)
        content = "\n".join(m["content"] for m in messages)
        for noise in ("好回答", "坏回答", "复制", "分享", "新建对话"):
            self.assertNotIn(noise, content)

    def test_data_uri_image_skipped(self):
        html = """
        <user-query>
          <img src="data:image/png;base64,AAAA" alt="占位">
          <div class="query-text">只有文字</div>
        </user-query>
        """
        messages = gemini.parse_messages(soup(html), {})
        self.assertEqual(messages[0]["role"], "User")
        self.assertIn("只有文字", messages[0]["content"])
        self.assertNotIn("data:image", messages[0]["content"])

    def test_fragment_html_from_collector_parses(self):
        messages = gemini.parse_messages(soup(GEMINI_FRAGMENT_HTML), {})
        self.assertEqual(
            [(m["role"]) for m in messages],
            ["User", "AI"],
        )
        self.assertIn("片段里的提问", messages[0]["content"])
        self.assertIn("片段里的回答", messages[1]["content"])

    def test_empty_turns_skipped(self):
        html = """
        <user-query><div class="query-text">  </div></user-query>
        <message-content><div class="model-response-text"><p>实际回答</p></div></message-content>
        """
        messages = gemini.parse_messages(soup(html), {})
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "AI")


# 真实分享页（2026-09 校准）结构：response-element 富内容包装器包裹 code-block；
# 用户附件在 user-query-file-preview（文档不可下载 / 图片走灯箱）；公式用
# math-inline / math-block class；屏幕阅读器节点含“你说”前缀；装饰图来自 gstatic。
REALISTIC_HTML = """
<share-viewer><section>
  <user-query>
    <span class="user-query-container"><user-query-content>
      <div class="user-query-container">
        <div class="file-preview-container">
          <user-query-file-carousel>
            <user-query-file-preview>
              <div data-test-id="uploaded-file" class="new-file-preview-container">
                <button aria-label="无法查看或下载共享对话中的文件">
                  <div class="extension-label">PDF</div>
                  <div class="filename-label">Lab 1</div>
                </button>
              </div>
            </user-query-file-preview>
            <user-query-file-preview>
              <div data-test-id="uploaded-file">
                <button aria-label="以灯箱形式显示上传的图片">
                  <img src="https://lh3.googleusercontent.com/gg/REALUSERIMG" alt="上传截图">
                </button>
              </div>
            </user-query-file-preview>
          </user-query-file-carousel>
        </div>
        <div class="query-content"><div class="query-text gds-body-l">
          <h5 class="cdk-visually-hidden screen-reader-user-query-label"><span>你说</span> 帮我看看这个lab </h5>
          <p class="query-text-line">帮我看看这个lab</p>
        </div></div>
      </div>
    </user-query-content></span>
  </user-query>
  <message-content>
    <div class="markdown markdown-main-panel">
      <p>复杂度 <span class="math-inline" data-math="O(n^2)">O(n²)</span> 如下。</p>
      <p><response-element class="no-md"><source-footnote><sup></sup></source-footnote></response-element></p>
      <response-element class="no-md">
        <code-block>
          <div class="code-block">
            <div class="code-block-decoration header-formatted"><span>Python</span></div>
            <pre><code class="code-container formatted" data-test-id="code-content">print("hi")</code></pre>
          </div>
        </code-block>
      </response-element>
      <div><span class="math-block" data-math="\\sum_{i=1}^{n} i">Σ</span></div>
      <img src="https://www.gstatic.com/lamda/images/sparkle.svg" alt="">
      <img src="https://lh3.googleusercontent.com/gg/AIPIC" alt="AI生成图">
    </div>
  </message-content>
</section></share-viewer>
"""


class GeminiRealisticDomTests(unittest.TestCase):
    def setUp(self):
        self.messages = gemini.parse_messages(soup(REALISTIC_HTML), {})

    def test_roles(self):
        self.assertEqual(
            [m["role"] for m in self.messages], ["User", "AI"]
        )

    def test_code_block_inside_response_element_not_dropped(self):
        # 回归：code-block 被 <response-element> 包裹时曾被当噪声整体删除。
        ai = self.messages[1]["content"]
        self.assertIn("```Python", ai)
        self.assertIn('print("hi")', ai)

    def test_math_inline_and_block_by_class(self):
        ai = self.messages[1]["content"]
        self.assertIn("$O(n^2)$", ai)
        self.assertIn("$$", ai)
        self.assertIn(r"\sum_{i=1}^{n} i", ai)
        self.assertNotIn("Σ", ai)

    def test_decorative_images_removed_real_images_kept(self):
        ai = self.messages[1]["content"]
        self.assertNotIn("gstatic.com", ai)
        self.assertIn("lh3.googleusercontent.com/gg/AIPIC", ai)

    def test_empty_footnote_stripped(self):
        ai = self.messages[1]["content"]
        self.assertNotIn("<sup", ai)
        self.assertNotIn("source-footnote", ai)

    def test_user_document_attachment_placeholder(self):
        user = self.messages[0]["content"]
        self.assertIn("[上传文档]", user)
        self.assertIn("Lab 1.pdf", user)
        # 图片型附件（灯箱）保留为图片而非文档占位。
        self.assertIn("lh3.googleusercontent.com/gg/REALUSERIMG", user)
        self.assertNotIn("无法查看或下载", user)

    def test_screen_reader_prefix_removed(self):
        user = self.messages[0]["content"]
        self.assertNotIn("你说", user)
        self.assertIn("帮我看看这个lab", user)


class GeminiRegistryTests(unittest.TestCase):
    def test_provider_registered(self):
        # 新平台按约定追加到末尾（当前末位为 Kimi），Gemini 保持已注册即可。
        self.assertIn(gemini, PROVIDERS)
        self.assertEqual(gemini.DISPLAY_NAME, "Gemini")
        self.assertIn("user-query", WAIT_SELECTOR)
        self.assertIn("message-content", WAIT_SELECTOR)


class GeminiAuthRoutingTests(unittest.TestCase):
    def test_private_app_requires_login(self):
        self.assertTrue(requires_authenticated_browser(
            "https://gemini.google.com/app/c9f0d4Abc123def456"
        ))

    def test_share_link_does_not_require_login(self):
        self.assertFalse(requires_authenticated_browser(
            "https://gemini.google.com/share/abcdef123456"
        ))

    def test_share_short_domain_does_not_require_login(self):
        # share.gemini.google/<id> 短链跳转后为公开分享页，无需登录。
        self.assertFalse(requires_authenticated_browser(
            "https://share.gemini.google/qER3ntHn8kOP"
        ))

    def test_blank_app_and_prefill_not_treated_as_conversation(self):
        self.assertFalse(requires_authenticated_browser(
            "https://gemini.google.com/app"
        ))
        self.assertFalse(requires_authenticated_browser(
            "https://gemini.google.com/app?q=hello"
        ))

    def test_existing_platforms_unaffected(self):
        self.assertTrue(requires_authenticated_browser(
            "https://chatgpt.com/c/abc-123"
        ))
        self.assertTrue(requires_authenticated_browser(
            "https://chat.deepseek.com/a/chat/s/abc-123"
        ))
        self.assertFalse(requires_authenticated_browser(
            "https://www.doubao.com/thread/abc123"
        ))


if __name__ == "__main__":
    unittest.main()
