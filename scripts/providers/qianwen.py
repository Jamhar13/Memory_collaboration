"""通义千问（qianwen.com / qianwen.my.cn）页面采集与消息解析。

千问分享页是 React SPA（Alibaba ICE 框架 + CSS Modules scoped 类名）。
据 2026-09 真机校准（有效分享链接实测）：

- 公开分享页 ``/share/chat/<32位hex-id>``：无需登录，SPA 渲染 DOM；
  但更高效的方式是直接调 API 获取完整对话 JSON（AI 回答为 markdown）。
- 公开 API：``POST https://chat2-api.qianwen.com/api/v1/share/info``
  body: ``{"share_id": "<id>", "biz_id": "ai_qwen"}``
  有效链接返回 ``data.session.record_list``（完整对话）；
  过期链接返回 ``data.expired=true``。
- 私有会话 ``/chat/<session_id>``：需登录，API 为
  ``GET https://chat2-api.qianwen.com/api/v1/session/msg/list?session_id=<id>&biz_id=ai_qwen&return_response_messages=true``
  返回 ``data.list``（与 record_list 相同结构）。请求会自动携带浏览器 Cookie。
- ``record_list`` / ``list`` 中每条记录的 ``response_messages`` 里，
  ``mime_type == "multi_load/iframe"`` 且 ``status == "complete"`` 的是 AI 回答（markdown）；
  ``signal/post``、``bar/workflow``、``survey/card`` 为空壳信号/思考轨迹/问卷，跳过。
- 用户提问在 ``request_messages[0].content``（``mime_type="text/plain"``）。
"""

import html as _html
import json
import re

DISPLAY_NAME = "千问"
WAIT_SELECTOR = ".question-text-card, .chat-answers-card-wrap"
HOSTS = ("qianwen.com", "www.qianwen.com", "qianwen.my.cn")

_SHARE_API_URL = "https://chat2-api.qianwen.com/api/v1/share/info?pr=qwen&fr=pc"
_PRIVATE_API_URL = "https://chat2-api.qianwen.com/api/v1/session/msg/list"
_SHARE_ID_PATTERN = re.compile(r"/share/chat/([0-9a-f]{32})", re.IGNORECASE)
_SESSION_ID_PATTERN = re.compile(r"(?<!/share)/chat/([0-9a-f]{32})", re.IGNORECASE)
_AI_MIME_TYPE = "multi_load/iframe"


async def _collect_from_share(page):
    """通过公开分享 API 采集对话。"""
    match = _SHARE_ID_PATTERN.search(page.url)
    if not match:
        return None
    share_id = match.group(1)

    try:
        resp = await page.request.post(
            _SHARE_API_URL,
            data=json.dumps({"share_id": share_id, "biz_id": "ai_qwen"}),
            headers={"content-type": "application/json"},
        )
        raw = await resp.text()
        data = json.loads(raw)
    except Exception:
        return None

    if data.get("data", {}).get("expired", True):
        return None

    return data.get("data", {}).get("session", {}).get("record_list", [])


async def _collect_from_private(page):
    """通过私有会话 API 采集对话（需登录态）。

    API 需要 ``ut`` 参数（用户 token），从浏览器 Cookie ``b-user-id`` 提取；
    其余查询参数（chat_client、device 等）为固定值。
    """
    match = _SESSION_ID_PATTERN.search(page.url)
    if not match:
        return None
    session_id = match.group(1)

    # 从 Cookie 提取 ut（b-user-id）
    try:
        cookies = await page.context.cookies()
    except Exception:
        return None
    ut = ""
    for c in cookies:
        if c.get("name") == "b-user-id":
            ut = c.get("value", "")
            break
    if not ut:
        return None

    try:
        resp = await page.request.get(
            _PRIVATE_API_URL,
            params={
                "session_id": session_id,
                "biz_id": "ai_qwen",
                "return_response_messages": "true",
                "event_filter": "all",
                "chat_client": "h5",
                "device": "pc",
                "fr": "pc",
                "pr": "qwen",
                "ut": ut,
                "la": "zh-CN",
                "tz": "Asia/Shanghai",
            },
        )
        raw = await resp.text()
        data = json.loads(raw)
    except Exception:
        return None

    if not data.get("data") or data.get("code") != 0:
        return None

    return data.get("data", {}).get("list", [])


async def collect_html(page):
    """通过 API 采集千问对话；页面不属于千问或已过期时返回 None。

    与其他平台适配器不同，千问不走 DOM 采集——直接调 API 获取完整对话
    JSON，AI 回答已经是 markdown 格式，无需 markdownify。合成一个简单
    HTML 结构（``<div class="qk-user/qk-ai" data-content="...">``）供
    :func:`parse_messages` 解析。

    先尝试公开分享 API，再尝试私有会话 API。
    """
    records = await _collect_from_share(page)
    if records is None:
        records = await _collect_from_private(page)
    if not records:
        return None

    fragments = []
    for rec in records:
        # --- 用户提问 ---
        for req in rec.get("request_messages", []):
            if req.get("mime_type") == "text/plain":
                content = (req.get("content") or "").strip()
                if content:
                    fragments.append(
                        f'<div class="qk-user" data-content="'
                        f"{_html.escape(content, quote=True)}"
                        f'"></div>'
                    )

        # --- AI 回答 ---
        for resp_msg in rec.get("response_messages", []):
            if (
                resp_msg.get("mime_type") == _AI_MIME_TYPE
                and resp_msg.get("status") == "complete"
            ):
                content = (resp_msg.get("content") or "").strip()
                if content:
                    fragments.append(
                        f'<div class="qk-ai" data-content="'
                        f"{_html.escape(content, quote=True)}"
                        f'"></div>'
                    )

    if not fragments:
        return None

    return (
        "<!DOCTYPE html><html><body>"
        + "\n".join(fragments)
        + "</body></html>"
    )


def parse_messages(soup, image_map=None):
    """解析千问消息；页面不属于千问时返回 None。

    :param soup: BeautifulSoup 对象（由 :func:`collect_html` 合成的 HTML）
    :param image_map: 可选的 ``{remote_url: local_path}`` 映射，用于将
        markdown 中的远程图片 URL 替换为本地路径。
    :return: ``[{"role": "User"|"AI", "content": "..."}]`` 或 None
    """
    if soup.select_one(".qk-user, .qk-ai") is None:
        return None

    messages = []
    for node in soup.select(".qk-user, .qk-ai"):
        classes = node.get("class") or []
        content = node.get("data-content", "")
        if not content:
            continue
        if "qk-user" in classes:
            messages.append({"role": "User", "content": content})
        elif "qk-ai" in classes:
            # 将 markdown 中的远程图片 URL 替换为本地路径
            if image_map:
                for remote_src, local_src in image_map.items():
                    content = content.replace(remote_src, local_src)
            messages.append({"role": "AI", "content": content})

    return messages or None
