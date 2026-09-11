"""AI 平台适配器注册表。

新增平台时，在本目录创建模块并加入 PROVIDERS 即可。
"""

from . import chatgpt, deepseek, doubao, gemini, kimi, qianwen


PROVIDERS = (chatgpt, deepseek, doubao, gemini, kimi, qianwen)
WAIT_SELECTOR = ", ".join(provider.WAIT_SELECTOR for provider in PROVIDERS)


def provider_for_host(host):
    """按页面域名返回适配器模块；未识别平台返回 None。

    仅供抓取前选择等待目标（WAIT_SELECTOR）使用：平台识别本身仍以 DOM
    为准（collect_html / parse_messages 自动探测），host 只决定"等哪个
    选择器出现"，不参与解析路由。host 可含端口/大小写，会被归一化。
    """
    normalized = (host or "").lower().split(":", 1)[0]
    for provider in PROVIDERS:
        if normalized in getattr(provider, "HOSTS", ()):
            return provider
    return None


async def collect_virtualized_html(page):
    """按注册顺序调用需要虚拟列表采集的平台。"""
    for provider in PROVIDERS:
        collector = getattr(provider, "collect_html", None)
        if collector is None:
            continue
        html = await collector(page)
        if html is not None:
            return html
    return None


def parse_messages(soup, image_map):
    """返回 (provider, messages)；未识别平台时返回 (None, None)。"""
    for provider in PROVIDERS:
        messages = provider.parse_messages(soup, image_map)
        if messages is not None:
            return provider, messages
    return None, None
