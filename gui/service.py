"""GUI 业务桥接服务模块。

纯新增模块，负责连接图形界面与底层的抓取、总结和文件保存流水线。
不修改任何原有模块，纯组合调用现有能力。
"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import ipaddress
import mimetypes
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Optional
from urllib.parse import parse_qs, quote, urlparse
from urllib.parse import unquote, urljoin

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from gui.credential_store import normalize_provider_order
from scripts.markdown_exporter import display_and_export
from scripts.project_paths import (
    BROWSER_USER_DATA_DIR,
    DEBUG_HTML_FILE,
    DEFAULT_EXPORT_FILE,
    IMAGES_DIR,
    PROJECT_ROOT,
)
from scripts.providers import WAIT_SELECTOR, collect_virtualized_html, parse_messages


SAVE_DEBUG_SNAPSHOT = os.getenv("AI_MEMORY_SAVE_DEBUG_HTML", "").strip() == "1"
BROWSER_MODE_ENV = "AI_MEMORY_BROWSER_MODE"
LITE_BROWSER_CHANNELS = ("msedge", "chrome")
BROWSER_CHANNEL_LABELS = {
    "chromium": "内置 Chromium",
    "msedge": "Microsoft Edge",
    "chrome": "Google Chrome",
}

PRIVATE_CONVERSATION_PATTERNS = (
    re.compile(r"^/c/[0-9a-f-]+/?$", re.IGNORECASE),
    re.compile(r"^/a/chat/s/[0-9a-f-]+/?$", re.IGNORECASE),
)
# Gemini：gemini.google.com（短链 share.gemini.google 跳转而来）。
# 公开分享 /share/<id> 无需登录；账号内会话 /app/<id> 需登录。
# 裸 /app（新对话）与 /app?q=（预填提问）不是历史会话。
GEMINI_HOSTS = {"gemini.google.com", "share.gemini.google"}
GEMINI_PRIVATE_CONVERSATION_PATTERN = re.compile(
    r"^/app/[0-9A-Za-z_-]{8,}/?$"
)
# Kimi：www.kimi.com / kimi.com（旧域名 kimi.moonshot.cn 跳转而来）。
# 公开分享 /share/[<lang>/]<id> 无需登录；账号内会话 /chat/<id> 需登录
# （未登录访问会重定向首页）。裸 /chat、/chat/history（7 字符，短于
# {8,}）不是历史会话。
KIMI_HOSTS = {"kimi.com", "www.kimi.com", "kimi.moonshot.cn"}
KIMI_PRIVATE_CONVERSATION_PATTERN = re.compile(
    r"^/chat/[0-9A-Za-z-]{8,}/?$"
)
# 千问：qianwen.com / qianwen.my.cn（原 tongyi.aliyun.com 301 跳转）。
# 公开分享 /share/chat/<32hex> 无需登录；私有会话 /chat 需登录。
QIANWEN_HOSTS = {"qianwen.com", "www.qianwen.com", "qianwen.my.cn"}
QIANWEN_PRIVATE_CONVERSATION_PATTERN = re.compile(
    r"^/chat(?:[/?]|$)"
)
# Grok：grok.com。公开分享 /share/<id> 无需登录；私有会话 /c/<uuid> 需登录。
GROK_HOSTS = {"grok.com", "www.grok.com"}
GROK_PRIVATE_CONVERSATION_PATTERN = re.compile(r"^/c/[0-9a-f-]{8,}")
DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".csv", ".md", ".rtf",
}
DOCUMENT_MIME_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
    "application/rtf": ".rtf",
}
GUI_DOCUMENT_MAX_BYTES = 25 * 1024 * 1024
UTF8_TEXT_DOCUMENT_EXTENSIONS = {".txt", ".csv", ".md"}
DOUBAO_DOCUMENT_API_PATH = "/alice/message/get_file_url"
DOUBAO_AI_DOCUMENT_MAX_COUNT = 12
CHATGPT_CARD_REFERENCE_PREFIX = "chatgpt-card:"
DEEPSEEK_CARD_REFERENCE_PREFIX = "deepseek-card:"
CHATGPT_ESCAPED_QUOTE = re.escape(chr(92) + '"')
CHATGPT_EMBEDDED_DOCUMENT_PATTERNS = (
    re.compile(
        rf'{CHATGPT_ESCAPED_QUOTE}name{CHATGPT_ESCAPED_QUOTE},'
        rf'{CHATGPT_ESCAPED_QUOTE}(?P<filename>.{{1,240}}?[.](?:pdf|docx?|xlsx?|xls|pptx?|ppt|txt|csv|md|rtf))'
        rf'{CHATGPT_ESCAPED_QUOTE},{CHATGPT_ESCAPED_QUOTE}'
        rf'(?P<file_id>file[_-][A-Za-z0-9_-]{{12,}}){CHATGPT_ESCAPED_QUOTE}',
        re.IGNORECASE,
    ),
    re.compile(
        r'"name","(?P<filename>[^"]{1,240}[.](?:pdf|docx?|xlsx?|xls|pptx?|ppt|txt|csv|md|rtf))'
        r'","(?P<file_id>file[_-][A-Za-z0-9_-]{12,})"',
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class DocumentCandidate:
    reference: str
    url: str
    filename: str


def _chatgpt_shared_conversation_id(url: str) -> str:
    parsed = urlparse(str(url or ""))
    host = parsed.netloc.lower().split(":", 1)[0]
    match = re.fullmatch(
        r"/share/(?P<share_id>[0-9a-f-]{32,36})/?",
        parsed.path,
        re.IGNORECASE,
    )
    if host not in {"chatgpt.com", "chat.openai.com"} or not match:
        return ""
    return match.group("share_id")



def _iter_json_mappings(value: Any):
    """递归遍历平台响应中的字典节点，不记录或输出原始响应。"""
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _iter_json_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_json_mappings(child)


def _collect_response_assets(
    payload: Any,
    page_url: str,
    document_candidates: list[DocumentCandidate],
    image_references: set[str],
) -> None:
    """从页面已获授权的 JSON 响应提取附件元数据，不额外上传数据。"""
    parsed = urlparse(page_url)
    host = parsed.netloc.lower().split(":", 1)[0]
    origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    share_id = _chatgpt_shared_conversation_id(page_url)
    share_query = (
        f"?shared_conversation_id={quote(share_id, safe='')}"
        if share_id else ""
    )
    for item in _iter_json_mappings(payload):
        filename = str(
            item.get("file_name") or item.get("name") or ""
        ).strip()
        mime_type = str(item.get("mime_type") or "").lower()
        suffix = Path(filename).suffix.lower()

        signed_path = str(item.get("signed_path") or "").strip()
        if (
            host == "chat.deepseek.com"
            and signed_path
            and filename
            and suffix in DOCUMENT_EXTENSIONS
            and str(item.get("status") or "SUCCESS").upper() == "SUCCESS"
        ):
            if signed_path.startswith(("http://", "https://")):
                download_url = signed_path
            else:
                download_url = (
                    "https://files.deepseeksvc.com/api/"
                    + signed_path.lstrip("/")
                )
            separator = "&" if "?" in download_url else "?"
            if not re.search(r"(?:[?&])ty=", download_url):
                download_url = f"{download_url}{separator}ty=r"
            document_candidates.append(DocumentCandidate(
                signed_path,
                download_url,
                _safe_document_filename(filename, suffix),
            ))

        if host in {"doubao.com", "www.doubao.com"} and filename:
            uri = str(item.get("uri") or item.get("key") or "").strip()
            if suffix in DOCUMENT_EXTENSIONS and _is_safe_doubao_file_uri(uri):
                document_candidates.append(DocumentCandidate(
                    uri,
                    f"{origin}{DOUBAO_DOCUMENT_API_PATH}",
                    _safe_document_filename(filename, suffix),
                ))

        if host in {"chatgpt.com", "chat.openai.com"} and filename:
            reference_ids = []
            for key in ("id", "file_id", "library_file_id"):
                reference = str(item.get(key) or "").strip()
                if reference.startswith(("file_", "file-")):
                    reference_ids.append(reference)
            if suffix in DOCUMENT_EXTENSIONS:
                for reference in dict.fromkeys(reference_ids):
                    document_candidates.append(DocumentCandidate(
                        reference,
                        f"{origin}/backend-api/files/download/"
                        f"{quote(reference, safe='')}{share_query}",
                        _safe_document_filename(filename, suffix),
                    ))
            if mime_type.startswith("image/") or suffix in {
                ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"
            }:
                image_references.update(reference_ids or {filename.lower()})


async def _capture_response_assets(
    response: Any,
    page_url: str,
    document_candidates: list[DocumentCandidate],
    image_references: set[str],
) -> None:
    try:
        payload = await response.json()
    except Exception:
        return
    _collect_response_assets(
        payload,
        page_url,
        document_candidates,
        image_references,
    )


def _document_response_cache_keys(response_url: str) -> tuple[str, ...]:
    parsed = urlparse(str(response_url or ""))
    keys = [str(response_url or ""), parsed.path]
    chatgpt_match = re.match(
        r"^/backend-api/files/download/(?P<file_id>[^/]+)$",
        parsed.path,
    )
    if chatgpt_match:
        keys.append(unquote(chatgpt_match.group("file_id")))
    if parsed.path == "/backend-api/estuary/content":
        file_id = parse_qs(parsed.query).get("id", [""])[0]
        if file_id:
            keys.append(unquote(file_id))
    return tuple(key for key in dict.fromkeys(keys) if key)


def _is_document_content_response(response_url: str) -> bool:
    parsed = urlparse(str(response_url or ""))
    host = parsed.netloc.lower().split(":", 1)[0]
    if host in {"chatgpt.com", "chat.openai.com"}:
        return bool(
            re.match(r"^/backend-api/files/download/[^/]+$", parsed.path)
            or parsed.path == "/backend-api/estuary/content"
        )
    return host == "files.deepseeksvc.com" and parsed.path == "/api/file"


async def _capture_document_content_response(
    response: Any,
    cache: dict[str, tuple[bytes, dict[str, str]]],
) -> None:
    if getattr(response, "status", 0) != 200:
        return
    headers = dict(getattr(response, "headers", {}) or {})
    content_type = headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type in {"text/html", "application/json"}:
        return
    length = headers.get("content-length", "").strip()
    if length.isdigit() and int(length) > GUI_DOCUMENT_MAX_BYTES:
        return
    try:
        body = await response.body()
    except Exception:
        return
    if not body or len(body) > GUI_DOCUMENT_MAX_BYTES:
        return
    captured = (body, headers)
    for key in _document_response_cache_keys(response.url):
        cache[key] = captured


def _is_asset_metadata_response(response_url: str) -> bool:
    parsed = urlparse(str(response_url or ""))
    host = parsed.netloc.lower().split(":", 1)[0]
    path = parsed.path.lower()
    if host in {"chatgpt.com", "chat.openai.com"}:
        return bool(re.match(r"^/backend-api/conversations/[^/]+/?$", path))
    if host == "chat.deepseek.com":
        return "/api/v0/" in path and any(
            token in path for token in ("share/content", "history", "message")
        )
    if host in {"doubao.com", "www.doubao.com"}:
        return any(
            token in path
            for token in ("/im/chain/single", "/im/chain/batch_single")
        )
    return False


async def _drain_response_tasks(tasks: set[asyncio.Task]) -> None:
    pending = list(tasks)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _page_has_conversation_content(page: Any, page_url: str) -> bool:
    host = urlparse(page_url).netloc.lower().split(":", 1)[0]
    if host in {"chatgpt.com", "chat.openai.com"}:
        selector = "[data-message-author-role]"
    elif host == "chat.deepseek.com":
        selector = "[data-virtual-list-item-key] .ds-message"
    elif host in {"doubao.com", "www.doubao.com"}:
        selector = (
            ".message-item, "
            "div[class*='message-list-'] div.my-0.w-full.mx-auto "
            "div.flex.flex-row.w-full"
        )
    elif host in GEMINI_HOSTS:
        selector = "user-query, message-content, model-response"
    elif host in KIMI_HOSTS:
        # 不能用联合选择器：Kimi 首页营销壳含 .message-item，
        # 会误中豆包选择器，把未登录重定向误判成会话已就绪。
        selector = ".segment-user, .segment-assistant"
    elif host in QIANWEN_HOSTS:
        selector = ".question-text-card, .chat-answers-card-wrap"
    elif host in GROK_HOSTS:
        selector = ".message-bubble[role='article']"
    else:
        selector = WAIT_SELECTOR

    # 私有会话未登录会被重定向到首页/登录页：URL 一旦离开原会话路径，
    # 立即判定未就绪，不空等选择器超时。反之 URL 仍在会话页时，无头
    # 冷启动 hydration 可能很慢（Gemini 实测约 25 秒、Grok 约 40 秒），
    # 等待过短会把"已登录但渲染慢"误判成需要重新登录。
    timeout_ms = 10000
    requested = urlparse(page_url)
    current = urlparse(getattr(page, "url", page_url))
    current_host = current.netloc.lower().split(":", 1)[0]
    private_checks = (
        (GEMINI_HOSTS, GEMINI_PRIVATE_CONVERSATION_PATTERN),
        (KIMI_HOSTS, KIMI_PRIVATE_CONVERSATION_PATTERN),
        (QIANWEN_HOSTS, QIANWEN_PRIVATE_CONVERSATION_PATTERN),
        (GROK_HOSTS, GROK_PRIVATE_CONVERSATION_PATTERN),
    )
    for hosts, pattern in private_checks:
        if host in hosts and pattern.match(requested.path):
            timeout_ms = 30000 if hosts is GEMINI_HOSTS else 10000
            if hosts is GROK_HOSTS:
                timeout_ms = 45000
            if current_host not in hosts or not pattern.match(current.path):
                return False
            break

    try:
        await page.wait_for_selector(
            selector, state="attached", timeout=timeout_ms
        )
        return await page.locator(selector).count() > 0
    except Exception:
        return False


async def _chatgpt_assets_need_rehydrate(
    page: Any,
    page_url: str,
    document_candidates: list[DocumentCandidate],
    image_references: set[str],
) -> bool:
    host = urlparse(page_url).netloc.lower().split(":", 1)[0]
    if host not in {"chatgpt.com", "chat.openai.com"}:
        return False
    try:
        body_text = await page.locator("body").inner_text(timeout=3000)
        rendered_images = await page.locator(
            "[data-message-author-role='user'] img"
        ).count()
    except Exception:
        return False
    if "上传文件" in body_text:
        return True
    if image_references and rendered_images < len(image_references):
        return True
    visible_names = {
        candidate.filename.lower()
        for candidate in document_candidates
        if candidate.filename and candidate.filename.lower() in body_text.lower()
    }
    return bool(document_candidates and not visible_names)


def requires_authenticated_browser(url: str) -> bool:
    """账号内会话链接需要检测持久登录态。"""
    parsed = urlparse(str(url or "").strip())
    host = parsed.netloc.lower().split(":", 1)[0]
    if host in {"chatgpt.com", "chat.openai.com"}:
        return bool(PRIVATE_CONVERSATION_PATTERNS[0].match(parsed.path))
    if host == "chat.deepseek.com":
        return bool(PRIVATE_CONVERSATION_PATTERNS[1].match(parsed.path))
    if host in GEMINI_HOSTS:
        return bool(GEMINI_PRIVATE_CONVERSATION_PATTERN.match(parsed.path))
    if host in KIMI_HOSTS:
        return bool(KIMI_PRIVATE_CONVERSATION_PATTERN.match(parsed.path))
    if host in QIANWEN_HOSTS:
        return bool(QIANWEN_PRIVATE_CONVERSATION_PATTERN.match(parsed.path))
    if host in GROK_HOSTS:
        return bool(GROK_PRIVATE_CONVERSATION_PATTERN.match(parsed.path))
    return False


def browser_channel_candidates() -> tuple[str, ...]:
    """返回当前发布变体允许使用的浏览器通道。"""
    mode = os.getenv(BROWSER_MODE_ENV, "full").strip().lower()
    return LITE_BROWSER_CHANNELS if mode == "lite" else ("chromium",)


def _browser_profile_directory(
    channel: str,
    profile_root: Optional[Path] = None,
) -> Path:
    root = Path(profile_root or BROWSER_USER_DATA_DIR).resolve()
    if browser_channel_candidates() == LITE_BROWSER_CHANNELS:
        return root / channel
    return root


async def launch_browser_context(
    playwright: Any,
    *,
    headless: bool,
    viewport: Optional[dict[str, int]],
    no_viewport: bool,
    start_minimized: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    profile_root: Optional[Path] = None,
) -> tuple[Any, str]:
    """启动当前变体的浏览器；轻量版按 Edge、Chrome 顺序回退。"""
    channels = browser_channel_candidates()
    last_error: Optional[BaseException] = None
    for index, channel in enumerate(channels):
        try:
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(
                    _browser_profile_directory(channel, profile_root)
                ),
                headless=headless,
                channel=channel,
                viewport=viewport,
                no_viewport=no_viewport,
                ignore_default_args=["--enable-automation"],
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-service-autorun",
                    *(["--start-minimized"] if start_minimized else []),
                ],
            )
            if logger:
                logger(f"正在使用 {BROWSER_CHANNEL_LABELS[channel]}。")
            return context, channel
        except Exception as error:
            last_error = error
            if logger and index + 1 < len(channels):
                next_channel = channels[index + 1]
                logger(
                    f"未能启动 {BROWSER_CHANNEL_LABELS[channel]}，"
                    f"正在尝试 {BROWSER_CHANNEL_LABELS[next_channel]}..."
                )

    if channels == LITE_BROWSER_CHANNELS:
        raise RuntimeError(
            "未检测到可用的 Microsoft Edge 或 Google Chrome，"
            "请下载安装其中一个浏览器，或下载全量版。"
        ) from None
    if last_error is not None:
        raise last_error
    raise RuntimeError("未能启动浏览器。")


async def _set_browser_window_state(page: Any, state: str) -> None:
    """最小化后台兼容窗口，确需登录时再恢复。"""
    session = None
    try:
        session = await page.context.new_cdp_session(page)
        window = await session.send("Browser.getWindowForTarget")
        await session.send("Browser.setWindowBounds", {
            "windowId": window["windowId"],
            "bounds": {"windowState": state},
        })
    except Exception:
        pass
    finally:
        if session is not None:
            try:
                await asyncio.wait_for(session.detach(), timeout=3)
            except Exception:
                pass


@dataclass
class FetchResult:
    html: Optional[str]
    image_map: dict[str, str]
    messages: list[dict[str, str]]
    document_map: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    user_wait_seconds: float = 0.0


@dataclass
class GenerationBundle:
    """一次 GUI 生成任务实际落盘的文件与展示选择。"""

    saved_files: list[Path]
    selected_sections: tuple[str, ...] = ()
    selected_topics: tuple[str, ...] = ()
    summary_result: Optional[dict[str, Any]] = None


DEFAULT_MODE_MARKDOWN_FILENAMES = {
    "raw": "AI_memory_export.md",
    "normal": "AI_memory_summary.md",
    "simple": "AI_memory_simple.md",
    "detailed": "AI_memory_detailed_summary.md",
}

MODE_FILENAME_SUFFIXES = {
    "raw": "_export",
    "normal": "_summary",
    "simple": "_simple",
    "detailed": "_detailed_summary",
}

GUI_IMAGE_DOWNLOAD_CONCURRENCY = 4
GUI_IMAGE_DOWNLOAD_ATTEMPTS = 2
GUI_IMAGE_DOWNLOAD_TIMEOUT_MS = 10000
GUI_REQUEST_TIMEOUT_SECONDS = 120
SILICONFLOW_FREE_SUMMARY_MODELS = (
    "Qwen/Qwen3-8B",
)


def default_output_filename(modes: Mapping[str, bool]) -> str:
    """返回保存对话框应展示的默认 Markdown 文件名。"""
    enabled_modes = [
        mode for mode in DEFAULT_MODE_MARKDOWN_FILENAMES
        if modes.get(mode)
    ]
    if len(enabled_modes) == 1:
        return DEFAULT_MODE_MARKDOWN_FILENAMES[enabled_modes[0]]
    return "AI_memory.md"


def default_summary_result_cache_dir(
    runtime_data_dir: Optional[Path] = None,
) -> Path:
    """返回 GUI 私有的本机完成结果缓存目录。"""
    if runtime_data_dir is not None:
        return Path(runtime_data_dir).resolve() / "summary_results"
    base = os.getenv("LOCALAPPDATA") or os.getenv("XDG_CACHE_HOME")
    if base:
        return Path(base) / "AI Memory Summary" / "summary_results"
    return Path.home() / ".cache" / "ai-memory-summary" / "summary_results"


def normalize_markdown_filename(filename: str) -> str:
    """只保留文件名，并将任意扩展名强制规范为小写 ``.md``。"""
    safe_name = Path(str(filename).strip()).name
    if not safe_name:
        raise ValueError("文件名不能为空。")
    path = Path(safe_name)
    if path.suffix:
        path = path.with_suffix(".md")
    else:
        path = Path(f"{safe_name}.md")
    return path.name


def build_image_asset_directory(
    save_dir: Path,
    output_filename: str,
) -> Path:
    """返回与本次 Markdown 主文件配套的便携图片目录。"""
    normalized = normalize_markdown_filename(output_filename)
    return Path(save_dir) / f"{Path(normalized).stem}_images"

def build_document_asset_directory(
    save_dir: Path,
    output_filename: str,
) -> Path:
    """返回与本次 Markdown 主文件配套的便携附件目录。"""
    normalized = normalize_markdown_filename(output_filename)
    return Path(save_dir) / f"{Path(normalized).stem}_files"



def build_markdown_asset_prefix(
    asset_dir: Path,
    markdown_dir: Path,
) -> str:
    """构造相对于 Markdown 所在目录、可安全包含中文和空格的 URL 路径。"""
    relative_dir = Path(
        os.path.relpath(Path(asset_dir).resolve(), Path(markdown_dir).resolve())
    ).as_posix()
    encoded_dir = quote(relative_dir, safe="/-_.~")
    return encoded_dir if encoded_dir.startswith(".") else f"./{encoded_dir}"


def build_output_paths(
    save_dir: Path,
    modes: Mapping[str, bool],
    output_filename: Optional[str] = None,
) -> dict[str, Path]:
    """计算各模式的输出路径；未指定名称时完全沿用旧命名规则。"""
    save_dir = Path(save_dir)
    enabled_modes = [
        mode for mode in DEFAULT_MODE_MARKDOWN_FILENAMES
        if modes.get(mode)
    ]
    markdown_names = dict(DEFAULT_MODE_MARKDOWN_FILENAMES)

    if output_filename is not None:
        normalized = normalize_markdown_filename(output_filename)
        stem = Path(normalized).stem
        if len(enabled_modes) == 1:
            markdown_names[enabled_modes[0]] = normalized
        else:
            for mode in enabled_modes:
                markdown_names[mode] = (
                    f"{stem}{MODE_FILENAME_SUFFIXES[mode]}.md"
                )

    def json_name(mode: str, legacy_name: str) -> str:
        if output_filename is None:
            return legacy_name
        markdown_path = Path(markdown_names[mode])
        if len(enabled_modes) == 1:
            return markdown_path.with_suffix(".json").name
        stem = Path(normalize_markdown_filename(output_filename)).stem
        suffix = "_result" if mode == "normal" else "_detailed_result"
        return f"{stem}{suffix}.json"

    return {
        "raw_markdown": save_dir / markdown_names["raw"],
        "normal_json": save_dir / json_name(
            "normal", "AI_memory_result.json"
        ),
        "normal_markdown": save_dir / markdown_names["normal"],
        "detailed_json": save_dir / json_name(
            "detailed", "AI_memory_detailed_result.json"
        ),
        "detailed_markdown": save_dir / markdown_names["detailed"],
        "simple_markdown": save_dir / markdown_names["simple"],
        "intermediate_json": save_dir / ".gui_intermediate_result.json",
        "intermediate_markdown": save_dir / ".gui_intermediate_summary.md",
    }


def gui_summary_config_candidates(base_config: Any) -> list[Any]:
    """构造同一提供商内的 GUI 快速回退链，避免额度等待像界面卡死。"""
    candidates: list[Any] = []

    def append(candidate: Any) -> None:
        key = (candidate.provider, candidate.model)
        if key not in {
            (item.provider, item.model) for item in candidates
        }:
            candidates.append(replace(
                candidate,
                retries=1,
                rate_limit_wait_seconds=min(
                    int(candidate.rate_limit_wait_seconds), 5
                ),
                request_timeout_seconds=min(
                    int(candidate.request_timeout_seconds),
                    GUI_REQUEST_TIMEOUT_SECONDS,
                ),
            ))

    append(base_config)
    if base_config.provider == "gemini":
        append(replace(base_config, model="gemini-3.6-flash"))
        append(replace(base_config, model="gemini-3.5-flash-lite"))
    elif base_config.provider == "siliconflow":
        for model in SILICONFLOW_FREE_SUMMARY_MODELS:
            append(replace(base_config, model=model))
    return candidates


def resolve_gui_summary_config(
    base_config: Any,
    api_keys: Mapping[str, str],
) -> Any:
    """按用户设置的优先级选择已配置的提供商。"""
    provider_order = normalize_provider_order(api_keys)
    normalized = {
        provider: str(api_keys.get(provider) or "").strip()
        for provider in provider_order
        if str(api_keys.get(provider) or "").strip()
    }
    if not normalized:
        from scripts.gemini_summarizer import GeminiSummaryError
        raise GeminiSummaryError("请先配置API KEY。")

    preferred_provider = next(
        provider for provider in provider_order if provider in normalized
    )
    if preferred_provider == base_config.provider:
        return base_config

    from scripts.gemini_summarizer import (
        DEEPSEEK_DEFAULT_MODEL,
        DEFAULT_MODEL,
        SILICONFLOW_DEFAULT_MODEL,
    )
    model = {
        "gemini": DEFAULT_MODEL,
        "siliconflow": SILICONFLOW_DEFAULT_MODEL,
        "deepseek": DEEPSEEK_DEFAULT_MODEL,
    }[preferred_provider]
    return replace(base_config, provider=preferred_provider, model=model)


def gui_summary_attempt_configs(
    base_config: Any,
    api_keys: Mapping[str, str],
) -> list[Any]:
    """按用户顺序构造已配置提供商的回退链。"""
    from scripts.gemini_summarizer import (
        DEEPSEEK_DEFAULT_MODEL,
        DEFAULT_MODEL,
        SILICONFLOW_DEFAULT_MODEL,
    )

    primary = resolve_gui_summary_config(base_config, api_keys)
    provider_defaults = {
        "gemini": DEFAULT_MODEL,
        "siliconflow": SILICONFLOW_DEFAULT_MODEL,
        "deepseek": DEEPSEEK_DEFAULT_MODEL,
    }
    provider_order = [
        provider
        for provider in normalize_provider_order(api_keys)
        if str(api_keys.get(provider) or "").strip()
    ]
    attempts: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for provider in provider_order:
        provider_base = (
            primary
            if provider == primary.provider
            else replace(
                primary,
                provider=provider,
                model=provider_defaults[provider],
            )
        )
        for candidate in gui_summary_config_candidates(provider_base):
            key = (candidate.provider, candidate.model)
            if key not in seen:
                seen.add(key)
                attempts.append(candidate)
    return attempts


async def goto_with_retry_gui(page, url: str, attempts: int = 3, logger: Optional[Callable[[str], None]] = None):
    for attempt in range(1, attempts + 1):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            return
        except Exception as e:
            if attempt >= attempts:
                raise
            if logger:
                logger(f"页面加载超时，正在进行第 {attempt + 1}/{attempts} 次尝试...")
            await page.wait_for_timeout(2000 * attempt)


async def _rehydrate_chatgpt_conversation(
    page: Any,
    conversation_url: str,
    logger: Optional[Callable[[str], None]] = None,
) -> None:
    """站内切到新对话再返回；失败时才用完整导航恢复原会话。"""
    parsed = urlparse(str(conversation_url or ""))
    host = parsed.netloc.lower().split(":", 1)[0]
    if host not in {"chatgpt.com", "chat.openai.com"}:
        return
    await _set_browser_window_state(page, "minimized")

    # 只进入空白“新对话”视图，不访问用户的其他历史对话内容。
    try:
        new_chat_links = page.locator("a[href='/']")
        if await new_chat_links.count() > 0:
            await new_chat_links.first.click(force=True, timeout=5000)
            await _set_browser_window_state(page, "minimized")
            await page.wait_for_timeout(900)
            if urlparse(page.url).path != parsed.path:
                await page.go_back(
                    wait_until="domcontentloaded",
                    timeout=45000,
                )
                await _set_browser_window_state(page, "minimized")
                await page.wait_for_timeout(900)
                returned = urlparse(page.url)
                if (
                    returned.netloc.lower() == parsed.netloc.lower()
                    and returned.path == parsed.path
                ):
                    return
    except Exception:
        pass

    neutral_url = f"{parsed.scheme or 'https'}://{parsed.netloc}/"
    await page.goto(neutral_url, wait_until="domcontentloaded", timeout=45000)
    await _set_browser_window_state(page, "minimized")
    await page.wait_for_timeout(700)
    await goto_with_retry_gui(page, conversation_url, logger=logger)
    await _set_browser_window_state(page, "minimized")


def parse_fallback_messages_gui(soup: BeautifulSoup) -> list[dict[str, str]]:
    text_content = soup.get_text(separator="\n", strip=True)
    lines = text_content.split("\n")
    parsed_messages = []
    current_block = []
    is_user = True

    ignored_text = [
        "复制", "重新生成", "点赞", "踩", "分享",
        "已采纳", "查看更多", "编辑", "朗读"
    ]
    for line in lines:
        line = line.strip()
        if len(line) < 2 or line in ignored_text:
            continue
        if (
            line.startswith("回答")
            or line.startswith("好的")
            or line.startswith("字数")
            or line.endswith("字以内)")
        ):
            if current_block:
                parsed_messages.append({
                    "role": "User" if is_user else "AI",
                    "content": "\n".join(current_block)
                })
                current_block = []
                is_user = not is_user
        current_block.append(line)

    if current_block:
        parsed_messages.append({
            "role": "User" if is_user else "AI",
            "content": "\n".join(current_block)
        })

    return parsed_messages


async def _authenticated_page_get(page: Any, url: str, timeout: int):
    """同源资源由已登录网页发起请求，跨域资源沿用请求上下文。"""
    page_origin = urlparse(str(getattr(page, "url", "") or ""))
    target_origin = urlparse(str(url or ""))
    same_origin = (
        page_origin.scheme == target_origin.scheme
        and page_origin.netloc.lower() == target_origin.netloc.lower()
        and page_origin.scheme in {"http", "https"}
    )
    if not same_origin:
        return await page.request.get(url, timeout=timeout)

    async def fetch_from_page(
        target: str,
        prepare: Optional[str] = None,
    ):
        async with page.expect_response(
            lambda response: response.url == target,
            timeout=timeout,
        ) as response_info:
            await page.evaluate(
                """async ({resource, prepare}) => {
                    const options = {credentials: 'include'};
                    if (prepare) {
                        const resourceUrl = new URL(resource, location.href);
                        const prepareUrl = new URL(prepare, location.href);
                        if (
                            resourceUrl.origin === location.origin
                            && prepareUrl.origin === location.origin
                        ) {
                            const sessionResponse = await fetch(
                                '/api/auth/session',
                                {credentials: 'include'}
                            );
                            if (sessionResponse.ok) {
                                const session = await sessionResponse.json();
                                const token = (
                                    session.accessToken
                                    || session.access_token
                                    || ''
                                );
                                if (token) {
                                    options.headers = {
                                        Authorization: `Bearer ${token}`
                                    };
                                }
                            }
                            const prepared = await fetch(prepareUrl, options);
                            await prepared.arrayBuffer();
                        }
                    }
                    const response = await fetch(
                        resource,
                        options
                    );
                    await response.arrayBuffer();
                }""",
                {"resource": target, "prepare": prepare},
            )
        return await response_info.value

    # ChatGPT 的文件服务要求先访问 simple 端点准备当前会话授权，
    # 随后 download 端点才会返回原文件；网页点击文件卡片也是此顺序。
    download_match = re.match(
        r"^/backend-api/files/download/(?P<file_id>[^/]+)$",
        target_origin.path,
    )
    if (
        download_match
        and target_origin.netloc.lower()
        in {"chatgpt.com", "chat.openai.com"}
    ):
        simple_url = (
            f"{target_origin.scheme}://{target_origin.netloc}"
            f"/backend-api/files/"
            f"{download_match.group('file_id')}/simple"
        )
        if target_origin.query:
            simple_url = f"{simple_url}?{target_origin.query}"
        return await fetch_from_page(url, simple_url)
    return await fetch_from_page(url)


def _image_extension(src: str) -> str:
    lowered = src.lower()
    if ".jpg" in lowered or ".jpeg" in lowered:
        return "jpg"
    if ".webp" in lowered:
        return "webp"
    return "png"


def _is_decorative_image_candidate(src: str) -> bool:
    """识别 ChatGPT 引用卡片使用的 Google favicon，避免下载无关图标。"""
    parsed = urlparse(src)
    return (
        parsed.netloc.lower() in {"google.com", "www.google.com"}
        and parsed.path.rstrip("/").lower() == "/s2/favicons"
    )


def _ordered_image_sources(candidates: list[str]) -> list[str]:
    """按 DOM 首次出现顺序去重，并排除确定无语义的装饰图片。"""
    return list(dict.fromkeys(
        src for src in candidates
        if src and not _is_decorative_image_candidate(src)
    ))


def _extract_chatgpt_shared_image_sources(
    page_html: str,
    conversation_url: str,
) -> list[str]:
    """从 ChatGPT 分享页状态中还原未渲染的用户图片下载地址。"""
    parsed = urlparse(conversation_url)
    share_id = _chatgpt_shared_conversation_id(conversation_url)
    if not share_id:
        return []

    origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    sources = []
    for match in re.finditer(
        r"sediment://(?P<file_id>file_[A-Za-z0-9_-]+)"
        r"\?shared_conversation_id=(?P<share_id>[0-9a-f-]{32,36})",
        page_html or "",
        re.IGNORECASE,
    ):
        if match.group("share_id").lower() != share_id.lower():
            continue
        sources.append(
            f"{origin}/backend-api/files/download/"
            f"{quote(match.group('file_id'), safe='')}"
            f"?shared_conversation_id={quote(share_id, safe='')}"
        )
    # ChatGPT 的压缩映射按当前节点向父节点写入，附件在源码中是倒序。
    # 这里只作为页面运行时状态不可用时的兜底。
    return list(reversed(dict.fromkeys(sources)))


async def _chatgpt_message_asset_groups(
    page: Any,
    conversation_url: str,
) -> tuple[list[list[str]], list[list[DocumentCandidate]]]:
    """读取 ChatGPT 已解码的线性对话，保留每条用户消息的附件归属。"""
    parsed = urlparse(str(conversation_url or ""))
    if parsed.netloc.lower().split(":", 1)[0] not in {
        "chatgpt.com", "chat.openai.com"
    }:
        return [], []
    try:
        groups = await page.evaluate(
            """() => {
                const root = window.__reactRouterDataRouter?.state?.loaderData;
                const seen = new WeakSet();
                let linear = null;
                function find(value) {
                    if (linear || !value || typeof value !== "object"
                        || seen.has(value)) return;
                    seen.add(value);
                    if (Array.isArray(value)) {
                        for (const item of value) find(item);
                        return;
                    }
                    if (Array.isArray(value.linear_conversation)) {
                        linear = value.linear_conversation;
                        return;
                    }
                    for (const item of Object.values(value)) find(item);
                }
                find(root);
                return (linear || [])
                    .filter(item => item?.message?.author?.role === "user")
                    .map(item => {
                        const message = item.message;
                        const parts = Array.isArray(message.content?.parts)
                            ? message.content.parts : [];
                        const attachments = Array.isArray(
                            message.metadata?.attachments
                        ) ? message.metadata.attachments : [];
                        return {
                            images: parts
                                .filter(part => part && typeof part === "object"
                                    && part.content_type === "image_asset_pointer")
                                .map(part => part.asset_pointer)
                                .filter(Boolean),
                            attachments: attachments.map(attachment => ({
                                id: attachment.id || attachment.file_id || "",
                                name: attachment.name || attachment.file_name || "",
                            })),
                        };
                    });
            }"""
        )
    except Exception:
        return [], []

    origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    share_id = _chatgpt_shared_conversation_id(conversation_url)
    share_query = (
        f"?shared_conversation_id={quote(share_id, safe='')}"
        if share_id else ""
    )
    image_groups: list[list[str]] = []
    document_groups: list[list[DocumentCandidate]] = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, Mapping):
            continue
        images = []
        for pointer in group.get("images") or []:
            match = re.match(
                r"^(?:sediment|file-service)://"
                r"(?P<file_id>file[_-][A-Za-z0-9_-]+)",
                str(pointer or ""),
                re.IGNORECASE,
            )
            if match:
                images.append(
                    f"{origin}/backend-api/files/download/"
                    f"{quote(match.group('file_id'), safe='')}{share_query}"
                )
        documents = []
        for attachment in group.get("attachments") or []:
            if not isinstance(attachment, Mapping):
                continue
            file_id = str(attachment.get("id") or "").strip()
            filename = str(attachment.get("name") or "").strip()
            suffix = Path(filename).suffix.lower()
            if (
                not re.fullmatch(r"file[_-][A-Za-z0-9_-]+", file_id)
                or suffix not in DOCUMENT_EXTENSIONS
            ):
                continue
            documents.append(DocumentCandidate(
                file_id,
                f"{origin}/backend-api/files/download/"
                f"{quote(file_id, safe='')}{share_query}",
                _safe_document_filename(filename, suffix),
            ))
        image_groups.append(list(dict.fromkeys(images)))
        document_groups.append(list(dict.fromkeys(documents)))
    return image_groups, document_groups


def _inject_chatgpt_message_images(
    html: str,
    sources_by_message: list[list[str]],
) -> str:
    """按线性对话中的用户消息序号补回图片，不再猜全局附件顺序。"""
    if not html or not sources_by_message:
        return html
    soup = BeautifulSoup(html, "html.parser")
    user_messages = soup.find_all(attrs={"data-message-author-role": "user"})
    for target, sources in zip(user_messages, sources_by_message):
        existing = target.find_all("img")
        if len(existing) >= len(sources):
            continue
        missing = sources[len(existing):]
        for source in reversed(missing):
            image = soup.new_tag("img", src=source, alt="已上传的图片")
            target.insert(0, image)
    return str(soup)


def _inject_chatgpt_shared_images(html: str, sources: list[str]) -> str:
    """将分享页占位图片补回对应用户消息，供现有解析器生成 Markdown。"""
    if not html or not sources:
        return html
    soup = BeautifulSoup(html, "html.parser")
    placeholders = [
        node for node in soup.find_all("span")
        if node.get_text(" ", strip=True) == "已上传图片"
    ]
    for placeholder, source in zip(placeholders, sources):
        turn = placeholder.find_parent(
            attrs={"data-testid": re.compile(r"^conversation-turn-")}
        )
        target = (
            turn.find(attrs={"data-message-author-role": "user"})
            if turn else None
        )
        if target is None or target.find("img", src=source):
            continue
        image = soup.new_tag("img", src=source, alt="已上传的图片")
        target.insert(0, image)
    return str(soup)


def _image_file_index(path: Path) -> int:
    match = re.match(r"img_(\d+)_", path.name)
    return int(match.group(1)) if match else 0


def _is_supported_image_body(body: bytes) -> bool:
    return bool(
        body.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"BM"))
        or (len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WEBP")
    )


def _existing_image_for_source(images_dir: Path, src: str) -> Optional[Path]:
    """按 URL 哈希寻找此前已成功下载的同一资源，不依赖旧顺序编号。"""
    url_hash = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    extension = _image_extension(src)
    matches = sorted(
        images_dir.glob(f"img_*_{url_hash}.{extension}"),
        key=lambda path: (_image_file_index(path), path.name),
    ) if images_dir.is_dir() else []
    for path in matches:
        try:
            if path.is_file():
                with path.open("rb") as stream:
                    if _is_supported_image_body(stream.read(16)):
                        return path
        except OSError:
            continue
    return None


async def _download_image_candidates(
    page: Any,
    candidates: list[str],
    images_dir: Path,
    image_reference_prefix: str,
    concurrency: int = GUI_IMAGE_DOWNLOAD_CONCURRENCY,
    warning_collector: Optional[list[str]] = None,
) -> dict[str, str]:
    """复用已有文件并受限并发下载唯一真实图片，稳定保持 DOM 顺序。"""
    images_dir = Path(images_dir)
    ordered_sources = _ordered_image_sources(candidates)
    resolved_references: dict[str, str] = {}
    pending_sources: list[str] = []
    for src in ordered_sources:
        existing = _existing_image_for_source(images_dir, src)
        if existing is None:
            pending_sources.append(src)
        else:
            resolved_references[src] = (
                f"{image_reference_prefix}/{existing.name}"
            )

    limit = max(1, min(int(concurrency), 8))
    semaphore = asyncio.Semaphore(limit)

    async def download(
        src: str,
    ) -> tuple[str, Optional[bytes], Optional[str]]:
        failure_reason: Optional[str] = None
        for _attempt in range(GUI_IMAGE_DOWNLOAD_ATTEMPTS):
            try:
                async with semaphore:
                    response = await _authenticated_page_get(
                        page, src, GUI_IMAGE_DOWNLOAD_TIMEOUT_MS
                    )
                    if response.ok:
                        headers = dict(getattr(response, "headers", {}) or {})
                        content_type = headers.get("content-type", "").split(
                            ";", 1
                        )[0].lower()
                        if content_type == "application/json":
                            payload = await response.json()
                            download_url = _document_download_url_from_payload(payload)
                            if not download_url:
                                failure_reason = "not_an_image"
                                continue
                            response = await _authenticated_page_get(
                                page, download_url, GUI_IMAGE_DOWNLOAD_TIMEOUT_MS
                            )
                            if not response.ok:
                                status = getattr(response, "status", None)
                                failure_reason = (
                                    f"http_{status}" if status else "http_error"
                                )
                                continue
                        body = await response.body()
                        if _is_supported_image_body(body):
                            return src, body, None
                        failure_reason = "empty_body" if not body else "not_an_image"
                    else:
                        status = getattr(response, "status", None)
                        failure_reason = (
                            f"http_{status}" if status else "http_error"
                        )
            except Exception as error:
                failure_reason = type(error).__name__
                continue
        return src, None, failure_reason or "unknown_error"

    download_results = await asyncio.gather(*(
        download(src) for src in pending_sources
    ))
    downloaded = {
        src: body for src, body, _reason in download_results
    }
    failed_reasons = [
        reason for _src, body, reason in download_results
        if body is None and reason
    ]
    if failed_reasons and warning_collector is not None:
        reason_counts: dict[str, int] = {}
        for reason in failed_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reason_summary = "、".join(
            f"{reason}×{count}"
            for reason, count in sorted(reason_counts.items())
        )
        warning_collector.append(
            f"{len(failed_reasons)} 个真实图片资源下载失败"
            f"（{reason_summary}）；对话文字抓取继续保留。"
        )
    existing_indices = [
        _image_file_index(path)
        for path in images_dir.glob("img_*")
    ] if images_dir.is_dir() else []
    img_index = max(existing_indices, default=0) + 1
    for src in ordered_sources:
        if src in resolved_references:
            continue
        body = downloaded.get(src)
        if body is None:
            continue
        url_hash = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
        filename = f"img_{img_index}_{url_hash}.{_image_extension(src)}"
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / filename).write_bytes(body)
        resolved_references[src] = f"{image_reference_prefix}/{filename}"
        img_index += 1
    return {
        src: resolved_references[src]
        for src in ordered_sources
        if src in resolved_references
    }


def _document_filename_from_text(value: str) -> str:
    match = re.search(
        r'([^\\/:*?"<>|\r\n]{1,180}\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|md|rtf))',
        str(value or ""),
        re.IGNORECASE,
    )
    return Path(match.group(1).strip()).name if match else ""


def _document_filename_from_disposition(value: str) -> str:
    encoded = re.search(
        r"filename\*\s*=\s*(?:UTF-8'')?([^;]+)",
        str(value or ""),
        re.IGNORECASE,
    )
    if encoded:
        return Path(unquote(encoded.group(1).strip().strip('"'))).name
    plain = re.search(
        r'filename\s*=\s*"?([^";]+)',
        str(value or ""),
        re.IGNORECASE,
    )
    return Path(plain.group(1).strip()).name if plain else ""


def _document_suffix(value: str) -> str:
    suffix = Path(unquote(urlparse(str(value or "")).path)).suffix.lower()
    return suffix if suffix in DOCUMENT_EXTENSIONS else ""


def _safe_document_filename(name: str, fallback_suffix: str = "") -> str:
    normalized = str(name or "").replace("\\", "/")
    filename = Path(normalized).name.strip()
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).strip(" .")
    suffix = Path(filename).suffix.lower()
    if suffix not in DOCUMENT_EXTENSIONS:
        suffix = fallback_suffix if fallback_suffix in DOCUMENT_EXTENSIONS else ""
        stem = filename or "attachment"
        filename = f"{stem}{suffix}"
    if not filename:
        filename = f"attachment{fallback_suffix or '.bin'}"
    suffix = Path(filename).suffix
    stem = Path(filename).stem[: max(1, 180 - len(suffix))]
    return f"{stem}{suffix}"


def _is_safe_doubao_file_uri(value: str) -> bool:
    """只接受豆包返回的公开对象标识，拒绝绝对地址和路径穿越。"""
    normalized = str(value or "").strip()
    return bool(
        normalized.startswith("tos-")
        and ".." not in normalized
        and re.fullmatch(r"[A-Za-z0-9_./-]{1,500}", normalized)
        and Path(normalized).suffix.lower() in DOCUMENT_EXTENSIONS
    )


def _repair_downloaded_text_mojibake(
    body: bytes,
    filename: str,
    content_type: str = "",
) -> bytes:
    """仅在严格可逆且乱码特征显著下降时修复 UTF-8 二次解码。"""
    suffix = Path(filename).suffix.lower()
    if (
        suffix not in UTF8_TEXT_DOCUMENT_EXTENSIONS
        and not str(content_type or "").lower().startswith("text/")
    ):
        return body
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return body

    markers = "ÃÂâðåæçèéïä"

    def score(value: str) -> int:
        return (
            sum(value.count(marker) for marker in markers)
            + 3 * sum(1 for char in value if 0x80 <= ord(char) <= 0x9F)
        )

    before_score = score(text)
    if before_score < 3:
        return body

    reconstructed = bytearray()
    try:
        for char in text:
            if ord(char) <= 0xFF:
                reconstructed.append(ord(char))
            else:
                encoded = char.encode("cp1252")
                if len(encoded) != 1:
                    return body
                reconstructed.extend(encoded)
        repaired = bytes(reconstructed).decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return body

    return repaired.encode("utf-8") if score(repaired) < before_score else body


def _decode_doubao_embedded_text(value: str) -> str:
    decoded = str(value or "")
    for _ in range(4):
        updated = html_lib.unescape(decoded)
        updated = updated.replace(r"\/", "/").replace(r"\u0026", "&")
        updated = updated.replace(r"\u002F", "/").replace(r"\u002f", "/")
        updated = updated.replace(r"\&quot;", '"').replace(r'\"', '"')
        if updated == decoded:
            break
        decoded = updated
    return decoded


def _extract_doubao_embedded_document_candidates(
    html: str,
    origin: str,
) -> list[DocumentCandidate]:
    """从豆包分享页内嵌状态中提取文件名和原始对象标识。"""
    decoded = _decode_doubao_embedded_text(html)
    extension_pattern = r"(?:pdf|docx?|xlsx?|xls|pptx?|ppt|txt|csv|md|rtf)"
    patterns = (
        re.compile(
            rf'"(?:name|file_name)"\s*:\s*"(?P<filename>[^"\r\n]{{1,240}}\.{extension_pattern})"'
            rf'.{{0,1600}}?"(?:uri|key)"\s*:\s*"(?P<uri>tos-[A-Za-z0-9_./-]+\.{extension_pattern})"',
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf'"(?:uri|key)"\s*:\s*"(?P<uri>tos-[A-Za-z0-9_./-]+\.{extension_pattern})"'
            rf'.{{0,1600}}?"(?:name|file_name)"\s*:\s*"(?P<filename>[^"\r\n]{{1,240}}\.{extension_pattern})"',
            re.IGNORECASE | re.DOTALL,
        ),
    )
    candidates: list[DocumentCandidate] = []
    seen: set[tuple[str, str]] = set()
    for pattern in patterns:
        for match in pattern.finditer(decoded):
            uri = match.group("uri")
            filename = _safe_document_filename(match.group("filename"))
            key = (uri, filename.lower())
            if key in seen or not _is_safe_doubao_file_uri(uri):
                continue
            seen.add(key)
            candidates.append(DocumentCandidate(
                uri,
                f"{origin}{DOUBAO_DOCUMENT_API_PATH}",
                filename,
            ))
    return candidates


def _extract_doubao_ai_document_titles(
    html: str,
    base_url: str,
) -> list[str]:
    """只识别豆包 AI 回答里的在线文档产品卡片标题。"""
    parsed = urlparse(str(base_url or ""))
    if parsed.netloc.lower().split(":", 1)[0] not in {
        "doubao.com", "www.doubao.com"
    }:
        return []
    soup = BeautifulSoup(html or "", "html.parser")
    titles: list[str] = []
    seen: set[str] = set()
    for card in soup.find_all(
        "div", class_=re.compile(r"^product-card-")
    ):
        # 在线文档卡片同时具备专用标题节点与“创建时间”。严格限定结构，
        # 避免把回答中的普通标题、文件名或其他产品卡片误当作文档。
        if "创建时间" not in card.get_text(" ", strip=True):
            continue
        title_node = card.find(
            "div",
            class_=re.compile(r"^card-content-info-title-text-"),
        )
        title = " ".join(
            (title_node.get_text(" ", strip=True) if title_node else "").split()
        )
        if not title or len(title) > 180 or title.lower() in seen:
            continue
        seen.add(title.lower())
        titles.append(title)
    return titles[:DOUBAO_AI_DOCUMENT_MAX_COUNT]


def _extract_doubao_ai_document_resources(
    html: str,
    base_url: str,
) -> dict[str, str]:
    """从分享页状态提取豆包在线文档的公开资源地址。"""
    parsed = urlparse(str(base_url or ""))
    host = parsed.netloc.lower().split(":", 1)[0]
    if host not in {"doubao.com", "www.doubao.com"}:
        return {}
    origin = f"{parsed.scheme or 'https'}://{parsed.netloc}"
    decoded = _decode_doubao_embedded_text(html)
    resources: dict[str, str] = {}
    for marker in re.finditer(r'"artifact_block"\s*:\s*\{', decoded):
        block = decoded[marker.end():marker.end() + 12000]
        finish = block.find('"is_finish"')
        if finish >= 0:
            block = block[:finish]
        if not re.search(r'"resource_type"\s*:\s*10', block):
            continue
        resource_match = re.search(
            r'"resource_id"\s*:\s*"(?P<id>[A-Za-z0-9_-]{8,128})"',
            block,
        )
        title_match = re.search(
            r'"title"\s*:\s*"(?P<title>[^"\r\n]{1,240})"',
            block,
        )
        if not resource_match or not title_match:
            continue
        title = " ".join(title_match.group("title").split())
        if title:
            resources.setdefault(
                title,
                f"{origin}/docx/{resource_match.group('id')}",
            )
    return resources


def _is_safe_document_url(value: str) -> bool:
    """拒绝可能访问用户本机、内网或在 URL 中夹带凭据的附件地址。"""
    try:
        parsed = urlparse(str(value or ""))
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return False
    if port not in {None, 80, 443}:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        not host
        or host == "localhost"
        or host.endswith((".localhost", ".local", ".internal"))
    ):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return address.is_global


def _extract_chatgpt_document_card_candidates(
    html: str,
    base_url: str,
) -> list[DocumentCandidate]:
    """仅从 ChatGPT 真实文件标题节点建立点击下载候选。"""
    parsed = urlparse(str(base_url or ""))
    if parsed.netloc.lower().split(":", 1)[0] not in {
        "chatgpt.com", "chat.openai.com"
    }:
        return []
    candidates: list[DocumentCandidate] = []
    seen_names: set[str] = set()
    soup = BeautifulSoup(html or "", "html.parser")
    for title in soup.select("div.truncate.font-semibold"):
        filename = _document_filename_from_text(
            " ".join(title.get_text(" ", strip=True).split())
        )
        if (
            not filename
            or Path(filename).suffix.lower() not in DOCUMENT_EXTENSIONS
        ):
            continue
        filename = _safe_document_filename(filename)
        lowered = filename.lower()
        if lowered in seen_names:
            continue
        seen_names.add(lowered)
        candidates.append(DocumentCandidate(
            f"{CHATGPT_CARD_REFERENCE_PREFIX}{lowered}",
            str(base_url),
            filename,
        ))
    return candidates


def _extract_deepseek_document_card_candidates(
    html: str,
    base_url: str,
) -> list[DocumentCandidate]:
    """从 DeepSeek 私有会话可见文件卡片建立点击下载候选。"""
    parsed = urlparse(str(base_url or ""))
    if parsed.netloc.lower().split(":", 1)[0] != "chat.deepseek.com":
        return []
    filename_pattern = re.compile(
        r'[^\\/:*?"<>|\r\n]{1,180}\.'
        r'(?:pdf|docx?|xlsx?|pptx?|txt|csv|md|rtf)',
        re.IGNORECASE,
    )
    size_pattern = re.compile(
        r'^(?:PDF|DOCX?|XLSX?|PPTX?|TXT|CSV|MD|RTF)\s+'
        r'\d+(?:\.\d+)?\s*(?:B|KB|MB|GB)$',
        re.IGNORECASE,
    )
    candidates: list[DocumentCandidate] = []
    seen_names: set[str] = set()
    soup = BeautifulSoup(html or "", "html.parser")
    for message in soup.select(
        "[data-virtual-list-item-key] .ds-message"
    ):
        lines = [
            line.strip()
            for line in message.get_text(separator="\n", strip=True).splitlines()
            if line.strip()
        ]
        for index, line in enumerate(lines[:-1]):
            if (
                not filename_pattern.fullmatch(line)
                or not size_pattern.fullmatch(lines[index + 1])
            ):
                continue
            filename = _safe_document_filename(line)
            lowered = filename.lower()
            if lowered in seen_names:
                continue
            seen_names.add(lowered)
            candidates.append(DocumentCandidate(
                f"{DEEPSEEK_CARD_REFERENCE_PREFIX}{lowered}",
                str(base_url),
                filename,
            ))
    return candidates


def _extract_document_candidates(
    html: str,
    base_url: str,
) -> list[DocumentCandidate]:
    """从完整消息快照中提取真实 HTTP(S) 文档引用。"""
    soup = BeautifulSoup(html or "", "html.parser")
    candidates: list[DocumentCandidate] = []
    seen: set[tuple[str, str]] = set()
    url_attributes = (
        "href", "data-url", "data-href", "data-download-url",
        "data-file-url", "data-resource-url",
    )
    for element in soup.find_all(True):
        label = " ".join(element.get_text(" ", strip=True).split())
        declared_name = str(element.get("download") or "").strip()
        for attribute in url_attributes:
            reference = str(element.get(attribute) or "").strip()
            if not reference:
                continue
            absolute_url = urljoin(base_url, reference)
            if not _is_safe_document_url(absolute_url):
                continue
            filename = (
                _document_filename_from_text(declared_name)
                or _document_filename_from_text(label)
                or Path(unquote(urlparse(absolute_url).path)).name
            )
            if (
                Path(filename).suffix.lower() not in DOCUMENT_EXTENSIONS
                and not _document_suffix(absolute_url)
            ):
                continue
            filename = _safe_document_filename(
                filename,
                _document_suffix(absolute_url),
            )
            key = (absolute_url, filename.lower())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(DocumentCandidate(reference, absolute_url, filename))

    parsed_base = urlparse(base_url)
    host = parsed_base.netloc.lower().split(":", 1)[0]
    if host in {"doubao.com", "www.doubao.com"}:
        origin = f"{parsed_base.scheme or 'https'}://{parsed_base.netloc}"
        for candidate in _extract_doubao_embedded_document_candidates(html, origin):
            key = (candidate.url, candidate.filename.lower())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    if host in {"chatgpt.com", "chat.openai.com"}:
        origin = f"{parsed_base.scheme or 'https'}://{parsed_base.netloc}"
        share_id = _chatgpt_shared_conversation_id(base_url)
        share_query = (
            f"?shared_conversation_id={quote(share_id, safe='')}"
            if share_id else ""
        )
        for pattern in CHATGPT_EMBEDDED_DOCUMENT_PATTERNS:
            for match in pattern.finditer(html or ""):
                filename = _safe_document_filename(match.group("filename"))
                file_id = match.group("file_id")
                absolute_url = (
                    f"{origin}/backend-api/files/download/"
                    f"{quote(file_id, safe='')}{share_query}"
                )
                key = (absolute_url, filename.lower())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    DocumentCandidate(file_id, absolute_url, filename)
                )
    return candidates


def _inject_chatgpt_attachment_names(
    html: str,
    candidates: list[DocumentCandidate],
    candidates_by_message: Optional[list[list[DocumentCandidate]]] = None,
) -> str:
    """ChatGPT 只显示“上传文件”时补入已获元数据中的真实文件名。"""
    if not html or not candidates:
        return html
    soup = BeautifulSoup(html, "html.parser")
    user_messages = soup.find_all(attrs={"data-message-author-role": "user"})
    if candidates_by_message:
        for target, message_candidates in zip(
            user_messages, candidates_by_message
        ):
            existing_text = target.get_text(" ", strip=True).lower()
            for candidate in dict.fromkeys(message_candidates):
                if (
                    not candidate.filename
                    or candidate.filename.lower() in existing_text
                ):
                    continue
                marker = soup.new_tag("div")
                marker["class"] = [
                    "attachment", "document", "api-attachment-name"
                ]
                marker.string = candidate.filename
                target.append(marker)
        return str(soup)
    placeholders = [
        message for message in user_messages
        if "上传文件" in message.get_text(" ", strip=True)
    ]
    if not placeholders:
        return html
    existing_text = soup.get_text(" ", strip=True).lower()
    missing_names = list(dict.fromkeys(
        candidate.filename
        for candidate in candidates
        if candidate.filename
        and candidate.filename.lower() not in existing_text
    ))
    if not missing_names:
        return html
    target = placeholders[0]
    for filename in missing_names:
        marker = soup.new_tag("div")
        marker["class"] = ["attachment", "document", "api-attachment-name"]
        marker.string = filename
        target.append(marker)
    return str(soup)
def _document_download_url_from_payload(payload: Any) -> str:
    """只接受平台下载凭证中明确标注的安全公网 HTTP(S) 地址。"""
    preferred_keys = {
        "download_url", "signed_url", "file_url", "url", "href"
    }
    for item in _iter_json_mappings(payload):
        ordered_items = sorted(
            item.items(),
            key=lambda pair: str(pair[0]).lower() not in preferred_keys,
        )
        for key, value in ordered_items:
            key_text = str(key).lower()
            if not isinstance(value, str):
                continue
            if not (
                key_text in preferred_keys
                or "download" in key_text
                or key_text.endswith("url")
            ):
                continue
            candidate = value.strip()
            if _is_safe_document_url(candidate):
                return candidate
    return ""


async def _scroll_to_chatgpt_file_card(
    page: Any,
    filename: str,
) -> bool:
    """逐屏挂载 ChatGPT 虚拟列表，直到目标历史文件卡片出现。"""
    cards = page.locator(
        "div.truncate.font-semibold",
        has_text=filename,
    )
    if await cards.count() > 0:
        return True
    role_messages = page.locator("[data-message-author-role]")
    try:
        await role_messages.first.wait_for(state="attached", timeout=12000)
        scroll_container = await role_messages.first.evaluate_handle(
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
    except Exception:
        return False

    try:
        scroll_top = 0
        for _ in range(100):
            await scroll_container.evaluate(
                "(element, top) => element.scrollTo(0, top)",
                scroll_top,
            )
            await page.wait_for_timeout(450)
            if await cards.count() > 0:
                await cards.first.scroll_into_view_if_needed(timeout=3000)
                return True
            metrics = await scroll_container.evaluate(
                """element => ({
                    scrollHeight: element.scrollHeight,
                    clientHeight: element.clientHeight
                })"""
            )
            maximum = max(
                0,
                int(metrics["scrollHeight"]) - int(metrics["clientHeight"]),
            )
            if scroll_top >= maximum:
                break
            step = max(int(metrics["clientHeight"] * 0.55), 700)
            next_top = min(scroll_top + step, maximum)
            if next_top <= scroll_top:
                break
            scroll_top = next_top
    finally:
        await scroll_container.dispose()
    return False


async def _chatgpt_document_card_get(
    page: Any,
    candidate: DocumentCandidate,
    timeout: int,
):
    """点击 ChatGPT 已渲染的文件卡片并接收网页自身的授权下载响应。"""
    parsed = urlparse(candidate.url)
    is_card_candidate = candidate.reference.startswith(
        CHATGPT_CARD_REFERENCE_PREFIX
    )
    if (
        parsed.netloc.lower().split(":", 1)[0]
        not in {"chatgpt.com", "chat.openai.com"}
        or (
            not is_card_candidate
            and not re.match(
                r"^/backend-api/files/download/[^/]+$", parsed.path
            )
        )
    ):
        return None
    cards = page.locator(
        "div.truncate.font-semibold",
        has_text=candidate.filename,
    )
    await _set_browser_window_state(page, "minimized")
    if not await _scroll_to_chatgpt_file_card(page, candidate.filename):
        return None
    try:
        expected_path = parsed.path
        async with page.expect_response(
            lambda response: (
                (
                    is_card_candidate
                    and response.status == 200
                    and _is_document_content_response(response.url)
                )
                or (
                    not is_card_candidate
                    and urlparse(response.url).path == expected_path
                )
            ),
            timeout=min(timeout, 15000),
        ) as response_info:
            await cards.first.click(force=True, timeout=5000)
            await _set_browser_window_state(page, "minimized")
        return await response_info.value
    except Exception:
        return None
    finally:
        await _set_browser_window_state(page, "minimized")


async def _chatgpt_document_card_file_id(
    page: Any,
    candidate: DocumentCandidate,
) -> str:
    """从已挂载文件卡片的 React 属性读取 ID，不触发可见下载。"""
    if not candidate.reference.startswith(CHATGPT_CARD_REFERENCE_PREFIX):
        return ""
    if not await _scroll_to_chatgpt_file_card(page, candidate.filename):
        return ""
    cards = page.locator(
        "div.truncate.font-semibold",
        has_text=candidate.filename,
    )
    try:
        return str(await cards.first.evaluate(
            """(element, targetName) => {
                const pattern = /(?:^|[^A-Za-z0-9])(file[_-][A-Za-z0-9_-]{12,})/g;
                const seen = new WeakSet();
                let visited = 0;
                const scan = (value, depth) => {
                    if (typeof value === 'string') {
                        const ids = Array.from(
                            value.matchAll(pattern), match => match[1]
                        ).filter(id => /\d/.test(id));
                        const hasName = value.includes(targetName);
                        return {
                            hasName,
                            ids,
                            match: hasName && ids.length ? ids[0] : ''
                        };
                    }
                    if (
                        value == null
                        || depth > 10
                        || visited++ > 6000
                        || (typeof value !== 'object'
                            && typeof value !== 'function')
                        || seen.has(value)
                    ) {
                        return {hasName: false, ids: [], match: ''};
                    }
                    seen.add(value);
                    let hasName = false;
                    const ids = [];
                    let keys = [];
                    try { keys = Object.keys(value); } catch { return {hasName, ids, match: ''}; }
                    for (const key of keys.slice(0, 240)) {
                        if (key === 'accessToken' || key === 'access_token') continue;
                        let child;
                        try { child = scan(value[key], depth + 1); } catch { continue; }
                        if (child.match) return child;
                        hasName ||= child.hasName;
                        ids.push(...child.ids);
                    }
                    return {
                        hasName,
                        ids,
                        match: hasName && ids.length ? ids[0] : ''
                    };
                };
                let current = element;
                for (let depth = 0; current && depth < 8; depth++) {
                    const keys = Object.keys(current);
                    const reactKeys = [
                        ...keys.filter(key => key.startsWith('__reactProps')),
                        ...keys.filter(key => (
                            key.startsWith('__react')
                            && !key.startsWith('__reactProps')
                        ))
                    ];
                    for (const key of reactKeys) {
                        const result = scan(current[key], 0);
                        if (result.match) return result.match;
                    }
                    current = current.parentElement;
                }
                return '';
            }""",
            candidate.filename,
        ) or "")
    except Exception:
        return ""


def _deepseek_document_card_locator(page: Any, filename: str):
    return page.get_by_text(filename, exact=True).locator(
        "xpath=ancestor::*[@tabindex='0'][1]"
    )


async def _scroll_to_deepseek_file_card(
    page: Any,
    filename: str,
) -> bool:
    cards = _deepseek_document_card_locator(page, filename)
    if await cards.count() > 0:
        await cards.first.scroll_into_view_if_needed(timeout=3000)
        return True
    messages = page.locator("[data-virtual-list-item-key] .ds-message")
    try:
        await messages.first.wait_for(state="attached", timeout=12000)
        scroll_container = await messages.first.evaluate_handle(
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
    except Exception:
        return False
    try:
        scroll_top = 0
        for _ in range(500):
            await scroll_container.evaluate(
                "(element, top) => element.scrollTo(0, top)",
                scroll_top,
            )
            await page.wait_for_timeout(450)
            if await cards.count() > 0:
                await cards.first.scroll_into_view_if_needed(timeout=3000)
                return True
            metrics = await scroll_container.evaluate(
                """element => ({
                    scrollHeight: element.scrollHeight,
                    clientHeight: element.clientHeight
                })"""
            )
            maximum = max(
                0,
                int(metrics["scrollHeight"]) - int(metrics["clientHeight"]),
            )
            if scroll_top >= maximum:
                break
            step = max(int(metrics["clientHeight"] * 0.55), 700)
            next_top = min(scroll_top + step, maximum)
            if next_top <= scroll_top:
                break
            scroll_top = next_top
    finally:
        await scroll_container.dispose()
    return False


async def _deepseek_document_card_get(
    page: Any,
    candidate: DocumentCandidate,
    timeout: int,
):
    """点击 DeepSeek 私有会话文件卡片并接收其已授权文件响应。"""
    parsed = urlparse(candidate.url)
    if (
        parsed.netloc.lower().split(":", 1)[0] != "chat.deepseek.com"
        or not candidate.reference.startswith(DEEPSEEK_CARD_REFERENCE_PREFIX)
    ):
        return None
    if not await _scroll_to_deepseek_file_card(page, candidate.filename):
        return None
    cards = _deepseek_document_card_locator(page, candidate.filename)
    try:
        async with page.expect_response(
            lambda response: (
                urlparse(response.url).netloc.lower().split(":", 1)[0]
                == "files.deepseeksvc.com"
                and urlparse(response.url).path == "/api/file"
            ),
            timeout=min(timeout, 15000),
        ) as response_info:
            await cards.first.click(force=True, timeout=5000)
        return await response_info.value
    except Exception:
        return None


async def _doubao_document_api_get(
    page: Any,
    candidate: DocumentCandidate,
    timeout: int,
):
    """用豆包页面自身的同源接口把对象标识兑换为短期下载地址。"""
    parsed = urlparse(candidate.url)
    if (
        parsed.netloc.lower().split(":", 1)[0]
        not in {"doubao.com", "www.doubao.com"}
        or parsed.path != DOUBAO_DOCUMENT_API_PATH
        or not _is_safe_doubao_file_uri(candidate.reference)
    ):
        return None
    try:
        payload = await page.evaluate(
            """async ({endpoint, uri}) => {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    credentials: 'include',
                    headers: {'content-type': 'application/json'},
                    body: JSON.stringify({
                        uris: [uri],
                        type: 'file',
                        expire_second: 604800
                    })
                });
                if (!response.ok) return null;
                return await response.json();
            }""",
            {"endpoint": candidate.url, "uri": candidate.reference},
        )
    except Exception:
        return None
    download_url = _document_download_url_from_payload(payload)
    if not download_url:
        return None
    return await _authenticated_page_get(page, download_url, timeout)


def _normalize_doubao_ai_document_text(parts: Collection[str]) -> str:
    """清理豆包编辑器正文，并去除虚拟分页产生的重复文本。"""
    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = str(part or "").replace("\u200b", "").replace("\ufeff", "")
        text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return "\n\n".join(normalized)


async def _read_doubao_ai_document_body(document_page: Any) -> str:
    """从豆包在线文档的同源内嵌编辑器读取用户可见正文。"""
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        for frame in document_page.frames:
            parsed = urlparse(str(frame.url or ""))
            if (
                parsed.netloc.lower().split(":", 1)[0]
                not in {"doubao.com", "www.doubao.com"}
                or not parsed.path.startswith("/partner/ccm-docx/docx/")
            ):
                continue
            try:
                parts = await frame.locator(
                    ".render-unit-wrapper"
                ).evaluate_all(
                    "elements => elements.map(element => element.innerText || '')"
                )
            except Exception:
                continue
            body = _normalize_doubao_ai_document_text(parts)
            if body:
                return body
        await document_page.wait_for_timeout(500)
    return ""


async def _save_doubao_ai_documents(
    page: Any,
    html: str,
    documents_dir: Path,
    document_reference_prefix: str,
    warning_collector: Optional[list[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> tuple[dict[str, str], int]:
    """打开豆包 AI 在线文档卡片，将可见正文保存为本地 Markdown。"""
    titles = _extract_doubao_ai_document_titles(html, page.url)
    resource_urls = _extract_doubao_ai_document_resources(html, page.url)
    known_titles = {title.lower() for title in titles}
    titles.extend(
        title for title in resource_urls if title.lower() not in known_titles
    )
    if not titles:
        return {}, 0
    if logger:
        logger(f"检测到 {len(titles)} 个豆包 AI 生成文档，正在读取正文...")

    resolved: dict[str, str] = {}
    failures = 0
    documents_dir = Path(documents_dir)
    for title in titles:
        document_page = None
        inline_document_opened = False
        document_url = ""
        try:
            direct_document_url = resource_urls.get(title, "")
            if direct_document_url:
                document_page = await page.context.new_page()
                await document_page.goto(
                    direct_document_url,
                    wait_until="domcontentloaded",
                    timeout=10000,
                )
                document_url = str(document_page.url or "")
                body_source = document_page
            else:
                title_nodes = page.get_by_text(title, exact=True)
                if await title_nodes.count() == 0:
                    failures += 1
                    continue
                card = title_nodes.first.locator(
                    "xpath=ancestor::div[contains(@class,'product-card-')][1]"
                )
                if await card.count() == 0:
                    failures += 1
                    continue
                is_direct_chat = urlparse(str(page.url or "")).path.startswith(
                    "/chat/"
                )
                if is_direct_chat:
                    await title_nodes.first.scroll_into_view_if_needed(
                        timeout=3000
                    )
                    await title_nodes.first.click(timeout=5000)
                    inline_document_opened = True
                    body_source = page
                else:
                    async with page.expect_popup(timeout=10000) as popup_info:
                        await card.click(force=True, timeout=5000)
                    document_page = await popup_info.value
                    await document_page.wait_for_url(
                        re.compile(r"^https://(?:www\.)?doubao\.com/docx/"),
                        timeout=10000,
                    )
                    await document_page.wait_for_load_state(
                        "domcontentloaded", timeout=10000
                    )
                    document_url = str(document_page.url or "")
                    body_source = document_page

            parsed_document_url = urlparse(document_url)
            if document_url and (
                parsed_document_url.netloc.lower().split(":", 1)[0]
                not in {"doubao.com", "www.doubao.com"}
                or not parsed_document_url.path.startswith("/docx/")
            ):
                failures += 1
                continue

            body = await _read_doubao_ai_document_body(body_source)
            if not body:
                failures += 1
                continue
            if not document_url:
                document_url = next((
                    str(frame.url or "")
                    for frame in page.frames
                    if re.match(
                        r"^/partner/ccm-docx/docx/[^/]+$",
                        urlparse(str(frame.url or "")).path,
                    )
                ), "")
            payload = f"# {title}\n\n{body}\n".encode("utf-8")
            if len(payload) > GUI_DOCUMENT_MAX_BYTES:
                failures += 1
                continue

            filename = _safe_document_filename(f"{title}.md", ".md")
            documents_dir.mkdir(parents=True, exist_ok=True)
            target = documents_dir / filename
            if target.exists():
                try:
                    same_content = target.read_bytes() == payload
                except OSError:
                    same_content = False
                if not same_content:
                    digest = hashlib.sha256(
                        (document_url or title).encode("utf-8")
                    ).hexdigest()[:8]
                    target = target.with_name(
                        f"{target.stem}_{digest}{target.suffix}"
                    )
            if not target.exists():
                target.write_bytes(payload)
            local_reference = (
                f"{document_reference_prefix}/"
                f"{quote(target.name, safe='-_.~')}"
            )
            for key in {
                title.lower(),
                f"doubao-ai-doc:{title}",
                document_url,
            }:
                if key:
                    resolved[key] = local_reference
        except Exception:
            failures += 1
        finally:
            if document_page is not None:
                try:
                    if not document_page.is_closed():
                        await document_page.close()
                except Exception:
                    pass
            elif inline_document_opened:
                try:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(300)
                except Exception:
                    pass

    if failures and warning_collector is not None:
        warning_collector.append(
            f"{failures} 个豆包 AI 生成文档未能读取正文；"
            "对话文字与其余附件仍已保留。"
        )
    if logger:
        logger(
            f"豆包 AI 生成文档处理完成：发现 {len(titles)} 个，"
            f"成功保存 {len(set(resolved.values()))} 个。"
        )
    return resolved, len(titles)


async def _download_document_candidates(
    page: Any,
    candidates: list[DocumentCandidate],
    documents_dir: Path,
    document_reference_prefix: str,
    concurrency: int = 3,
    warning_collector: Optional[list[str]] = None,
    conversation_url: Optional[str] = None,
    captured_documents: Optional[
        Mapping[str, tuple[bytes, Mapping[str, str]]]
    ] = None,
) -> dict[str, str]:
    """使用页面登录态下载可用附件，并返回 URL/文件名到本地相对链接的映射。"""
    documents_dir = Path(documents_dir)
    ordered = list(dict.fromkeys(candidates))
    has_chatgpt_candidates = any(
        urlparse(candidate.url).netloc.lower().split(":", 1)[0]
        in {"chatgpt.com", "chat.openai.com"}
        for candidate in ordered
    )
    has_doubao_candidates = any(
        urlparse(candidate.url).netloc.lower().split(":", 1)[0]
        in {"doubao.com", "www.doubao.com"}
        and urlparse(candidate.url).path == DOUBAO_DOCUMENT_API_PATH
        for candidate in ordered
    )
    has_deepseek_card_candidates = any(
        candidate.reference.startswith(DEEPSEEK_CARD_REFERENCE_PREFIX)
        for candidate in ordered
    )
    effective_concurrency = (
        1
        if (
            has_chatgpt_candidates
            or has_doubao_candidates
            or has_deepseek_card_candidates
        )
        else max(1, min(int(concurrency), 4))
    )
    semaphore = asyncio.Semaphore(effective_concurrency)
    chatgpt_card_lock = asyncio.Lock()
    chatgpt_share_id = _chatgpt_shared_conversation_id(conversation_url or "")
    chatgpt_share_query = (
        f"?shared_conversation_id={quote(chatgpt_share_id, safe='')}"
        if chatgpt_share_id else ""
    )

    async def download(candidate: DocumentCandidate):
        try:
            cached = None
            if captured_documents:
                for key in (
                    candidate.reference,
                    candidate.url,
                    urlparse(candidate.url).path,
                ):
                    if key and key in captured_documents:
                        cached = captured_documents[key]
                        break
            if cached is not None:
                body, headers = cached
                return candidate, body, dict(headers), None
            async with semaphore:
                response = None
                parsed_candidate = urlparse(candidate.url)
                candidate_host = parsed_candidate.netloc.lower().split(":", 1)[0]
                is_doubao_candidate = (
                    candidate_host in {"doubao.com", "www.doubao.com"}
                    and parsed_candidate.path == DOUBAO_DOCUMENT_API_PATH
                )
                is_deepseek_card_candidate = (
                    candidate_host == "chat.deepseek.com"
                    and candidate.reference.startswith(
                        DEEPSEEK_CARD_REFERENCE_PREFIX
                    )
                )
                is_chatgpt_direct_candidate = (
                    candidate_host in {"chatgpt.com", "chat.openai.com"}
                    and bool(re.match(
                        r"^/backend-api/files/download/[^/]+$",
                        parsed_candidate.path,
                    ))
                )
                if is_doubao_candidate:
                    response = await _doubao_document_api_get(
                        page, candidate, 20000
                    )
                elif is_deepseek_card_candidate:
                    response = await _deepseek_document_card_get(
                        page, candidate, 20000
                    )
                elif is_chatgpt_direct_candidate:
                    response = await _authenticated_page_get(
                        page, candidate.url, 20000
                    )
                elif candidate_host in {"chatgpt.com", "chat.openai.com"}:
                    async with chatgpt_card_lock:
                        file_id = await _chatgpt_document_card_file_id(
                            page, candidate
                        )
                        if file_id:
                            origin = (
                                f"{parsed_candidate.scheme or 'https'}://"
                                f"{parsed_candidate.netloc}"
                            )
                            response = await _authenticated_page_get(
                                page,
                                f"{origin}/backend-api/files/download/"
                                f"{quote(file_id, safe='')}"
                                f"{chatgpt_share_query}",
                                20000,
                            )
                        # 每次文件点击后 ChatGPT 可能卸载其余虚拟卡片；
                        # 切到平台首页再返回原会话，让当前文件卡片重新挂载。
                        if response is None and conversation_url:
                            try:
                                await _rehydrate_chatgpt_conversation(
                                    page, conversation_url
                                )
                                await page.locator(
                                    "[data-message-author-role]"
                                ).first.wait_for(
                                    state="attached",
                                    timeout=15000,
                                )
                            except Exception:
                                pass
                        if response is None:
                            response = await _chatgpt_document_card_get(
                                page, candidate, 20000
                            )
                if response is None:
                    if is_doubao_candidate or is_deepseek_card_candidate:
                        return candidate, None, {}, "not_a_document"
                    response = await _authenticated_page_get(
                        page, candidate.url, 20000
                    )
            if not response.ok:
                return candidate, None, {}, f"http_{response.status}"
            headers = dict(getattr(response, "headers", {}) or {})
            length = headers.get("content-length", "").strip()
            if length.isdigit() and int(length) > GUI_DOCUMENT_MAX_BYTES:
                return candidate, None, headers, "too_large"
            content_type = headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type == "application/json":
                try:
                    payload = await response.json()
                except Exception:
                    return candidate, None, headers, "not_a_document"
                download_url = _document_download_url_from_payload(payload)
                if not download_url:
                    return candidate, None, headers, "not_a_document"
                response = await _authenticated_page_get(
                    page, download_url, 20000
                )
                if not response.ok:
                    return candidate, None, {}, f"http_{response.status}"
                headers = dict(getattr(response, "headers", {}) or {})
                length = headers.get("content-length", "").strip()
                if length.isdigit() and int(length) > GUI_DOCUMENT_MAX_BYTES:
                    return candidate, None, headers, "too_large"
                content_type = headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type in {"text/html", "application/json"}:
                return candidate, None, headers, "not_a_document"
            body = await response.body()
            if not body:
                return candidate, None, headers, "empty_body"
            if len(body) > GUI_DOCUMENT_MAX_BYTES:
                return candidate, None, headers, "too_large"
            return candidate, body, headers, None
        except Exception as error:
            return candidate, None, {}, type(error).__name__

    if (
        has_chatgpt_candidates
        or has_doubao_candidates
        or has_deepseek_card_candidates
    ):
        results = []
        downloaded_names: set[str] = set()
        for candidate in ordered:
            if (
                candidate.reference.startswith(CHATGPT_CARD_REFERENCE_PREFIX)
                and candidate.filename.lower() in downloaded_names
            ):
                continue
            result = await download(candidate)
            results.append(result)
            if result[1] is not None:
                downloaded_names.add(candidate.filename.lower())
    else:
        results = await asyncio.gather(*(
            download(candidate) for candidate in ordered
        ))
    resolved: dict[str, str] = {}
    failures: dict[str, int] = {}
    successful_names = {
        candidate.filename.lower()
        for candidate, body, _headers, _reason in results
        if body is not None
    }
    for candidate, body, headers, reason in results:
        if body is None:
            if candidate.filename.lower() in successful_names:
                continue
            failures[reason or "unknown_error"] = failures.get(
                reason or "unknown_error", 0
            ) + 1
            continue
        disposition_name = _document_filename_from_disposition(
            headers.get("content-disposition", "")
        )
        content_type = headers.get("content-type", "").split(";", 1)[0].lower()
        inferred_suffix = (
            DOCUMENT_MIME_EXTENSIONS.get(content_type)
            or _document_suffix(candidate.url)
            or mimetypes.guess_extension(content_type)
            or ""
        )
        filename = _safe_document_filename(
            disposition_name or candidate.filename,
            inferred_suffix,
        )
        if Path(filename).suffix.lower() not in DOCUMENT_EXTENSIONS:
            failures["unsupported_type"] = failures.get("unsupported_type", 0) + 1
            continue
        body = _repair_downloaded_text_mojibake(body, filename, content_type)
        documents_dir.mkdir(parents=True, exist_ok=True)
        target = documents_dir / filename
        if target.exists():
            try:
                same_content = target.read_bytes() == body
            except OSError:
                same_content = False
            if not same_content:
                digest = hashlib.sha256(candidate.url.encode("utf-8")).hexdigest()[:8]
                target = target.with_name(f"{target.stem}_{digest}{target.suffix}")
        if not target.exists():
            target.write_bytes(body)
        local_reference = (
            f"{document_reference_prefix}/"
            f"{quote(target.name, safe='-_.~')}"
        )
        for key in {
            candidate.reference,
            candidate.url,
            candidate.filename.lower(),
            filename.lower(),
        }:
            if key:
                resolved[key] = local_reference

    if failures and warning_collector is not None:
        reason_summary = "、".join(
            f"{reason}×{count}" for reason, count in sorted(failures.items())
        )
        warning_collector.append(
            f"{sum(failures.values())} 个文档附件未能保存到本地"
            f"（{reason_summary}）；对话文字抓取继续保留。"
        )
    return resolved

async def _close_browser_context_safely(
    context: Any,
    warnings: list[str],
    logger: Optional[Callable[[str], None]] = None,
) -> None:
    """关闭浏览器；驱动已断开时保留此前结果并记录清理警告。"""
    try:
        await asyncio.wait_for(context.close(), timeout=10)
    except Exception as error:
        warning = f"浏览器清理异常（已保留此前抓取结果）：{error}"
        if warning not in warnings:
            warnings.append(warning)
        if logger:
            try:
                logger(warning)
            except Exception:
                pass


async def fetch_chat_pipeline(
    url: str,
    need_login: bool = False,
    login_ready_event: Optional[asyncio.Event] = None,
    login_required_callback: Optional[Callable[[], None]] = None,
    logger: Optional[Callable[[str], None]] = None,
    image_output_dir: Optional[Path] = None,
    image_reference_base: Optional[Path] = None,
    document_output_dir: Optional[Path] = None,
    document_reference_base: Optional[Path] = None,
    document_download_concurrency: int = 3,
    image_download_concurrency: int = GUI_IMAGE_DOWNLOAD_CONCURRENCY,
    browser_profile_root: Optional[Path] = None,
    debug_html_file: Optional[Path] = None,
) -> FetchResult:
    """异步抓取并提取网页中的对话和图片。"""
    fetch_warnings: list[str] = []
    completed_result: Optional[FetchResult] = None
    user_wait_seconds = 0.0
    if image_output_dir is None:
        resolved_images_dir = Path(IMAGES_DIR).resolve()
        image_reference_prefix = "./images"
    else:
        resolved_images_dir = Path(image_output_dir).resolve()
        reference_base = Path(
            image_reference_base or resolved_images_dir.parent
        ).resolve()
        image_reference_prefix = build_markdown_asset_prefix(
            resolved_images_dir,
            reference_base,
        )
    if document_output_dir is None:
        resolved_documents_dir = Path(PROJECT_ROOT, "attachments").resolve()
        document_reference_prefix = "./attachments"
    else:
        resolved_documents_dir = Path(document_output_dir).resolve()
        document_base = Path(
            document_reference_base or resolved_documents_dir.parent
        ).resolve()
        document_reference_prefix = build_markdown_asset_prefix(
            resolved_documents_dir,
            document_base,
        )
    requires_login_probe = requires_authenticated_browser(url)
    requested_host = urlparse(url).netloc.lower().split(":", 1)[0]
    chatgpt_minimized = (
        not need_login
        and requires_login_probe
        and requested_host in {"chatgpt.com", "chat.openai.com"}
    )
    headless = not need_login and not chatgpt_minimized

    viewport_config = None if need_login or chatgpt_minimized else {
        "width": 1920,
        "height": 10800
    }

    if logger:
        if need_login:
            logger("正在启动浏览器供您登录或读取已保存的登录状态...")
        elif chatgpt_minimized:
            logger("正在最小化浏览器中读取已保存的 ChatGPT 登录状态...")
        elif requires_login_probe:
            logger("正在后台读取已保存的 AI 登录状态...")
        else:
            logger("正在启动无头浏览器加载分享页...")

    try:
        async with async_playwright() as playwright:
            context, _browser_channel = await launch_browser_context(
                playwright,
                headless=headless,
                viewport=viewport_config,
                no_viewport=need_login or chatgpt_minimized,
                start_minimized=chatgpt_minimized,
                logger=logger,
                profile_root=browser_profile_root,
            )
            page = context.pages[0] if context.pages else await context.new_page()
            if chatgpt_minimized:
                await _set_browser_window_state(page, "minimized")
            response_document_candidates: list[DocumentCandidate] = []
            response_image_references: set[str] = set()
            response_tasks: set[asyncio.Task] = set()
            captured_document_responses: dict[
                str, tuple[bytes, dict[str, str]]
            ] = {}
            authorized_content_responses: set[str] = set()
            chatgpt_assets_rehydrated = False
            pre_rehydrate_chat_html: Optional[str] = None

            def capture_response_assets(response: Any) -> None:
                task = None
                if _is_asset_metadata_response(response.url):
                    if response.status == 200:
                        authorized_content_responses.add(response.url)
                    task = asyncio.create_task(_capture_response_assets(
                        response,
                        url,
                        response_document_candidates,
                        response_image_references,
                    ))
                elif _is_document_content_response(response.url):
                    task = asyncio.create_task(
                        _capture_document_content_response(
                            response, captured_document_responses
                        )
                    )
                if task is not None:
                    response_tasks.add(task)
                    task.add_done_callback(response_tasks.discard)

            page.on("response", capture_response_assets)

            try:
                if logger:
                    host = urlparse(url).netloc.lower() or "AI"
                    logger(f"正在加载 {host} 分享页...")
                await goto_with_retry_gui(page, url, logger=logger)

                if need_login or requires_login_probe:
                    await page.wait_for_timeout(1800)
                    await _drain_response_tasks(response_tasks)
                    content_ready = (
                        await _page_has_conversation_content(page, url)
                        or bool(authorized_content_responses)
                    )
                    if content_ready:
                        if requested_host in {"chatgpt.com", "chat.openai.com"}:
                            await _set_browser_window_state(page, "minimized")
                        if logger:
                            logger("已复用此前保存的登录状态，无需重复授权。")
                    else:
                        if not need_login and not chatgpt_minimized:
                            if logger:
                                logger(
                                    "该平台无法在无界面模式读取会话，"
                                    "正在最小化浏览器中继续尝试..."
                                )
                            await _close_browser_context_safely(
                                context, fetch_warnings, logger
                            )
                            context, _browser_channel = (
                                await launch_browser_context(
                                    playwright,
                                    headless=False,
                                    viewport=None,
                                    no_viewport=True,
                                    start_minimized=True,
                                    logger=logger,
                                    profile_root=browser_profile_root,
                                )
                            )
                            page = (
                                context.pages[0]
                                if context.pages
                                else await context.new_page()
                            )
                            await _set_browser_window_state(page, "minimized")
                            page.on("response", capture_response_assets)
                            await goto_with_retry_gui(page, url, logger=logger)
                            await page.wait_for_timeout(1800)
                            await _drain_response_tasks(response_tasks)
                            content_ready = (
                                await _page_has_conversation_content(page, url)
                                or bool(authorized_content_responses)
                            )
                            if content_ready:
                                if logger:
                                    logger(
                                        "已在最小化浏览器中复用登录状态，"
                                        "无需重复授权。"
                                    )
                            else:
                                need_login = True
                                await _set_browser_window_state(page, "normal")
                        elif not need_login:
                            need_login = True
                            await _set_browser_window_state(page, "normal")
                        if not content_ready:
                            if logger:
                                logger(
                                    "当前登录状态无法读取该会话，请在浏览器中登录后"
                                    "点击【登录完毕】..."
                                )
                            if login_required_callback is not None:
                                login_required_callback()
                            if login_ready_event:
                                login_wait_started = time.perf_counter()
                                await login_ready_event.wait()
                                user_wait_seconds += (
                                    time.perf_counter() - login_wait_started
                                )
                            if logger:
                                logger("登录确认完毕，正在继续抓取对话数据...")
                            await page.wait_for_timeout(1200)

                            # 登录流程可能跳回平台首页；确认后重新打开原始会话。
                            await goto_with_retry_gui(page, url, logger=logger)
                            await page.wait_for_timeout(1800)
                            await _drain_response_tasks(response_tasks)
                            if requested_host in {
                                "chatgpt.com", "chat.openai.com"
                            }:
                                await _set_browser_window_state(
                                    page, "minimized"
                                )
                            if logger:
                                logger(
                                    "已重新打开原始对话链接，正在读取完整内容..."
                                )

                await page.wait_for_timeout(1000)
                await _drain_response_tasks(response_tasks)
                current_host = urlparse(url).netloc.lower().split(":", 1)[0]
                if current_host in {"chatgpt.com", "chat.openai.com"}:
                    initial_snapshot = await page.content()
                    response_document_candidates.extend(
                        _extract_document_candidates(initial_snapshot, page.url)
                    )
                    response_document_candidates[:] = list(dict.fromkeys(
                        response_document_candidates
                    ))
                if await _chatgpt_assets_need_rehydrate(
                    page,
                    url,
                    response_document_candidates,
                    response_image_references,
                ):
                    try:
                        await page.wait_for_selector(
                            WAIT_SELECTOR, state="attached", timeout=15000
                        )
                        pre_rehydrate_chat_html = (
                            await collect_virtualized_html(page)
                        )
                    except Exception:
                        pre_rehydrate_chat_html = None
                    if logger:
                        logger(
                            "检测到附件尚未完成渲染，正在切到平台首页后返回原对话..."
                        )
                    await _rehydrate_chatgpt_conversation(page, url, logger=logger)
                    chatgpt_assets_rehydrated = True
                    await page.wait_for_timeout(2500)
                    await _drain_response_tasks(response_tasks)

                if logger:
                    logger("正在等待动态内容渲染...")
                try:
                    await page.wait_for_selector(
                        WAIT_SELECTOR,
                        state="attached",
                        timeout=15000
                    )
                except Exception:
                    if logger:
                        logger("等待动态节点超时，可能网页结构有所变化或需登录访问。")
                await page.wait_for_timeout(2000)

                current_host = urlparse(url).netloc.lower().split(":", 1)[0]
                if current_host in {"chatgpt.com", "chat.openai.com"}:
                    late_snapshot = await page.content()
                    response_document_candidates.extend(
                        _extract_document_candidates(late_snapshot, page.url)
                    )
                    response_document_candidates[:] = list(dict.fromkeys(
                        response_document_candidates
                    ))
                    if (
                        not chatgpt_assets_rehydrated
                        and await _chatgpt_assets_need_rehydrate(
                            page,
                            url,
                            response_document_candidates,
                            response_image_references,
                        )
                    ):
                        if pre_rehydrate_chat_html is None:
                            try:
                                pre_rehydrate_chat_html = (
                                    await collect_virtualized_html(page)
                                )
                            except Exception:
                                pass
                        if logger:
                            logger(
                                "动态页面仍只显示附件占位，正在切换页面后返回原对话..."
                            )
                        await _rehydrate_chatgpt_conversation(
                            page, url, logger=logger
                        )
                        await page.wait_for_timeout(3000)
                        await _drain_response_tasks(response_tasks)
                        try:
                            await page.wait_for_selector(
                                WAIT_SELECTOR,
                                state="attached",
                                timeout=10000,
                            )
                        except Exception:
                            pass

                page_snapshot_html = await page.content()
                html = await collect_virtualized_html(page)
                if html is None:
                    html = page_snapshot_html
                if (
                    pre_rehydrate_chat_html
                    and pre_rehydrate_chat_html.count("data-message-author-role")
                    > html.count("data-message-author-role")
                ):
                    html = pre_rehydrate_chat_html
                chatgpt_image_groups, chatgpt_document_groups = (
                    await _chatgpt_message_asset_groups(page, url)
                )
                if chatgpt_image_groups:
                    html = _inject_chatgpt_message_images(
                        html, chatgpt_image_groups
                    )
                else:
                    html = _inject_chatgpt_shared_images(
                        html,
                        _extract_chatgpt_shared_image_sources(
                            page_snapshot_html, url
                        ),
                    )
                soup_pre = BeautifulSoup(html, "html.parser")
                image_candidates: list[str] = []
                if logger:
                    logger("正在检查并下载页面中的图片资产...")

                for img in soup_pre.find_all(["img", "source"]):
                    src_candidates = [img.get("src"), img.get("data-src")]
                    srcset = img.get("srcset")
                    if srcset:
                        for item in srcset.split(","):
                            parts = item.strip().split()
                            if parts:
                                src_candidates.append(parts[0])

                    for src in src_candidates:
                        alt = str(img.get("alt") or "").strip().lower()
                        if not (
                            src
                            and src.startswith("http")
                            and not src.startswith("data:image/svg")
                        ):
                            continue
                        if (
                            alt == "asset cover"
                            or "doc-canvas-card-fallback" in src.lower()
                        ):
                            continue
                        image_candidates.append(src)

                download_started = time.perf_counter()
                image_map = await _download_image_candidates(
                    page,
                    image_candidates,
                    resolved_images_dir,
                    image_reference_prefix,
                    image_download_concurrency,
                    fetch_warnings,
                )
                if logger:
                    usable_sources = _ordered_image_sources(image_candidates)
                    logger(
                        f"图片资产处理完成：发现 {len(image_candidates)} 个引用，"
                        f"去重并过滤装饰图后 {len(usable_sources)} 个，"
                        f"成功下载或复用 {len(image_map)} 个，耗时 "
                        f"{time.perf_counter() - download_started:.1f} 秒。"
                    )
                document_started = time.perf_counter()
                await _drain_response_tasks(response_tasks)
                document_candidates = list(dict.fromkeys([
                    *(
                        candidate
                        for group in chatgpt_document_groups
                        for candidate in group
                    ),
                    *response_document_candidates,
                    *_extract_document_candidates(
                        page_snapshot_html + chr(10) + html,
                        page.url,
                    ),
                ]))
                document_candidates.extend(
                    _extract_chatgpt_document_card_candidates(html, page.url)
                )
                document_candidates = list(dict.fromkeys(document_candidates))
                existing_document_names = {
                    candidate.filename.lower()
                    for candidate in document_candidates
                }
                for candidate in _extract_deepseek_document_card_candidates(
                    html, page.url
                ):
                    if candidate.filename.lower() in existing_document_names:
                        continue
                    document_candidates.append(candidate)
                    existing_document_names.add(candidate.filename.lower())
                document_map = await _download_document_candidates(
                    page,
                    document_candidates,
                    resolved_documents_dir,
                    document_reference_prefix,
                    document_download_concurrency,
                    fetch_warnings,
                    conversation_url=url,
                    captured_documents=captured_document_responses,
                )
                doubao_ai_document_count = 0
                if current_host in {"doubao.com", "www.doubao.com"}:
                    ai_document_map, doubao_ai_document_count = (
                        await _save_doubao_ai_documents(
                            page,
                            html,
                            resolved_documents_dir,
                            document_reference_prefix,
                            fetch_warnings,
                            logger,
                        )
                    )
                    document_map.update(ai_document_map)

                    # 豆包混合对话中，用户附件卡片可能先渲染文件名，原始
                    # 对象 uri 则在 AI 在线文档读取期间才写入页面状态。
                    # 仅在豆包内补取一次内嵌附件，避免影响其他平台与通用
                    # 文档候选规则。
                    try:
                        late_doubao_html = await page.content()
                    except Exception:
                        late_doubao_html = ""
                    page_url = urlparse(page.url)
                    late_doubao_candidates = (
                        _extract_doubao_embedded_document_candidates(
                            late_doubao_html,
                            f"{page_url.scheme or 'https'}://{page_url.netloc}",
                        )
                        if late_doubao_html
                        else []
                    )
                    known_doubao_candidates = set(document_candidates)
                    late_doubao_candidates = [
                        candidate
                        for candidate in late_doubao_candidates
                        if candidate not in known_doubao_candidates
                    ]
                    if late_doubao_candidates:
                        document_map.update(
                            await _download_document_candidates(
                                page,
                                late_doubao_candidates,
                                resolved_documents_dir,
                                document_reference_prefix,
                                document_download_concurrency,
                                fetch_warnings,
                                conversation_url=url,
                                captured_documents=(
                                    captured_document_responses
                                ),
                            )
                        )
                        document_candidates.extend(late_doubao_candidates)

                if logger:
                    logger(
                        "文档附件处理完成：发现 "
                        f"{len(document_candidates) + doubao_ai_document_count} "
                        "个可用引用，"
                        f"成功下载或复用 {len(set(document_map.values()))} 个，耗时 "
                        f"{time.perf_counter() - document_started:.1f} 秒。"
                    )

                snapshot_saved = False
                if SAVE_DEBUG_SNAPSHOT:
                    try:
                        snapshot_path = Path(
                            debug_html_file or DEBUG_HTML_FILE
                        ).resolve()
                        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(
                            snapshot_path, "w", encoding="utf-8"
                        ) as output:
                            output.write(html)
                        snapshot_saved = True
                    except Exception:
                        pass

                html = _inject_chatgpt_attachment_names(
                    html,
                    document_candidates,
                    chatgpt_document_groups,
                )
                soup = BeautifulSoup(html, "html.parser")
                parser_asset_map = {**image_map, **document_map}
                provider, parsed_messages = parse_messages(
                    soup, parser_asset_map
                )
                if provider is not None:
                    if logger:
                        logger(f"检测到 {provider.DISPLAY_NAME} 对话格式，使用专用解析器。")
                else:
                    if logger:
                        logger("未识别出平台标志性类名，使用降级解析。")
                    parsed_messages = parse_fallback_messages_gui(soup)

                if not parsed_messages:
                    completed_result = FetchResult(
                        html=html,
                        image_map=image_map,
                        messages=[],
                        document_map=document_map,
                        error=(
                            "未能提取到有效对话内容。"
                            + (
                                "调试快照已保存到本地。"
                                if snapshot_saved else ""
                            )
                        ),
                        warnings=fetch_warnings,
                        user_wait_seconds=user_wait_seconds,
                    )
                    return completed_result

                completed_result = FetchResult(
                    html=html,
                    image_map=image_map,
                    messages=parsed_messages,
                    document_map=document_map,
                    warnings=fetch_warnings,
                    user_wait_seconds=user_wait_seconds,
                )
                return completed_result

            finally:
                await _close_browser_context_safely(
                    context, fetch_warnings, logger
                )
    except Exception as e:
        if completed_result is not None:
            warning = (
                "Playwright 驱动退出异常（已保留此前抓取结果）："
                f"{e}"
            )
            if warning not in fetch_warnings:
                fetch_warnings.append(warning)
            if logger:
                try:
                    logger(warning)
                except Exception:
                    pass
            return completed_result
        return FetchResult(
            html=None,
            image_map={},
            messages=[],
            error=re.sub(r"https?://[^\s)\]]+", "<分享链接>", str(e)),
            warnings=fetch_warnings,
            user_wait_seconds=user_wait_seconds,
        )


def generate_raw_markdown(messages: list[dict[str, str]], target_path: Path) -> Path:
    """生成原始对话 Markdown 并保存到目标路径。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as file:
        file.write("# AI 对话记忆导出\n\n")
        for item in messages:
            if item["role"] == "User":
                file.write(
                    '\n<hr style="border: 0; border-top: 5px solid #2563EB; '
                    'margin: 48px 0 24px 0;">\n\n'
                    f"## 🔵 👤 用户提问\n\n{item['content']}\n\n"
                )
            else:
                file.write(
                    '\n<hr style="border: 0; border-top: 5px solid #9333EA; '
                    'margin: 48px 0 24px 0;">\n\n'
                    f"## 🟣 🤖 AI 回答\n\n{item['content']}\n\n"
                )
    return target_path


def generate_output_bundle(
    messages: list[dict[str, str]],
    modes: Mapping[str, bool],
    save_dir: Path,
    *,
    output_filename: Optional[str] = None,
    project_dir: Path = PROJECT_ROOT,
    section_selector: Optional[
        Callable[[dict[str, Any]], Collection[str] | str]
    ] = None,
    topic_selector: Optional[
        Callable[[dict[str, Any]], Collection[str] | str]
    ] = None,
    progress: Callable[[str], None] = lambda _message: None,
    config: Any = None,
    gateway: Any = None,
    api_keys: Optional[Mapping[str, str]] = None,
    result_cache_dir: Optional[Path] = None,
    source_platform: Optional[str] = None,
    source_name: Optional[str] = None,
    source_dir: Optional[Path] = None,
) -> GenerationBundle:
    """按 GUI 模式生成文件，并让普通/详细版复用同一份语义结果。

    选择器会在后端已经完成主题与分类综合、尚未写出 Markdown 时调用。
    GUI 使用 ``topic_selector`` 选择具体主题；旧的 ``section_selector`` 继续
    兼容命令行式分类选择。普通版和详细版复用同一选择结果；极简版和仅抓取
    模式不会触发选择。
    """
    if not messages:
        raise ValueError("没有可生成的对话消息。")
    if not any(bool(value) for value in modes.values()):
        raise ValueError("至少需要选择一种生成模式。")

    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(save_dir, modes, output_filename)
    saved_files: list[Path] = []

    if modes.get("raw"):
        raw_path = generate_raw_markdown(
            messages, output_paths["raw_markdown"]
        )
        saved_files.append(raw_path)
        progress(f"已生成 raw 对话文件：{raw_path.name}")

    needs_summary = any(
        modes.get(key) for key in ("normal", "simple", "detailed")
    )
    if not needs_summary:
        return GenerationBundle(saved_files=saved_files)

    from scripts.gemini_summarizer import (
        SUMMARY_SECTION_LABELS,
        SummaryConfig,
        create_gateway,
        normalize_summary_sections,
        normalize_summary_topics,
        summarize_conversation,
        write_summary_outputs,
    )

    config_was_provided = config is not None
    gateway_was_provided = gateway is not None
    config = config or SummaryConfig.from_env()
    user_api_keys = (
        None
        if api_keys is None
        else {
            provider: str(api_keys.get(provider) or "").strip()
            for provider in normalize_provider_order(api_keys)
            if str(api_keys.get(provider) or "").strip()
        }
    )
    if user_api_keys is not None:
        config = resolve_gui_summary_config(config, user_api_keys)
    selected_sections: tuple[str, ...] = ()
    selected_topics: tuple[str, ...] = ()

    def capture_selection(result: dict[str, Any]) -> tuple[str, ...]:
        nonlocal selected_sections
        raw_selection: Collection[str] | str = ()
        if section_selector is not None:
            raw_selection = section_selector(result)
        normalized = normalize_summary_sections(raw_selection)
        selected_sections = tuple(
            key for key in SUMMARY_SECTION_LABELS if key in normalized
        )
        return selected_sections

    def capture_topic_selection(result: dict[str, Any]) -> tuple[str, ...]:
        nonlocal selected_topics
        raw_selection: Collection[str] | str = ()
        if topic_selector is not None:
            raw_selection = topic_selector(result)
        selected_topics = normalize_summary_topics(result, raw_selection)
        return selected_topics

    has_selectable_output = bool(modes.get("normal") or modes.get("detailed"))
    section_callback = (
        capture_selection
        if has_selectable_output and section_selector is not None
        else None
    )
    topic_callback = (
        capture_topic_selection
        if has_selectable_output and topic_selector is not None
        else None
    )

    if modes.get("normal"):
        primary_json = output_paths["normal_json"]
        primary_markdown = output_paths["normal_markdown"]
        primary_includes_details = False
        intermediate_only = False
    elif modes.get("detailed"):
        primary_json = output_paths["detailed_json"]
        primary_markdown = output_paths["detailed_markdown"]
        primary_includes_details = True
        intermediate_only = False
    else:
        # 极简版仍需先生成完整语义结果，但不应额外留下普通版 Markdown。
        primary_json = output_paths["intermediate_json"]
        primary_markdown = output_paths["intermediate_markdown"]
        primary_includes_details = False
        intermediate_only = True

    if gateway_was_provided:
        attempts = [(config, gateway)]
    else:
        candidate_configs = (
            [config]
            if config_was_provided
            else (
                gui_summary_attempt_configs(config, user_api_keys)
                if user_api_keys is not None
                else gui_summary_config_candidates(config)
            )
        )
        attempts = [(candidate, None) for candidate in candidate_configs]

    from scripts.gemini_summarizer import (
        GeminiSummaryError,
        SummaryRequestTimeoutError,
        safe_error_message,
    )

    base_result = None
    last_error: Optional[BaseException] = None
    timed_out_providers: set[str] = set()

    def next_usable_config(start_index: int) -> Optional[Any]:
        for later_config, _later_gateway in attempts[start_index:]:
            if later_config.provider not in timed_out_providers:
                return later_config
        return None

    for attempt_index, (candidate_config, candidate_gateway) in enumerate(
        attempts, start=1
    ):
        if candidate_config.provider in timed_out_providers:
            continue
        if candidate_gateway is None:
            if user_api_keys is None:
                candidate_gateway = create_gateway(candidate_config)
            else:
                candidate_gateway = create_gateway(
                    candidate_config,
                    api_key=user_api_keys[candidate_config.provider],
                )
        progress(
            f"正在调用 {candidate_config.provider}/{candidate_config.model} "
            "生成结构化分层总结..."
        )
        try:
            base_result = summarize_conversation(
                messages=messages,
                project_dir=Path(project_dir).resolve(),
                source_dir=(
                    Path(source_dir).resolve()
                    if source_dir is not None
                    else save_dir
                ),
                source_name=source_name or output_paths["raw_markdown"].name,
                output_json=primary_json,
                output_markdown=primary_markdown,
                config=candidate_config,
                gateway=candidate_gateway,
                progress=progress,
                include_details=primary_includes_details,
                selected_sections=(),
                section_selector=section_callback,
                selected_topics=(),
                topic_selector=topic_callback,
                result_cache_dir=(
                    Path(result_cache_dir).resolve()
                    if result_cache_dir is not None
                    else default_summary_result_cache_dir()
                ),
                source_platform=source_platform,
            )
            config = candidate_config
            gateway = candidate_gateway
            break
        except SummaryRequestTimeoutError as error:
            # 同一提供商的其他模型通常共享网络入口；超时后跳过该提供商，
            # 但若用户还配置了另一提供商，则立即切换，避免 120 秒后直接失败。
            last_error = error
            timed_out_providers.add(candidate_config.provider)
            next_config = next_usable_config(attempt_index)
            if next_config is None:
                raise
            safe_message = safe_error_message(
                error,
                tuple(user_api_keys.values()) if user_api_keys else (),
            )
            progress(
                f"{candidate_config.provider} 请求超时（{safe_message}），"
                f"正在切换到 {next_config.provider}/{next_config.model}..."
            )
        except GeminiSummaryError as error:
            last_error = error
            next_config = next_usable_config(attempt_index)
            if next_config is None:
                raise
            safe_message = safe_error_message(
                error,
                tuple(user_api_keys.values()) if user_api_keys else (),
            )
            progress(
                f"当前模型不可用（{safe_message}），"
                f"正在切换到 {next_config.provider}/{next_config.model}..."
            )

    if base_result is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("总结后端未返回结果。")

    if modes.get("normal"):
        saved_files.extend([primary_markdown, primary_json])

    if modes.get("detailed"):
        if primary_includes_details:
            detailed_json = primary_json
            detailed_markdown = primary_markdown
        else:
            detailed_json = output_paths["detailed_json"]
            detailed_markdown = output_paths["detailed_markdown"]
            write_summary_outputs(
                base_result,
                detailed_json,
                detailed_markdown,
                include_details=True,
                selected_sections=selected_sections,
                selected_topics=selected_topics,
            )
        saved_files.extend([detailed_markdown, detailed_json])

    if modes.get("simple"):
        from scripts.simple_summarizer import (
            build_simple_metadata,
            generate_simple_overview,
            write_simple_markdown,
        )

        progress("正在生成极简版总览...")
        simple_overview = generate_simple_overview(
            base_result, messages, gateway
        )
        simple_meta = build_simple_metadata(
            base_result,
            provider=config.provider,
            model=config.model,
        )
        simple_markdown = output_paths["simple_markdown"]
        write_simple_markdown(simple_markdown, simple_overview, simple_meta)
        saved_files.append(simple_markdown)

    if intermediate_only:
        for intermediate_path in (primary_json, primary_markdown):
            try:
                intermediate_path.unlink()
            except FileNotFoundError:
                pass

    return GenerationBundle(
        saved_files=saved_files,
        selected_sections=selected_sections,
        selected_topics=selected_topics,
        summary_result=base_result,
    )
