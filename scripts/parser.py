import argparse
import asyncio
import hashlib
import os
import sys

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from rich.console import Console
from rich.panel import Panel
from urllib.parse import urlparse

from .markdown_exporter import display_and_export
from .project_paths import (
    BROWSER_USER_DATA_DIR,
    DEBUG_HTML_FILE,
    DEFAULT_EXPORT_FILE,
    IMAGES_DIR,
    PROJECT_ROOT,
)
from .providers import (
    WAIT_SELECTOR,
    collect_virtualized_html,
    parse_messages,
    provider_for_host,
)


console = Console()


async def goto_with_retry(page, url, attempts=3):
    """加载分享页；网络偶发超时时自动重试，并明确显示进度。"""
    for attempt in range(1, attempts + 1):
        try:
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=45000
            )
            return
        except Exception:
            if attempt >= attempts:
                raise
            console.print(
                f"[yellow]页面加载超时，正在进行第 "
                f"{attempt + 1}/{attempts} 次尝试...[/yellow]"
            )
            await page.wait_for_timeout(2000 * attempt)


async def _fetch_image_bytes(page, src):
    """下载一张图片并返回字节；两种通道互为兜底。

    首选 Playwright 的 ``page.request``（独立请求栈）。部分网络环境下该栈
    会被重置（如 ``ECONNRESET``），此时回退到页面内 ``fetch``——它走浏览器
    自身的网络栈，自动携带 Cookie/Referer，与页面正常加载图片同一通道，
    对需要签名或同源策略宽松的图床（如 lh3.googleusercontent.com）更可靠。
    任一通道失败均返回 ``None``，由调用方降级为保留远程图片链接。
    """
    try:
        response = await page.request.get(src, timeout=10000)
        if response.ok:
            return await response.body()
    except Exception:
        pass

    try:
        data_url = await page.evaluate(
            """async (u) => {
                const resp = await fetch(u, { credentials: 'include' });
                if (!resp.ok) return null;
                const buf = await resp.arrayBuffer();
                const bytes = new Uint8Array(buf);
                let bin = '';
                const chunk = 0x8000;
                for (let i = 0; i < bytes.length; i += chunk) {
                    bin += String.fromCharCode.apply(
                        null, bytes.subarray(i, i + chunk)
                    );
                }
                return 'data:b;base64,' + btoa(bin);
            }""",
            src,
        )
        if data_url and "," in data_url:
            import base64
            return base64.b64decode(data_url.split(",", 1)[1])
    except Exception:
        pass

    return None


async def _wait_for_conversation_content(page):
    """等待对话内容挂载到 DOM（状态绑定，而非固定 sleep）。

    短链跳转完成后以 ``page.url`` 的域名锁定平台适配器，等待该平台自己的
    ``WAIT_SELECTOR`` 出现——比全平台联合选择器更准确，且日志能说明在等
    哪个平台；域名无法识别时回退到联合选择器。等待成功/超时都有明确日志。
    """
    host = urlparse(page.url).netloc.lower().split(":", 1)[0]
    provider = provider_for_host(host)

    if provider is not None:
        console.print(
            f"[dim]正在等待 {provider.DISPLAY_NAME} 对话内容渲染...[/dim]"
        )
        try:
            await page.wait_for_selector(
                provider.WAIT_SELECTOR,
                state="attached",
                timeout=30000,
            )
            console.print(
                f"[dim]{provider.DISPLAY_NAME} 对话节点已出现。[/dim]"
            )
            return
        except Exception:
            console.print(
                f"[yellow]等待 {provider.DISPLAY_NAME} 对话节点超时，"
                "正在尝试通用检测...[/yellow]"
            )
    else:
        console.print("[dim]正在等待动态内容渲染...[/dim]")

    try:
        await page.wait_for_selector(
            WAIT_SELECTOR,
            state="attached",
            timeout=15000,
        )
    except Exception:
        console.print(
            "[dim]提示: 等待动态节点超时，可能网页结构有所变化"
            "或需登录访问。[/dim]"
        )


async def fetch_chat_content(url, need_login=False):
    user_data_dir = str(BROWSER_USER_DATA_DIR)
    images_dir = str(IMAGES_DIR)
    os.makedirs(images_dir, exist_ok=True)

    headless = not need_login
    if need_login:
        console.print(
            "[bold yellow]正在启动浏览器供您登录账号...[/bold yellow]"
        )
        console.print("[dim]请在弹出的浏览器中登录您的 AI 账号。[/dim]")

    console.print(f"[bold cyan]正在加载页面:[/bold cyan] {url}")

    # 无头抓取模式下使用大视口以渲染虚拟列表
    viewport_config = None if need_login else {
        'width': 1920,
        'height': 10800
    }

    async with async_playwright() as playwright:
        # 使用持久化上下文保存/读取登录 Cookie 及 Session
        # 配合反自动化伪装参数，绕过登录页面的自动化拦截
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            channel="chromium",
            viewport=viewport_config,
            no_viewport=need_login,
            ignore_default_args=["--enable-automation"],
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-service-autorun"
            ]
        )
        page = (
            context.pages[0]
            if context.pages
            else await context.new_page()
        )

        try:
            # 使用 domcontentloaded，避免现代页面长连接导致 networkidle 超时
            await goto_with_retry(page, url)

            if need_login:
                console.print(
                    "\n[bold green]>>> 提示：请在弹出的浏览器界面中完成登录，"
                    "完成后按【回车键 (Enter)】继续抓取... <<<[/bold green]"
                )
                input()
                console.print("[dim]登录确认成功，继续抓取对话数据...[/dim]")
                await page.wait_for_timeout(2000)

            # 按当前平台适配器的选择器等待内容挂载（短链跳转后以
            # page.url 实际域名为准）；未识别平台时内部回退联合选择器。
            await _wait_for_conversation_content(page)
            await page.wait_for_timeout(4000)

            # 需要虚拟列表处理的平台会逐屏收集；其他平台使用当前快照。
            html = await collect_virtualized_html(page)
            if html is None:
                html = await page.content()
            soup_pre = BeautifulSoup(html, "html.parser")

            # 自动提取并本地化保存网页中的所有真实图片
            # 支持 src、data-src 和 srcset。
            image_map = {}
            img_index = 1
            console.print("[dim]正在检查并下载页面中的图片资产...[/dim]")

            for img in soup_pre.find_all(["img", "source"]):
                src_candidates = [img.get("src"), img.get("data-src")]
                srcset = img.get("srcset")
                if srcset:
                    for item in srcset.split(","):
                        parts = item.strip().split()
                        if parts:
                            src_candidates.append(parts[0])

                for src in src_candidates:
                    if not (
                        src
                        and src.startswith("http")
                        and not src.startswith("data:image/svg")
                        and src not in image_map
                    ):
                        continue

                    url_hash = hashlib.md5(
                        src.encode('utf-8')
                    ).hexdigest()[:8]
                    ext = "png"
                    if ".jpg" in src.lower() or ".jpeg" in src.lower():
                        ext = "jpg"
                    elif ".webp" in src.lower():
                        ext = "webp"

                    filename = f"img_{img_index}_{url_hash}.{ext}"
                    filepath = os.path.join(images_dir, filename)

                    if not os.path.exists(filepath):
                        try:
                            image_bytes = await _fetch_image_bytes(page, src)
                            if image_bytes:
                                with open(filepath, "wb") as image_file:
                                    image_file.write(image_bytes)
                                image_map[src] = f"./images/{filename}"
                                img_index += 1
                        except Exception:
                            pass
                    else:
                        image_map[src] = f"./images/{filename}"

            return html, image_map
        except Exception as error:
            console.print(
                f"[bold red]加载页面时发生错误:[/bold red] {error}"
            )
            return None, {}
        finally:
            await context.close()


def parse_fallback_messages(soup):
    """保留原有的通用降级提取方案。"""
    console.print(
        "[dim]未能找到标志性的对话类，启用降级提取模式...[/dim]"
    )
    text_content = soup.get_text(separator='\n', strip=True)
    lines = text_content.split('\n')
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
                    'role': 'User' if is_user else 'AI',
                    'content': '\n'.join(current_block)
                })
                current_block = []
                is_user = not is_user
        current_block.append(line)

    if current_block:
        parsed_messages.append({
            'role': 'User' if is_user else 'AI',
            'content': '\n'.join(current_block)
        })

    return parsed_messages


def parse_and_display(
    html,
    image_map=None,
    summarize=False,
    gemini_model=None,
    summary_provider=None,
    include_details=False,
    summary_sections=None,
    prompt_section_selection=True
):
    if not html:
        return False

    if image_map is None:
        image_map = {}

    # 保留一份供调试的本地快照
    with open(DEBUG_HTML_FILE, "w", encoding="utf-8") as debug_file:
        debug_file.write(html)

    soup = BeautifulSoup(html, 'html.parser')
    console.print(
        "[bold green]✅ 页面加载完毕，正在解析对话内容...[/bold green]\n"
    )

    provider, parsed_messages = parse_messages(soup, image_map)
    if provider is not None:
        console.print(
            f"[dim]检测到 {provider.DISPLAY_NAME} 对话格式，"
            f"使用专用解析器...[/dim]"
        )
    else:
        parsed_messages = parse_fallback_messages(soup)

    if not parsed_messages:
        console.print(
            "[bold red]未能提取到有效对话内容。网页快照已保存到 "
            "debug_last_fetch.html，供排查。[/bold red]"
        )
        return False

    export_filename = str(DEFAULT_EXPORT_FILE)
    exported = display_and_export(
        parsed_messages,
        image_map,
        export_filename,
        console
    )
    if not exported or not summarize:
        return exported

    try:
        from .gemini_summarizer import (
            SummaryConfig,
            normalize_summary_sections,
            prompt_summary_sections,
            summarize_conversation,
        )

        console.print(
            "\n[bold cyan]开始生成多模态分层总结...[/bold cyan]"
        )
        summarize_conversation(
            messages=parsed_messages,
            project_dir=PROJECT_ROOT,
            source_dir=PROJECT_ROOT,
            source_name=os.path.basename(export_filename),
            config=SummaryConfig.from_env(
                model_override=gemini_model,
                provider_override=summary_provider
            ),
            include_details=include_details,
            selected_sections=normalize_summary_sections(summary_sections),
            section_selector=(
                prompt_summary_sections
                if prompt_section_selection and summary_sections is None
                else None
            ),
            progress=lambda message: console.print(
                message,
                style="dim"
            )
        )
        return True
    except Exception as error:
        try:
            from .gemini_summarizer import safe_error_message
            message = safe_error_message(error)
        except Exception:
            message = (
                f"{type(error).__name__}，请检查总结模型配置。"
            )
        console.print(
            f"[bold red]总结失败：[/bold red]{message}"
        )
        console.print(
            "[dim]原始对话 Markdown 已保留，不受本次失败影响。[/dim]"
        )
        return False


async def main(
    summarize=False, gemini_model=None, summary_provider=None,
    include_details=False, summary_sections=None
):
    console.print(Panel.fit(
        "[bold yellow]🚀 AI 记忆协同管理工具 - 对话提取与可视化工具 "
        "(V0.2)[/bold yellow]",
        border_style="green"
    ))
    console.print(
        "本程序使用 Playwright + 登录凭证保存技术，支持抓取带图片/附件"
        "的完整对话。\n"
    )

    console.print("[bold cyan]请选择模式:[/bold cyan]")
    console.print("  [1] 直接抓取 (推荐，自动使用已保存的登录凭证)")
    console.print("  [2] 弹出浏览器登录账号 (首次使用/需要更新登录状态时选此项)")

    mode = input("\n请选择模式 (1 或 2，默认 1): ").strip()
    need_login = mode == "2"

    url = input("请输入 AI 的对话/分享链接 (例如 https://...): ").strip()
    if not url:
        console.print("[bold red]链接不能为空，程序退出。[/bold red]")
        sys.exit(1)

    html, image_map = await fetch_chat_content(
        url,
        need_login=need_login
    )
    success = parse_and_display(
        html,
        image_map,
        summarize=summarize,
        gemini_model=gemini_model,
        summary_provider=summary_provider,
        include_details=include_details,
        summary_sections=summary_sections
    )
    if not success:
        console.print(
            "\n[bold red]处理失败，未生成有效的导出文件。[/bold red]"
        )
        sys.exit(1)

    console.print("\n[bold green]处理完成！[/bold green]")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="抓取 AI 对话并可选使用已配置模型生成分层总结。"
    )
    argument_parser.add_argument(
        "--summarize",
        action="store_true",
        help="抓取完成后调用已配置总结后端；默认不调用 API"
    )
    argument_parser.add_argument(
        "--provider",
        choices=("gemini", "siliconflow", "deepseek"),
        help="总结后端，仅在 --summarize 时使用"
    )
    argument_parser.add_argument(
        "--model",
        help="临时覆盖总结模型，仅在 --summarize 时使用"
    )
    argument_parser.add_argument(
        "--include-details",
        action="store_true",
        help="总结中附加精简的细节记忆；仅在 --summarize 时使用"
    )
    argument_parser.add_argument(
        "--expand",
        help=(
            "指定要展开的分类板块，多个键用逗号分隔；未提供时在模型完成"
            "分类后由终端选择，传 none 表示全部不选"
        )
    )
    arguments = argument_parser.parse_args()
    asyncio.run(main(
        summarize=arguments.summarize,
        gemini_model=arguments.model,
        summary_provider=arguments.provider,
        include_details=arguments.include_details,
        summary_sections=arguments.expand
    ))
