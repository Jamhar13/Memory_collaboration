"""轻量版浏览器选择与回退测试。"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from gui.service import (
    BROWSER_MODE_ENV,
    browser_channel_candidates,
    fetch_chat_pipeline,
    launch_browser_context,
)


class FakeChromium:
    def __init__(self, failures=()):
        self.failures = set(failures)
        self.calls = []

    async def launch_persistent_context(self, **kwargs):
        self.calls.append(kwargs)
        channel = kwargs["channel"]
        if channel in self.failures:
            raise RuntimeError(f"sensitive failure from {channel}")
        return object()


class FakePlaywright:
    def __init__(self, chromium):
        self.chromium = chromium


class LiteBrowserSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def _launch(self, chromium, mode):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {BROWSER_MODE_ENV: mode}):
                return await launch_browser_context(
                    FakePlaywright(chromium),
                    headless=True,
                    viewport={"width": 1280, "height": 720},
                    no_viewport=False,
                    profile_root=Path(temp_dir),
                )

    def test_full_variant_still_uses_only_bundled_chromium(self):
        with patch.dict(os.environ, {BROWSER_MODE_ENV: "full"}):
            self.assertEqual(browser_channel_candidates(), ("chromium",))

    def test_lite_variant_declares_edge_then_chrome(self):
        with patch.dict(os.environ, {BROWSER_MODE_ENV: "lite"}):
            self.assertEqual(
                browser_channel_candidates(),
                ("msedge", "chrome"),
            )

    async def test_lite_prefers_edge(self):
        chromium = FakeChromium()
        _context, channel = await self._launch(chromium, "lite")

        self.assertEqual(channel, "msedge")
        self.assertEqual(
            [call["channel"] for call in chromium.calls],
            ["msedge"],
        )
        self.assertTrue(
            chromium.calls[0]["user_data_dir"].endswith("msedge")
        )

    async def test_lite_falls_back_to_chrome(self):
        chromium = FakeChromium(failures={"msedge"})
        _context, channel = await self._launch(chromium, "lite")

        self.assertEqual(channel, "chrome")
        self.assertEqual(
            [call["channel"] for call in chromium.calls],
            ["msedge", "chrome"],
        )
        self.assertTrue(
            chromium.calls[1]["user_data_dir"].endswith("chrome")
        )

    async def test_lite_reports_full_download_without_leaking_errors(self):
        chromium = FakeChromium(failures={"msedge", "chrome"})
        with self.assertRaisesRegex(
            RuntimeError,
            "请下载安装其中一个浏览器，或下载全量版",
        ) as raised:
            await self._launch(chromium, "lite")

        self.assertNotIn("sensitive failure", str(raised.exception))

    async def test_chatgpt_private_url_respects_no_login_mode(self):
        class FakePage:
            def on(self, *_args):
                pass

            async def wait_for_timeout(self, _milliseconds):
                pass

            async def content(self):
                raise RuntimeError("stop after login-mode check")

        class FakeContext:
            def __init__(self):
                self.pages = [FakePage()]

            async def close(self):
                pass

        class FakePlaywrightManager:
            async def __aenter__(self):
                return object()

            async def __aexit__(self, *_args):
                return False

        for has_conversation_content in (True, False):
            with self.subTest(
                has_conversation_content=has_conversation_content
            ):
                contexts = [FakeContext()]
                callback = Mock()
                confirmation_callback = Mock(return_value=False)
                login_event = asyncio.Event()
                login_event.set()
                launcher = AsyncMock(side_effect=[
                    (context, "chromium") for context in contexts
                ])
                with patch(
                    "gui.service.async_playwright",
                    return_value=FakePlaywrightManager(),
                ), patch(
                    "gui.service.launch_browser_context",
                    new=launcher,
                ), patch(
                    "gui.service.goto_with_retry_gui",
                    new=AsyncMock(),
                ), patch(
                    "gui.service._drain_response_tasks",
                    new=AsyncMock(),
                ), patch(
                    "gui.service._page_has_conversation_content",
                    new=AsyncMock(return_value=has_conversation_content),
                ), patch(
                    "gui.service._set_browser_window_state",
                    new=AsyncMock(),
                ):
                    if has_conversation_content:
                        await fetch_chat_pipeline(
                            "https://chatgpt.com/c/"
                            "11111111-2222-3333-4444-555555555555",
                            need_login=False,
                            login_ready_event=login_event,
                            login_required_callback=callback,
                            login_confirmation_callback=confirmation_callback,
                        )
                    else:
                        result = await fetch_chat_pipeline(
                            "https://chatgpt.com/c/"
                            "11111111-2222-3333-4444-555555555555",
                            need_login=False,
                            login_ready_event=login_event,
                            login_required_callback=callback,
                            login_confirmation_callback=confirmation_callback,
                        )
                        self.assertIn("已取消打开登录浏览器", result.error)

                self.assertEqual(
                    [
                        call.kwargs["headless"]
                        for call in launcher.await_args_list
                    ],
                    [True],
                )
                self.assertFalse(
                    launcher.await_args_list[0].kwargs["start_minimized"]
                )
                self.assertEqual(callback.call_count, 0)
                self.assertEqual(
                    confirmation_callback.call_count,
                    0 if has_conversation_content else 1,
                )
