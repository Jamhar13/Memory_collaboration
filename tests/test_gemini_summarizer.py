import asyncio
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup

from scripts import gemini_summarizer as summary
from scripts.providers import chatgpt, deepseek, doubao


class LocalAssetPathTests(unittest.TestCase):
    def test_file_uri_is_not_prefixed_with_project_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            asset = Path(temp_dir) / "image.png"
            asset.write_bytes(b"image")
            reference = asset.as_uri()

            project_dir = Path(temp_dir) / "project"
            self.assertEqual(
                summary._resolve_local_asset(
                    reference,
                    project_dir=project_dir,
                    source_dir=Path(temp_dir),
                ),
                asset.resolve(),
            )
            self.assertEqual(
                summary._resolve_local_asset(
                    str(asset),
                    project_dir=project_dir,
                    source_dir=Path(temp_dir),
                ),
                asset.resolve(),
            )


class FakeGateway:
    def __init__(self):
        self.calls = []
        self.prompts = []

    def generate_json(self, prompt, schema, media_assets=None):
        self.calls.append((schema, media_assets or []))
        self.prompts.append((schema, prompt))
        if schema is summary.MEDIA_SCHEMA:
            return {
                "items": [
                    {
                        "media_id": asset.media_id,
                        "description": "截图显示一段测试文字。",
                        "status": "described"
                    }
                    for asset in media_assets or []
                ]
            }
        if schema is summary.CHUNK_SCHEMA:
            return {
                "title": "测试主题",
                "summary": "这是分块摘要。",
                "conversation_types": ["programming"],
                "memory_items": [{
                    "topic": "错误处理",
                    "memory_type": "assistant_suggestion",
                    "content": "上一 AI 建议增加异常捕获，用户尚未确认执行。",
                    "source": "assistant",
                    "status": "suggested",
                    "message_ids": [1, 2, 9999],
                    "evidence_quote": "建议增加 try-except"
                }],
                "learning_records": [],
                "calculation_records": [],
                "programming_records": [{
                    "topic": "测试程序",
                    "code_state": "已有原始版本",
                    "constraints": ["保持路径"],
                    "bug_or_issue": "future.result 报错",
                    "assistant_diagnosis": "上一 AI 推测为 API 非 JSON 返回",
                    "implemented_changes": [],
                    "pending_validation": ["尚未验证建议"],
                    "message_ids": [1, 2]
                }],
                "decision_records": [],
                "contextual_messages": [],
                "progressions": [],
                "source_text_issues": [{
                    "original_text": "500MB 以下，有点大了",
                    "issue_description": "阈值重复且语义矛盾，疑似笔误",
                    "inferred_correction": "第二处可能应为 500MB 以上",
                    "source": "assistant",
                    "status": "uncertain",
                    "message_ids": [2]
                }],
                "media_links": [
                    {
                        "media_id": asset.media_id,
                        "user_message_id": 1,
                        "assistant_message_ids": [2],
                        "assistant_conclusion": "上一 AI 根据图片作出测试结论",
                        "conclusion_status": "suggested"
                    }
                    for asset in media_assets or []
                ],
                "current_progress": {
                    "current_activity": "测试代码",
                    "reached_stage": "收到 AI 建议",
                    "completed_actions": [],
                    "suggested_but_unconfirmed": ["增加异常捕获"],
                    "unresolved": ["用户是否执行"],
                    "last_user_intent": "请求解决报错",
                    "message_ids": [1, 2]
                }
            }
        if schema is summary.FINAL_SCHEMA:
            return {
                "overall_summary": "用户正在测试程序，目前只收到上一 AI 的建议，尚无执行证据。",
                "conversation_types": ["programming"],
                "current_state": {
                    "current_activity": {
                        "content": "测试程序",
                        "source": "user",
                        "status": "confirmed",
                        "message_ids": [1]
                    },
                    "reached_stage": {
                        "content": "收到异常捕获建议",
                        "source": "assistant",
                        "status": "suggested",
                        "message_ids": [2]
                    },
                    "completed": [],
                    "pending": [{
                        "content": "验证建议是否解决问题",
                        "source": "assistant",
                        "status": "unresolved",
                        "message_ids": [2]
                    }],
                    "next_step": {
                        "content": "等待用户验证",
                        "source": "inferred",
                        "status": "uncertain",
                        "message_ids": [2]
                    },
                    "last_user_message_id": 9999,
                    "last_user_intent": "请求解决报错",
                    "breakpoint_status": "waiting_verification"
                },
                "topics": [{
                    "title": "程序异常处理",
                    "summary": "上一 AI 提供了尚未验证的异常捕获建议。",
                    "memory_ids": ["C001M001", "BAD"],
                    "source_message_ids": [1, 2, 9999]
                }]
            }
        raise AssertionError("unexpected schema")


class GeminiSummarizerTests(unittest.TestCase):
    def test_progress_cache_is_shared_across_fallback_models(self):
        messages = [{"role": "User", "content": "请总结这段内容"}]
        first = summary.SummaryConfig(
            provider="gemini",
            model="gemini-3.5-flash",
            chunk_chars=12000,
        )
        fallback = summary.SummaryConfig(
            provider="deepseek",
            model="deepseek-v4-pro",
            chunk_chars=12000,
        )
        different_chunks = summary.SummaryConfig(
            provider="deepseek",
            model="deepseek-v4-pro",
            chunk_chars=8000,
        )

        self.assertEqual(
            summary._summary_fingerprint(messages, first),
            summary._summary_fingerprint(messages, fallback),
        )
        self.assertNotEqual(
            summary._summary_fingerprint(messages, first),
            summary._summary_fingerprint(messages, different_chunks),
        )

    def test_fallback_model_reuses_completed_chunks(self):
        class FailingFinalGateway(FakeGateway):
            def generate_json(self, prompt, schema, media_assets=None):
                if schema is summary.FINAL_SCHEMA:
                    raise RuntimeError("temporary provider failure")
                return super().generate_json(prompt, schema, media_assets)

        messages = [
            {"role": "User", "content": "请解决报错"},
            {"role": "AI", "content": "建议增加 try-except"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            output_json = project / "result.json"
            output_markdown = project / "summary.md"
            with self.assertRaises(RuntimeError):
                summary.summarize_conversation(
                    messages=messages,
                    project_dir=project,
                    output_json=output_json,
                    output_markdown=output_markdown,
                    config=summary.SummaryConfig(
                        provider="gemini", model="gemini-3.5-flash"
                    ),
                    gateway=FailingFinalGateway(),
                    progress=lambda _message: None,
                )

            fallback_gateway = FakeGateway()
            progress_messages = []
            summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                output_json=output_json,
                output_markdown=output_markdown,
                config=summary.SummaryConfig(
                    provider="deepseek", model="deepseek-v4-pro"
                ),
                gateway=fallback_gateway,
                progress=progress_messages.append,
            )

        self.assertEqual(
            [schema for schema, _assets in fallback_gateway.calls],
            [summary.FINAL_SCHEMA],
        )
        self.assertTrue(any(
            "恢复细粒度记忆" in message for message in progress_messages
        ))

    def test_completed_result_cache_is_exact_and_invalidates_on_input_change(self):
        messages = [
            {"role": "User", "content": "请解决报错"},
            {"role": "AI", "content": "建议增加 try-except"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            cache_dir = project / "completed-cache"
            first_gateway = FakeGateway()
            first = summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                output_json=project / "first.json",
                output_markdown=project / "first.md",
                config=summary.SummaryConfig(),
                gateway=first_gateway,
                result_cache_dir=cache_dir,
                progress=lambda _message: None,
            )
            self.assertEqual(len(first_gateway.calls), 2)
            self.assertFalse(first["processing"]["cache_hit"])
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)

            second_gateway = FakeGateway()
            second = summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                source_name="另一个输出名.md",
                output_json=project / "second.json",
                output_markdown=project / "second.md",
                config=summary.SummaryConfig(),
                gateway=second_gateway,
                result_cache_dir=cache_dir,
                progress=lambda _message: None,
            )
            self.assertEqual(second_gateway.calls, [])
            self.assertTrue(second["processing"]["cache_hit"])
            for key in (
                "overall_summary",
                "current_state",
                "topics",
                "memory_items",
                "typed_records",
                "query_index",
                "recent_context",
                "media",
            ):
                self.assertEqual(second[key], first[key])
            self.assertEqual(second["source"], "另一个输出名.md")
            self.assertTrue((project / "second.json").is_file())
            self.assertTrue((project / "second.md").is_file())

            changed_gateway = FakeGateway()
            summary.summarize_conversation(
                messages=messages + [
                    {"role": "User", "content": "我已经验证，问题解决。"}
                ],
                project_dir=project,
                output_json=project / "changed.json",
                output_markdown=project / "changed.md",
                config=summary.SummaryConfig(),
                gateway=changed_gateway,
                result_cache_dir=cache_dir,
                progress=lambda _message: None,
            )
            self.assertGreater(len(changed_gateway.calls), 0)

    def test_chatgpt_snapshot_upgrade_preserves_text_and_images(self):
        existing = {"text_length": 100, "image_score": 1}
        self.assertTrue(chatgpt._prefer_snapshot(
            {"text_length": 200, "image_score": 1}, existing
        ))
        self.assertTrue(chatgpt._prefer_snapshot(
            {"text_length": 100, "image_score": 2}, existing
        ))
        self.assertFalse(chatgpt._prefer_snapshot(
            {"text_length": 90, "image_score": 2}, existing
        ))
        self.assertFalse(chatgpt._prefer_snapshot(
            {"text_length": 200, "image_score": 0}, existing
        ))
        self.assertTrue(chatgpt._prefer_snapshot(
            {"text_length": 50, "image_score": 0},
            {"text_length": 0, "image_score": 1},
        ))

    def test_detailed_outputs_use_separate_directory(self):
        project = Path("project")
        normal = summary.default_summary_paths(project, "测试.md")
        detailed = summary.default_summary_paths(
            project, "测试.md", include_details=True
        )

        self.assertEqual(normal[0].parent.name, "summary")
        self.assertEqual(detailed[0].parent.name, "summary_detailed")
        self.assertEqual(normal[0].parent.parent.name, "results")
        self.assertEqual(detailed[0].parent.parent.name, "results")
        self.assertEqual(normal[0].name, detailed[0].name)
        self.assertEqual(normal[1].name, detailed[1].name)

    def test_message_fingerprint_is_stable_and_content_sensitive(self):
        first = [
            {'role': 'User', 'content': '问题'},
            {'role': 'AI', 'content': '回答'}
        ]
        second = [dict(item) for item in first]
        self.assertEqual(
            summary.messages_fingerprint(first),
            summary.messages_fingerprint(second)
        )
        second[1]['content'] = '不同回答'
        self.assertNotEqual(
            summary.messages_fingerprint(first),
            summary.messages_fingerprint(second)
        )

    def test_default_model_is_non_lite_gemini(self):
        with patch.dict(os.environ, {}, clear=True):
            config = summary.SummaryConfig.from_env()
        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, "gemini-3.5-flash")

    def test_chunking_keeps_user_and_following_ai_in_same_chunk(self):
        messages = [
            {"role": "User", "content": "A" * 30},
            {"role": "AI", "content": "B" * 30},
            {"role": "User", "content": "无饮水机，无电风扇"},
            {"role": "AI", "content": "新条件下约 50-100 度"}
        ]
        chunks = summary.chunk_messages(messages, max_chars=100)
        containing_user = next(
            chunk for chunk in chunks if 3 in chunk.message_indices
        )
        self.assertIn(4, containing_user.message_indices)

    def test_deepseek_dense_conversation_uses_smaller_chunks_only_there(self):
        messages = [
            {
                "role": "User" if index % 2 == 0 else "AI",
                "content": str(index) + "X" * 700,
            }
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            regular = summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                output_json=project / "regular.json",
                output_markdown=project / "regular.md",
                config=summary.SummaryConfig(),
                gateway=FakeGateway(),
                progress=lambda _message: None,
            )
            deepseek_result = summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                output_json=project / "deepseek.json",
                output_markdown=project / "deepseek.md",
                config=summary.SummaryConfig(),
                gateway=FakeGateway(),
                progress=lambda _message: None,
                source_platform="deepseek",
            )

        self.assertEqual(regular["conversation"]["chunk_count"], 1)
        self.assertGreater(deepseek_result["conversation"]["chunk_count"], 1)

    def test_model_and_chunk_size_are_configurable(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_MODEL": "future-model",
                "GEMINI_CHUNK_CHARS": "12345"
            },
            clear=False
        ):
            config = summary.SummaryConfig.from_env()
        self.assertEqual(config.model, "future-model")
        self.assertEqual(config.chunk_chars, 12345)

    def test_thinking_level_is_configurable(self):
        with patch.dict(
            os.environ, {"GEMINI_THINKING_LEVEL": "high"}, clear=True
        ):
            config = summary.SummaryConfig.from_env()
        self.assertEqual(config.thinking_level, "high")

    def test_siliconflow_provider_is_inferred_from_qwen_model(self):
        with patch.dict(
            os.environ,
            {"SUMMARY_MODEL": "Qwen/Qwen3.5-397B-A17B"},
            clear=True
        ):
            config = summary.SummaryConfig.from_env()
        self.assertEqual(config.provider, "siliconflow")
        self.assertEqual(config.model, "Qwen/Qwen3.5-397B-A17B")

    def test_deepseek_provider_is_inferred_from_deepseek_model(self):
        with patch.dict(
            os.environ,
            {"SUMMARY_MODEL": "deepseek-v4-pro"},
            clear=True,
        ):
            config = summary.SummaryConfig.from_env()
        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-v4-pro")

    def test_safe_error_message_redacts_siliconflow_key(self):
        secret = "sk-test-silicon-key-that-must-not-leak"
        with patch.dict(os.environ, {"Silicon_API_KEY": secret}, clear=True):
            message = summary.safe_error_message(
                summary.GeminiSummaryError(f"请求失败：{secret}")
            )
        self.assertNotIn(secret, message)
        self.assertIn("<redacted>", message)

    def test_siliconflow_gateway_uses_json_chat_completions(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}]
                }).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        config = summary.SummaryConfig(
            provider="siliconflow",
            model="Qwen/Qwen3.5-397B-A17B",
            retries=1,
            request_timeout_seconds=37,
        )
        with patch.dict(os.environ, {"Silicon_API_KEY": "test-key"}, clear=True):
            with patch.object(summary, "urlopen", side_effect=fake_urlopen):
                gateway = summary.SiliconFlowGateway(config)
                result = gateway.generate_json("测试", {"type": "object"})
        self.assertEqual(result, {"ok": True})
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["payload"]["model"], config.model)
        self.assertEqual(captured["timeout"], 37)
        response_format = captured["payload"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            response_format["json_schema"]["schema"], {"type": "object"}
        )
        self.assertTrue(captured["payload"]["enable_thinking"])
        self.assertGreaterEqual(captured["payload"]["thinking_budget"], 128)

    def test_deepseek_gateway_uses_user_key_and_thinking_payload(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "{\"ok\": true}"}}]
                }).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        config = summary.SummaryConfig(
            provider="deepseek",
            model=summary.DEEPSEEK_DEFAULT_MODEL,
            thinking_level="high",
            retries=1,
        )
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "developer-environment-key"},
            clear=True,
        ), patch.object(summary, "urlopen", side_effect=fake_urlopen):
            gateway = summary.create_gateway(
                config, api_key="user-deepseek-key"
            )
            result = gateway.generate_json("测试 JSON", {"type": "object"})

        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            captured["url"],
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(
            captured["authorization"], "Bearer user-deepseek-key"
        )
        self.assertEqual(captured["payload"]["model"], "deepseek-v4-pro")
        self.assertEqual(
            captured["payload"]["response_format"], {"type": "json_object"}
        )
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")
        self.assertEqual(
            captured["payload"]["thinking"], {"type": "enabled"}
        )

    def test_gemini_client_receives_millisecond_request_timeout(self):
        captured = {}
        from google import genai

        def fake_client(**kwargs):
            captured.update(kwargs)
            return object()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            with patch.object(genai, "Client", side_effect=fake_client):
                summary.GeminiGateway(summary.SummaryConfig(
                    request_timeout_seconds=37
                ))

        self.assertEqual(captured["http_options"].timeout, 37_000)

    def test_gateway_timeout_has_specific_recoverable_error(self):
        class ReadTimeout(Exception):
            pass

        class FakeModels:
            def generate_content(self, **_kwargs):
                raise ReadTimeout("request timed out")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            gateway = summary.GeminiGateway(summary.SummaryConfig(
                retries=1,
                request_timeout_seconds=12,
            ))
        gateway._client = type("Client", (), {"models": FakeModels()})()
        with self.assertRaises(summary.SummaryRequestTimeoutError) as raised:
            gateway.generate_json("test", {"type": "object"})
        self.assertIn("12 秒", str(raised.exception))

    def test_rate_limit_retry_uses_long_wait(self):
        class RateLimitError(Exception):
            code = 429
            status = "RESOURCE_EXHAUSTED"

        class FakeModels:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RateLimitError()
                return type("Response", (), {"text": "{}"})()

        waits = []
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            gateway = summary.GeminiGateway(
                summary.SummaryConfig(
                    retries=2,
                    rate_limit_wait_seconds=65
                ),
                sleep=waits.append
            )
        gateway._client = type("Client", (), {"models": FakeModels()})()
        result = gateway.generate_json("test", {"type": "object"})

        self.assertEqual(result, {})
        self.assertEqual(waits, [65])

    def test_safe_error_message_redacts_api_key(self):
        fake_key = "AIza" + "A" * 30
        with patch.dict(os.environ, {"GEMINI_API_KEY": fake_key}, clear=False):
            message = summary.safe_error_message(
                summary.GeminiSummaryError("request failed with " + fake_key)
            )
        self.assertNotIn(fake_key, message)
        self.assertIn("<redacted>", message)

    def test_media_is_resolved_enriched_and_marked_reverifiable(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            images = project / "images"
            images.mkdir()
            (images / "sample.png").write_bytes(b"not-a-real-png")
            config = summary.SummaryConfig()
            messages = [{
                "role": "User",
                "content": "请看 ![截图](../images/sample.png)"
            }]
            assets = summary.discover_media(messages, project, project, config)
            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0].status, "ready")
            gateway = FakeGateway()
            warnings = summary.describe_media(
                assets, gateway, config, lambda _message: None
            )
            enriched = summary.enrich_messages(messages, assets)
            public = assets[0].public_dict()

        self.assertEqual(warnings, [])
        self.assertIn("截图显示一段测试文字", enriched[0]["content"])
        self.assertIn(
            "图片要描述主要对象、可读文字、数据/界面和与对话可能有关的信息；"
            "PDF 要概括主题、关键事实和结论。看不清时 status 使用 unclear。",
            gateway.prompts[0][1],
        )
        self.assertNotIn("extracted_document_text", gateway.prompts[0][1])
        self.assertIn("- M001：用户上传了一张图片", enriched[0]["content"])
        self.assertNotIn("｜媒体说明", enriched[0]["content"])
        self.assertEqual(public["access_status"], "available_local")
        self.assertTrue(public["can_reverify"])
        self.assertEqual(public["source_role"], "user")

    def test_media_api_failure_keeps_local_file_reverifiable_and_retryable(self):
        class FailingGateway:
            def generate_json(self, prompt, schema, media_assets=None):
                raise summary.GeminiSummaryError("temporary media outage")

        with tempfile.TemporaryDirectory() as temp:
            local_path = Path(temp) / "sample.png"
            local_path.write_bytes(b"image")
            asset = summary.MediaAsset(
                media_id="M001",
                message_index=1,
                kind="image",
                label="截图",
                reference="./sample.png",
                local_path=local_path,
                mime_type="image/png",
                status="ready",
            )
            warnings = summary.describe_media(
                [asset],
                FailingGateway(),
                summary.SummaryConfig(),
                lambda _message: None,
            )
            public = asset.public_dict()

            retry_asset = summary.MediaAsset(
                media_id="M001",
                message_index=1,
                kind="image",
                label="截图",
                reference="./sample.png",
                local_path=local_path,
                mime_type="image/png",
                status="ready",
            )
            summary._apply_cached_media(
                [retry_asset],
                [{
                    "media_id": "M001",
                    "reference": "./sample.png",
                    "status": "analysis_failed",
                    "description": asset.description,
                }],
            )

        self.assertEqual(len(warnings), 1)
        self.assertEqual(asset.status, "analysis_failed")
        self.assertEqual(public["access_status"], "available_local")
        self.assertTrue(public["can_reverify"])
        self.assertIn("本地文件仍可访问", asset.description)
        self.assertEqual(retry_asset.status, "ready")

    def test_assistant_citation_icons_are_ignored(self):
        messages = [
            {
                "role": "User",
                "content": "请看 ![截图](./images/sample.png)",
            },
            {
                "role": "AI",
                "content": (
                    "参考资料：[![](https://www.google.com/s2/favicons?"
                    "domain=https://www.reddit.com&sz=128)Reddit]"
                    "(https://www.reddit.com/example)\n"
                    "以及 [![](./images/site-logo.png)Hugging Face]"
                    "(https://huggingface.co/example)"
                ),
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            images = project / "images"
            images.mkdir()
            (images / "sample.png").write_bytes(b"sample")
            (images / "site-logo.png").write_bytes(b"logo")
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
            self.assertTrue(assets[0].public_dict()["can_reverify"])

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].message_index, 1)
        self.assertEqual(assets[0].source_role, "user")

    def test_meaningful_assistant_image_keeps_its_source_role(self):
        messages = [{
            "role": "AI",
            "content": "这是结果图：![部署架构图](./images/diagram.png)",
        }]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            images = project / "images"
            images.mkdir()
            (images / "diagram.png").write_bytes(b"diagram")
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
            summary.describe_media(
                assets,
                FakeGateway(),
                summary.SummaryConfig(),
                lambda _message: None,
            )

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].source_role, "assistant")
        self.assertTrue(
            assets[0].description.startswith("AI 回答中包含一张图片")
        )

    def test_assistant_media_attribution_is_corrected_across_generated_fields(self):
        asset = summary.MediaAsset(
            media_id="M001",
            message_index=6,
            kind="image",
            label="image",
            reference="./images/portrait.jpg",
            source_role="assistant",
            status="described",
        )
        result = {
            "overall_summary": "AI结合用户上传的图片回答了问题。",
            "current_state": {
                "reached_stage": {
                    "content": "AI结合用户上传的图片给出介绍。",
                    "message_ids": [6],
                },
                "completed": [{
                    "content": "分析并识别了用户上传的免冠证件照。",
                    "message_ids": [6],
                }],
            },
            "topics": [{
                "title": "人物识别",
                "summary": "根据用户提供的两张照片介绍人物。",
                "source_message_ids": [5, 6],
            }],
            "memory_items": [{
                "content": "用户发来的肖像照展示一名男性。",
                "evidence_quote": "用户上传的图片",
                "message_ids": [6],
            }],
            "typed_records": {
                "context_references": [{
                    "resolved_reference": "用户展示的截图",
                    "message_ids": [6],
                }],
            },
            "media": [asset.public_dict()],
            "recent_context": [{
                "message_id": 6,
                "content": "用户上传的图片",
            }],
        }
        summary._correct_media_role_attribution(result, [asset])

        generated_text = json.dumps({
            key: result[key] for key in (
                "overall_summary", "current_state", "topics",
                "memory_items", "typed_records",
            )
        }, ensure_ascii=False)
        self.assertNotIn("用户上传", generated_text.replace(
            '"evidence_quote": "用户上传的图片"', ""
        ))
        self.assertIn("AI 回答中提供的免冠证件照", generated_text)
        self.assertIn("AI 结合回答中提供的图片", generated_text)
        self.assertEqual(
            result["memory_items"][0]["evidence_quote"],
            "用户上传的图片",
        )
        self.assertEqual(
            result["recent_context"][0]["content"],
            "用户上传的图片",
        )

    def test_user_or_mixed_media_attribution_is_not_rewritten(self):
        user_asset = summary.MediaAsset(
            media_id="M001",
            message_index=1,
            kind="image",
            label="用户图片",
            reference="./images/user.jpg",
            source_role="user",
        )
        assistant_asset = summary.MediaAsset(
            media_id="M002",
            message_index=2,
            kind="image",
            label="AI 图片",
            reference="./images/ai.jpg",
            source_role="assistant",
        )
        result = {
            "overall_summary": "用户上传的图片与 AI 图片被共同讨论。",
            "current_state": {
                "completed": [{
                    "content": "分析了用户上传的图片。",
                    "message_ids": [1],
                }, {
                    "content": "综合了用户上传的图片与 AI 图片。",
                    "message_ids": [1, 2],
                }],
            },
        }
        summary._correct_media_role_attribution(
            result, [user_asset, assistant_asset]
        )
        self.assertEqual(
            result["current_state"]["completed"][0]["content"],
            "分析了用户上传的图片。",
        )
        self.assertEqual(
            result["current_state"]["completed"][1]["content"],
            "综合了用户上传的图片与 AI 图片。",
        )
        self.assertEqual(
            result["overall_summary"],
            "用户上传的图片与 AI 图片被共同讨论。",
        )

    def test_downloaded_docx_text_is_extracted_safely(self):
        class DocumentGateway:
            def __init__(self):
                self.prompt = ""
                self.media_assets = []

            def generate_json(self, prompt, schema, media_assets=None):
                self.prompt = prompt
                self.media_assets = list(media_assets or [])
                return {"items": [{
                    "media_id": "M001",
                    "description": "这份报告包含两段正文，并提供总结所需信息。",
                    "status": "described",
                }]}

        document_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>附件中的第一段正文</w:t></w:r></w:p>
            <w:p><w:r><w:t>第二段包含总结所需信息</w:t></w:r></w:p>
          </w:body>
        </w:document>"""
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            document = project / "report.docx"
            with zipfile.ZipFile(document, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
            messages = [{
                "role": "User",
                "content": "[report.docx](./report.docx)"
            }]
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
            self.assertEqual(assets[0].status, "ready")
            self.assertIn("附件中的第一段正文", assets[0].extracted_text)
            self.assertIn("第二段包含总结所需信息", assets[0].extracted_text)
            gateway = DocumentGateway()
            warnings = summary.describe_media(
                assets,
                gateway,
                summary.SummaryConfig(),
                lambda _message: None,
            )
            self.assertTrue(assets[0].public_dict()["can_reverify"])

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].status, "described")
        self.assertEqual(warnings, [])
        self.assertIn("主要内容为", assets[0].description)
        self.assertIn("两段正文", assets[0].description)
        self.assertNotIn("附件中的第一段正文", assets[0].description)
        self.assertIn("附件中的第一段正文", gateway.prompt)
        self.assertEqual(gateway.media_assets, [])
        self.assertNotIn("extracted_text", assets[0].public_dict())

    def test_text_document_is_summarized_before_conversation_chunks(self):
        class PipelineGateway(FakeGateway):
            def __init__(self):
                super().__init__()
                self.captured_prompts = []

            def generate_json(self, prompt, schema, media_assets=None):
                self.captured_prompts.append(
                    (schema, prompt, list(media_assets or []))
                )
                if schema is summary.MEDIA_SCHEMA:
                    return {"items": [{
                        "media_id": "M001",
                        "description": (
                            "这是一份旧对话记录，涉及英语词汇和部署案例；"
                            "它在本次对话中仅作为待审查附件。"
                        ),
                        "status": "described",
                    }]}
                return super().generate_json(prompt, schema, media_assets)

        raw_only_marker = "ATTACHMENT_RAW_ONLY_MARKER"
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            document = project / "历史记录.md"
            document.write_text(
                "# 英语词汇\n\n" + raw_only_marker + "\n\n# 部署任务",
                encoding="utf-8",
            )
            messages = [{
                "role": "User",
                "content": "[历史记录.md](./历史记录.md)\n请比较这份文件的总结质量",
            }, {
                "role": "AI",
                "content": "已比较文件结构和状态标注。",
            }]
            gateway = PipelineGateway()
            summary.summarize_conversation(
                messages,
                project_dir=project,
                source_dir=project,
                source_name="test.md",
                output_json=project / "result.json",
                output_markdown=project / "result.md",
                config=summary.SummaryConfig(),
                gateway=gateway,
                progress=lambda _message: None,
            )

        media_prompts = [
            prompt for schema, prompt, _assets in gateway.captured_prompts
            if schema is summary.MEDIA_SCHEMA
        ]
        chunk_prompts = [
            prompt for schema, prompt, _assets in gateway.captured_prompts
            if schema is summary.CHUNK_SCHEMA
        ]
        final_prompts = [
            prompt for schema, prompt, _assets in gateway.captured_prompts
            if schema is summary.FINAL_SCHEMA
        ]
        self.assertEqual(len(media_prompts), 1)
        self.assertIn(raw_only_marker, media_prompts[0])
        self.assertTrue(chunk_prompts)
        self.assertTrue(final_prompts)
        self.assertTrue(all(raw_only_marker not in prompt for prompt in chunk_prompts))
        self.assertTrue(all(raw_only_marker not in prompt for prompt in final_prompts))
        self.assertIn("附件上下文（不是独立对话主题）", chunk_prompts[0])
        self.assertIn("不得把文档内部", chunk_prompts[0])
        self.assertIn("AI 对附件内部子主题的逐项复述", chunk_prompts[0])
        self.assertIn("文档摘要中的内部章节", final_prompts[0])
        self.assertIn("上一 AI 在识别、概括、总结或审查附件时", final_prompts[0])

    def test_attachment_content_cannot_create_language_topic(self):
        memories = [
            {
                "memory_id": "M1", "topic": "文档识别：mddd.md 与 豆包豆包.md",
                "content": "用户上传的文档记录了庄园英文翻译和其他历史问答。",
                "memory_type": "user_condition", "source": "user",
                "message_ids": [1],
            },
            {
                "memory_id": "M2", "topic": "文档识别：mddd.md 与 豆包豆包.md",
                "content": "AI 对两份文档进行了识别和结构化概括。",
                "memory_type": "verification", "source": "assistant",
                "message_ids": [2],
            },
            {
                "memory_id": "M3", "topic": "军训衣服丢失处理",
                "content": "用户询问军训衣服丢了怎么办。",
                "memory_type": "user_condition", "source": "user",
                "message_ids": [3],
            },
            {
                "memory_id": "M4", "topic": "军训衣服丢失处理",
                "content": "AI 给出了寻找和补购军训服的建议。",
                "memory_type": "assistant_suggestion", "source": "assistant",
                "message_ids": [4],
            },
        ]
        topics = [
            {
                "title": "英语词汇、翻译与表达学习",
                "memory_ids": ["M1", "M3"],
                "source_message_ids": [1, 2, 3, 4],
            },
            {
                "title": "文档识别与概括（mddd.md与豆包豆包.md）",
                "memory_ids": ["M2"], "source_message_ids": [1, 2],
            },
            {
                "title": "军训衣服丢失应对建议",
                "memory_ids": ["M4"], "source_message_ids": [3, 4],
            },
        ]

        normalized = summary._normalize_topic_assignments(
            topics, memories, message_count=4
        )

        self.assertFalse(any(
            topic["title"] == "英语词汇、翻译与表达学习"
            for topic in normalized
        ))
        by_title = {topic["title"]: topic for topic in normalized}
        self.assertEqual(
            set(by_title[
                "文档识别与概括（mddd.md与豆包豆包.md）"
            ]["memory_ids"]),
            {"M1", "M2"},
        )
        self.assertEqual(
            set(by_title["军训衣服丢失应对建议"]["memory_ids"]),
            {"M3", "M4"},
        )

    def test_platform_parsers_use_downloaded_document_links(self):
        local_reference = "./result_files/report.docx"
        asset_map = {
            "/download/report": local_reference,
            "report.docx": local_reference,
        }
        chatgpt_html = """
        <div data-testid="conversation-turn-1">
          <div data-message-author-role="user">请总结附件</div>
          <a href="/download/report">report.docx</a>
        </div>
        """
        chatgpt_messages = chatgpt.parse_messages(
            BeautifulSoup(chatgpt_html, "html.parser"), asset_map
        )
        self.assertIn(local_reference, chatgpt_messages[0]["content"])

        deepseek_html = """
        <div data-virtual-list-item-key="1">
          <div class="ds-message">
            <div>report.docx</div><div>DOCX 1 KB</div>
          </div>
        </div>
        """
        deepseek_messages = deepseek.parse_messages(
            BeautifulSoup(deepseek_html, "html.parser"), asset_map
        )
        self.assertIn(local_reference, deepseek_messages[0]["content"])

        doubao_html = """
        <div class="message-item justify-end">
          <div class="doc-card">report.docx</div>
          请总结附件
        </div>
        """
        doubao_messages = doubao.parse_messages(
            BeautifulSoup(doubao_html, "html.parser"), asset_map
        )
        self.assertIn(local_reference, doubao_messages[0]["content"])
    def test_missing_document_gets_explicit_description(self):
        marker = chr(96)
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            messages = [{
                "role": "User",
                "content": "📎 **[上传文档]** " + marker + "missing.docx" + marker
            }]
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].status, "unavailable")
        self.assertIn("当前导出结果无法访问文档原件", assets[0].description)
        self.assertIn("不代表历史对话中的 AI 当时未读取", assets[0].description)
        self.assertFalse(assets[0].public_dict()["can_reverify"])

    def test_message_range_is_deterministic(self):
        # message_range 由程序根据消息编号确定性生成。
        self.assertEqual(
            summary._message_range([1, 2, 3, 5, 7, 8]),
            '1~3, 5, 7, 8'
        )
        self.assertEqual(
            summary._message_range([
                *range(1, 11), 17, 18, *range(29, 87),
                *range(89, 187), *range(213, 229), *range(231, 237)
            ]),
            '1~10, 17, 18, 29~86, 89~186, 213~228, 231~236'
        )

    def test_chatgpt_nested_role_nodes_are_not_duplicated(self):
        html = """
        <section data-message-author-role="user">
          <div data-message-author-role="user">生成一张图片给我</div>
        </section>
        """
        self.assertEqual(chatgpt.parse_messages(
            BeautifulSoup(html, "html.parser"),
        ), [{"role": "User", "content": "生成一张图片给我"}])

    def test_chatgpt_generated_image_only_answer_is_preserved(self):
        source = "https://chatgpt.com/backend-api/estuary/content?id=file_image"
        html = f"""
        <section data-message-author-role="assistant">
          <img src="{source}" alt="已生成图片：学习桌">
          <img src="{source}" alt="已生成图片：学习桌">
          <img src="{source}" alt="已生成图片：学习桌">
        </section>
        """
        messages = chatgpt.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {source: "./images/generated.png"},
        )
        self.assertEqual(messages, [{
            "role": "AI",
            "content": "![已生成图片：学习桌](./images/generated.png)",
        }])

    def test_chatgpt_math_nodes_restore_latex_before_markdown_conversion(self):
        html = r"""
        <div data-message-author-role="assistant">
          <p>行内公式
            <span role="math" data-math-source="n\ge3">
              <span class="katex">n ≥ 3</span>
            </span>
          </p>
          <span role="math" data-math-source="\boxed{m-n+p}"
                style="display: block;">
            <span class="katex-display"><span class="katex">m−n+p</span></span>
          </span>
          <span class="katex">
            <span class="katex-mathml"><math><semantics>
              <annotation encoding="application/x-tex">\binom{6}{3}</annotation>
            </semantics></math></span>
          </span>
        </div>
        """
        messages = chatgpt.parse_messages(
            BeautifulSoup(html, "html.parser"), {}
        )
        content = messages[0]["content"]
        self.assertIn(r"$n\ge3$", content)
        self.assertIn("$$\n\\boxed{m-n+p}\n$$", content)
        self.assertIn(r"$\binom{6}{3}$", content)
        self.assertNotIn("m−n+p", content)

    def test_upload_placeholder_is_preserved_as_unavailable_document(self):
        messages = [{
            'role': 'User',
            'content': '  上传文件  ' + chr(10) + '请总结这个文档'
        }]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].kind, 'document')
        self.assertEqual(assets[0].status, 'unavailable')
        self.assertIn('只保留了“上传文件”占位', assets[0].description)

    def test_bare_document_filename_in_user_message_is_preserved(self):
        messages = [{
            'role': 'User',
            'content': '附件1-复旦大学“思源计划”第十四期申请表.docx'
        }]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].kind, 'document')
        self.assertEqual(assets[0].message_index, 1)
        self.assertIn('当前导出结果无法访问文档原件', assets[0].description)

        ai_only = [{'role': 'AI', 'content': '请打开 example.docx 查看'}]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            self.assertEqual(
                summary.discover_media(
                    ai_only, project, project, summary.SummaryConfig()
                ),
                []
            )

    def test_document_link_and_card_are_deduplicated(self):
        marker = chr(96)
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            messages = [{
                "role": "User",
                "content": (
                    "[report.pdf](https://example.com/download/report.pdf)"
                    "\n📎 **[上传文档]** " + marker + "report.pdf" + marker
                )
            }]
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].label, "report.pdf")

    def test_deepseek_assistant_web_citation_is_not_a_document(self):
        messages = [{
            "role": "AI",
            "content": (
                "[-1](https://example.com/reference/article.html)\n"
                "[report.pdf](https://example.com/files/report.pdf)"
            ),
        }]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            regular = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
            deepseek_assets = summary.discover_media(
                messages,
                project,
                project,
                summary.SummaryConfig(),
                source_platform="deepseek",
            )

        self.assertEqual([asset.label for asset in regular], ["-1", "report.pdf"])
        self.assertEqual([asset.label for asset in deepseek_assets], ["report.pdf"])

    def test_doubao_collector_scrolls_every_message_for_lazy_images(self):
        class FakeMessage:
            def __init__(self, page, index):
                self.page = page
                self.index = index

            async def scroll_into_view_if_needed(self, timeout):
                self.page.scrolled.append((self.index, timeout))

        class FakeMessages:
            def __init__(self, page):
                self.page = page

            async def count(self):
                return self.page.message_count

            def nth(self, index):
                return FakeMessage(self.page, index)

        class FakePage:
            def __init__(self, message_count):
                self.message_count = message_count
                self.scrolled = []
                self.waits = []

            def locator(self, selector):
                self.selector = selector
                return FakeMessages(self)

            async def wait_for_timeout(self, milliseconds):
                self.waits.append(milliseconds)

            async def content(self):
                return "<div class='message-item'>完整页面</div>"

        page = FakePage(4)
        html = asyncio.run(doubao.collect_html(page))
        self.assertEqual(page.selector, doubao.SHARE_MESSAGE_SELECTOR)
        self.assertEqual(page.scrolled, [
            (0, 3000), (1, 3000), (2, 3000), (3, 3000)
        ])
        self.assertEqual(page.waits, [120, 120, 120, 120, 700])
        self.assertIn("完整页面", html)

        unrelated_page = FakePage(0)
        self.assertIsNone(asyncio.run(doubao.collect_html(unrelated_page)))
        self.assertEqual(unrelated_page.scrolled, [])
        self.assertEqual(unrelated_page.waits, [])

    def test_direct_doubao_collector_merges_history_loaded_while_scrolling(self):
        def fragment(number):
            return {
                "key": str(number),
                "html": (
                    '<div class="message-item" data-doubao-role="assistant">'
                    f'消息{number}</div>'
                ),
            }

        class EmptyShareMessages:
            async def count(self):
                return 0

        class FirstMessage:
            def __init__(self, page):
                self.page = page

            async def evaluate(self, script):
                self.page.round += 1

        class VisibleMessage:
            async def scroll_into_view_if_needed(self, timeout):
                pass

        class DirectMessages:
            def __init__(self, page):
                self.page = page
                self.first = FirstMessage(page)

            async def count(self):
                return len(self.page.snapshots[min(
                    self.page.round, len(self.page.snapshots) - 1
                )])

            def nth(self, index):
                return VisibleMessage()

            async def evaluate_all(self, script):
                return self.page.snapshots[min(
                    self.page.round, len(self.page.snapshots) - 1
                )]

        class SelectedList:
            def __init__(self, page):
                self.page = page

            def locator(self, selector):
                self.page.direct_selector = selector
                return DirectMessages(self.page)

        class Lists:
            def __init__(self, page):
                self.page = page

            async def evaluate_all(self, script):
                return [{"index": 0, "count": 2, "textLength": 10}]

            def nth(self, index):
                return SelectedList(self.page)

        class FakePage:
            def __init__(self):
                self.round = 0
                self.waits = []
                self.direct_selector = None
                self.snapshots = [
                    [fragment(3), fragment(4)],
                    [fragment(1), fragment(2), fragment(3), fragment(4)],
                    [fragment(1), fragment(2), fragment(3), fragment(4)],
                    [fragment(1), fragment(2), fragment(3), fragment(4)],
                ]

            def locator(self, selector):
                if selector == doubao.SHARE_MESSAGE_SELECTOR:
                    return EmptyShareMessages()
                return Lists(self)

            async def wait_for_timeout(self, milliseconds):
                self.waits.append(milliseconds)

        page = FakePage()
        html = asyncio.run(doubao.collect_html(page))

        self.assertEqual(page.direct_selector, doubao.DIRECT_MESSAGE_SELECTOR)
        self.assertEqual(page.round, 3)
        self.assertEqual(html.count('class="message-item"'), 4)
        self.assertLess(html.index("消息1"), html.index("消息4"))

    def test_direct_doubao_collector_captures_generated_iframe_charts(self):
        png = b"\x89PNG\r\n\x1a\nchart"

        class Preview:
            async def screenshot(self, type):
                return png

        class Previews:
            def __init__(self, count):
                self._count = count

            async def count(self):
                return self._count

            def nth(self, index):
                return Preview()

        class Message:
            def __init__(self, has_preview=False):
                self.has_preview = has_preview

            async def scroll_into_view_if_needed(self, timeout):
                pass

            async def evaluate(self, script):
                pass

            def locator(self, selector):
                return Previews(int(self.has_preview))

        class Messages:
            first = Message()

            async def count(self):
                return 2

            def nth(self, index):
                return Message(has_preview=index == 1)

            async def evaluate_all(self, script):
                return [{
                    "index": 1,
                    "key": "1",
                    "html": (
                        '<div class="message-item" data-doubao-role="assistant">'
                        '<iframe src="https://marscode-static.doubaocdn.com/'
                        'html_preview/?channelId=chart"></iframe></div>'
                    ),
                }]

        class SelectedList:
            def locator(self, selector):
                return Messages()

        class Lists:
            async def evaluate_all(self, script):
                return [{"index": 0, "count": 1, "textLength": 0}]

            def nth(self, index):
                return SelectedList()

        class EmptyShareMessages:
            async def count(self):
                return 0

        class Page:
            def locator(self, selector):
                return EmptyShareMessages() if selector == doubao.SHARE_MESSAGE_SELECTOR else Lists()

            async def wait_for_timeout(self, milliseconds):
                pass

        html = asyncio.run(doubao.collect_html(Page()))
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("生成图表", html)
        self.assertNotIn("<iframe", html)

    def test_direct_doubao_synthetic_roles_do_not_depend_on_css_role_class(self):
        html = """
        <div class="message-item" data-doubao-role="user">用户问题</div>
        <div class="message-item" data-doubao-role="assistant">豆包回答</div>
        """
        messages = doubao.parse_messages(
            BeautifulSoup(html, "html.parser"), {}
        )
        self.assertEqual(
            [message["role"] for message in messages],
            ["User", "AI"],
        )
        self.assertIn("用户问题", messages[0]["content"])
        self.assertIn("豆包回答", messages[1]["content"])

    def test_deepseek_markdown_attachment_uses_downloaded_local_file(self):
        html = """
        <div data-virtual-list-item-key="1">
          <div class="ds-message">
            <div>课堂资料.md</div><div>MD 19.49KB</div>
            <div>识别文档</div>
          </div>
        </div>
        """
        messages = deepseek.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {"课堂资料.md": "./files/%E8%AF%BE%E5%A0%82.md"},
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("./files/%E8%AF%BE%E5%A0%82.md", messages[0]["content"])
        self.assertIn("识别文档", messages[0]["content"])

    def test_chatgpt_code_filenames_are_not_guessed_as_attachments(self):
        html = """
        <div data-testid="conversation-turn-0">
          <div data-message-author-role="user" data-message-id="m0">
            <pre>from reportlab.pdfgen import canvas
sign = hashlib.md5(data).hexdigest()
output = "Harry Potter_translated.pdf"</pre>
          </div>
        </div>
        """
        messages = chatgpt.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {},
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("from reportlab.pdfgen", messages[0]["content"])
        self.assertIn("sign = hashlib.md5", messages[0]["content"])
        self.assertIn("Harry Potter_translated.pdf", messages[0]["content"])
        self.assertNotIn("[上传文档]", messages[0]["content"])

    def test_chatgpt_parenthesized_file_card_uses_downloaded_local_file(self):
        html = """
        <div data-testid="conversation-turn-0">
          <div data-message-author-role="user" data-message-id="m0">
            <div class="truncate font-semibold">fuckk(1).md</div>
            <div>文件</div>
            <div>忘记 haiku，重新比较</div>
          </div>
        </div>
        """
        local_reference = "./files/fuckk%281%29.md"
        messages = chatgpt.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {"fuckk(1).md": local_reference},
        )
        self.assertEqual(len(messages), 1)
        self.assertIn(
            f"[📄 fuckk(1).md]({local_reference})",
            messages[0]["content"],
        )
        self.assertIn("忘记 haiku，重新比较", messages[0]["content"])

    def test_doubao_document_covers_are_not_user_images(self):
        html = """
        <div class="message-item justify-end">
          <div class="doc-card">
            <img alt="Asset cover" src="data:image/png;base64,AAAA" />
            <img alt="Asset cover" src="//cdn/doc-canvas-card-fallback-light.png" />
            申请表.docx
          </div>
          请分析这个文档
        </div>
        """
        messages = doubao.parse_messages(BeautifulSoup(html, "html.parser"), {})
        self.assertEqual(len(messages), 1)
        self.assertNotIn("![用户图片]", messages[0]["content"])
        self.assertIn("申请表.docx", messages[0]["content"])
        self.assertIn("📎 **[上传文档]**", messages[0]["content"])
        self.assertEqual(messages[0]["content"].count("申请表.docx"), 1)

        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            legacy = [{
                "role": "User",
                "content": (
                    "![Asset cover](data:image/png;base64,AAAA)\n"
                    "![Asset cover](//cdn/doc-canvas-card-fallback-light.png)"
                )
            }]
            assets = summary.discover_media(
                legacy, project, project, summary.SummaryConfig()
            )
        self.assertEqual(assets, [])

    def test_doubao_ai_document_card_links_to_saved_markdown(self):
        html = """
        <div class="message-item" data-doubao-role="assistant">
          <p>回答正文</p>
          <div class="product-card-real123">
            <div class="card-content-info-title-text-real123">在线报告</div>
            <div>创建时间：07-08 10:06</div>
          </div>
        </div>
        """
        local_reference = "./result_files/%E5%9C%A8%E7%BA%BF%E6%8A%A5%E5%91%8A.md"
        messages = doubao.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {"在线报告": local_reference},
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("回答正文", messages[0]["content"])
        self.assertIn("查看 AI 生成文档：在线报告", messages[0]["content"])
        self.assertIn(local_reference, messages[0]["content"])

    def test_doubao_state_only_document_links_to_saved_markdown(self):
        html = """
        <div class="message-item" data-doubao-role="assistant">
          <p>已创建在线报告</p>
        </div>
        """
        local_reference = "./result_files/report.md"
        messages = doubao.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {"doubao-ai-doc:在线报告": local_reference},
        )
        self.assertIn(
            f"[📄 查看 AI 生成文档：在线报告]({local_reference})",
            messages[0]["content"],
        )

    def test_saved_doubao_ai_document_is_read_with_assistant_attribution(self):
        class DocumentGateway:
            def generate_json(self, _prompt, _schema, media_assets=None):
                self.media_assets = list(media_assets or [])
                return {"items": [{
                    "media_id": "M001",
                    "description": "这是一份由 AI 生成的在线报告。",
                    "status": "described",
                }]}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = root / "result_files"
            files.mkdir()
            (files / "online.md").write_text(
                "# 在线报告\n\nAI 生成的正文内容",
                encoding="utf-8",
            )
            assets = summary.discover_media(
                [{
                    "role": "AI",
                    "content": (
                        "[📄 查看 AI 生成文档：在线报告]"
                        "(./result_files/online.md)"
                    ),
                }],
                root,
                root,
                summary.SummaryConfig(),
            )
            self.assertEqual(assets[0].status, "ready")
            self.assertIn("AI 生成的正文内容", assets[0].extracted_text)
            gateway = DocumentGateway()
            summary.describe_media(
                assets,
                gateway,
                summary.SummaryConfig(),
                lambda _message: None,
            )

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].source_role, "assistant")
        self.assertEqual(assets[0].status, "described")
        self.assertIn("AI 回答中包含文档", assets[0].description)
        self.assertIn("AI 生成的在线报告", assets[0].description)
        self.assertNotIn("AI 生成的正文内容", assets[0].description)
        self.assertEqual(gateway.media_assets, [])

    def test_doubao_generated_image_prefers_original_over_thumbnail(self):
        thumbnail = (
            "https://cdn.example/rc_gen_image/same.jpeg"
            "~tplv-x-cthumb_lwm3.png?thumb"
        )
        original = (
            "https://cdn.example/rc_gen_image/same.jpeg"
            "~tplv-x-cgen_lwm3.png?original"
        )
        html = f"""
        <div class="message-item">
          <img alt="image" src="{thumbnail}" />
          <img src="{original}" />
        </div>
        """
        messages = doubao.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {thumbnail: "./images/thumb.jpg", original: "./images/original.jpg"},
        )
        self.assertNotIn("thumb.jpg", messages[0]["content"])
        self.assertIn("original.jpg", messages[0]["content"])
        self.assertEqual(
            doubao.prefer_original_generated_images([thumbnail, original]),
            [original],
        )

    def test_doubao_ai_images_drop_empty_placeholders_and_repeated_ui_icons(self):
        placeholder = (
            "data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27"
            "%20version=%271.1%27%20width=%27256%27%20height=%27192%27/%3e"
        )
        html = f"""
        <div class="message-item">
          <span><img src="{placeholder}" /></span>
          <picture><img alt="image" src="https://cdn.example/first.jpg" /></picture>
          <span><img src="{placeholder}" /></span>
          <picture><img alt="image" src="https://cdn.example/second.jpg" /></picture>
          <div><img class="img-z0eKj1" src="https://cdn.example/icon.png" /></div>
          <div><img class="img-z0eKj1" src="https://cdn.example/icon.png" /></div>
          <div><img class="img-z0eKj1" src="https://cdn.example/icon.png" /></div>
          正文说明
        </div>
        """
        image_map = {
            "https://cdn.example/first.jpg": "./images/first.jpg",
            "https://cdn.example/second.jpg": "./images/second.jpg",
            "https://cdn.example/icon.png": "./images/icon.png",
        }
        messages = doubao.parse_messages(
            BeautifulSoup(html, "html.parser"), image_map
        )
        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        self.assertEqual(content.count("![image]"), 2)
        self.assertIn("./images/first.jpg", content)
        self.assertIn("./images/second.jpg", content)
        self.assertNotIn("data:image/svg+xml", content)
        self.assertNotIn("./images/icon.png", content)

    def test_doubao_filter_keeps_meaningful_svg_and_nonrepeated_image(self):
        meaningful_svg = (
            "data:image/svg+xml,%3Csvg%20xmlns=%27http://www.w3.org/2000/svg%27%3E"
            "%3Cpath%20d=%27M0%200h10v10z%27/%3E%3C/svg%3E"
        )
        html = f"""
        <div class="message-item">
          <img alt="diagram" src="{meaningful_svg}" />
          <img class="img-z0eKj1" src="https://cdn.example/unique.png" />
        </div>
        """
        messages = doubao.parse_messages(
            BeautifulSoup(html, "html.parser"),
            {"https://cdn.example/unique.png": "./images/unique.png"},
        )
        content = messages[0]["content"]
        self.assertIn("data:image/svg+xml", content)
        self.assertIn("./images/unique.png", content)

    def test_unavailable_image_marker_becomes_non_reverifiable_media(self):
        messages = [{
            "role": "User",
            "content": "🖼️ **[用户上传图片]**（DeepSeek 分享页原图已失效或加载失败）"
        }]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            assets = summary.discover_media(
                messages, project, project, summary.SummaryConfig()
            )
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].status, "unavailable")
        self.assertFalse(assets[0].public_dict()["can_reverify"])
        self.assertIn("不可重新验证", assets[0].description)

    def test_long_messages_are_split_without_losing_text(self):
        original = "A" * 900 + "\n\n" + "B" * 900
        chunks = summary.chunk_messages(
            [{"role": "AI", "content": original}], max_chars=1000
        )
        self.assertGreaterEqual(len(chunks), 2)
        merged = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("A" * 900, merged)
        self.assertIn("B" * 900, merged)
        self.assertTrue(all(chunk.message_indices == (1,) for chunk in chunks))

    def test_query_index_preserves_every_user_message_and_recent_order(self):
        messages = [
            {"role": "User" if index % 2 else "AI", "content": f"消息-{index}"}
            for index in range(1, 25)
        ]
        queries = summary._build_query_index(messages)
        recent = summary._build_recent_context(messages, max_messages=20)
        self.assertEqual(queries[0]["raw_user_message"], "消息-1")
        self.assertEqual(queries[-1]["raw_user_message"], "消息-23")
        self.assertEqual([item["message_id"] for item in recent], list(range(5, 25)))
        self.assertEqual(recent[-1]["content"], "消息-24")

    def test_terminal_section_selector_uses_only_available_numbered_items(self):
        result = {
            "typed_records": {
                "programming": [{"topic": "x"}],
                "learning": [{"topic": "y"}],
                "calculations": [], "decisions": [],
                "context_references": [], "progressions": [],
                "source_text_issues": []
            },
            "media": [{"media_id": "MEDIA_001"}]
        }
        output = []
        selected = summary.prompt_summary_sections(
            result,
            input_fn=lambda _prompt: "1, 2, 9",
            output_fn=output.append
        )
        self.assertEqual(selected, ("programming", "learning"))
        self.assertEqual(
            summary.available_summary_sections(result),
            ["programming", "learning"]
        )
        self.assertTrue(any("[1]" in line for line in output))
        self.assertEqual(
            summary.prompt_summary_sections(
                result, input_fn=lambda _prompt: "", output_fn=lambda _line: None
            ),
            ()
        )

    def test_topic_selector_uses_concrete_topics_and_stable_ids(self):
        result = {
            "topics": [
                {"topic_id": "topic_code", "title": "代码实现"},
                {"title": "部署流程"},
            ]
        }
        available = summary.available_summary_topics(result)
        self.assertEqual(
            [(item["topic_id"], item["title"]) for item in available],
            [("topic_code", "代码实现"), ("topic_2", "部署流程")],
        )
        self.assertEqual(
            summary.normalize_summary_topics(result, ("部署流程",)),
            ("topic_2",),
        )
        self.assertEqual(
            summary.normalize_summary_topics(result, "all"),
            ("topic_code", "topic_2"),
        )

    def test_selected_topic_expands_only_its_related_details(self):
        result = {
            "provider": "gemini",
            "model": "test-model",
            "source": "test.md",
            "conversation": {
                "message_count": 4,
                "chunk_count": 1,
                "conversation_types": ["programming"],
                "programming_mode": "task",
            },
            "overall_summary": "总览",
            "current_state": {},
            "topics": [
                {
                    "topic_id": "topic_1",
                    "title": "代码实现",
                    "summary": "代码主题摘要。",
                    "memory_ids": ["M1"],
                    "source_message_ids": [1, 2],
                },
                {
                    "topic_id": "topic_2",
                    "title": "部署流程",
                    "summary": "部署主题摘要。",
                    "memory_ids": ["M2"],
                    "source_message_ids": [3, 4],
                },
            ],
            "memory_items": [
                {
                    "memory_id": "M1",
                    "topic": "代码关键细节",
                    "content": "只属于代码主题的重点细节。",
                    "source": "assistant",
                    "status": "delivered",
                    "message_ids": [1, 2],
                    "evidence_quote": "代码证据",
                },
                {
                    "memory_id": "M2",
                    "topic": "部署关键细节",
                    "content": "不应随代码主题展开的部署细节。",
                    "source": "assistant",
                    "status": "delivered",
                    "message_ids": [3, 4],
                    "evidence_quote": "部署证据",
                },
            ],
            "typed_records": {
                "programming": [
                    {
                        "topic": "代码主题记录",
                        "code_state": "已有脚本",
                        "bug_or_issue": "存在报错",
                        "assistant_diagnosis": "路径问题",
                        "constraints": [],
                        "implemented_changes": [],
                        "proposed_changes": ["修复路径"],
                        "pending_validation": ["重新运行"],
                        "implementation_status": "unconfirmed",
                        "message_ids": [1, 2],
                    },
                    {
                        "topic": "部署主题记录",
                        "code_state": "待部署",
                        "bug_or_issue": "无",
                        "assistant_diagnosis": "无",
                        "constraints": [],
                        "implemented_changes": [],
                        "proposed_changes": ["发布"],
                        "pending_validation": [],
                        "implementation_status": "unconfirmed",
                        "message_ids": [3, 4],
                    },
                ]
            },
            "query_index": [],
            "media": [],
            "processing": {"warnings": []},
        }

        rendered = summary.render_summary_markdown(
            result, selected_topics=("topic_1",)
        )
        self.assertIn("## 分主题摘要", rendered)
        self.assertIn("### 代码实现", rendered)
        self.assertIn("### 部署流程", rendered)
        self.assertIn("## 重点主题详情", rendered)
        self.assertIn("只属于代码主题的重点细节", rendered)
        self.assertNotIn("不应随代码主题展开的部署细节", rendered)
        self.assertIn("#### 编程任务记录", rendered)
        self.assertIn("##### 代码主题记录", rendered)
        self.assertNotIn("##### 部署主题记录", rendered)
        self.assertIn("## 媒体与附件说明（开发检查）", rendered)

        without_selection = summary.render_summary_markdown(result)
        self.assertNotIn("## 重点主题详情", without_selection)
        self.assertNotIn("只属于代码主题的重点细节", without_selection)

    def test_summary_markdown_links_every_media_reference(self):
        result = {
            "model": "test-model", "source": "DeepSeek.md",
            "conversation": {
                "message_count": 2, "chunk_count": 1,
                "conversation_types": ["document_analysis"],
            },
            "overall_summary": "附件测试。",
            "current_state": {}, "topics": [], "memory_items": [],
            "typed_records": {}, "query_index": [],
            "media": [{
                "media_id": "M001", "message_index": 1,
                "kind": "document", "source_role": "user",
                "label": "result1.xlsx",
                "reference": "./AI_memory_summary_files/result1.xlsx",
                "status": "unavailable", "can_reverify": True,
                "description": "格式暂未接入解析。",
            }, {
                "media_id": "M002", "message_index": 2,
                "kind": "image", "source_role": "user",
                "label": "用户图片",
                "reference": "./AI_memory_summary_images/image.png",
                "status": "described", "can_reverify": True,
                "description": "图片内容。",
            }],
            "processing": {"warnings": []},
        }

        rendered = summary.render_summary_markdown(result)

        self.assertIn(
            "- 文件：[result1.xlsx](./AI_memory_summary_files/result1.xlsx)",
            rendered,
        )
        self.assertIn(
            "- 图片：![用户图片](./AI_memory_summary_images/image.png)",
            rendered,
        )

    def test_default_outputs_and_v8_structure(self):
        messages = [
            {"role": "User", "content": "请解决报错"},
            {"role": "AI", "content": "建议增加 try-except"}
        ]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            result = summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                source_name="ChatGPT_编程.md",
                config=summary.SummaryConfig(),
                gateway=FakeGateway(),
                progress=lambda _message: None
            )
            json_path = (
                project / "results" / "summary"
                / "ChatGPT_编程_result.json"
            )
            markdown_path = (
                project / "results" / "summary"
                / "ChatGPT_编程_summary.md"
            )
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(result["schema_version"], 8)
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["memory_items"][0]["message_range"], "2")
        self.assertEqual(saved["source"], "ChatGPT_编程.md")
        self.assertEqual(saved["memory_items"][0]["source"], "assistant")
        self.assertEqual(saved["memory_items"][0]["status"], "suggested")
        self.assertEqual(saved["memory_items"][0]["message_ids"], [2])
        self.assertEqual(saved["current_state"]["last_user_message_id"], 1)
        self.assertEqual(
            saved['typed_records']['programming'][0]['implementation_status'],
            'unconfirmed'
        )
        self.assertEqual(saved["current_state"]["latest_message_id"], 2)
        self.assertEqual(saved["current_state"]["latest_message_role"], "AI")
        self.assertTrue(saved["current_state"]["last_user_turn_answered"])
        self.assertNotIn("细粒度记忆索引", markdown)
        self.assertNotIn("用户原始查询索引", markdown)
        self.assertNotIn("最近上下文（按时间顺序，高保真）", markdown)
        self.assertNotIn("## 编程任务记录", markdown)
        self.assertIn("媒体与附件说明（开发检查）", markdown)
        self.assertIn("（未检测到图片或附件。）", markdown)

        expanded = summary.render_summary_markdown(
            saved, selected_sections="all"
        )
        self.assertIn("## 编程任务记录", expanded)
        self.assertIn("500MB 以下，有点大了", expanded)
        self.assertIn("可能应为 500MB 以上", expanded)

        saved["topics"] = [
            {"title": "主题一", "summary": "主题一摘要。",
             "source_message_ids": [1], "memory_ids": []},
            {"title": "主题二", "summary": "主题二摘要。",
             "source_message_ids": [2], "memory_ids": []}
        ]
        without_records = summary.render_summary_markdown(saved)
        with_programming = summary.render_summary_markdown(
            saved, selected_sections={"programming"}
        )
        for rendered in (without_records, with_programming):
            self.assertIn("### 主题一", rendered)
            self.assertIn("### 主题二", rendered)
            self.assertLess(rendered.index("### 主题一"), rendered.index("### 主题二"))
            self.assertIn("## 媒体与附件说明（开发检查）", rendered)
        self.assertNotIn("## 编程任务记录", without_records)
        self.assertIn("## 编程任务记录", with_programming)

        detailed = summary.render_summary_markdown(
            saved, include_details=True
        )
        self.assertIn("细节记忆（可选）", detailed)
        self.assertNotIn("细粒度记忆索引", detailed)
        self.assertNotIn("用户原始查询索引", detailed)
        self.assertLessEqual(
            len(summary._select_detail_memory(saved)),
            8
        )

    def test_short_conversation_has_no_topic_sections_but_keeps_memory(self):
        messages = [
            {"role": "User", "content": "短问题"},
            {"role": "AI", "content": "短回答"}
        ]
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            result = summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                output_json=project / "short.json",
                output_markdown=project / "short.md",
                config=summary.SummaryConfig(short_conversation_chars=1000),
                gateway=FakeGateway(),
                progress=lambda _message: None
            )
        self.assertEqual(result["topics"], [])
        self.assertEqual(len(result["memory_items"]), 1)
        self.assertEqual(len(result["query_index"]), 1)
        self.assertEqual(len(result["recent_context"]), 2)

    def test_context_references_are_filtered_compact_and_limited(self):
        records = [
            {
                "raw_message": "portion",
                "resolved_reference": "英语词汇查询",
                "resolution_status": "certain"
            },
            {
                "raw_message": "1",
                "resolved_reference": "上轮列出的第一个方案",
                "resolution_status": "certain"
            },
            {
                "raw_message": "无饮水机，无电风扇",
                "resolved_reference": "在之前的宿舍用电估算基础上排除两项设备",
                "resolution_status": "certain"
            },
            {
                "raw_message": "无吹风机",
                "resolved_reference": "在之前的条件基础上继续排除吹风机",
                "resolution_status": "certain"
            },
            {
                "raw_message": "这个呢",
                "resolved_reference": "指上轮第二个方案",
                "resolution_status": "uncertain"
            },
            {
                "raw_message": "1",
                "resolved_reference": "AI 将其解释为英语单词 one",
                "resolution_status": "certain"
            }
        ]
        lines = []
        summary._render_context_records(lines, records)
        markdown = chr(10).join(lines)
        bullets = [line for line in lines if line.startswith("- “")]

        self.assertEqual(len(bullets), 3)
        self.assertNotIn("portion", markdown)
        self.assertNotIn("AI 当时的理解", markdown)
        self.assertNotIn("上下文消息", markdown)
        self.assertIn("“1” → 上轮列出的第一个方案", markdown)
        self.assertNotIn("英语单词 one", markdown)

    def test_minor_typos_are_hidden_but_material_conflicts_are_rendered(self):
        records = [
            {
                "original_text": "obessesive",
                "issue_description": "普通拼写错误",
                "inferred_correction": "obsessive",
                "source": "user",
                "status": "uncertain",
                "message_ids": [1]
            },
            {
                "original_text": "500MB 以下",
                "issue_description": "相同阈值对应相反建议，形成语义冲突",
                "inferred_correction": "第二处可能是 500MB 以上",
                "source": "assistant",
                "status": "uncertain",
                "message_ids": [2]
            }
        ]
        lines = []
        summary._render_source_text_issues(lines, records)
        markdown = chr(10).join(lines)

        self.assertNotIn("obessesive", markdown)
        self.assertIn("500MB 以下", markdown)
        self.assertIn("语义冲突", markdown)

    def test_language_details_are_aggregated_instead_of_listed_one_by_one(self):
        memories = []
        for index, word in enumerate(
            ("portion", "uniformly", "hub", "category", "implementation"),
            start=1
        ):
            memories.append({
                "topic": word,
                "memory_type": "fact",
                "content": f"{word} 是一个英语单词，含义已讲解。",
                "source": "assistant",
                "status": "uncertain",
                "message_ids": [index]
            })
        result = {
            "conversation": {"conversation_types": ["language_learning"]},
            "memory_items": memories,
            "query_index": [{
                "message_id": 99,
                "raw_user_message": "bony"
            }]
        }
        details = summary._select_detail_memory(result)

        self.assertEqual(len(details), 1)
        self.assertEqual(details[0][0], "学习概览")
        self.assertIn("集中讨论 5 项", details[0][2])
        self.assertNotIn("bony", str(details))

    def test_answered_open_question_is_not_in_detail_memory(self):
        result = {
            "conversation": {"conversation_types": ["ordinary"]},
            "current_state": {
                "last_user_message_id": 1,
                "last_user_turn_answered": True
            },
            "memory_items": [{
                "topic": "已回答的问题",
                "memory_type": "open_question",
                "content": "用户询问文档是否由 AI 撰写。",
                "source": "user",
                "status": "unresolved",
                "message_ids": [1]
            }],
            "query_index": []
        }

        self.assertEqual(summary._select_detail_memory(result), [])

    def test_assistant_media_conclusion_is_not_confirmed_without_user_evidence(self):
        record = summary._normalize_media_links([
            {
                'media_id': 'M001',
                'user_message_id': 1,
                'assistant_message_ids': [2],
                'assistant_conclusion': '上一 AI 的视觉判断',
                'conclusion_status': 'confirmed'
            }
        ], message_count=2)[0]
        self.assertEqual(record['conclusion_status'], 'uncertain')

    def test_unavailable_attachment_keeps_nearby_historical_answer(self):
        asset = summary.MediaAsset(
            media_id='M001',
            message_index=1,
            kind='document',
            label='未保留文件名的上传文档',
            reference='unavailable://upload-placeholder',
            status='unavailable'
        )
        media = summary._bind_media_results(
            [asset],
            [],
            [
                {'role': 'User', 'content': '上传文件'},
                {'role': 'User', 'content': '请检查申请表'},
                {'role': 'AI', 'content': '检查结果如下'}
            ]
        )
        binding = media[0]['assistant_bindings'][0]
        self.assertEqual(binding['assistant_message_ids'], [3])
        self.assertEqual(binding['conclusion_status'], 'unavailable')
        self.assertIn('不能重新验证', binding['assistant_conclusion'])

    def test_old_unconfirmed_code_changes_render_as_ai_proposal(self):
        lines = []
        summary._render_programming_records(lines, [{
            'topic': '旧结果', 'code_state': '', 'bug_or_issue': '',
            'assistant_diagnosis': '', 'constraints': [],
            'implemented_changes': ['修改解析器'],
            'pending_validation': [], 'message_ids': [1, 2],
            'implementation_status': 'unconfirmed'
        }])
        markdown = chr(10).join(lines)
        self.assertIn('- 已实施修改：无用户确认', markdown)
        self.assertIn('- AI 已提供/建议的修改：修改解析器', markdown)

    def test_confirmed_code_change_keeps_proposal_and_avoids_conflicting_text(self):
        lines = []
        summary._render_programming_records(lines, [{
            'topic': '已确认修改', 'code_state': '', 'bug_or_issue': '',
            'assistant_diagnosis': '', 'constraints': [],
            'implemented_changes': [], 'proposed_changes': ['修改解析器'],
            'pending_validation': [], 'message_ids': [1, 2, 3],
            'implementation_status': 'confirmed_by_user'
        }])
        markdown = chr(10).join(lines)
        self.assertIn(
            '- 已实施修改：用户已确认实施，具体修改项未单独提取',
            markdown
        )
        self.assertIn('- AI 已提供/建议的修改：修改解析器', markdown)
        self.assertIn('- 用户执行状态：用户明确确认已实施', markdown)
        self.assertNotIn('已实施修改：无用户确认', markdown)

    def test_normalization_preserves_ai_proposal_after_user_confirmation(self):
        messages = [
            {'role': 'User', 'content': '请修改解析器'},
            {'role': 'AI', 'content': '建议修改解析器的边界判断'},
            {'role': 'User', 'content': '已经按这个方案修改好了'},
        ]
        normalized = summary._normalize_final_summary(
            {
                'overall_summary': '用户已完成解析器修改。',
                'conversation_types': ['programming'],
                'current_state': {
                    'current_activity': {}, 'reached_stage': {},
                    'completed': [{
                        'content': '已按方案修改解析器',
                        'source': 'user', 'status': 'executed',
                        'message_ids': [3]
                    }],
                    'pending': [], 'next_step': {},
                    'last_user_intent': '完成修改',
                    'breakpoint_status': 'completed'
                },
                'topics': []
            },
            chunk_summaries=[{
                'conversation_types': ['programming'],
                'programming_records': [{
                    'topic': '解析器修改', 'code_state': '已有解析器',
                    'constraints': [], 'bug_or_issue': '边界判断错误',
                    'assistant_diagnosis': '需要修正边界判断',
                    'assistant_proposed_changes': ['修改解析器的边界判断'],
                    'implemented_changes': [], 'pending_validation': [],
                    'message_ids': [1, 2, 3]
                }]
            }],
            messages=messages,
            assets=[],
            short_conversation=True
        )
        record = normalized['typed_records']['programming'][0]
        self.assertEqual(record['implementation_status'], 'confirmed_by_user')
        self.assertEqual(
            record['proposed_changes'], ['修改解析器的边界判断']
        )

    def test_old_calculation_is_not_claimed_as_program_verified(self):
        lines = []
        summary._render_calculation_records(lines, [{
            'topic': '旧计算', 'user_conditions': [],
            'assistant_assumptions': [], 'result': '约 55',
            'confidence': 'medium', 'message_ids': [1, 2]
        }])
        self.assertIn(
            '- 数值忠实度：旧结果未执行程序校验',
            chr(10).join(lines)
        )

    def test_mixed_turn_assistant_action_is_not_labeled_as_user_confirmed(self):
        item = {
            'content': '润色了用户的邮件',
            'source': 'user',
            'status': 'confirmed',
            'message_ids': [1, 2]
        }
        reconciled = summary._reconcile_source_status(
            item,
            [
                {'role': 'User', 'content': '帮我润色邮件'},
                {'role': 'AI', 'content': '这是润色后的版本'}
            ]
        )
        self.assertEqual(reconciled['source'], 'assistant')
        self.assertEqual(reconciled['status'], 'delivered')
        self.assertEqual(reconciled['message_ids'], [2])

    def test_ai_explanation_does_not_prove_user_mastery(self):
        item = {
            'content': '掌握了类变量与实例变量的区别',
            'source': 'user',
            'status': 'confirmed',
            'message_ids': [1, 2]
        }
        reconciled = summary._reconcile_source_status(
            item,
            [
                {'role': 'User', 'content': '类变量与实例变量有什么区别？'},
                {'role': 'AI', 'content': '区别如下……'}
            ]
        )
        self.assertEqual(reconciled['source'], 'assistant')
        self.assertEqual(reconciled['status'], 'answered')
        self.assertIn('没有用户掌握程度的证据', reconciled['content'])

    def test_code_translation_memory_is_not_vocabulary(self):
        self.assertFalse(summary._is_vocabulary_memory({
            'topic': 'Python PDF翻译工具代码',
            'content': 'ThreadPoolExecutor 并发翻译 future.result 报错',
            'memory_type': 'fact'
        }, language_heavy=True))

    def test_calculation_unsupported_numbers_are_flagged(self):
        records = [{
            'result': '月用电量约 55 至 100 度',
            'confidence': 'medium',
            'message_ids': [1, 2]
        }]
        summary._validate_calculation_records(
            records,
            [
                {'role': 'User', 'content': '无饮水机，无电风扇'},
                {'role': 'AI', 'content': '通常 50 到 100，最常见 60 到 80 度'}
            ]
        )
        self.assertEqual(records[0]['source_fidelity'], 'unsupported_numbers')
        self.assertEqual(records[0]['unsupported_numbers'], ['55'])
        self.assertEqual(records[0]['confidence'], 'low')
        self.assertIn('已从正式总结移除', records[0]['result'])
        self.assertEqual(
            records[0]['model_result_rejected'],
            '月用电量约 55 至 100 度'
        )

    def test_courtesy_followup_is_removed_from_open_state(self):
        messages = [
            {'role': 'User', 'content': '这个目录能删吗？'},
            {
                'role': 'AI',
                'content': (
                    '可以删除缓存，但请先退出微信。'
                    '如果你愿意，我还能教你自动清理旧缓存的设置。'
                )
            }
        ]
        state = {
            'pending': [{
                'content': '用户是否需要了解自动清理设置',
                'source': 'assistant', 'status': 'suggested',
                'message_ids': [2]
            }],
            'next_step': {
                'content': '等待用户回复是否需要继续',
                'source': 'assistant', 'status': 'suggested',
                'message_ids': [2]
            },
            'breakpoint_status': 'waiting_user',
            'last_user_turn_answered': True
        }
        summary._filter_spurious_open_state(state, messages)
        self.assertEqual(state['pending'], [])
        self.assertEqual(state['next_step']['content'], '未明确')
        self.assertEqual(state['breakpoint_status'], 'complete')

    def test_waiting_for_next_new_question_is_removed(self):
        messages = [
            {'role': 'User', 'content': 'one'},
            {'role': 'AI', 'content': 'one 表示一。'}
        ]
        state = {
            'pending': [],
            'next_step': {
                'content': '等待用户提出下一个新问题',
                'source': 'inferred', 'status': 'unresolved',
                'message_ids': [2]
            },
            'breakpoint_status': 'waiting_user',
            'last_user_turn_answered': True
        }
        summary._filter_spurious_open_state(state, messages)
        self.assertEqual(state['next_step']['content'], '未明确')
        self.assertEqual(state['breakpoint_status'], 'complete')

    def test_accepted_assistant_followup_is_kept(self):
        messages = [
            {'role': 'User', 'content': '这个目录能删吗？'},
            {
                'role': 'AI',
                'content': '可以删除。如果你愿意，我还能教你自动清理设置。'
            },
            {'role': 'User', 'content': '可以，请继续教我'}
        ]
        claim = {
            'content': '继续讲解自动清理设置',
            'source': 'assistant', 'status': 'suggested',
            'message_ids': [2, 3]
        }
        self.assertFalse(
            summary._claim_is_spurious_open_state(claim, messages)
        )

    def test_user_explicitly_deferred_choice_is_kept(self):
        messages = [{
            'role': 'User',
            'content': '两个邮件版本我稍后决定是否采用'
        }]
        claim = {
            'content': '决定是否采用两个邮件版本',
            'source': 'user', 'status': 'unresolved',
            'message_ids': [1]
        }
        self.assertFalse(
            summary._claim_is_spurious_open_state(claim, messages)
        )

    def test_programming_attempt_preserves_ai_changes_without_confirming_completion(self):
        messages = [
            {'role': 'User', 'content': '请把脚本改成并发版'},
            {'role': 'AI', 'content': '已提供 ThreadPoolExecutor 版本'},
            {'role': 'User', 'content': '现在运行到 future.result 报错'}
        ]
        self.assertTrue(summary._user_attempted_programming_change(
            {'message_ids': [1, 2, 3]}, messages
        ))

    def test_topic_fallback_keeps_independent_topics(self):
        chunk = {
            'title': '多项测试',
            'memory_items': [
                {
                    'memory_id': 'C001M001', 'topic': '庄园翻译',
                    'content': '庄园可译为 manor。', 'message_ids': [1, 2]
                },
                {
                    'memory_id': 'C001M002', 'topic': '微信界面',
                    'content': '分析微信界面截图。', 'message_ids': [3, 4]
                },
                {
                    'memory_id': 'C001M003', 'topic': '复旦 eHall',
                    'content': '识别网上办事大厅。', 'message_ids': [5, 6]
                }
            ]
        }
        topics = summary._ensure_topic_coverage(
            topics=[],
            chunk_summaries=[chunk],
            message_count=6,
            short_conversation=False
        )
        self.assertEqual(len(topics), 3)
        self.assertEqual(
            {topic['title'] for topic in topics},
            {'庄园翻译', '微信界面', '复旦 eHall'}
        )

    def test_multichunk_language_topics_are_merged_but_other_topic_is_kept(self):
        chunks = [
            {
                'title': '第一批词汇',
                'conversation_types': ['language_learning'],
                'memory_items': [{
                    'memory_id': 'C001M001', 'topic': 'dynamic 词汇查询',
                    'content': 'dynamic 的含义与用法。',
                    'memory_type': 'assistant_suggestion',
                    'message_ids': [1, 2]
                }]
            },
            {
                'title': '第二批混合内容',
                'conversation_types': ['language_learning', 'ordinary'],
                'memory_items': [
                    {
                        'memory_id': 'C002M001', 'topic': 'flourish 词汇查询',
                        'content': 'flourish 的含义与用法。',
                        'memory_type': 'fact', 'message_ids': [3, 4]
                    },
                    {
                        'memory_id': 'C002M002', 'topic': '宿舍用电量估算',
                        'content': '非空调季节用电量估算。',
                        'memory_type': 'calculation', 'message_ids': [5, 6]
                    }
                ]
            }
        ]
        topics = summary._ensure_topic_coverage(
            topics=[], chunk_summaries=chunks,
            message_count=6, short_conversation=False
        )
        self.assertEqual(len(topics), 2)
        self.assertEqual(
            {topic['title'] for topic in topics},
            {'英语词汇、翻译与表达学习', '宿舍用电量估算'}
        )

    def test_topic_cleanup_then_merge_produces_one_language_topic(self):
        memories = [
            {
                'memory_id': 'M1', 'topic': 'word one',
                'content': '英语单词 one 的含义。',
                'memory_type': 'learning_point', 'message_ids': [1, 2]
            },
            {
                'memory_id': 'M2', 'topic': 'word two',
                'content': '英语单词 two 的含义。',
                'memory_type': 'learning_point', 'message_ids': [3, 4]
            },
            {
                'memory_id': 'M3', 'topic': '宿舍用电',
                'content': '四人宿舍用电估算。',
                'memory_type': 'calculation', 'message_ids': [5, 6]
            }
        ]
        result = {
            'conversation': {'message_count': 6},
            'memory_items': memories,
            'topics': [
                {
                    'title': '英语词汇学习一', 'memory_ids': ['M1', 'M3'],
                    'source_message_ids': [1, 2, 5, 6]
                },
                {
                    'title': '英语词汇学习二', 'memory_ids': ['M2'],
                    'source_message_ids': [3, 4]
                }
            ]
        }
        summary.renormalize_result(result)
        language_topics = [
            topic for topic in result['topics']
            if topic['title'] == '英语词汇、翻译与表达学习'
        ]
        self.assertEqual(len(language_topics), 1)
        self.assertEqual(set(language_topics[0]['memory_ids']), {'M1', 'M2'})
        self.assertTrue(any('宿舍用电' in topic['title'] for topic in result['topics']))

    def test_language_topic_does_not_absorb_chinese_writing_or_geography(self):
        memories = [
            {
                'memory_id': 'M1', 'topic': '软件权限英文翻译',
                'content': '翻译英文权限提示。', 'memory_type': 'correction',
                'message_ids': [1, 2]
            },
            {
                'memory_id': 'M2', 'topic': '甘肃天祝行政级别',
                'content': '天祝是县级藏族自治县。', 'memory_type': 'fact',
                'message_ids': [3, 4]
            },
            {
                'memory_id': 'M3', 'topic': '大学经历材料润色',
                'content': 'AI 提供五公里健康跑材料。',
                'memory_type': 'assistant_suggestion', 'message_ids': [5, 6]
            }
        ]
        topics = summary._normalize_topic_assignments([{
            'topic_id': 'topic_1',
            'title': '英语词汇、翻译与表达学习',
            'summary': '混合主题',
            'memory_ids': ['M1', 'M2', 'M3'],
            'source_message_ids': [1, 2, 3, 4, 5, 6]
        }], memories, 6)
        language = topics[0]
        self.assertEqual(language['memory_ids'], ['M1'])
        self.assertEqual(set(language['source_message_ids']), {1, 2})
        self.assertTrue(any('天祝' in topic['title'] for topic in topics))
        self.assertTrue(any('大学经历' in topic['title'] for topic in topics))

    def test_calculation_merge_keeps_latest_conditions_and_answer(self):
        messages = [
            {'role': 'User', 'content': '四人宿舍用电量？'},
            {'role': 'AI', 'content': '通常 80-200 度。'},
            {'role': 'User', 'content': '无饮水机，无电风扇'},
            {'role': 'AI', 'content': '新条件下通常 50-100 度，常见 60-80 度。'}
        ]
        records = [
            {
                'topic': '四人宿舍用电量估算',
                'user_conditions': ['四人宿舍'],
                'assistant_assumptions': [], 'result': '80-200 度',
                'confidence': 'medium', 'message_ids': [1, 2]
            },
            {
                'topic': '宿舍用电估算（无饮水机和电风扇）',
                'user_conditions': ['四人宿舍', '无饮水机', '无电风扇'],
                'assistant_assumptions': [], 'result': '50-100 度，常见 60-80 度',
                'confidence': 'high', 'message_ids': [3, 4]
            }
        ]
        merged = summary._merge_calculation_records(records, messages)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['result'], '50-100 度，常见 60-80 度')
        self.assertEqual(merged[0]['message_ids'], [3, 4])

    def test_user_question_is_not_labeled_as_ai_answer(self):
        claim = {
            'content': '询问 __str__ 和运算符重载',
            'source': 'user', 'status': 'answered', 'message_ids': [1]
        }
        summary._separate_answered_user_claim(
            claim, [{'role': 'User', 'content': '这是什么？'}]
        )
        self.assertEqual(claim['source'], 'user')
        self.assertEqual(claim['status'], 'confirmed')

    def test_question_sourced_fact_is_bound_to_following_ai_answer(self):
        item = {
            'topic': 'Tim Peters', 'memory_type': 'fact',
            'content': 'Tim Peters 是 Python 核心开发者。',
            'source': 'user', 'status': 'confirmed', 'message_ids': [1]
        }
        reconciled = summary._reconcile_source_status(item, [
            {'role': 'User', 'content': 'Tim Peters 是谁？'},
            {'role': 'AI', 'content': '他是 Python 核心开发者。'}
        ])
        self.assertEqual(reconciled['source'], 'assistant')
        self.assertEqual(reconciled['status'], 'answered')
        self.assertEqual(reconciled['message_ids'], [2])

    def test_assistant_fact_detail_uses_answer_label(self):
        self.assertEqual(summary._detail_label({
            'memory_type': 'fact', 'source': 'assistant', 'status': 'answered'
        }), '上一 AI 回答')

    def test_semantically_covered_query_is_not_added_as_missing(self):
        query = {'raw_user_message': '那以后会涨到多少？会涨多快？'}
        memory = {
            'topic': '微信缓存增长规律',
            'content': '解答了清理后缓存回涨的速度和上限。'
        }
        self.assertTrue(summary._query_covered_by_memory(query, memory))

    def test_following_answer_message_counts_as_query_coverage(self):
        result = {
            'memory_items': [{
                'topic': '页面识别', 'content': '识别为复旦 eHall。',
                'memory_type': 'media_finding', 'message_ids': [2],
                'source': 'assistant', 'status': 'answered'
            }],
            'current_state': {
                'last_user_message_id': 1, 'last_user_turn_answered': True
            },
            'conversation': {'conversation_types': []},
            'query_index': [{
                'message_id': 1, 'raw_user_message': '这是什么'
            }]
        }
        details = summary._select_detail_memory(result)
        self.assertFalse(any(topic == '未被记忆覆盖' for _label, topic, _content in details))

    def test_english_global_summary_gets_chinese_fallback(self):
        value = summary._ensure_chinese_summary(
            'The user is studying many English words and phrases.',
            [{'summary': '用户持续进行英语词汇学习。'}]
        )
        self.assertEqual(value, '用户持续进行英语词汇学习。')

    def test_programming_learning_is_not_treated_as_project_code_state(self):
        messages = [
            {'role': 'User', 'content': '我在学习 Python。i 是类变量，对吗？'},
            {'role': 'AI', 'content': '对，它定义在类中、方法外。'},
            {'role': 'User', 'content': '__getitem__ 是什么，有什么作用？'},
            {'role': 'AI', 'content': '它是索引取值时触发的魔术方法。'}
        ]
        chunk = {
            'conversation_types': ['programming'],
            'memory_items': [{
                'memory_id': 'C001M001',
                'topic': '类变量判断',
                'memory_type': 'code_state',
                'content': '类中、方法外定义的 i 是类变量。',
                'source': 'assistant',
                'status': 'suggested',
                'message_ids': [1, 2]
            }],
            'programming_records': [{
                'topic': '类变量判断',
                'code_state': 'class MyClass: i = 12345',
                'constraints': ['理解类变量与实例变量'],
                'bug_or_issue': '用户询问 i 是否为类变量',
                'assistant_diagnosis': 'i 是类变量',
                'implemented_changes': [],
                'pending_validation': [],
                'message_ids': [1, 2]
            }],
            'learning_records': [], 'calculation_records': [],
            'decision_records': [], 'contextual_messages': [],
            'progressions': [], 'source_text_issues': [], 'media_links': []
        }
        normalized = summary._normalize_final_summary(
            {
                'overall_summary': '用户正在学习 Python 面向对象编程。',
                'conversation_types': ['programming', 'language_learning'],
                'current_state': {
                    'current_activity': {
                        'content': '学习 Python 面向对象编程',
                        'source': 'user', 'status': 'confirmed',
                        'message_ids': [1, 3]
                    },
                    'reached_stage': {}, 'completed': [], 'pending': [],
                    'next_step': {}, 'last_user_intent': '理解魔术方法',
                    'breakpoint_status': 'ongoing'
                },
                'topics': []
            },
            chunk_summaries=[chunk],
            messages=messages,
            assets=[],
            short_conversation=True
        )
        result = {
            'model': 'test-model', 'source': 'DeepSeek_含图片.md',
            'conversation': {
                'message_count': 4, 'chunk_count': 1,
                'conversation_types': normalized['conversation_types'],
                'programming_mode': normalized['programming_mode']
            },
            **{
                key: normalized[key] for key in (
                    'overall_summary', 'current_state', 'topics',
                    'memory_items', 'typed_records'
                )
            },
            'query_index': []
        }
        markdown = summary.render_summary_markdown(
            result,
            include_details=True,
            selected_sections={"programming"}
        )

        self.assertEqual(normalized['programming_mode'], 'learning')
        self.assertEqual(
            normalized['conversation_types'], ['programming_learning']
        )
        self.assertEqual(
            normalized['memory_items'][0]['memory_type'], 'learning_point'
        )
        self.assertIn('对话类型：编程学习', markdown)
        self.assertIn('## 编程学习记录', markdown)
        self.assertIn('学习要点｜类变量判断', markdown)
        self.assertNotIn('## 编程任务记录', markdown)
        self.assertNotIn('代码状态｜', markdown)

    def test_real_programming_fix_stays_a_programming_task(self):
        mode = summary._infer_programming_mode(
            conversation_types=['programming_learning'],
            current_state={
                'current_activity': {'content': '修复代码解析器'},
                'reached_stage': {'content': '复现测试失败'},
                'next_step': {'content': '运行回归测试'},
                'last_user_intent': '修复该问题'
            },
            overall_summary='用户正在修复项目中的解析错误。',
            messages=[
                {'role': 'User', 'content': '运行失败了，请修复这个问题'}
            ]
        )
        self.assertEqual(mode, 'task')

    def test_old_learning_result_is_reclassified_when_rendered(self):
        result = {
            'model': 'test-model', 'source': 'old.json',
            'conversation': {
                'message_count': 2, 'chunk_count': 1,
                'conversation_types': ['programming']
            },
            'overall_summary': '用户正在学习 Python 面向对象编程。',
            'current_state': {
                'current_activity': {'content': '学习 Python 面向对象编程'}
            },
            'topics': [], 'typed_records': {'programming': []},
            'memory_items': [{
                'memory_id': 'C001M001', 'topic': '类变量',
                'memory_type': 'code_state',
                'content': 'i 是类变量。', 'source': 'assistant',
                'status': 'suggested', 'message_ids': [1, 2]
            }],
            'query_index': []
        }
        markdown = summary.render_summary_markdown(
            result, include_details=True
        )
        self.assertIn('对话类型：编程学习', markdown)
        self.assertIn('学习要点｜类变量', markdown)
        self.assertNotIn('代码状态｜', markdown)

    def test_answered_last_user_turn_is_not_described_as_waiting_reply(self):
        normalized = summary._normalize_final_summary(
            {
                'overall_summary': '测试',
                'conversation_types': ['ordinary'],
                'current_state': {
                    'current_activity': {
                        'content': '用户已发送数字 1 等待回复',
                        'source': 'user',
                        'status': 'confirmed',
                        'message_ids': [1, 2]
                    },
                    'reached_stage': {}, 'completed': [], 'pending': [],
                    'next_step': {}, 'last_user_intent': '查询 1',
                    'breakpoint_status': 'waiting_user'
                },
                'topics': []
            },
            chunk_summaries=[],
            messages=[
                {'role': 'User', 'content': '1'},
                {'role': 'AI', 'content': 'One 的释义'}
            ],
            assets=[],
            short_conversation=True
        )
        state = normalized['current_state']
        self.assertTrue(state['last_user_turn_answered'])
        self.assertEqual(state['current_activity']['source'], 'user')
        self.assertEqual(state['current_activity']['message_ids'], [1])
        self.assertEqual(state['reached_stage']['source'], 'assistant')
        self.assertEqual(state['reached_stage']['message_ids'], [2])

    def test_non_language_topics_are_not_swallowed_in_language_heavy_details(self):
        unrelated = [
            {
                "topic": "PDF 翻译程序开发",
                "content": "用户请求编写调用百度翻译 API 的脚本。",
                "memory_type": "user_condition"
            },
            {
                "topic": "未来三年大学生活规划",
                "content": "用户希望规划健康跑、学习与辅导员材料。",
                "memory_type": "assistant_suggestion"
            }
        ]
        for item in unrelated:
            self.assertFalse(summary._is_vocabulary_memory(item, True))
        self.assertTrue(summary._is_vocabulary_memory({
            "topic": "obsessive 的拼写与含义",
            "content": "解释英语单词 obsessive 的含义。",
            "memory_type": "learning_point"
        }, True))
        self.assertFalse(summary._is_vocabulary_memory({
            "topic": "PDF 提取文本方法优化",
            "content": "AI 建议在翻译脚本中使用 page.extract_text()。",
            "memory_type": "assistant_suggestion"
        }, True))

    def test_natural_language_detection_ignores_programming_translation_task(self):
        self.assertFalse(summary._looks_like_language_query(
            "请用 Python 写 PDF 翻译程序，调用百度翻译API"
        ))
        self.assertTrue(summary._looks_like_language_query("obessesive"))

    def test_topic_sources_expand_to_complete_question_answer_pairs(self):
        messages = [
            {"role": "User", "content": "问题 19"},
            {"role": "AI", "content": "回答 20"},
            {"role": "User", "content": "问题 21"},
            {"role": "AI", "content": "回答 22"}
        ]
        topics = [{
            "memory_ids": ["M1"], "source_message_ids": [2, 4],
            "summary": "学习两个概念"
        }]
        expanded = summary._expand_topic_sources(
            topics,
            [{"memory_id": "M1", "message_ids": [2, 4]}],
            messages
        )
        self.assertEqual(expanded[0]["source_message_ids"], [1, 2, 3, 4])

    def test_single_programming_task_subtopics_are_merged(self):
        memories = [
            {
                "memory_id": f"M{index}", "topic": title,
                "content": title, "message_ids": [index]
            }
            for index, title in enumerate((
                "Python PDF 翻译脚本", "并发异常处理",
                "代码路径约束", "PDF 排版优化"
            ), 1)
        ]
        topics = [
            {
                "title": memory["topic"], "summary": memory["content"],
                "memory_ids": [memory["memory_id"]],
                "source_message_ids": memory["message_ids"]
            }
            for memory in memories
        ]
        merged = summary._merge_programming_task_topics(
            topics, memories, "task"
        )
        self.assertEqual(len(merged), 1)
        self.assertIn("PDF 翻译脚本", merged[0]["title"])

    def test_explicit_language_user_request_can_join_language_topic(self):
        self.assertTrue(summary._is_vocabulary_memory({
            "topic": "英文系统权限提示翻译",
            "content": "用户请求翻译英文 scopes 提示语。",
            "memory_type": "user_condition"
        }, True))

    def test_answer_claim_bound_to_user_query_moves_to_following_ai(self):
        item = summary._reconcile_source_status({
            "memory_type": "fact",
            "content": "查询 One 的含义并回答",
            "source": "user", "status": "answered", "message_ids": [1]
        }, [
            {"role": "User", "content": "1"},
            {"role": "AI", "content": "One 的释义"}
        ])
        self.assertEqual(item["source"], "assistant")
        self.assertEqual(item["status"], "answered")
        self.assertEqual(item["message_ids"], [2])

    def test_learning_record_distinguishes_answer_from_correction_adoption(self):
        records = summary._normalize_learning_records([{
            "topic": "庄园的英文",
            "user_original": "庄园英文",
            "assistant_revision": "manor",
            "rationale": "给出翻译",
            "adoption_status": "confirmed",
            "message_ids": [1, 2]
        }], 2)
        self.assertEqual(records[0]["record_kind"], "translation")
        self.assertEqual(records[0]["adoption_status"], "not_applicable")

    def test_explicit_spelling_corrections_are_deterministically_preserved(self):
        messages = [
            {"role": "User", "content": "form"},
            {"role": "AI", "content": "这里 form 应为 from。"},
            {"role": "User", "content": "obessesive"},
            {"role": "AI", "content": "obessesive 是拼写错误，正确拼写为 obsessive。"}
        ]
        records = summary._extract_explicit_corrections(messages)
        pairs = {
            (record["topic"], tuple(record["message_ids"])) for record in records
        }
        self.assertIn(("form → from", (1, 2)), pairs)
        self.assertIn(("obessesive → obsessive", (3, 4)), pairs)

    def test_unavailable_image_topic_is_qualified_as_historical_inference(self):
        topics = [{
            "summary": "用户上传了一张 Pottermore 宣传图片并讨论内容。",
            "source_message_ids": [1, 2]
        }]
        media = [{
            "message_index": 1, "kind": "image", "can_reverify": False
        }]
        qualified = summary._qualify_unavailable_media_topics(topics, media)
        self.assertIn("当前不可重新验证", qualified[0]["summary"])
        self.assertIn("历史 AI", qualified[0]["summary"])
        self.assertIn("原图不可重新验证", qualified[0]["title"])

    def test_answered_request_is_not_labeled_as_user_constraint(self):
        label = summary._detail_label({
            "source": "user", "status": "confirmed",
            "memory_type": "user_condition",
            "content": "用户询问未来三年的大学生活规划"
        })
        self.assertEqual(label, "历史问答")

    def test_user_fact_is_labeled_as_information_not_constraint(self):
        label = summary._detail_label({
            "source": "user", "status": "confirmed",
            "memory_type": "user_condition",
            "content": "用户确认自己几乎不用小程序"
        })
        self.assertEqual(label, "用户信息")

    def test_long_learning_records_render_only_representatives_and_corrections(self):
        records = [
            {
                "topic": f"word-{index}", "record_kind": "translation",
                "user_original": f"word-{index}", "assistant_revision": "answer",
                "rationale": "", "adoption_status": "not_applicable",
                "message_ids": [index * 2 - 1, index * 2]
            }
            for index in range(1, 21)
        ]
        records.insert(10, {
            "topic": "form → from", "record_kind": "correction",
            "user_original": "form", "assistant_revision": "from",
            "rationale": "拼写修正", "adoption_status": "unconfirmed",
            "message_ids": [21, 22]
        })
        selected = summary._select_learning_records_for_render(records)
        self.assertLessEqual(len(selected), 8)
        self.assertTrue(any(x["topic"] == "form → from" for x in selected))

    def test_assistant_suggestion_is_not_duplicated_as_pending_task(self):
        state = {
            "pending": [{
                "content": "用户可以按建议清理缓存", "source": "assistant",
                "status": "suggested", "message_ids": [2]
            }],
            "next_step": {
                "content": "按建议清理缓存", "source": "inferred",
                "status": "suggested", "message_ids": [2]
            },
            "breakpoint_status": "waiting_user",
            "last_user_turn_answered": True
        }
        summary._filter_spurious_open_state(state, [
            {"role": "User", "content": "怎么清理？"},
            {"role": "AI", "content": "建议完全退出后清理。"}
        ])
        self.assertEqual(state["pending"], [])

    def test_unconfirmed_programming_summary_does_not_claim_user_implemented(self):
        text = summary._qualify_unconfirmed_programming_text(
            "在 AI 建议下，用户为代码增加了异常处理。",
            [{"implementation_status": "unconfirmed"}]
        )
        self.assertNotIn("用户为代码增加", text)
        self.assertIn("上一 AI 提供", text)

    def test_answered_non_programming_turn_does_not_wait_for_new_topic(self):
        state = {
            "last_user_message_id": 1,
            "last_user_intent": "测试结束",
            "reached_stage": {"content": "已回答", "source": "assistant", "message_ids": [2]},
            "pending": [],
            "next_step": {
                "content": "等待用户发送新的有效指令或开启新的对话",
                "source": "inferred", "status": "unresolved", "message_ids": [2]
            },
            "breakpoint_status": "waiting_user"
        }
        messages = [
            {"role": "User", "content": "这是测试，结束"},
            {"role": "AI", "content": "收到，无需继续处理。"}
        ]
        summary._filter_spurious_open_state(state, messages)
        summary._normalize_latest_turn_state(state, messages)
        self.assertEqual(state["next_step"]["content"], "未明确")
        self.assertEqual(state["breakpoint_status"], "complete")
        cleaned = summary._sanitize_overall_completion(
            "任务已答复，当前处于等待用户开启新话题的状态。", state
        )
        self.assertNotIn("等待", cleaned)
        preserved = summary._sanitize_overall_completion(
            "AI均已完成详细解答与信息提取，目前所有任务已答复，等待用户开启新话题。",
            state
        )
        self.assertIn("详细解答与信息提取", preserved)
        self.assertNotIn("等待用户", preserved)

    def test_learning_summary_does_not_claim_user_mastery_without_confirmation(self):
        text = summary._qualify_unconfirmed_learning_text(
            "目前已完成了对类变量的学习，并深入理解了运算符重载。",
            [{"role": "User", "content": "什么是重载？"}]
        )
        self.assertNotIn("深入理解", text)
        self.assertIn("上一 AI 还讲解", text)

    def test_minor_typos_are_omitted_from_source_issue_section(self):
        records = summary._filter_minor_source_text_issues([
            {"issue_description": "姓名拼写错误", "inferred_correction": "Tim Peters"},
            {"issue_description": "阈值前后语义冲突，会影响执行判断", "inferred_correction": "500MB以上"}
        ])
        self.assertEqual(len(records), 1)
        self.assertIn("语义冲突", records[0]["issue_description"])

    def test_renormalize_old_result_adds_corrections_and_removes_false_language_type(self):
        result = {
            "model": "old-model", "source": "old.json",
            "conversation": {
                "message_count": 2,
                "conversation_types": ["programming", "language_learning"]
            },
            "overall_summary": "用户正在开发 PDF 翻译程序。",
            "current_state": {}, "topics": [], "memory_items": [],
            "typed_records": {"learning": [], "programming": []},
            "media": []
        }
        messages = [
            {"role": "User", "content": "请写一个 Python PDF 翻译程序"},
            {"role": "AI", "content": "下面是代码。"}
        ]
        normalized = summary.renormalize_result(result, messages)
        self.assertEqual(normalized["provider"], "gemini")
        self.assertNotIn(
            "language_learning",
            normalized["conversation"]["conversation_types"]
        )

    def test_multichunk_summary_keeps_topic_and_calls_final_schema(self):
        messages = [
            {"role": "User", "content": "U" * 700},
            {"role": "AI", "content": "A" * 700}
        ]
        gateway = FakeGateway()
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            result = summary.summarize_conversation(
                messages=messages,
                project_dir=project,
                output_json=project / "result.json",
                output_markdown=project / "result.md",
                config=summary.SummaryConfig(
                    chunk_chars=600,
                    short_conversation_chars=1000
                ),
                gateway=gateway,
                progress=lambda _message: None
            )
        self.assertGreater(result["conversation"]["chunk_count"], 1)
        self.assertEqual(result["topics"][0]["topic_id"], "topic_1")
        self.assertTrue(any(call[0] is summary.FINAL_SCHEMA for call in gateway.calls))

    def test_load_exported_markdown_removes_separators(self):
        exported = """# AI 对话记忆导出

<hr style="border: 0;">

## 🔵 👤 用户提问

问题

<hr style="border: 0;">

## 🟣 🤖 AI 回答

回答
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "export.md"
            path.write_text(exported, encoding="utf-8")
            messages = summary.load_exported_markdown(path)
        self.assertEqual(messages, [
            {"role": "User", "content": "问题"},
            {"role": "AI", "content": "回答"}
        ])

    def test_programming_topic_merge_never_hard_cuts_markdown(self):
        memories = [
            {
                "memory_id": "M1", "topic": "Python PDF 翻译脚本",
                "memory_type": "fact", "source": "assistant",
                "status": "answered", "message_ids": [2],
                "content": "用户编写了 Python PDF 翻译脚本。"
            },
            {
                "memory_id": "M2", "topic": "代码路径约束",
                "memory_type": "user_condition", "source": "user",
                "status": "confirmed", "message_ids": [3],
                "content": "用户要求保留硬编码路径。"
            },
            {
                "memory_id": "M3", "topic": "Python 完整代码交付",
                "memory_type": "action", "source": "assistant",
                "status": "delivered", "message_ids": [6],
                "content": "AI 已交付并发和重试版本。"
            },
            {
                "memory_id": "M4", "topic": "future.result 报错",
                "memory_type": "open_question", "source": "user",
                "status": "unresolved", "message_ids": [7],
                "content": "用户反馈 `future.result()` 报错。"
            },
            {
                "memory_id": "M5", "topic": "Python 异常修复",
                "memory_type": "assistant_suggestion", "source": "assistant",
                "status": "suggested", "message_ids": [8],
                "content": "AI 建议为 `future.result()` 增加异常保护。"
            }
        ]
        topics = [
            {
                "title": item["topic"], "summary": item["content"],
                "memory_ids": [item["memory_id"]],
                "source_message_ids": item["message_ids"]
            }
            for item in memories[:4]
        ]
        topics[-1]["memory_ids"].append("M5")
        merged = summary._merge_programming_task_topics(
            topics, memories, "task"
        )
        text = merged[0]["summary"]
        self.assertEqual(len(merged), 1)
        self.assertFalse(text.endswith("…"))
        self.assertEqual(text.count("`") % 2, 0)
        self.assertIn("future.result()", text)

    def test_latest_user_intent_uses_raw_complete_request(self):
        state = {
            "last_user_message_id": 1,
            "last_user_intent": "询问页面内容和具体功能分类"
        }
        summary._normalize_latest_turn_state(state, [
            {"role": "User", "content": "这是什么"},
            {"role": "AI", "content": "这是办事大厅，并介绍了功能分类。"}
        ])
        self.assertEqual(state["last_user_intent"], "这是什么")

    def test_latest_contextual_choice_keeps_resolved_intent(self):
        state = {
            "last_user_message_id": 1,
            "last_user_intent": "选择几乎不用小程序"
        }
        summary._normalize_latest_turn_state(state, [
            {"role": "User", "content": "1"},
            {"role": "AI", "content": "收到。"}
        ])
        self.assertEqual(state["last_user_intent"], "选择几乎不用小程序")

    def test_ai_recommendation_is_not_promoted_to_user_choice(self):
        records = [{
            "topic": "缓存清理", "options": ["退出微信后删除"],
            "user_choice": "退出微信后删除（AI 推荐）",
            "status": "suggested", "message_ids": [1, 2]
        }]
        reconciled = summary._reconcile_decision_records(records, [
            {"role": "User", "content": "1"},
            {"role": "AI", "content": "建议退出微信后删除。"}
        ])
        self.assertEqual(reconciled, [])

    def test_explicit_user_decision_is_recovered_from_source_message(self):
        records = [{
            "topic": "脚本路径", "options": ["跨平台路径", "保留硬编码路径"],
            "user_choice": "", "status": "suggested", "message_ids": [1, 2]
        }]
        reconciled = summary._reconcile_decision_records(records, [
            {"role": "User", "content": "我要保留硬编码路径"},
            {"role": "AI", "content": "收到，将保留硬编码路径。"}
        ])
        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0]["user_choice"], "我要保留硬编码路径")
        self.assertEqual(reconciled[0]["status"], "confirmed")

    def test_language_term_topics_merge_by_adjacent_user_query(self):
        memories = [
            {
                "memory_id": "M1", "topic": "英语词汇：stain",
                "content": "AI 解释了英语词汇 stain 的含义。",
                "memory_type": "learning_point", "message_ids": [2]
            },
            {
                "memory_id": "M2", "topic": "技术术语：bytecode",
                "content": "AI 解释了 bytecode 在 Python 中的含义。",
                "memory_type": "learning_point", "message_ids": [4]
            },
            {
                "memory_id": "M3", "topic": "Python 魔法方法",
                "content": "AI 讲解了 Python 的 __str__ 与运算符重载。",
                "memory_type": "learning_point", "message_ids": [6]
            }
        ]
        topics = [
            {"topic_id": "topic_1", "title": "英语词汇学习",
             "summary": "", "memory_ids": ["M1"], "source_message_ids": [1, 2]},
            {"topic_id": "topic_2", "title": "计算机术语",
             "summary": "", "memory_ids": ["M2"], "source_message_ids": [3, 4]},
            {"topic_id": "topic_3", "title": "编程学习",
             "summary": "", "memory_ids": ["M3"], "source_message_ids": [5, 6]}
        ]
        messages = [
            {"role": "User", "content": "stain"},
            {"role": "AI", "content": "stain 是污渍。"},
            {"role": "User", "content": "bytecode"},
            {"role": "AI", "content": "bytecode 是字节码。"},
            {"role": "User", "content": "str 是特有方法吗？什么叫重载？"},
            {"role": "AI", "content": "__str__ 是特殊方法。"},
            {"role": "User", "content": "slow"},
            {"role": "AI", "content": "slow 是缓慢的。"},
            {"role": "User", "content": "possess"},
            {"role": "AI", "content": "possess 是拥有。"},
            {"role": "User", "content": "bubble"},
            {"role": "AI", "content": "bubble 是气泡。"}
        ]
        merged = summary._merge_language_learning_topics(
            topics, memories, messages=messages
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["memory_ids"], ["M1", "M2"])
        self.assertEqual(merged[1]["memory_ids"], ["M3"])

    def test_language_exchange_is_counted_once(self):
        memories = [
            {
                "topic": "英文系统权限提示翻译", "source": "user",
                "message_ids": [16]
            },
            {
                "topic": "英文系统权限提示翻译", "source": "assistant",
                "message_ids": [17]
            }
        ]
        self.assertEqual(summary._semantic_memory_count(memories), 1)

    def test_mixed_language_topic_is_split_before_dorm_calculations_merge(self):
        memories = [
            {"memory_id": "M1", "topic": "word", "content": "bytecode",
             "memory_type": "learning_point", "message_ids": [2]},
            {"memory_id": "M2", "topic": "expression", "content": "phrase fix",
             "memory_type": "assistant_suggestion", "message_ids": [4]},
            {"memory_id": "M3", "topic": "power", "content": "4\u4eba\u5bbf\u820d\u7528\u7535 100 kWh",
             "memory_type": "calculation", "message_ids": [6]},
            {"memory_id": "M4", "topic": "power", "content": "4\u4eba\u5bbf\u820d\u7535\u91cf 60 kWh",
             "memory_type": "calculation", "message_ids": [8]}
        ]
        topics = [
            {"topic_id": "topic_1", "title": "mixed", "summary": "",
             "memory_ids": ["M1", "M3"], "source_message_ids": [1, 2, 5, 6]},
            {"topic_id": "topic_2", "title": "expression", "summary": "",
             "memory_ids": ["M2"], "source_message_ids": [3, 4]},
            {"topic_id": "topic_3", "title": "\u5bbf\u820d\u7528\u7535", "summary": "",
             "memory_ids": ["M4"], "source_message_ids": [7, 8]}
        ]
        messages = [
            {"role": "User", "content": "bytecode"},
            {"role": "AI", "content": "bytecode means ..."},
            {"role": "User", "content": "blessed name of the god"},
            {"role": "AI", "content": "a better phrase is ..."},
            {"role": "User", "content": "slow"}, {"role": "AI", "content": "..."},
            {"role": "User", "content": "possess"}, {"role": "AI", "content": "..."},
            {"role": "User", "content": "bubble"}, {"role": "AI", "content": "..."}
        ]
        merged = summary._merge_language_learning_topics(
            topics, memories, messages=messages
        )
        merged = summary._merge_dorm_electricity_topics(merged, memories)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["memory_ids"], ["M1", "M2"])
        self.assertEqual(set(merged[1]["memory_ids"]), {"M3", "M4"})

    def test_unavailable_document_title_says_original_document(self):
        topics = [{
            "title": "文档检查（仅历史 AI 判断，原图不可重新验证）",
            "summary": "用户上传文档并请求检查。",
            "source_message_ids": [1, 2]
        }]
        qualified = summary._qualify_unavailable_media_topics(topics, [{
            "message_index": 1, "kind": "document", "can_reverify": False
        }])
        self.assertIn("原文档不可重新验证", qualified[0]["title"])
        self.assertNotIn("原图不可重新验证", qualified[0]["title"])

    def test_detailed_media_section_does_not_repeat_document_body(self):
        body = "这是已经进入总结输入的原始文档正文，不应在详细版中重复。"
        for description in (
            f"用户上传了文档“报告.md”。程序已提取文本，内容是：{body}",
            (
                "用户上传了文档“报告.docx”。"
                f"程序已安全提取 DOCX 正文，内容是：{body}"
            ),
        ):
            with self.subTest(description=description[:30]):
                rendered = summary._media_description_for_markdown({
                    "kind": "document",
                    "description": description,
                })
                self.assertNotIn(body, rendered)
                self.assertIn("已提取正文供总结使用", rendered)
                self.assertEqual(
                    description.split("程序已", 1)[0],
                    rendered.split("程序已", 1)[0],
                )

        image_description = "图片识别结果保持原样"
        self.assertEqual(
            summary._media_description_for_markdown({
                "kind": "image",
                "description": image_description,
            }),
            image_description,
        )

    def test_compacted_topic_balances_chinese_quotes(self):
        compacted = summary._compact_balanced_topic(
            "英文短语“with less fear of the consequences”解析", 24
        )
        self.assertEqual(compacted.count("“"), compacted.count("”"))

    def test_overall_summary_does_not_reverse_answering_role(self):
        cleaned = summary._sanitize_overall_role_attribution(
            "此外，用户解答了关于甘肃天祝行政区划的疑问，"
            "并翻译解析了关于 Scopes 权限的英文提示。"
        )
        self.assertIn("用户获得了关于甘肃天祝行政区划的解答", cleaned)
        self.assertNotIn("用户解答了", cleaned)

    def test_generated_summary_removes_conflicting_sentence_punctuation(self):
        cleaned = summary._clean_generated_punctuation(
            "AI 已完整解答，当前处于任务完成、。"
        )
        self.assertEqual(cleaned, "AI 已完整解答，当前处于任务完成。")


if __name__ == "__main__":
    unittest.main()
