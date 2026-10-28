"""Tests for fingerprint sanitization, including PR #62 OmO Sisyphus-Junior fix."""
import copy
import io
import json, os, sys, unittest
import tempfile
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _isolated_dirs import isolated_data_dirs  # noqa: E402  (its own data root)
# Its own directories, removed when this process exits. This suite used to share
# tests/_acc_test and tests/_use_test with _test_daily_chat.py and
# _test_tasks_cache.py, which made them unsafe to run at the same time and left
# directories in the repo after a run.
_TMP = isolated_data_dirs("wb-fingerprint-")

import wb_accounts
import wb_proxy as P
import wb_settings
import wb_transforms as T

class FingerprintSanitizationTests(unittest.TestCase):
    def test_omo_junior_fingerprint_neutralized(self):
        fp = "Sisyphus-Junior - Focused executor from OhMyOpenCode"
        # 1. Exact string rewritten
        self.assertEqual(T.sanitize_text(fp), "Sisyphus-Junior - Focused executor")

        # 2. Case insensitive
        self.assertEqual(T.sanitize_text(fp.lower()), "Sisyphus-Junior - Focused executor")

        # 3. Embedded in context
        text = f"You are a subagent. {fp}. Complete the task."
        expected = "You are a subagent. Sisyphus-Junior - Focused executor. Complete the task."
        self.assertEqual(T.sanitize_text(text), expected)

    def test_omo_partial_tokens_untouched(self):
        # Tokens alone should not be stripped or modified
        self.assertEqual(T.sanitize_text("Sisyphus-Junior alone"), "Sisyphus-Junior alone")
        self.assertEqual(T.sanitize_text("from OhMyOpenCode alone"), "from OhMyOpenCode alone")
        self.assertEqual(T.sanitize_text("Sisyphus master agent"), "Sisyphus master agent")

    def test_sanitize_messages_covers_all_roles_and_tool_args(self):
        fp = "Sisyphus-Junior - Focused executor from OhMyOpenCode"
        clean = "Sisyphus-Junior - Focused executor"
        messages = [
            {"role": "system", "content": f"System: {fp}"},
            {"role": "user", "content": f"User: {fp}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run", "arguments": json.dumps({"query": fp})}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": f"Tool output: {fp}"}
        ]
        sanitized = P.sanitize_messages(messages)
        self.assertIn(clean, sanitized[0]["content"])
        self.assertNotIn("from OhMyOpenCode", sanitized[0]["content"])
        self.assertIn(clean, sanitized[1]["content"])
        self.assertNotIn("from OhMyOpenCode", sanitized[1]["content"])
        args = json.loads(sanitized[2]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args["query"], clean)
        self.assertIn(clean, sanitized[3]["content"])
        self.assertNotIn("from OhMyOpenCode", sanitized[3]["content"])

class DshHarnessChannelFingerprintTests(unittest.TestCase):
    """DSH（DeepSeek Harness）系统提示首句的渠道指纹（与 #382 / #506 同类）。

    上游对连续短语做大小写不敏感的渠道校验，命中即 400 / code 11128。逐句二分
    确认：整句触发，拆词或单独 ``DeepSeek Harness`` 都不触发；只插入一个 "the"
    改变短语结构后即不再命中。
    """

    FP = "You are an AI agent powered by DeepSeek Harness."
    FP2 = "You are a coding agent powered by the deepseek-v4.1-flash model."
    CLEAN = "You are a helpful AI assistant."

    def test_exact_phrase_rewritten(self):
        self.assertEqual(T.sanitize_text(self.FP), self.CLEAN)

    def test_case_insensitive(self):
        self.assertEqual(T.sanitize_text(self.FP.upper()), self.CLEAN)
        self.assertEqual(T.sanitize_text(self.FP.lower()), self.CLEAN)

    def test_coding_agent_sentence_rewritten(self):
        self.assertEqual(T.sanitize_text(self.FP2), self.CLEAN)

    def test_built_on_variant_rewritten(self):
        self.assertEqual(T.sanitize_text("You are an AI assistant built on DeepSeek."),
                         self.CLEAN)

    def test_embedded_in_system_prompt(self):
        text = self.FP + "\n\nYou are a coding agent."
        out = T.sanitize_text(text)
        self.assertEqual(out, self.CLEAN + "\n\nYou are a coding agent.")
        self.assertFalse(T.has_fingerprint(out))

    def test_idempotent(self):
        once = T.sanitize_text(self.FP)
        self.assertEqual(T.sanitize_text(once), once)

    def test_partial_phrases_untouched(self):
        # 只有连续整句是已确认的指纹；不构成自述句的文本不整句替换。
        for text in ("You are an AI agent.",
                     "You are free to use tools built by the user."):
            self.assertEqual(T.sanitize_text(text), text)
            self.assertFalse(T.has_fingerprint(text))

    def test_sanitize_messages_covers_system_role(self):
        messages = [
            {"role": "system", "content": self.FP},
            {"role": "user", "content": "hi"},
        ]
        out = P.sanitize_messages(messages)
        self.assertEqual(out[0]["content"], self.CLEAN)
        self.assertEqual(out[1]["content"], "hi")

    def test_identity_rewrite_covers_system_developer_and_assistant(self):
        # assistant 历史里的身份句同样命中上游校验（#574：同一句 400/400/400，
        # 放进 user 则 200/200/200）。user / tool 仍是用户数据，不能改写（#537）。
        messages = [
            {"role": "system", "content": self.FP},
            {"role": "developer", "content": self.FP2},
            {"role": "user", "content": self.FP},
            {"role": "assistant", "content": self.FP2},
            {"role": "tool", "content": self.FP},
        ]
        out = P.sanitize_messages(messages)
        self.assertEqual(out[0]["content"], self.CLEAN)
        self.assertEqual(out[1]["content"], self.CLEAN)
        self.assertEqual(out[2]["content"], self.FP)
        self.assertEqual(out[3]["content"], self.CLEAN)
        self.assertEqual(out[4]["content"], self.FP)


class DshGuiParagraphTests(unittest.TestCase):
    """DSH 系统提示里 GUI 描述段的渠道指纹（#570）。

    A 句 + B 句连续出现即命中 11-128；实测改 A 句、改 B 句、或在两句之间插东西都会
    失配。#579 只调整 A 句尾的语序，实测仍 400；#574 进一步确认品牌词是必要条件，
    于是把散文里的 `DeepSeek Harness` 缩成 `DeepSeek`。两处都只动已确认的片段，
    连字符 / 下划线形式的路径与包名保持原样。
    """

    A = ('You are interacting with the user through the DeepSeek Harness Web GUI '
         'whose URL the runtime context names.')
    A_REWORDED = ('You are interacting with the user through the DeepSeek Web GUI '
                  'whose URL is named in the runtime context.')
    B = ('When the user refers to "this page", "this GUI", or "this app" without naming '
         'another target, they mean this GUI.')
    GUI = A + " " + B

    def test_gui_paragraph_in_system_reworded(self):
        out = P.sanitize_messages([{"role": "system", "content": self.GUI}])
        self.assertEqual(out[0]["content"], self.A_REWORDED + " " + self.B)

    def test_brand_word_reduced(self):
        # #579 只改语序，实测仍 400；#574 确认品牌词是必要条件，缩成
        # `DeepSeek` 后通过，句子其余部分不动。
        out = T.sanitize_text(self.GUI)
        self.assertIn("DeepSeek Web GUI", out)
        self.assertNotIn("DeepSeek Harness", out)
        self.assertNotIn("whose URL the runtime context names", out)

    def test_gui_paragraph_in_developer_reworded(self):
        out = P.sanitize_messages([{"role": "developer", "content": self.GUI}])
        self.assertEqual(out[0]["content"], self.A_REWORDED + " " + self.B)

    def test_user_and_tool_untouched(self):
        out = P.sanitize_messages([
            {"role": "user", "content": self.GUI},
            {"role": "tool", "content": self.GUI},
        ])
        self.assertEqual(out[0]["content"], self.GUI)
        self.assertEqual(out[1]["content"], self.GUI)

    def test_brand_word_reduced_outside_the_sentence(self):
        # 品牌词缩减不锚定句子：`DeepSeek Harness` 与 GUI 描述句组合才触发，
        # 但只缩一个词、语义不变，比按句子形状做特判更稳。
        for text, want in (("DeepSeek Harness", "DeepSeek"),
                           ("the DeepSeek Harness project", "the DeepSeek project"),
                           ("You are interacting with the user through the DeepSeek Harness CLI.",
                            "You are interacting with the user through the DeepSeek CLI.")):
            self.assertEqual(T.sanitize_text(text), want, text)

    def test_hyphenated_identifiers_untouched(self):
        # 路径 / 包名 / 仓库地址里的连字符形式必须原样保留。
        for text in (
            "github.com/deepseek-ai/deepseek-harness/tree/main",
            "import deepseek_harness as dsh",
            "/opt/deepseek-harness/app",
        ):
            self.assertEqual(T.sanitize_text(text), text, text)

    def test_case_insensitive_and_idempotent(self):
        out = T.sanitize_text(self.GUI.lower())
        self.assertEqual(
            out,
            self.GUI.lower().replace("whose url the runtime context names",
                                     "whose URL is named in the runtime context")
                          .replace("deepseek harness", "DeepSeek"))
        self.assertEqual(T.sanitize_text(out), out)

    def test_wire_body_carries_the_rewrite(self):
        # 走完整链路再断言一次：规则接上了还不够，得确认 build_upstream_body
        # 真的用它构造出站 body（漏接时单测全绿、线上照旧 400）。
        # 软打码默认开启会先把空格换成空格+零宽字符，品牌句不再命中这条改写；
        # 这里关掉软打码，单独钉住旧清洗路径。
        with mock.patch.dict(os.environ, {"WB_SOFT_MASK_ENABLED": "0"}):
            body = P.build_upstream_body({
                "model": "deepseek-v4.1-flash",
                "messages": [{"role": "system", "content": self.GUI},
                             {"role": "user", "content": "hi"}]})
        system = body["messages"][0]["content"]
        self.assertIn("DeepSeek Web GUI", system)
        self.assertNotIn("DeepSeek Harness", system)
        self.assertIn("whose URL is named in the runtime context", system)


class AssistantHistoryFingerprintTests(unittest.TestCase):
    """#574：assistant 历史里的身份句 / 品牌 + GUI 组合。

    同一句放在 user 里 200/200/200，放在 assistant 里 400/400/400；只把
    `DeepSeek Harness` 缩成 `DeepSeek` 就能让真实 129KB 请求通过。
    """

    IDENTITY = "You are an AI agent powered by DeepSeek Harness."
    CLEAN = "You are a helpful AI assistant."
    GUI = ('You are interacting with the user through the DeepSeek Harness Web GUI '
           'whose URL the runtime context names. When the user refers to "this page", '
           '"this GUI", or "this app", they mean this GUI.')

    def test_assistant_identity_sentence_rewritten(self):
        out = P.sanitize_messages([{"role": "assistant", "content": self.IDENTITY}])
        self.assertEqual(out[0]["content"], self.CLEAN)

    def test_assistant_gui_paragraph_rewritten(self):
        out = P.sanitize_messages([{"role": "assistant", "content": self.GUI}])
        self.assertNotIn("DeepSeek Harness", out[0]["content"])
        self.assertIn("DeepSeek Web GUI", out[0]["content"])

    def test_user_and_tool_history_untouched(self):
        out = P.sanitize_messages([
            {"role": "user", "content": self.IDENTITY},
            {"role": "tool", "content": self.GUI},
        ])
        self.assertEqual(out[0]["content"], self.IDENTITY)
        self.assertEqual(out[1]["content"], self.GUI)

    def test_hyphenated_paths_in_assistant_untouched(self):
        for text in ("github.com/deepseek-ai/deepseek-harness/tree/main",
                     "import deepseek_harness as dsh",
                     "/opt/deepseek-harness/app"):
            out = P.sanitize_messages([{"role": "assistant", "content": text}])
            self.assertEqual(out[0]["content"], text, text)

    def test_wire_body_carries_assistant_rewrite(self):
        # 软打码默认开启会先改空格，身份句不再命中旧改写；这里关掉软打码，
        # 单独钉住 assistant 历史的旧清洗路径。
        with mock.patch.dict(os.environ, {"WB_SOFT_MASK_ENABLED": "0"}):
            body = P.build_upstream_body({
                "model": "deepseek-v4.1-flash",
                "messages": [
                    {"role": "assistant", "content": self.IDENTITY},
                    {"role": "user", "content": self.IDENTITY},
                ]})
        by_role = {}
        for m in body["messages"]:
            by_role.setdefault(m.get("role"), m.get("content"))
        self.assertEqual(by_role["assistant"], self.CLEAN)
        self.assertEqual(by_role["user"], self.IDENTITY)


class IdentityVerbVariantsTests(unittest.TestCase):
    """同类自述句的其它动词搭配（built by / created by / …）与省略名词的形态。

    上游只按「自我介绍句式 + 品牌」判，不区分动词；1.7.18 只覆盖 powered by /
    built on 两种，补全动词表后这些变体同样应被中性化，而分词修饰宾语的句子
    （"tools built by …"）不能被误伤。
    """

    CLEAN = "You are a helpful AI assistant."

    def test_built_by_variant_rewritten(self):
        # 专名 + 逗号 + built by 的形态（客户端默认身份句常见写法）。
        self.assertEqual(
            T.sanitize_text("You are Example Agent, built by Example Corp. Be direct."),
            self.CLEAN + " Be direct.")

    def test_created_developed_made_trained_variants(self):
        for sentence in (
            "You are a coding assistant created by Example Corp.",
            "You are an AI agent developed by Example Corp.",
            "You are a chat model made by Example Corp.",
            "You are an assistant trained by Example Corp.",
        ):
            self.assertEqual(T.sanitize_text(sentence), self.CLEAN, sentence)

    def test_bare_powered_by_without_noun(self):
        self.assertEqual(
            T.sanitize_text("You are powered by the model named glm-5.3-flash."),
            self.CLEAN)

    def test_participle_modifying_object_is_untouched(self):
        for sentence in (
            "You are free to use tools built by the user.",
            "You are a coding agent pairing with the user inside their codebase.",
        ):
            self.assertEqual(T.sanitize_text(sentence), sentence, sentence)

    def test_variants_scoped_to_system_and_developer(self):
        fp = "You are Example Agent, built by Example Corp."
        out = P.sanitize_messages([
            {"role": "system", "content": fp},
            {"role": "developer", "content": fp},
            {"role": "user", "content": fp},
        ])
        self.assertEqual(out[0]["content"], self.CLEAN)
        self.assertEqual(out[1]["content"], self.CLEAN)
        self.assertEqual(out[2]["content"], fp)


class SoftMaskTests(unittest.TestCase):
    def test_apply_soft_mask_only_changes_spaces(self):
        self.assertEqual(T.apply_soft_mask("one two  three"),
                         "one \u200btwo \u200b \u200bthree")
        for value in (None, 1, {"text": "one two"}):
            self.assertEqual(T.apply_soft_mask(value), value)

    def test_roles_and_sensitive_terms(self):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "developer", "content": "developer prompt"},
            {"role": "user", "content": "mention DeepSeek Harness here"},
            {"role": "assistant", "content": "Tokens prefixed with @ are sensitive"},
            {"role": "user", "content": "ordinary user text"},
            {"role": "assistant", "content": "ordinary assistant text"},
        ]
        before = copy.deepcopy(messages)
        out = T.soft_mask_messages(messages)
        self.assertEqual(out[0]["content"], "system \u200bprompt")
        self.assertEqual(out[1]["content"], "developer \u200bprompt")
        self.assertEqual(out[2]["content"], "mention \u200bDeepSeek \u200bHarness \u200bhere")
        self.assertEqual(out[3]["content"],
                         "Tokens \u200bprefixed \u200bwith \u200b@ \u200bare \u200bsensitive")
        self.assertEqual(out[4]["content"], messages[4]["content"])
        self.assertEqual(out[5]["content"], messages[5]["content"])
        self.assertEqual(messages, before)

    def test_additional_sensitive_terms(self):
        for text in ("paths explicitly referenced by the user",
                     "resolve from the workspace root"):
            with self.subTest(text=text):
                out = T.soft_mask_messages([{"role": "user", "content": text}])
                self.assertEqual(out[0]["content"], T.apply_soft_mask(text))

    def test_tool_messages_and_calls_are_never_masked(self):
        arguments = '{"query":"DeepSeek Harness and Tokens prefixed with @"}'
        messages = [
            {"role": "assistant", "content": "DeepSeek Harness stays here",
             "tool_calls": [{"function": {"name": "run", "arguments": arguments}}]},
            {"role": "tool", "content": "tool result with spaces"},
            {"role": "function", "content": "legacy tool result"},
        ]
        for force in (False, True):
            with self.subTest(force=force):
                out = T.soft_mask_messages(messages, force=force)
                self.assertEqual(out, messages)
                self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], arguments)

    def test_multipart_masks_only_text_parts(self):
        image = {"type": "image_url", "image_url": {"url": "https://example.invalid/a b"}}
        out = T.soft_mask_messages([
            {"role": "system", "content": [{"type": "text", "text": "two words"}, image]},
            {"role": "user", "content": [{"type": "text", "text": "DeepSeek Harness"}, image]},
        ])
        self.assertEqual(out[0]["content"][0]["text"], "two \u200bwords")
        self.assertEqual(out[1]["content"][0]["text"], "DeepSeek \u200bHarness")
        self.assertIs(out[0]["content"][1], image)
        self.assertIs(out[1]["content"][1], image)

    def test_force_masks_ordinary_roles_but_not_tools(self):
        out = T.soft_mask_messages([
            {"role": "user", "content": "ordinary user text"},
            {"role": "assistant", "content": "ordinary assistant text"},
            {"role": "tool", "content": "ordinary tool text"},
        ], force=True)
        self.assertEqual(out[0]["content"], "ordinary \u200buser \u200btext")
        self.assertEqual(out[1]["content"], "ordinary \u200bassistant \u200btext")
        self.assertEqual(out[2]["content"], "ordinary tool text")

    def test_wire_body_opt_in_and_sensitive_assistant(self):
        payload = {"model": "deepseek-v4.1-flash", "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "assistant", "content": "DeepSeek Harness history"},
            {"role": "user", "content": "ordinary user text"},
        ]}
        with mock.patch.dict(os.environ, {"WB_SOFT_MASK_ENABLED": "0"}):
            body = P.build_upstream_body(payload)
            self.assertNotIn("\u200b", body["messages"][0]["content"])
        with mock.patch.dict(os.environ, {"WB_SOFT_MASK_ENABLED": "1"}):
            body = P.build_upstream_body(payload)
            self.assertEqual(body["messages"][0]["content"], "system \u200bprompt")
            self.assertEqual(body["messages"][1]["content"],
                             "DeepSeek \u200bHarness \u200bhistory")
            self.assertEqual(body["messages"][2]["content"], "ordinary user text")

    def test_panel_setting_defaults_on_and_off_stops_masking(self):
        payload = {"model": "deepseek-v4.1-flash", "messages": [
            {"role": "system", "content": "system prompt"}]}
        old_dir = P.ACCOUNTS_DIR
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"WB_SOFT_MASK_ENABLED": ""}):
            P.ACCOUNTS_DIR = directory
            try:
                self.assertTrue(wb_settings.soft_mask_enabled(directory))
                body = P.build_upstream_body(payload)
                self.assertEqual(body["messages"][0]["content"],
                                 "system \u200bprompt")
                wb_settings.set_soft_mask_enabled(directory, False)
                self.assertFalse(wb_settings.soft_mask_enabled(directory))
                body = P.build_upstream_body(payload)
                self.assertEqual(body["messages"][0]["content"], "system prompt")
                with mock.patch.dict(os.environ, {"WB_SOFT_MASK_ENABLED": "1"}):
                    body = P.build_upstream_body(payload)
                    self.assertEqual(body["messages"][0]["content"],
                                     "system \u200bprompt")
            finally:
                P.ACCOUNTS_DIR = old_dir


class SoftMaskRetryTests(unittest.TestCase):
    def make_pool(self, account):
        class Pool(object):
            accounts = [account]
            affinity = type("Affinity", (), {
                "unbind": lambda self, key: None,
                "demote": lambda self, key, uid=None: None,
            })()

            def count_ready(self, realm, model=None):
                return 1

            def pick_for_session(self, realm, session_key=None, exclude=(),
                                 model=None, page=None):
                return account

            def apply_remaining_weights(self, weights=None):
                return weights or {}

        return Pool()

    def run_request(self, payload, failures, realm="intl"):
        bodies = []

        class Response(object):
            def close(self):
                pass

        def fake_urlopen(req, timeout=30, proxy=""):
            bodies.append(json.loads(req.data.decode("utf-8")))
            index = len(bodies) - 1
            if index < len(failures):
                raise urllib.error.HTTPError(
                    "https://upstream.invalid", 400, "Bad Request", {},
                    io.BytesIO(failures[index].encode("utf-8")))
            return Response()

        with tempfile.TemporaryDirectory() as directory:
            account = wb_accounts.Account(
                {"uid": "soft-mask-retry", "accessToken": "t", "realm": realm},
                os.path.join(directory, "soft-mask-retry.json"))
            with mock.patch.object(P, "POOL", self.make_pool(account)), \
                    mock.patch.object(P, "ACCOUNTS_DIR", directory), \
                    mock.patch.object(wb_accounts, "urlopen", fake_urlopen), \
                    mock.patch.dict(os.environ, {"WB_SOFT_MASK_ENABLED": "1"}):
                try:
                    response, _account, _effort = P.open_upstream(
                        payload, target_realm=realm)
                    response.close()
                    return bodies, None
                except urllib.error.HTTPError as exc:
                    return bodies, exc

    def payload(self):
        arguments = '{"query":"keep every space"}'
        return {"model": "deepseek-v4.1-flash", "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "ordinary user text"},
            {"role": "assistant", "content": "ordinary assistant text"},
            {"role": "assistant", "content": "tool call content",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "run", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "tool result text"},
        ]}

    def test_11128_retries_once_with_all_ordinary_content_masked(self):
        bodies, error = self.run_request(
            self.payload(), ['{"code":11128,"message":"illegal channel"}'])
        self.assertIsNone(error)
        self.assertEqual(len(bodies), 2)
        first, second = bodies
        self.assertEqual(first["messages"][0]["content"], "system \u200bprompt")
        self.assertEqual(first["messages"][1]["content"], "ordinary user text")
        self.assertEqual(second["messages"][1]["content"], "ordinary \u200buser \u200btext")
        self.assertEqual(second["messages"][2]["content"],
                         "ordinary \u200bassistant \u200btext")
        self.assertEqual(second["messages"][3]["content"], "tool call content")
        self.assertEqual(second["messages"][3]["tool_calls"][0]["function"]["arguments"],
                         '{"query":"keep every space"}')
        self.assertEqual(second["messages"][4]["content"], "tool result text")

    def test_11128_retry_also_applies_to_china_realm(self):
        bodies, error = self.run_request(
            self.payload(), ['{"code":11128,"message":"illegal channel"}'],
            realm="cn")
        self.assertIsNone(error)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[1]["messages"][1]["content"],
                         "ordinary \u200buser \u200btext")

    def test_second_11128_is_returned_without_another_retry(self):
        failure = '{"code":11128,"message":"illegal channel"}'
        bodies, error = self.run_request(self.payload(), [failure, failure])
        self.assertIsNotNone(error)
        self.assertEqual(len(bodies), 2)
        self.assertIn("11128", P.read_http_error(error).decode("utf-8"))

    def test_unrelated_400_is_not_retried(self):
        bodies, error = self.run_request(
            self.payload(), ['{"code":11101,"message":"bad request"}'])
        self.assertIsNotNone(error)
        self.assertEqual(len(bodies), 1)


if __name__ == "__main__":
    unittest.main()

