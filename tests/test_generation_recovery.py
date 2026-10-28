"""Regress interrupted generation and bounded recovery (no real network)."""

import http.client
import io
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from generator_runtime import GenerationResult, GeneratorRuntime


class FakeResponse:
    def __init__(self, text="ok", finish_reason="stop", fail_read=None, raw=None):
        self.text = text
        self.finish_reason = finish_reason
        self.fail_read = fail_read
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if self.fail_read is not None:
            raise self.fail_read
        if self.raw is not None:
            return self.raw
        return json.dumps({
            "choices": [{
                "message": {"content": self.text},
                "finish_reason": self.finish_reason,
            }]
        }).encode("utf-8")


def state():
    return {
        "utterance_plan": {
            "contract": "mouth_only_v1",
            "ready": True,
            "speech_act": "return_greeting",
            "content_units": [{
                "kind": "communicative_act",
                "value": "Return greeting.",
            }],
            "style": {"language": "ko"},
        }
    }


def setup_runtime(**extra):
    environment = {
        "FLYGPT_GENERATOR_URL": "https://example.invalid/v1/chat/completions",
        "FLYGPT_GENERATOR_MODEL": "test-model",
        "FLYGPT_GENERATOR_RETRY_DELAY": "0",
        "FLYGPT_GENERATOR_MAX_TOKENS": "128",
    }
    environment.update(extra)
    with patch.dict(os.environ, environment, clear=True):
        return GeneratorRuntime()


class RuntimeRecoveryTests(unittest.TestCase):
    def test_real_http_body_deadline_and_normal_completion(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = json.dumps({"choices": [{"message": {"content": "안녕하세요!"},
                                               "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    if self.path == "/slow":
                        for byte in raw:
                            self.wfile.write(bytes([byte]))
                            self.wfile.flush()
                            time.sleep(0.02)
                    else:
                        self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            runtime = setup_runtime()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with patch("generator_runtime.urllib.request.urlopen", side_effect=opener.open):
                runtime.url = f"http://127.0.0.1:{server.server_port}/ok"
                normal = runtime._generate_once(model="test", messages=[], timeout_seconds=1)
                self.assertTrue(normal.used)
                self.assertEqual(normal.answer, "안녕하세요!")
                runtime.url = f"http://127.0.0.1:{server.server_port}/slow"
                started = time.perf_counter()
                slow = runtime._generate_once(model="test", messages=[], timeout_seconds=0.15)
                elapsed = time.perf_counter() - started
                self.assertEqual(slow.error_code, "timeout")
                self.assertFalse(slow.used)
                self.assertIsNone(slow.answer)
                self.assertLess(elapsed, 1.0)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_late_success_is_timeout_and_does_not_retry(self):
        runtime = setup_runtime()
        clock = [0.0]

        def late_success(**kwargs):
            clock[0] = 8.0
            return GenerationResult(used=True, provider="test", model="test-model",
                                    answer="안녕하세요!")

        with patch("generator_runtime.time.perf_counter", side_effect=lambda: clock[0]), \
             patch.object(runtime, "_generate_once", side_effect=late_success) as generate:
            result = runtime.generate("안녕", "general", brain_state=state(), budget_seconds=7)
        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error_code, "timeout")
        self.assertEqual(result.attempts, 1)
        generate.assert_called_once()

    def test_expired_budget_makes_no_provider_request(self):
        runtime = setup_runtime()
        with patch.object(runtime, "_generate_once") as generate:
            result = runtime.generate("안녕", "general", brain_state=state(), budget_seconds=0)
        self.assertEqual(result.error_code, "timeout")
        self.assertEqual(result.attempts, 0)
        generate.assert_not_called()

    def test_remote_disconnect_during_body_read_is_retried(self):
        calls = 0

        def open_response(request, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return FakeResponse(fail_read=http.client.IncompleteRead(b"partial", 99))
            return FakeResponse(text="복구 성공!")

        with patch("generator_runtime.urllib.request.urlopen", side_effect=open_response):
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertTrue(result.used)
        self.assertEqual(result.answer, "복구 성공!")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(calls, 2)

    def test_timeout_during_body_read_is_retried(self):
        calls = 0

        def open_response(request, timeout):
            nonlocal calls
            calls += 1
            return FakeResponse(
                fail_read=TimeoutError("timed out") if calls == 1 else None,
                text="정상 응답",
            )

        with patch("generator_runtime.urllib.request.urlopen", side_effect=open_response):
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertTrue(result.used)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.answer, "정상 응답")

    def test_rate_limit_is_retryable_only_once(self):
        calls = 0

        def open_response(request, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.HTTPError(
                    url="https://example.invalid", code=429, msg="Rate limited",
                    hdrs=None, fp=io.BytesIO(b'{"error":{"message":"rate limited"}}'),
                )
            return FakeResponse(text="다시 응답!")

        with patch("generator_runtime.urllib.request.urlopen", side_effect=open_response):
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertTrue(result.used)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(calls, 2)

    def test_bad_json_response_retries_once(self):
        calls = 0

        def open_response(request, timeout):
            nonlocal calls
            calls += 1
            return FakeResponse(raw=b"{broken" if calls == 1 else None, text="유효한 응답")

        with patch("generator_runtime.urllib.request.urlopen", side_effect=open_response):
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertTrue(result.used)
        self.assertEqual(result.attempts, 2)

    def test_length_truncation_retries_with_more_tokens(self):
        tokens = []

        def open_response(request, timeout):
            payload = json.loads(request.data)
            tokens.append(payload["max_tokens"])
            if len(tokens) == 1:
                return FakeResponse(text="잘린 부분", finish_reason="length")
            return FakeResponse(text="완성된 답변")

        with patch("generator_runtime.urllib.request.urlopen", side_effect=open_response):
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertTrue(result.used)
        self.assertEqual(result.answer, "완성된 답변")
        self.assertEqual(tokens, [128, 256])
        self.assertEqual(result.attempts, 2)

    def test_repeated_length_truncation_never_shows_partial_as_complete(self):
        with patch("generator_runtime.urllib.request.urlopen", return_value=FakeResponse(
            text="답변의 앞부분만...", finish_reason="length"
        )) as opener:
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertFalse(result.used)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error_code, "truncated")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(opener.call_count, 2)

    def test_persistent_disconnect_fails_without_unhandled_exception(self):
        with patch("generator_runtime.urllib.request.urlopen", return_value=FakeResponse(
            fail_read=http.client.RemoteDisconnected("Remote end closed connection")
        )) as opener:
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertFalse(result.used)
        self.assertEqual(result.error_code, "connection_lost")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(opener.call_count, 2)

    def test_nonretryable_auth_failure_does_not_retry(self):
        def fail(request, timeout):
            raise urllib.error.HTTPError(
                url="https://example.invalid", code=401, msg="Unauthorized",
                hdrs=None, fp=io.BytesIO(b'{"error":{"message":"invalid key"}}'),
            )

        with patch("generator_runtime.urllib.request.urlopen", side_effect=fail) as opener:
            result = setup_runtime().generate("안녕", "general", brain_state=state())

        self.assertFalse(result.used)
        self.assertEqual(result.error_code, "http_401")
        self.assertEqual(result.attempts, 1)
        opener.assert_called_once()


if __name__ == "__main__":
    unittest.main()
