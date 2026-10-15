"""The trace stream must reuse the actual chat route and never invent activity."""
import asyncio
import json
import unittest
from unittest.mock import patch

import app as api


async def collect_events(response):
    decoded = []
    async for raw in response.body_iterator:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        for record in raw.split("\n\n"):
            if not record.startswith("data: "):
                continue
            decoded.append(json.loads(record[6:]))
    return decoded


class ChatTraceStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_real_frames_and_one_final_answer(self):
        invocations = []

        def actual_chat(req, request):
            invocations.append(req.message)
            api._emit_live_trace({"type": "status", "phase": "router_loading"})
            api._emit_live_trace({
                "type": "trace",
                "frame": {"stage": "input", "activations": [0.1, 1.0]},
            })
            api._emit_live_trace({
                "type": "trace",
                "frame": {"stage": "step_1", "activations": [0.3, 1.0]},
            })
            api._emit_live_trace({
                "type": "route",
                "route": "general", "confidence": 0.8,
                "accepted": True, "semantic_override": False,
            })
            api._emit_live_trace({"type": "status", "phase": "generation"})
            return {
                "answer": "안녕!",
                "type": "dispatch",
                "router": {"route": "general", "trace": [
                    {"stage": "input", "activations": [0.1, 1.0]},
                    {"stage": "step_1", "activations": [0.3, 1.0]},
                ]},
            }

        with patch.object(api, "_require_user", return_value={"id": "test-user"}), \
             patch.object(api, "chat_endpoint", side_effect=actual_chat):
            response = await api.chat_stream_endpoint(api.ChatRequest(message="안녕"), object())
            events = await asyncio.wait_for(collect_events(response), timeout=5)

        self.assertEqual(invocations, ["안녕"])
        self.assertEqual([event["frame"]["stage"] for event in events if event["type"] == "trace"],
                         ["input", "step_1"])
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["response"]["answer"], "안녕!")

    async def test_deterministic_tool_does_not_fabricate_trace(self):
        def exact_tool(req, request):
            return {"answer": "🧮 3888784", "type": "math_fast_path"}

        with patch.object(api, "_require_user", return_value={"id": "test-user"}), \
             patch.object(api, "chat_endpoint", side_effect=exact_tool):
            response = await api.chat_stream_endpoint(api.ChatRequest(message="1972^2는?"), object())
            events = await asyncio.wait_for(collect_events(response), timeout=5)

        self.assertEqual([item["type"] for item in events], ["status", "result"])
        self.assertEqual(events[-1]["response"]["type"], "math_fast_path")

    async def test_authentication_fails_before_streaming(self):
        with patch.object(api, "_require_user", side_effect=api.HTTPException(status_code=401, detail="Not authorized")):
            with self.assertRaises(api.HTTPException) as caught:
                await api.chat_stream_endpoint(api.ChatRequest(message="안녕"), object())
        self.assertEqual(caught.exception.status_code, 401)

    async def test_provider_errors_are_not_returned_as_trace(self):
        def error_request(req, request):
            api._emit_live_trace({"type": "status", "phase": "router_loading"})
            raise api.HTTPException(status_code=503, detail="backend unavailable")
        with patch.object(api, "_require_user", return_value={"id": "test-user"}), \
             patch.object(api, "chat_endpoint", side_effect=error_request):
            response = await api.chat_stream_endpoint(api.ChatRequest(message="안녕"), object())
            events = await asyncio.wait_for(collect_events(response), timeout=5)
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["status"], 503)
        self.assertFalse(any(event["type"] == "trace" for event in events))


if __name__ == "__main__":
    unittest.main()
