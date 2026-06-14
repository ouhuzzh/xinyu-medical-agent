import os
import sys
import unittest
from unittest import mock

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.prod_acceptance_check import _parse_sse_events, _run_chat_smoke, main  # noqa: E402


class ProdAcceptanceCheckTests(unittest.TestCase):
    def test_parse_sse_events_extracts_event_names(self):
        body = (
            "event: session\n"
            'data: {"thread_id":"t1"}\n'
            "\n"
            "event: final\n"
            'data: {"done":true}\n'
            "\n"
        )

        events = _parse_sse_events(body)

        self.assertEqual([item["event"] for item in events], ["session", "final"])

    @mock.patch("scripts.prod_acceptance_check._request")
    @mock.patch("scripts.prod_acceptance_check._request_json")
    def test_run_chat_smoke_requires_final_without_app_error(self, mock_request_json, mock_request):
        mock_request_json.return_value = (200, {"thread_id": "thread-1"})
        mock_request.return_value = (
            200,
            "event: session\n"
            'data: {"thread_id":"thread-1"}\n'
            "\n"
            "event: final\n"
            'data: {"done":true}\n'
            "\n",
        )

        lines = _run_chat_smoke("https://api.example.com", "token", "hello")

        self.assertTrue(any("chat_session OK" in item for item in lines))
        self.assertTrue(any("chat_stream OK" in item for item in lines))

    @mock.patch("scripts.prod_acceptance_check._check_frontend", return_value=["frontend OK"])
    @mock.patch("scripts.prod_acceptance_check._check_public_api", return_value=["api_healthz OK"])
    def test_main_rejects_chat_smoke_without_token(self, _mock_public_api, _mock_frontend):
        rc = main(
            [
                "prod_acceptance_check.py",
                "--frontend-url",
                "https://medical.example.com",
                "--api-base-url",
                "https://api.example.com",
                "--chat-smoke",
            ]
        )

        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
