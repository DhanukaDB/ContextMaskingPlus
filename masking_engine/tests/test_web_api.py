"""
test_web_api.py — HTTP/JSON layer tests for web/server.py
R26-CS-012: Context-Aware Masking + Instruction Engine

Starts the real server on an ephemeral port in a background thread and
hits it over HTTP. Detection *logic* is already covered by
test_detector.py's 29 cases — this only checks that the web layer
(routing, JSON serialization, error handling) correctly wires requests
through to the real engine and back.
"""

import json
import os
import sys
import threading
import unittest
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"))

from http.server import ThreadingHTTPServer
import web.server as server_module


class TestWebAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def _get(self, path):
        with urllib.request.urlopen(self.base_url + path) as resp:
            return resp.status, dict(resp.headers), resp.read()

    def test_index_page_served(self):
        status, headers, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))

    def test_static_assets_served(self):
        status, headers, body = self._get("/style.css")
        self.assertEqual(status, 200)
        status, headers, body = self._get("/app.js")
        self.assertEqual(status, 200)

    def test_demo_cases_endpoint(self):
        status, headers, body = self._get("/api/demo-cases")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn("cases", data)
        self.assertGreater(len(data["cases"]), 0)
        self.assertIn("label", data["cases"][0])
        self.assertIn("text", data["cases"][0])

    def test_detect_valid_lk_phone_number(self):
        status, data = self._post("/api/detect", {"text": "0771234567"})
        self.assertEqual(status, 200)
        types = {e["entity_type"] for e in data["entities"]}
        self.assertIn("PHONE_LK", types)

    def test_detect_returns_matched_pattern_and_score_breakdown(self):
        status, data = self._post("/api/detect", {"text": "Call 0771234567 to confirm."})
        self.assertEqual(status, 200)
        phone = next(e for e in data["entities"] if e["entity_type"] == "PHONE_LK")
        self.assertIn("matched_pattern", phone)
        self.assertTrue(phone["matched_pattern"])
        self.assertIn("score_breakdown", phone)
        for factor in ("pattern_strength", "keyword_proximity", "co_occurrence", "format_validity"):
            self.assertIn(factor, phone["score_breakdown"])

    def test_detect_masked_text_present(self):
        status, data = self._post("/api/detect", {"text": "CVV 123 on file for the card."})
        self.assertEqual(status, 200)
        self.assertIn("masked_text", data)
        self.assertIn("overall_risk", data)

    def test_detect_empty_text_returns_400(self):
        status, data = self._post("/api/detect", {"text": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_detect_no_entities_for_plain_text(self):
        status, data = self._post("/api/detect", {"text": "Hello, how are you today?"})
        self.assertEqual(status, 200)
        self.assertEqual(data["entities"], [])

    def test_unknown_route_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/nonexistent")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
