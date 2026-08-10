"""
web/server.py — Local Web Demo Interface
R26-CS-012: Context-Aware Masking + Instruction Engine

Stdlib-only HTTP server (no external dependencies, matching engine/'s
design) that serves a small static frontend and a JSON API exercising
the REAL detection pipeline — not a reimplementation.

Usage:
    python masking_engine/web/server.py
    (open the printed URL in a browser)
"""

import sys
import os
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.normalizer import normalize
from engine.detector import detect, PATTERNS, NER_KEYWORDS, COMPILED_NER
from engine.confidence_scorer import score_all, resolve_overlapping_entities
from engine.masker import mask
from engine.token_registry import TokenRegistry
from main import DEMO_CASES

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8765

# entity_type -> raw regex pattern string, for the structured PATTERNS registry
_REGEX_PATTERN_BY_TYPE = {entity_type: pattern for entity_type, pattern, *_ in PATTERNS}


def _matched_pattern_for(entity) -> str:
    """Best-effort lookup of the exact regex responsible for a detection,
    for display purposes only (does not affect scoring/masking)."""
    et = entity.entity_type
    if et in _REGEX_PATTERN_BY_TYPE:
        return _REGEX_PATTERN_BY_TYPE[et]

    # NER types can have multiple candidate patterns per type — find the
    # one whose match span lines up with this entity's span.
    if et in NER_KEYWORDS:
        for raw, compiled in zip(NER_KEYWORDS[et], COMPILED_NER[et]):
            for m in compiled.finditer(entity.context_source or ""):
                if m.start() == entity.start and m.end() == entity.end:
                    return raw
        return NER_KEYWORDS[et][0]

    return "(unknown)"


def run_pipeline(text: str) -> dict:
    norm = normalize(text)
    raw_entities = detect(norm["normalized"], norm["despaced"], norm["despaced_map"])

    # Attach the normalized text to each entity so _matched_pattern_for
    # can re-locate NER matches without threading an extra parameter
    # through detect()'s public signature.
    for e in raw_entities:
        e.context_source = norm["normalized"]

    scored_entities = score_all(raw_entities, norm["normalized"])
    scored_entities = resolve_overlapping_entities(scored_entities)

    registry = TokenRegistry()
    registry.next_prompt()
    masked_result = mask(norm["normalized"], scored_entities, registry)

    entities_out = []
    for s in scored_entities:
        e = s.entity
        entities_out.append({
            "entity_type": e.entity_type,
            "value": e.value,
            "start": e.start,
            "end": e.end,
            "category": e.category,
            "sensitivity": e.sensitivity,
            "matched_pattern": _matched_pattern_for(e),
            "score": s.score,
            "score_breakdown": s.score_breakdown,
            "confidence_level": s.confidence_level,
            "action": s.action,
        })
    entities_out.sort(key=lambda x: x["start"])

    result = {
        "original": text,
        "normalized": norm["normalized"] if norm["normalized"] != text else None,
        "despaced": norm["despaced"] if norm["despaced"] != norm["normalized"] else None,
        "transformations": norm["transformations"],
        "context_type": norm["context_type"],
        "masked_text": masked_result.masked_text,
        "overall_risk": masked_result.overall_risk,
        "entities": entities_out,
    }
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout clean; errors still surface via do_GET/do_POST exceptions

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename: str, content_type: str):
        path = os.path.join(WEB_DIR, filename)
        if not os.path.isfile(path):
            self.send_error(404, f"{filename} not found")
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/":
            self._send_static("index.html", "text/html; charset=utf-8")
        elif route == "/style.css":
            self._send_static("style.css", "text/css; charset=utf-8")
        elif route == "/app.js":
            self._send_static("app.js", "application/javascript; charset=utf-8")
        elif route == "/api/demo-cases":
            self._send_json({"cases": DEMO_CASES})
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/detect":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body or b"{}")
            text = payload.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send_json({"error": "Provide non-empty 'text'."}, status=400)
                return
            result = run_pipeline(text)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"R26-CS-012 Masking Engine — web demo running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
