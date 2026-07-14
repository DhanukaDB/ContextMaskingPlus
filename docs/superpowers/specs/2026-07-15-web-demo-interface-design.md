# Web Demo Interface — Design Spec

**Date:** 2026-07-15
**Status:** Approved (design confirmed via user action — attempted to run the server before it existed)

## Purpose

A local, browser-based demo interface for the masking engine: submit a text prompt, see exactly what the real `engine/` pipeline detects, why (which regex/pattern, which taxonomy category), how confident it is (4-factor score breakdown), and the resulting masked output. For manually exercising the engine and demonstrating it (e.g. to a supervisor) without the terminal-only `main.py` CLI.

## Architecture

Stdlib-only Python HTTP server + static HTML/CSS/vanilla-JS frontend. No new dependencies — reuses `engine/` directly, matching its zero-dependency design.

```
masking_engine/web/
├── server.py       # http.server.BaseHTTPRequestHandler; serves static files + JSON API
├── index.html       # page structure: explainer + input + results panel
├── style.css
└── app.js           # fetch() calls, renders results
```

## Backend (`server.py`)

- `GET /` and static files (`style.css`, `app.js`) served from the `web/` directory.
- `GET /api/demo-cases` → returns `main.py`'s existing `DEMO_CASES` list (label + text) as JSON, for one-click quick-fill buttons.
- `POST /api/detect` → body `{"text": "..."}`. Runs `normalize → detect → score_all → resolve_overlapping_entities → mask` (imported directly from `engine/`). Returns JSON:
  - `original`, `normalized`, `despaced` (only if they differ from original — surfaces adversarial-normalization behavior)
  - `masked_text`, `overall_risk`
  - `entities`: list of `{entity_type, value, start, end, category, matched_pattern, score, score_breakdown: {pattern_strength, keyword_proximity, co_occurrence, format_validity}, confidence_level, action}`
    - `matched_pattern` is looked up from `detector.PATTERNS`/`NER_KEYWORDS` by entity type — the actual regex string (or `"NER: <description>"` for name/address/DOB matches) that fired.
- Port 8765 (arbitrary, unlikely to collide). Server prints the URL on startup.
- Each request creates its own `TokenRegistry` (stateless per request — no cross-request idempotency; this is a single-shot inspector, not a session simulator).

## Frontend

- Collapsible explainer section: 5-stage pipeline overview, 7 taxonomy categories (per earlier approval).
- Textarea + "Analyze" button + quick-fill buttons (from `/api/demo-cases`).
- Results panel: masked output first (prominent), then one card per detected entity showing type, value, matched pattern, category, score + factor breakdown, confidence level, action — color-coded by risk level (CRITICAL/HIGH/MEDIUM/LOW), matching `main.py`'s terminal color convention.
- Empty state ("no entities detected") and a visible error banner if the fetch fails.

## Testing

`masking_engine/tests/test_web_api.py`: starts the server in a background thread bound to an ephemeral port, hits `/api/detect` with known inputs (reusing a couple of the phone-number/CVV cases already covered by `test_detector.py`) to verify the HTTP/JSON serialization layer works end-to-end. Does not re-test detection logic itself (already covered by the 29 existing tests).

## Out of scope

Session/token-registry persistence across requests, authentication, deployment beyond localhost, styling frameworks/build tooling.
