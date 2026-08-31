"""
ml_anomaly.py — ML Safety-Net Layer (Hybrid Detection, Layer 2)
R26-CS-012: Context-Aware Masking + Instruction Engine

WHY THIS EXISTS (read alongside detector.py's "WHY REGEX + RULE-BASED NER"):
  Regex/rule-based detection (Layer 1) is deterministic, auditable, and
  covers every KNOWN sensitive-data shape defined in the taxonomy — but by
  construction it can only catch a pattern someone already wrote a rule
  for. It cannot generalize to a NOVEL secret format, an unlisted keyword,
  or content obfuscated just enough to dodge a specific regex.

  That is exactly the shape of failure behind real breaches such as the
  2022 Optus incident: a single detection/access-control layer with one
  gap was the whole security boundary, so one uncovered path was enough
  for ~9.8M customer records (including passport/driver's-licence numbers)
  to be exposed via an unauthenticated API. Defense in depth means no
  single layer's blind spot is the whole story for the system.

  Layer 2 here is a statistical classifier (scikit-learn, trained in
  research/Colab_Masking_Engine_Lab.ipynb on the SAME synthetic dataset
  used to evaluate Layer 1 — see evaluate.py) on 17 structural/entropy
  features (NOT raw text — no memorised literal secrets, see Section 3 of
  that notebook), used ONLY as a low-trust second opinion:

    - It NEVER produces a mask_immediate / CRITICAL action, and it never
      decides WHAT to redact from the model's output alone. When the
      classifier's probability clears ANOMALY_THRESHOLD and the structural
      evidence gate agrees, a SEPARATE deterministic rule
      (_find_suspicious_spans — the same regex-shaped signals the gate
      already checks: base64/hex runs, sk-/AKIA-prefixed strings, long
      opaque tokens) locates the specific span to redact, tokenized
      through the same TokenRegistry Layer 1 uses. If no such span can be
      located, it falls back to a review-only flag instead of guessing.
      So the ML score only ever answers "should Layer 2 act at all" — WHAT
      gets redacted still traces to a named, explainable rule, preserving
      the auditability property that ruled out ML as the PRIMARY detector.
      Every actual masking decision in the product still traces to either
      a named regex/NER rule (detector.py) or this named span rule.
    - It only acts when Layer 1 found nothing at all — so it's strictly
      additive recall for prompts that "look" sensitive by shape even
      though no known rule matched, never a second vote on something
      Layer 1 already handled.
    - It is optional at runtime: engine/ is otherwise a zero-dependency
      package (masking_engine/README.md) so it can run in an air-gapped
      environment; if scikit-learn/joblib or the trained model file are
      unavailable, the engine silently continues with Layer 1 only. A
      security control that hard-fails when an optional enhancement is
      missing is itself a bad security property — availability matters as
      much as detection here (see Model_Regex_Docs.md Section 6).

  Use case in one line: catch the "unknown unknowns" — content that is
  structurally sensitive-shaped (high entropy, long opaque tokens, secret-
  adjacent vocabulary) but doesn't match any of the ~30 named patterns —
  and flag it for human review instead of silently letting it through.
"""

import os
import re
import math
from typing import Optional

_MODEL = None
_MODEL_LOAD_ATTEMPTED = False
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research", "models_rf_classifier.pkl",
)

# Only surface a flag when the classifier is this confident — below this,
# the noise isn't worth a reviewer's attention. Chosen conservatively
# (higher than a bare 0.5 decision boundary) because false positives here
# cost a human review cycle, not a masking decision.
ANOMALY_THRESHOLD = 0.75


def _load_model():
    """Lazy, memoised, best-effort load. Never raises — a missing/broken
    optional dependency must never break the compliance-critical Layer 1
    path (see module docstring)."""
    global _MODEL, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL
    _MODEL_LOAD_ATTEMPTED = True
    try:
        import joblib
        if os.path.isfile(_MODEL_PATH):
            _MODEL = joblib.load(_MODEL_PATH)
    except Exception:
        _MODEL = None
    return _MODEL


def is_available() -> bool:
    """Whether the Layer-2 safety net can actually run in this environment."""
    return _load_model() is not None


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {c: text.count(c) / len(text) for c in set(text)}
    return -sum(p * math.log2(p) for p in freq.values() if p > 0)


# Deliberately duplicated from research/build_notebook.py's Section 3 /
# research/test_model.py rather than imported: engine/ must stay
# self-contained (its one dependency is the optional joblib/sklearn pair
# imported lazily above), and this feature set is a frozen training-time
# contract — it must match what models_rf_classifier.pkl was fit on
# exactly, in exactly this key order, independent of anything research/
# changes later.
def _extract_features(text: str) -> dict:
    tokens = text.split()
    entropies = [_shannon_entropy(t) for t in tokens] if tokens else [0]
    long_tokens = [t for t in tokens if len(t) > 15]
    return {
        'mean_token_entropy':  float(sum(entropies) / len(entropies)),
        'max_token_entropy':   float(max(entropies)),
        'high_entropy_ratio':  sum(1 for e in entropies if e > 3.5) / max(len(entropies), 1),
        'text_length':         len(text),
        'token_count':         len(tokens),
        'long_token_ratio':    len(long_tokens) / max(len(tokens), 1),
        'avg_token_length':    float(sum(len(t) for t in tokens) / len(tokens)) if tokens else 0,
        'n_digit_runs':        len(re.findall(r'\d{6,}', text)),
        'n_special_chars':     len(re.findall(r'[^a-zA-Z0-9\s]', text)),
        'n_at_symbols':        text.count('@'),
        'n_slashes':           text.count('/'),
        'n_equals':            text.count('='),
        'has_base64_pattern':  int(bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text))),
        'has_hex_run':         int(bool(re.search(r'[0-9a-fA-F]{12,}', text))),
        'has_bearer_keyword':  int(bool(re.search(r'\b(token|key|secret|password|bearer|jwt|api)\b', text, re.I))),
        'has_akia_prefix':     int('AKIA' in text),
        'has_sk_prefix':       int(bool(re.search(r'\bsk-[A-Za-z0-9]', text))),
    }


def score_anomaly(text: str) -> Optional[float]:
    """
    Returns P(sensitive) in [0,1] from the trained classifier, or None if
    the model/its dependencies aren't available in this environment —
    callers must treat None as "Layer 2 skipped", not as a low score.
    """
    model = _load_model()
    if model is None:
        return None
    try:
        import numpy as np
        features = _extract_features(text)
        X = np.array([list(features.values())])
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0][1])
        return float(model.predict(X)[0])
    except Exception:
        return None


# EVALUATED, DOCUMENTED LIMITATION — read before changing ANOMALY_THRESHOLD:
# the trained classifier scores 98.75% "accuracy" on the synthetic dataset
# it was trained/validated on (see evaluate.py's ml_safety_net metrics),
# but that number is inflated by the dataset's own template structure —
# it does NOT generalize cleanly out-of-distribution. Empirically, ordinary
# benign sentences with none of the 17 features' structural secret-markers
# (e.g. "The weather in Colombo today is sunny...") score 0.93-0.98
# "sensitive" from the classifier ALONE, which would make Layer 2 an
# alert-fatigue liability rather than a safety net if trusted by itself.
#
# So this layer requires the classifier's probability AND independent
# structural evidence (an actual secret-shaped signal from the same 17
# features) to agree before it is allowed to surface anything — never
# trust one opaque signal alone, the same principle behind requiring a
# second control before a high-stakes action. This is a deliberate,
# evaluated design choice, not an oversight; see docs/doc.md's ML section
# for the full write-up and the false-positive example above.
_STRUCTURAL_ANOMALY_KEYS = (
    "has_base64_pattern", "has_hex_run", "has_bearer_keyword",
    "has_akia_prefix", "has_sk_prefix",
)


def _has_structural_evidence(features: dict) -> bool:
    return (
        any(features[k] for k in _STRUCTURAL_ANOMALY_KEYS)
        or features["high_entropy_ratio"] > 0.3
        or features["long_token_ratio"] > 0.2
    )


# ─────────────────────────────────────────────
# SPAN LOCATION — deterministic, not the model
# ─────────────────────────────────────────────
#
# The classifier's 17 features are all AGGREGATE (mean entropy, ratios,
# counts) — it has no notion of "this substring, at this offset, is the
# secret." So once the score+gate above decide Layer 2 should act, a
# SEPARATE named rule (not the model) has to decide WHAT to redact. These
# patterns intentionally mirror the shapes _has_structural_evidence already
# gates on, so "why was THIS span picked" always has a one-line, regex-
# based answer — the same auditability property every Layer-1 pattern has.
_SPAN_CANDIDATE_PATTERNS = (
    re.compile(r'\bsk-[A-Za-z0-9]{6,}\b'),        # OpenAI-style key shape
    re.compile(r'\bAKIA[A-Za-z0-9]{6,}\b'),        # AWS access-key shape
    re.compile(r'[A-Za-z0-9+/]{20,}={0,2}'),       # base64-shaped run
    re.compile(r'[0-9a-fA-F]{12,}'),               # hex run
    re.compile(r'\b[A-Za-z0-9+/_\-]{16,}\b'),      # generic long opaque token
)


def _find_suspicious_spans(text: str):
    """Locate the specific substring(s) that make `text` look secret-
    shaped, using deterministic regex shapes only. Overlapping/adjacent
    candidate spans are merged so a token matched by more than one pattern
    (e.g. a long hex-looking base64 run) is only redacted once. Returns a
    list of (start, end) tuples, possibly empty."""
    spans = [
        (m.start(), m.end())
        for pattern in _SPAN_CANDIDATE_PATTERNS
        for m in pattern.finditer(text)
        if m.end() - m.start() >= 8   # ignore trivial/incidental short matches
    ]
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def get_ml_flag(masked_result):
    """
    Returns the single Layer-2 record for this result — whichever shape it
    took — or None if Layer 2 didn't fire at all:
      - a dict from masked_entities (has a "replacement" key) if Layer 2
        located and masked a span, or
      - a dict from skipped_entities (reason starts with
        "ml_anomaly_flagged") if it fired but couldn't locate a span to
        redact, so it only flagged for review.
    Centralizes this lookup so callers (web/server.py, evaluate.py,
    canonical_adapter.py, tests) don't each re-derive it.
    """
    for m in masked_result.masked_entities:
        if m.get("entity_type") == "ML_FLAGGED_ANOMALY":
            return m
    for sk in masked_result.skipped_entities:
        if sk.get("reason", "").startswith("ml_anomaly_flagged"):
            return sk
    return None


def apply_safety_net(text: str, masked_result, registry=None) -> None:
    """
    Mutates `masked_result` in place. If Layer 1 (regex/NER) already
    masked something, stays quiet — Layer 2 only ever runs on a Layer-1
    miss (see module docstring).

    Otherwise: scores the prompt, and if the classifier AND the
    structural-evidence gate both agree it's sensitive-shaped, locates the
    specific secret-shaped span(s) via `_find_suspicious_spans` — a
    deterministic rule, not the model — and masks just those, tokenizing
    through `registry` exactly like a Layer-1 tokenized value (same
    session-idempotency guarantee). If `registry` isn't supplied, or the
    gate passed but no locatable span was found, falls back to a
    review-only flag in `skipped_entities` rather than guessing at what to
    redact or masking the whole prompt.
    """
    if masked_result.masked_entities:
        return  # Layer 1 already found something; Layer 2 stays quiet.

    model = _load_model()
    if model is None:
        return

    features = _extract_features(text)
    try:
        import numpy as np
        X = np.array([list(features.values())])
        score = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") \
            else float(model.predict(X)[0])
    except Exception:
        return

    if score < ANOMALY_THRESHOLD or not _has_structural_evidence(features):
        return

    spans = _find_suspicious_spans(text) if registry is not None else []

    if not spans:
        masked_result.skipped_entities.append({
            "entity_type": "ML_FLAGGED_ANOMALY",
            "value"      : "(whole-prompt signal — no specific span located; see instructions)",
            "reason"     : "ml_anomaly_flagged_no_span",
            "score"      : round(score, 3),
        })
        return

    masked_text = masked_result.masked_text
    for start, end in sorted(spans, reverse=True):
        original_value = masked_text[start:end]
        replacement = registry.get_or_create_token(original_value, "ML_FLAGGED_ANOMALY")
        masked_text = masked_text[:start] + replacement + masked_text[end:]
        masked_result.masked_entities.append({
            "entity_type": "ML_FLAGGED_ANOMALY",
            "original"   : original_value,
            "replacement": replacement,
            "strategy"   : "tokenize",
            "sensitivity": "MEDIUM",
            "score"      : round(score, 3),
            "action"     : "ml_flagged_mask",
        })
    masked_result.masked_text = masked_text

    risk_priority = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    if risk_priority.get(masked_result.overall_risk, 0) < risk_priority["MEDIUM"]:
        masked_result.overall_risk = "MEDIUM"
