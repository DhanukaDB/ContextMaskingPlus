"""
detector.py — Entity Detection
R26-CS-012: Context-Aware Masking + Instruction Engine

Detects sensitive entities using:
  1. Compiled regex patterns (per taxonomy Category 1–7)
  2. Rule-based NER for FULL_NAME, HOME_ADDRESS, DATE_OF_BIRTH
     (lightweight substitute for spaCy — no external dependencies)

WHY REGEX + RULE-BASED NER (not ML model here):
  - Regex is deterministic and auditable — required for PCI-DSS/GDPR compliance
  - Each detection decision can be traced to a specific rule (explainability)
  - Zero external dependencies — runs in air-gapped banking environments
  - ML NER (spaCy) would be layered on top in production; rule-based is
    sufficient for a progress demo and covers the core research scope
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sl_names import SL_LAST_NAMES


# ─────────────────────────────────────────────
# DETECTION RESULT
# ─────────────────────────────────────────────

@dataclass
class DetectedEntity:
    entity_type  : str
    value        : str
    start        : int
    end          : int
    sensitivity  : str          # CRITICAL | HIGH | MEDIUM | LOW
    category     : str          # e.g. "1A", "2A"
    match_quality: str          # exact | partial
    context_tokens: List[str]   = field(default_factory=list)


# ─────────────────────────────────────────────
# TAXONOMY PATTERN REGISTRY
# ─────────────────────────────────────────────
# Each entry: (entity_type, pattern, sensitivity, category, match_quality)

PATTERNS = [

    # ── Category 1A: National & Government Identifiers ──────────────
    ("NIC_OLD",         r'\b\d{9}[vVxX]\b',             "HIGH",     "1A", "exact"),
    ("NIC_NEW",         r'\b\d{12}\b',                   "HIGH",     "1A", "partial"),  # needs confidence boost
    ("PASSPORT",        r'\b[A-Z]{1,2}\d{6,7}\b',        "HIGH",     "1A", "partial"),
    ("DRIVING_LICENSE", r'\b[A-Z]\d{7}\b',               "HIGH",     "1A", "partial"),
    ("TAX_ID",          r'\b\d{9}\b',                    "HIGH",     "1A", "partial"),

    # ── Category 1B: Contact Information ────────────────────────────
    ("PHONE_LK",        r'(?:\+94|\b0)[0-9]{9}\b',       "MEDIUM",   "1B", "exact"),
    ("PHONE_INTL",      r'\+(?!94)[1-9]\d{6,14}\b',      "MEDIUM",   "1B", "exact"),
    ("EMAIL",           r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', "MEDIUM", "1B", "exact"),

    # ── Category 2A: Card Data (PCI-DSS) ────────────────────────────
    # \d(?:[ \-]?\d){12,18} (not (?:\d[ \-]?){13,19}) so a trailing
    # separator can never be greedily absorbed with nothing after it —
    # the old form made PAN's span 1 char longer than an identical digit
    # run's BANK_ACCOUNT_NO span whenever followed by a space, breaking
    # same-span ambiguity resolution between the two types.
    ("PAN",             r'\b\d(?:[ \-]?\d){12,18}\b',    "CRITICAL", "2A", "partial"),
    ("CVV",             r'\b\d{3,4}\b',                  "CRITICAL", "2A", "partial"),  # needs keyword
    # (?<!\d/) excludes MM/YY-shaped fragments embedded inside a longer
    # DD/MM/YYYY date (e.g. the "09/2001" tail of "20/09/2001") — without
    # it, any date-of-birth collides with this pattern.
    ("CARD_EXPIRY",     r'(?<!\d/)\b(0[1-9]|1[0-2])\/\d{2,4}\b', "HIGH", "2A", "exact"),

    # ── Category 2B: Bank Account Data ──────────────────────────────
    ("BANK_ACCOUNT_NO", r'\b\d{10,16}\b',                "HIGH",     "2B", "partial"),
    ("IBAN",            r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,20}\b', "HIGH", "2B", "exact"),
    ("SWIFT_BIC",       r'\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b', "CRITICAL", "2B", "partial"),

    # ── Category 2C: Transaction / SWIFT ────────────────────────────
    ("SWIFT_MT103",     r'\bMT\s?103\b',                 "CRITICAL", "2C", "exact"),
    ("SWIFT_MT202",     r'\bMT\s?202\b',                 "CRITICAL", "2C", "exact"),

    # ── Category 3A: API Keys & Tokens ──────────────────────────────
    ("API_KEY_OPENAI",  r'\bsk-[A-Za-z0-9]{20,}\b',     "HIGH",     "3A", "exact"),
    ("API_KEY_GENERIC", r'\b[A-Za-z0-9_\-]{32,}\b',     "HIGH",     "3A", "partial"),  # needs keyword
    ("JWT_TOKEN",       r'\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b', "HIGH", "3A", "exact"),

    # ── Category 3B: Passwords ───────────────────────────────────────
    ("PASSWORD",        r'(?i)(?:password|passwd|pwd)\s*[=:]\s*\S+', "HIGH", "3B", "exact"),

    # ── Category 3C: Private Keys ───────────────────────────────────
    ("PRIVATE_KEY",     r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', "CRITICAL", "3C", "exact"),

    # ── Category 4B: Database Connection Strings ─────────────────────
    # Two DSN shapes: URI-style (postgresql://...) and ADO.NET/ODBC
    # key=value style (Server=x;Database=y;User=z;Password=w;) — the
    # latter is the standard SQL Server/ODBC connection-string format and
    # has no protocol:// prefix at all, so it needs its own alternative.
    ("DB_CONNECTION_STRING",
                        r'(?i)(?:(?:postgresql|mysql|mongodb|redis|mssql)://[^\s]+'
                        r'|(?:Server|Data Source)\s*=\s*[^;]+;(?:[^;]+;)*?(?:Password|Pwd)\s*=\s*[^;]+;?)',
                        "HIGH", "4B", "exact"),

    # ── Category 4C: Internal Network ───────────────────────────────
    ("INTERNAL_IP",     r'\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b',
                        "MEDIUM", "4C", "exact"),

    # ── Category 7A: Cloud Keys ──────────────────────────────────────
    ("AWS_ACCESS_KEY",  r'\bAKIA[A-Z0-9]{16}\b',         "HIGH",     "7A", "exact"),
    ("AWS_SECRET_KEY",  r'(?i)aws.{0,25}secret.{0,20}[=:]\s*[A-Za-z0-9+/]{40}\b', "HIGH", "7A", "exact"),

    # ── Category 7B: Cloud Storage ───────────────────────────────────
    ("S3_BUCKET_REF",   r's3://[a-zA-Z0-9.\-_/]+',       "MEDIUM",   "7B", "exact"),

    # ── Category 7C: Encoded Secrets ────────────────────────────────
    ("JWT_IN_LOG",      r'(?i)Authorization:\s*Bearer\s+eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+',
                        "HIGH", "7C", "exact"),
]

# Compile all patterns once at import time (efficiency)
COMPILED_PATTERNS = [
    (entity_type, re.compile(pattern), sensitivity, category, match_quality)
    for entity_type, pattern, sensitivity, category, match_quality in PATTERNS
]


# ─────────────────────────────────────────────
# KEYWORD MAPS (for NER and confidence boosting)
# ─────────────────────────────────────────────

_SURNAME_ALTERNATION = "|".join(
    re.escape(n) for n in sorted(SL_LAST_NAMES, key=len, reverse=True)
)

NER_KEYWORDS = {
    "FULL_NAME": [
        r'\b(Mr|Mrs|Ms|Dr|Rev)\.?\s+[A-Z][a-z]+ [A-Z][a-z]+',
        rf'\b[A-Z][a-z]+ (?:{_SURNAME_ALTERNATION})\b'
    ],
    "DATE_OF_BIRTH": [
        r'\b(?:dob|date of birth|born(?:\s+on)?|birth date)\s*[:\-]?\s*'
        r'(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b',
        r'\b\d{4}-\d{2}-\d{2}\b'
    ],
    "HOME_ADDRESS": [
        r'\b\d+,?\s+[A-Za-z ]+(?:Road|Street|Lane|Avenue|Rd|St|Ave|Mawatha)'
        r'(?:,\s*[A-Za-z ]+)?\b'
    ]
}

COMPILED_NER = {
    entity_type: [re.compile(p, re.IGNORECASE) for p in patterns]
    for entity_type, patterns in NER_KEYWORDS.items()
}

NER_SENSITIVITY = {
    "FULL_NAME"    : ("LOW",    "1C"),
    "DATE_OF_BIRTH": ("MEDIUM", "1C"),
    "HOME_ADDRESS" : ("MEDIUM", "1B"),
}

# Confidence boost keywords per entity type (for use by confidence scorer)
BOOST_KEYWORDS = {
    "NIC_OLD"        : ["nic", "id", "identity", "national", "customer id"],
    "NIC_NEW"        : ["nic", "id", "identity", "national", "customer id", "customer reference"],
    "PASSPORT"       : ["passport", "travel document"],
    "DRIVING_LICENSE": ["license", "driving"],
    "TAX_ID"         : ["tin", "tax", "vat"],
    "CVV"            : ["cvv", "cvc", "security code", "card verification"],
    "PAN"            : ["card", "credit", "debit", "pan", "card number"],
    "BANK_ACCOUNT_NO": ["account", "account number", "acc", "bank account"],
    "SWIFT_BIC"      : ["swift", "bic", "bank code"],
    "API_KEY_GENERIC": ["api_key", "api key", "apikey", "token", "secret", "key"],
    "API_KEY_OPENAI" : ["api_key", "api key", "apikey", "token", "secret", "key"],
    "PASSWORD"       : ["password", "passwd", "pwd", "credentials"],
    "AWS_ACCESS_KEY" : ["aws", "access key", "iam"],
    "AWS_SECRET_KEY" : ["aws", "secret", "credentials"],
    "S3_BUCKET_REF"  : ["s3", "bucket", "storage"],
    "INTERNAL_IP"    : ["server", "host", "ip", "internal", "network"],
    "PHONE_LK"       : ["phone", "call", "mobile", "contact", "sms", "otp", "dispatch"],
    "PHONE_INTL"     : ["phone", "call", "mobile", "contact", "international"],
    "CARD_EXPIRY"    : ["expiry", "expires", "exp", "valid thru", "valid until"],
}

# Entity types where a bare regex match is not a valid candidate detection
# without a nearby keyword — per taxonomy Section 5 (e.g. CVV is defined as
# "Regex + keyword"). Comments in PATTERNS above previously said "needs
# keyword" but nothing enforced it; this closes that gap at detection time
# rather than leaving it to scoring (a bare match already scores ~0.25,
# which evaluate.py counts as a positive detection, so scoring alone
# would not have suppressed these false positives).
REQUIRE_KEYWORD_AT_DETECTION = {"CVV", "TAX_ID", "BANK_ACCOUNT_NO", "API_KEY_GENERIC"}

# Entity type groups that are structurally ambiguous with each other
# (their regexes can match the exact same span). detect() lets both
# candidates survive for these groups instead of the first-registered
# pattern silently winning; confidence_scorer.resolve_overlapping_entities
# then picks the higher-scoring one after real scoring/validation.
AMBIGUOUS_TYPE_GROUPS = [
    frozenset({"NIC_NEW", "BANK_ACCOUNT_NO"}),
    frozenset({"TAX_ID", "NIC_OLD"}),
    frozenset({"PASSPORT", "DRIVING_LICENSE"}),
    frozenset({"PAN", "BANK_ACCOUNT_NO"}),
]


def _in_same_ambiguous_group(type_a: str, type_b: str) -> bool:
    return any(type_a in g and type_b in g for g in AMBIGUOUS_TYPE_GROUPS)


# Entity types whose match "claims" everything inside its span as a single
# unit — e.g. a spaced-out PAN like "4111 1111 1111 1111" must not also be
# torn into four separate CVV candidates ("4111", "1111"...) just because
# the word "CVV" happens to appear later in the same sentence (the ±5 token
# keyword-proximity window doesn't know the difference); likewise a JWT's
# three dot-separated base64 segments must not each also independently
# match API_KEY_GENERIC. Matches of other types that fall strictly *inside*
# a container match's span are dropped.
CONTAINER_TYPES = {"PAN", "JWT_TOKEN", "JWT_IN_LOG"}


# ─────────────────────────────────────────────
# DETECTION ENGINE
# ─────────────────────────────────────────────

def _has_gating_keyword(entity_type: str, source_text: str, start: int, end: int) -> bool:
    """Used only for REQUIRE_KEYWORD_AT_DETECTION types (imported lazily to
    avoid a hard circular import at module load time)."""
    from engine.normalizer import get_context_window
    keywords = BOOST_KEYWORDS.get(entity_type, [])
    if not keywords:
        return True
    window = get_context_window(source_text, start, end, window=5)
    ctx = " ".join(window).lower()
    # Also check the text immediately before the match. Handles glued
    # patterns like "api_key=VALUE" or "password:VALUE" where the keyword
    # and the value share a single whitespace-delimited token — the
    # token-based context window above excludes the entity's own token,
    # which would otherwise hide a keyword glued directly onto the value.
    adjacent = source_text[max(0, start - 20):start].lower()
    ctx = f"{ctx} {adjacent}"
    return any(kw.lower() in ctx for kw in keywords)


def detect(
    text: str,
    despaced_text: Optional[str] = None,
    despaced_map: Optional[List[int]] = None,
) -> List[DetectedEntity]:
    """
    Run all regex + NER patterns against text.
    Also runs against despaced_text for adversarial inputs.
    Returns a deduplicated list of DetectedEntity objects.

    Args:
        text         : normalized input text
        despaced_text: version with adversarial spacing removed (from normalizer)
        despaced_map : despaced-index -> text-index mapping (normalize()'s
                       "despaced_map") — despaced_text is shorter than text
                       whenever anything was removed, so a match's raw
                       start()/end() in despaced_text are NOT valid offsets
                       into `text`. Without this, entities found in the
                       despaced pass get positions that, when later used to
                       slice `text` for masking, land on the wrong
                       characters — corrupting the mask and leaking part of
                       the original value in plaintext. When omitted, raw
                       despaced-text offsets are used as-is (matches prior
                       behavior, only safe if despaced_text is same length
                       as text i.e. nothing was actually removed).
    """
    entities: List[DetectedEntity] = []
    seen_spans: dict = {}   # span -> set of entity_types already registered there
    # Shared (not reset per pass) so a container found while scanning `text`
    # still protects nested matches when the despaced pass re-scans the same
    # region — a match that's already `seen` short-circuits before it would
    # otherwise re-populate a per-pass-local container list.
    container_spans: List[tuple] = []

    # Containers are matched first (within each source-text pass) so their
    # spans are known before other patterns are considered for nesting.
    _ordered_patterns = sorted(
        COMPILED_PATTERNS, key=lambda p: p[0] not in CONTAINER_TYPES
    )

    def _run_patterns(source_text: str, translate=lambda s, e: (s, e)):
        for entity_type, compiled, sensitivity, category, match_quality in _ordered_patterns:
            for match in compiled.finditer(source_text):
                start, end = translate(match.start(), match.end())
                span  = (start, end)

                if entity_type in CONTAINER_TYPES:
                    if span not in container_spans:
                        container_spans.append(span)
                elif any(
                    c_start <= start and end <= c_end and (c_start, c_end) != span
                    for c_start, c_end in container_spans
                ):
                    continue

                existing_types = seen_spans.get(span, set())
                if entity_type in existing_types:
                    continue
                if existing_types and not any(
                    _in_same_ambiguous_group(entity_type, t) for t in existing_types
                ):
                    # Unrelated collision at an identical span: keep the
                    # existing (first-registered) match, same as before.
                    continue

                # Avoid very short generic matches without keyword context
                value = match.group(0).strip()
                if len(value) < 3:
                    continue

                if entity_type in REQUIRE_KEYWORD_AT_DETECTION:
                    if not _has_gating_keyword(entity_type, source_text, match.start(), match.end()):
                        continue

                seen_spans.setdefault(span, set()).add(entity_type)
                entities.append(DetectedEntity(
                    entity_type   = entity_type,
                    value         = value,
                    start         = start,
                    end           = end,
                    sensitivity   = sensitivity,
                    category      = category,
                    match_quality = match_quality,
                ))

    # Run on normalized text
    _run_patterns(text)

    # Also run on despaced version if provided (adversarial detection)
    if despaced_text and despaced_text != text:
        if despaced_map is not None:
            def _to_normalized_span(s: int, e: int) -> tuple:
                return despaced_map[s], despaced_map[e - 1] + 1
            _run_patterns(despaced_text, _to_normalized_span)
        else:
            _run_patterns(despaced_text)

    # Run NER patterns
    for entity_type, patterns in COMPILED_NER.items():
        sensitivity, category = NER_SENSITIVITY[entity_type]
        for pattern in patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.setdefault(span, set()).add(entity_type)
                entities.append(DetectedEntity(
                    entity_type   = entity_type,
                    value         = match.group(0).strip(),
                    start         = match.start(),
                    end           = match.end(),
                    sensitivity   = sensitivity,
                    category      = category,
                    match_quality = "partial",
                ))

    # Sort by position in text
    entities.sort(key=lambda e: e.start)
    return entities


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "NIC: 199012345V",
        "Card: 4111 1111 1111 1111, CVV 123, exp 12/27",
        "api_key=sk-abcdefghij1234567890abcdefghij12",
        "postgresql://admin:Pass99@10.0.0.5:5432/corebank",
        "AKIA1234567890ABCDEF",
        "s3://sl-banking-backups-123",
        "MT103 transfer of LKR 50000",
    ]

    for t in tests:
        found = detect(t)
        print(f"\nInput : {t}")
        for e in found:
            print(f"  [{e.entity_type}] '{e.value}' | {e.sensitivity} | {e.category}")
