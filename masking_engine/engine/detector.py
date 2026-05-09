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

import re
from dataclasses import dataclass, field
from typing import List, Optional


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
    ("PHONE_LK",        r'(\+94|0)[0-9]{9}\b',           "MEDIUM",   "1B", "exact"),
    ("PHONE_INTL",      r'\+[1-9]\d{6,14}\b',            "MEDIUM",   "1B", "exact"),
    ("EMAIL",           r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', "MEDIUM", "1B", "exact"),

    # ── Category 2A: Card Data (PCI-DSS) ────────────────────────────
    ("PAN",             r'\b(?:\d[ \-]?){13,19}\b',      "CRITICAL", "2A", "partial"),
    ("CVV",             r'\b\d{3,4}\b',                  "CRITICAL", "2A", "partial"),  # needs keyword
    ("CARD_EXPIRY",     r'\b(0[1-9]|1[0-2])\/\d{2,4}\b', "HIGH",    "2A", "exact"),

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
    ("DB_CONNECTION_STRING",
                        r'(?i)(postgresql|mysql|mongodb|redis|mssql)://[^\s]+', "HIGH", "4B", "exact"),

    # ── Category 4C: Internal Network ───────────────────────────────
    ("INTERNAL_IP",     r'\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b',
                        "MEDIUM", "4C", "exact"),

    # ── Category 7A: Cloud Keys ──────────────────────────────────────
    ("AWS_ACCESS_KEY",  r'\bAKIA[A-Z0-9]{16}\b',         "HIGH",     "7A", "exact"),
    ("AWS_SECRET_KEY",  r'(?i)aws.{0,20}secret.{0,5}[=:]\s*[A-Za-z0-9+/]{40}\b', "HIGH", "7A", "exact"),

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

NER_KEYWORDS = {
    "FULL_NAME": [
        r'\b(Mr|Mrs|Ms|Dr|Rev)\.?\s+[A-Z][a-z]+ [A-Z][a-z]+',
        r'\b[A-Z][a-z]+ (Perera|Silva|Fernando|Jayawardena|Wickramasinghe|'
        r'Gunawardena|Rajapaksa|Dissanayake|Bandara|Senanayake)\b'
    ],
    "DATE_OF_BIRTH": [
        r'\b(?:dob|date of birth|born on|birth date)\s*[:\-]?\s*'
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
    "PASSWORD"       : ["password", "passwd", "pwd", "credentials"],
    "AWS_ACCESS_KEY" : ["aws", "access key", "iam"],
    "AWS_SECRET_KEY" : ["aws", "secret", "credentials"],
    "S3_BUCKET_REF"  : ["s3", "bucket", "storage"],
    "INTERNAL_IP"    : ["server", "host", "ip", "internal", "network"],
}


# ─────────────────────────────────────────────
# DETECTION ENGINE
# ─────────────────────────────────────────────

def detect(text: str, despaced_text: Optional[str] = None) -> List[DetectedEntity]:
    """
    Run all regex + NER patterns against text.
    Also runs against despaced_text for adversarial inputs.
    Returns a deduplicated list of DetectedEntity objects.

    Args:
        text         : normalized input text
        despaced_text: version with adversarial spacing removed (from normalizer)
    """
    entities: List[DetectedEntity] = []
    seen_spans = set()

    def _run_patterns(source_text: str, offset: int = 0):
        for entity_type, compiled, sensitivity, category, match_quality in COMPILED_PATTERNS:
            for match in compiled.finditer(source_text):
                start = match.start() + offset
                end   = match.end()   + offset
                span  = (start, end)

                if span in seen_spans:
                    continue

                # Avoid very short generic matches without keyword context
                value = match.group(0).strip()
                if len(value) < 3:
                    continue

                seen_spans.add(span)
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
        _run_patterns(despaced_text)

    # Run NER patterns
    for entity_type, patterns in COMPILED_NER.items():
        sensitivity, category = NER_SENSITIVITY[entity_type]
        for pattern in patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
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
