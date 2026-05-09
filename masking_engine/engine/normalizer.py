"""
normalizer.py — Input Preprocessing
R26-CS-012: Context-Aware Masking + Instruction Engine

Preprocessing pipeline applied BEFORE detection:
  1. Input normalization  — strip adversarial spacing/dashes, decode encodings
  2. Context type classification — natural_language / source_code / config_file / log_output / mixed
  3. Token window builder — tokenizes text, provides ±N token context windows

WHY THIS APPROACH:
  - Normalization must happen before regex to catch adversarial obfuscation (Taxonomy Section 7C)
  - Context type drives which entity categories are prioritized (Taxonomy Section 8)
  - Rule-based classification is interpretable and auditable — critical for banking compliance
"""

import re
import base64
import urllib.parse


# ─────────────────────────────────────────────
# 1. CONTEXT TYPE CLASSIFICATION
# ─────────────────────────────────────────────

CONTEXT_INDICATORS = {
    "source_code": [
        r"\bdef\s+\w+\s*\(", r"\bimport\s+\w+", r"\bfunction\s+\w+\s*\(",
        r"\bconst\s+\w+\s*=", r"\bvar\s+\w+\s*=", r"\bclass\s+\w+",
        r"=>", r"\breturn\b", r"//\s", r"/\*", r"\bpublic\b|\bprivate\b"
    ],
    "config_file": [
        r"^[A-Z_]+=.+",             # KEY=VALUE
        r'"[^"]+"\s*:\s*"[^"]+"',   # JSON key-value
        r"^\w+:\s+\S+",             # YAML key: value
        r"\.env", r"config\.", r"settings\."
    ],
    "log_output": [
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",  # timestamps
        r"\b(INFO|ERROR|DEBUG|WARN|CRITICAL|FATAL)\b",
        r"Traceback \(most recent call",
        r"at \w+\.\w+\([\w.]+:\d+\)",  # stack trace line
        r"\[[\d\-:T]+\]"               # log prefix
    ],
    "natural_language": [
        r"\b(please|can you|could you|help me|I need|check|verify|update|send)\b",
        r"\b(customer|account|transfer|balance|loan|payment)\b"
    ]
}


def classify_context(text: str) -> str:
    """
    Classify the input text into one of:
    natural_language | source_code | config_file | log_output | mixed

    Algorithm: score each context type by counting pattern matches,
    return the highest scorer. If multiple tie → 'mixed'.
    Complexity: O(P * T) where P = number of patterns, T = text length.
    """
    scores = {ctx: 0 for ctx in CONTEXT_INDICATORS}

    for ctx, patterns in CONTEXT_INDICATORS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                scores[ctx] += 1

    max_score = max(scores.values())

    if max_score == 0:
        return "natural_language"  # default

    winners = [ctx for ctx, s in scores.items() if s == max_score]

    if len(winners) > 1:
        return "mixed"

    return winners[0]


# ─────────────────────────────────────────────
# 2. INPUT NORMALIZATION
# ─────────────────────────────────────────────

def decode_base64_segments(text: str) -> str:
    """
    Find and decode Base64 segments within the text.
    Replaces decoded content inline so downstream patterns can match.
    Heuristic: sequences of 20+ base64 chars are candidates.
    """
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')

    def try_decode(match):
        candidate = match.group(0)
        try:
            # Pad if needed
            padded  = candidate + "=" * ((4 - len(candidate) % 4) % 4)
            decoded = base64.b64decode(padded).decode("utf-8", errors="strict")
            # Only replace if decoded result looks like printable text
            if decoded.isprintable() and len(decoded) > 5:
                return decoded
        except Exception:
            pass
        return candidate

    return b64_pattern.sub(try_decode, text)


def decode_url_encoding(text: str) -> str:
    """URL-decode the text to expose obfuscated values."""
    try:
        decoded = urllib.parse.unquote(text)
        return decoded if decoded != text else text
    except Exception:
        return text


def decode_hex_segments(text: str) -> str:
    """
    Find hex-encoded segments and decode them.
    Heuristic: even-length hex strings of 10+ characters.
    """
    hex_pattern = re.compile(r'\b[0-9a-fA-F]{10,}\b')

    def try_decode(match):
        candidate = match.group(0)
        if len(candidate) % 2 != 0:
            return candidate
        try:
            decoded = bytes.fromhex(candidate).decode("utf-8", errors="strict")
            if decoded.isprintable() and len(decoded) > 4:
                return decoded
        except Exception:
            pass
        return candidate

    return hex_pattern.sub(try_decode, text)


def remove_adversarial_spacing(text: str) -> str:
    """
    Remove spaces/dashes inserted between digits or alphanumeric groups
    to evade pattern matching (Taxonomy Section 7C adversarial note).

    E.g.: '1 9 9 0 1 2 3 4 5 V' → '199012345V'
          '199-012-345-V'        → '199012345V'
    """
    # Remove spaces between single characters (spaced-out obfuscation)
    # Pattern: single char followed by space followed by single char, repeatedly
    spaced = re.sub(r'(?<=\w) (?=\w)', '', text)
    # Remove dashes between digit groups
    dashed = re.sub(r'(\d)-(\d)', r'\1\2', spaced)
    return dashed


def normalize(text: str) -> dict:
    """
    Full normalization pipeline. Returns both the normalized text
    and a log of which transformations were applied.

    Steps (in order, per Taxonomy Section 7C):
      1. URL decode
      2. Hex decode
      3. Base64 decode
      4. Adversarial spacing removal
    """
    transformations = []
    current         = text

    # Step 1 — URL decode
    url_decoded = decode_url_encoding(current)
    if url_decoded != current:
        transformations.append("url_decoded")
        current = url_decoded

    # Step 2 — Hex decode
    hex_decoded = decode_hex_segments(current)
    if hex_decoded != current:
        transformations.append("hex_decoded")
        current = hex_decoded

    # Step 3 — Base64 decode
    b64_decoded = decode_base64_segments(current)
    if b64_decoded != current:
        transformations.append("base64_decoded")
        current = b64_decoded

    # Step 4 — Adversarial spacing removal (run on a parallel copy for detection)
    despaced = remove_adversarial_spacing(current)
    if despaced != current:
        transformations.append("adversarial_spacing_removed")

    return {
        "original"        : text,
        "normalized"      : current,
        "despaced"        : despaced,
        "transformations" : transformations,
        "context_type"    : classify_context(text)
    }


# ─────────────────────────────────────────────
# 3. TOKEN WINDOW BUILDER
# ─────────────────────────────────────────────

def tokenize(text: str) -> list:
    """
    Simple whitespace + punctuation tokenizer.
    Returns list of (token, start_char, end_char) tuples.
    """
    tokens = []
    for m in re.finditer(r'\S+', text):
        tokens.append((m.group(), m.start(), m.end()))
    return tokens


def get_context_window(text: str, entity_start: int, entity_end: int, window: int = 5) -> list:
    """
    Returns the ±window tokens around the entity position.
    Used for Context Keyword Proximity scoring (Taxonomy Section 4).

    Args:
        text         : full input string
        entity_start : char start index of detected entity
        entity_end   : char end index of detected entity
        window       : number of tokens on each side (default 5)

    Returns:
        list of token strings in the context window
    """
    tokens = tokenize(text)

    # Find which token index overlaps with the entity span
    entity_token_idx = None
    for i, (tok, start, end) in enumerate(tokens):
        if start <= entity_start < end or entity_start <= start < entity_end:
            entity_token_idx = i
            break

    if entity_token_idx is None:
        return []

    lo  = max(0, entity_token_idx - window)
    hi  = min(len(tokens) - 1, entity_token_idx + window)

    context_tokens = [
        tokens[i][0] for i in range(lo, hi + 1)
        if i != entity_token_idx
    ]

    return context_tokens


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "Customer NIC: 199012345V, please check.",
        "api_key=sk-abc123",
        "password%3DSecure99%40",
        "def connect(db): return psycopg2.connect(DSN)",
        "2024-01-15 INFO: User logged in",
        "Hex encoded: 5061737377307264",
    ]

    for s in samples:
        result = normalize(s)
        print(f"\nInput   : {s}")
        print(f"Context : {result['context_type']}")
        print(f"Transforms: {result['transformations']}")
        if result['normalized'] != s:
            print(f"Normalized: {result['normalized']}")
