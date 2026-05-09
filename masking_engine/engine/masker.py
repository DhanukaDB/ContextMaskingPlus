"""
masker.py — Masking Strategy Engine
R26-CS-012: Context-Aware Masking + Instruction Engine

Applies one of four masking strategies (Taxonomy Section 9):
  1. Full Mask      — replaces value with asterisks
  2. Partial Mask   — preserves prefix/suffix, masks middle
  3. Tokenization   — replaces with <TYPE_N> token (session-traceable)
  4. Contextual     — mask only if co-occurring with HIGH/CRITICAL entity

Strategy selection is driven by:
  - Entity sensitivity level (CRITICAL / HIGH / MEDIUM / LOW)
  - Entity type (some types always tokenize for traceability)
  - Co-occurrence elevation flag from confidence scorer

WHY THESE STRATEGIES (not uniform full-masking):
  - Partial mask preserves enough info for LLM to understand structure
    (e.g. '077****567' still shows it's a phone number)
  - Tokenization allows session-coherent references (<API_KEY_1> = same key
    across prompts) — critical for multi-turn banking queries
  - Contextual masking prevents over-masking low-sensitivity data
    (names alone are legitimate in many banking queries)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from engine.confidence_scorer import ScoredEntity
from engine.token_registry import TokenRegistry


# ─────────────────────────────────────────────
# MASKING RESULT
# ─────────────────────────────────────────────

@dataclass
class MaskedResult:
    original_text   : str
    masked_text     : str
    masked_entities : List[dict]    = field(default_factory=list)
    skipped_entities: List[dict]    = field(default_factory=list)
    overall_risk    : str           = "LOW"   # LOW | MEDIUM | HIGH | CRITICAL


# ─────────────────────────────────────────────
# STRATEGY SELECTION RULES
# ─────────────────────────────────────────────

# Entity types that always use tokenization (need cross-prompt traceability)
ALWAYS_TOKENIZE = {
    "API_KEY_OPENAI", "API_KEY_GENERIC", "JWT_TOKEN", "JWT_IN_LOG",
    "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "PRIVATE_KEY",
    "BANK_ACCOUNT_NO", "IBAN", "S3_BUCKET_REF",
    "BASE64_ENCODED_SECRET", "HEX_ENCODED_SECRET",
    "DB_CONNECTION_STRING",
}

# Entity types that use partial masking
ALWAYS_PARTIAL = {
    "PHONE_LK", "PHONE_INTL", "EMAIL", "PAN",
}

# Contextual types — only mask if co-occurring with HIGH/CRITICAL
CONTEXTUAL_TYPES = {
    "FULL_NAME", "GENDER", "RACE_ETHNICITY",
}


def select_strategy(scored: ScoredEntity, has_high_cooccur: bool) -> str:
    """
    Determine masking strategy for a single entity.

    Returns: 'full' | 'partial' | 'tokenize' | 'contextual_skip' | 'ignore'
    """
    entity_type = scored.entity.entity_type
    action      = scored.action
    sensitivity = scored.entity.sensitivity

    # Below threshold — don't mask
    if action == "ignore":
        return "ignore"

    # Low confidence suspected — don't mask, just log
    if action == "log_suspected":
        return "log_suspected"

    # Contextual types: only mask if co-occurring with something HIGH/CRITICAL
    if entity_type in CONTEXTUAL_TYPES:
        if has_high_cooccur or sensitivity == "CRITICAL":
            return "full"
        else:
            return "contextual_skip"

    # Always tokenize types
    if entity_type in ALWAYS_TOKENIZE:
        return "tokenize"

    # Always partial types
    if entity_type in ALWAYS_PARTIAL:
        return "partial"

    # CRITICAL sensitivity → full mask
    if sensitivity == "CRITICAL":
        return "full"

    # HIGH sensitivity → full mask
    if sensitivity == "HIGH":
        return "full"

    # MEDIUM → partial
    if sensitivity == "MEDIUM":
        return "partial"

    # LOW → contextual
    return "contextual_skip"


# ─────────────────────────────────────────────
# MASKING IMPLEMENTATIONS
# ─────────────────────────────────────────────

def full_mask(value: str) -> str:
    """Replace entire value with asterisks. Length preserved for awareness."""
    return "*" * len(value)


def partial_mask(value: str, entity_type: str) -> str:
    """
    Preserve meaningful prefix/suffix, mask the middle.
    Rules vary by entity type.
    """
    v = value.strip()

    if entity_type == "EMAIL":
        # j***@domain.com
        parts = v.split("@")
        if len(parts) == 2:
            user   = parts[0]
            prefix = user[0] if user else "*"
            return f"{prefix}***@{parts[1]}"
        return "***@***.***"

    if entity_type in ("PHONE_LK", "PHONE_INTL"):
        # 077****567
        digits_only = re.sub(r'\D', '', v)
        if len(digits_only) >= 7:
            return digits_only[:3] + "****" + digits_only[-3:]
        return "****"

    if entity_type == "PAN":
        # 4111 **** **** 1111
        digits = re.sub(r'\D', '', v)
        if len(digits) >= 8:
            return digits[:4] + " **** **** " + digits[-4:]
        return "**** **** **** ****"

    # Generic partial — show first and last 2 chars
    if len(v) > 6:
        return v[:2] + ("*" * (len(v) - 4)) + v[-2:]
    return "*" * len(v)


def tokenize_value(value: str, entity_type: str, registry: TokenRegistry) -> str:
    """
    Replace value with a session-scoped idempotent token.
    E.g., 'sk-abc123...' → '<API_KEY_1>'
    Same value in same session always gets same token.
    """
    return registry.get_or_create_token(value, entity_type)


# ─────────────────────────────────────────────
# MAIN MASKER
# ─────────────────────────────────────────────

def mask(
    text          : str,
    scored_entities: List[ScoredEntity],
    registry      : TokenRegistry
) -> MaskedResult:
    """
    Apply masking to the input text based on scored entities.

    Strategy:
      - Process entities in reverse order (right to left) so char positions
        remain valid as we replace substrings.
      - Build a record of what was masked, skipped, and why.
    """
    if not scored_entities:
        return MaskedResult(
            original_text  = text,
            masked_text    = text,
            masked_entities= [],
            skipped_entities= [],
            overall_risk   = "LOW"
        )

    # Determine if any HIGH/CRITICAL co-occurring entity exists
    sensitivities     = {s.entity.sensitivity for s in scored_entities}
    has_high_cooccur  = bool(sensitivities & {"HIGH", "CRITICAL"})

    # Sort by position descending (right to left replacement)
    sorted_entities = sorted(scored_entities, key=lambda s: s.entity.start, reverse=True)

    masked_text      = text
    masked_records   = []
    skipped_records  = []
    risk_levels      = []

    for scored in sorted_entities:
        entity   = scored.entity
        strategy = select_strategy(scored, has_high_cooccur)

        if strategy in ("ignore", "contextual_skip"):
            skipped_records.append({
                "entity_type": entity.entity_type,
                "value"      : entity.value,
                "reason"     : strategy,
                "score"      : scored.score,
            })
            continue

        if strategy == "log_suspected":
            skipped_records.append({
                "entity_type": entity.entity_type,
                "value"      : entity.value,
                "reason"     : "low_confidence_logged",
                "score"      : scored.score,
            })
            risk_levels.append("LOW")
            continue

        # Apply the chosen strategy
        original_value = entity.value

        if strategy == "full":
            replacement = full_mask(original_value)
        elif strategy == "partial":
            replacement = partial_mask(original_value, entity.entity_type)
        elif strategy == "tokenize":
            replacement = tokenize_value(original_value, entity.entity_type, registry)
        else:
            replacement = full_mask(original_value)  # fallback

        # Replace in text (using char positions)
        masked_text = masked_text[:entity.start] + replacement + masked_text[entity.end:]

        masked_records.append({
            "entity_type" : entity.entity_type,
            "original"    : original_value,
            "replacement" : replacement,
            "strategy"    : strategy,
            "sensitivity" : entity.sensitivity,
            "score"       : scored.score,
            "action"      : scored.action,
        })
        risk_levels.append(entity.sensitivity)

    # Overall risk = highest sensitivity level found
    risk_priority = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    if risk_levels:
        overall_risk = max(risk_levels, key=lambda r: risk_priority.get(r, 0))
    else:
        overall_risk = "LOW"

    return MaskedResult(
        original_text   = text,
        masked_text     = masked_text,
        masked_entities = masked_records,
        skipped_entities= skipped_records,
        overall_risk    = overall_risk
    )


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from engine.normalizer import normalize
    from engine.detector import detect
    from engine.confidence_scorer import score_all
    from engine.token_registry import TokenRegistry

    registry = TokenRegistry(session_id="test_session")

    examples = [
        "Call 0771234567 or email saman.perera@seylan.lk",
        "Card 4111 1111 1111 1111 CVV 123 exp 12/27",
        "api_key=sk-abcdefghij1234567890abcdefghij12",
        "Saman Perera, NIC 199012345V — CRITICAL combo",
    ]

    for text in examples:
        norm    = normalize(text)
        raw     = detect(norm["normalized"], norm["despaced"])
        scored  = score_all(raw, norm["normalized"])
        result  = mask(norm["normalized"], scored, registry)

        print(f"\nOriginal : {result.original_text}")
        print(f"Masked   : {result.masked_text}")
        print(f"Risk     : {result.overall_risk}")
        for m in result.masked_entities:
            print(f"  [{m['entity_type']}] '{m['original']}' → '{m['replacement']}' ({m['strategy']})")
