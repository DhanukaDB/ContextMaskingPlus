"""
confidence_scorer.py — 4-Factor Confidence Scoring
R26-CS-012: Context-Aware Masking + Instruction Engine

Implements the confidence scoring framework from Taxonomy Section 4.

Factors (weighted):
  1. Pattern Match Strength   — 40%  (exact vs partial regex match)
  2. Context Keyword Proximity — 30%  (boost keywords within ±5 tokens)
  3. Co-occurrence Boost       — 20%  (other entities present in same prompt)
  4. Format Validity           — 10%  (structural checks: Luhn, NIC digit-sum)

Score Range → Action:
  0.90–1.00  Very High  → Mask immediately
  0.75–0.89  High       → Mask immediately
  0.50–0.74  Medium     → Mask with warning log
  0.25–0.49  Low        → Log as suspected; alert dashboard
  0.00–0.24  Very Low   → Treat as false positive; ignore

WHY WEIGHTED LINEAR COMBINATION:
  - Fully transparent and auditable (each factor traceable)
  - No training data needed — weights are domain-defined
  - Each factor maps directly to a compliance-relevant concern
  - Easily adjustable per regulatory update (e.g. CBSL policy change)
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from engine.detector import DetectedEntity, BOOST_KEYWORDS
from engine.normalizer import get_context_window


# ─────────────────────────────────────────────
# SCORING RESULT
# ─────────────────────────────────────────────

@dataclass
class ScoredEntity:
    entity           : DetectedEntity
    score            : float
    confidence_level : str       # Very High | High | Medium | Low | Very Low
    action           : str       # mask_immediate | mask_warn | log_suspected | ignore
    score_breakdown  : Dict      = field(default_factory=dict)


# ─────────────────────────────────────────────
# FACTOR WEIGHTS (Taxonomy Section 4)
# ─────────────────────────────────────────────

W_PATTERN_STRENGTH   = 0.40
W_KEYWORD_PROXIMITY  = 0.30
W_CO_OCCURRENCE      = 0.20
W_FORMAT_VALIDITY    = 0.10


# ─────────────────────────────────────────────
# FACTOR 1 — PATTERN MATCH STRENGTH
# ─────────────────────────────────────────────

def score_pattern_strength(entity: DetectedEntity) -> float:
    """
    Exact match (specific pattern, unambiguous) → 1.0 × W
    Partial match (generic pattern, needs context) → 0.5 × W
    """
    return 1.0 if entity.match_quality == "exact" else 0.5


# ─────────────────────────────────────────────
# FACTOR 2 — CONTEXT KEYWORD PROXIMITY
# ─────────────────────────────────────────────

def score_keyword_proximity(entity: DetectedEntity, text: str) -> float:
    """
    Check if any boost keyword for this entity type appears
    within ±5 tokens of the entity in the prompt.

    Full match  → 1.0
    Partial match (keyword substring) → 0.5
    No match → 0.0
    """
    keywords = BOOST_KEYWORDS.get(entity.entity_type, [])
    if not keywords:
        return 0.5  # neutral if no keywords defined

    context_tokens = get_context_window(text, entity.start, entity.end, window=5)
    context_lower  = " ".join(context_tokens).lower()

    for kw in keywords:
        if kw.lower() in context_lower:
            return 1.0  # exact keyword found

    # Partial: any word in context shares prefix with a keyword
    for kw in keywords:
        kw_root = kw.lower().split()[0][:4]  # first 4 chars of first word
        if kw_root in context_lower:
            return 0.5

    return 0.0


# ─────────────────────────────────────────────
# FACTOR 3 — CO-OCCURRENCE BOOST
# ─────────────────────────────────────────────

CO_OCCURRENCE_PAIRS = {
    # (type_a, type_b) → elevated sensitivity label
    frozenset(["FULL_NAME", "NIC_OLD"])         : "CRITICAL",
    frozenset(["FULL_NAME", "NIC_NEW"])         : "CRITICAL",
    frozenset(["FULL_NAME", "BANK_ACCOUNT_NO"]) : "CRITICAL",
    frozenset(["EMAIL",     "PASSWORD"])        : "CRITICAL",
    frozenset(["BANK_ACCOUNT_NO", "PAN"])       : "CRITICAL",
    frozenset(["INTERNAL_IP", "DB_CONNECTION_STRING"]) : "CRITICAL",
    frozenset(["AWS_ACCESS_KEY", "AWS_SECRET_KEY"])    : "CRITICAL",
}

def score_co_occurrence(entity: DetectedEntity, all_entities: List[DetectedEntity]) -> tuple:
    """
    Returns (score_0_to_1, elevated_sensitivity_or_None).

    Logic:
      - 3+ entities detected in one prompt → score 1.0 (Taxonomy Section 7)
      - Known dangerous pair → score 1.0
      - Any other co-occurrence → score 0.5
      - No co-occurrence → score 0.0
    """
    other_types = {e.entity_type for e in all_entities if e is not entity}

    # Rule: Any 3+ entities → CRITICAL regardless
    if len(all_entities) >= 3:
        return 1.0, "CRITICAL"

    if not other_types:
        return 0.0, None

    # Check known dangerous pairs
    my_type = entity.entity_type
    for pair, elevated in CO_OCCURRENCE_PAIRS.items():
        if my_type in pair:
            other_in_pair = pair - {my_type}
            if other_in_pair & other_types:
                return 1.0, elevated

    # Generic co-occurrence (other entities present but no special pair)
    return 0.5, None


# ─────────────────────────────────────────────
# FACTOR 4 — FORMAT VALIDITY
# ─────────────────────────────────────────────

def luhn_check(number_str: str) -> bool:
    """Luhn algorithm for PAN validation."""
    digits = [int(d) for d in re.sub(r'\D', '', number_str)]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def nic_old_check(value: str) -> bool:
    """
    Sri Lankan old NIC structural validation:
    - 9 digits followed by V or X
    - Digits 3–5 must be a valid day-of-year (001–366 for female: 501–866)
    """
    match = re.match(r'^(\d{2})(\d{3})(\d{4})[vVxX]$', value)
    if not match:
        return False
    days = int(match.group(2))
    return (1 <= days <= 366) or (501 <= days <= 866)


def nic_new_check(value: str) -> bool:
    """
    Sri Lankan new NIC (12 digits):
    - First 4 digits = birth year (1900–2010)
    - Digits 5–7 = day of year (001–366 or 501–866)
    """
    if not re.match(r'^\d{12}$', value):
        return False
    year = int(value[:4])
    days = int(value[4:7])
    return (1900 <= year <= 2010) and ((1 <= days <= 366) or (501 <= days <= 866))


def iban_check(value: str) -> bool:
    """Basic IBAN structural check (length + country code)."""
    iban = re.sub(r'\s', '', value).upper()
    if len(iban) < 15 or len(iban) > 34:
        return False
    if not re.match(r'^[A-Z]{2}\d{2}', iban):
        return False
    return True


FORMAT_VALIDATORS = {
    "PAN"            : luhn_check,
    "NIC_OLD"        : nic_old_check,
    "NIC_NEW"        : nic_new_check,
    "IBAN"           : iban_check,
}


def score_format_validity(entity: DetectedEntity) -> float:
    """
    Run structural validator if one exists for this entity type.
    Pass → 1.0, Fail → 0.0, No validator → 0.5 (neutral)
    """
    validator = FORMAT_VALIDATORS.get(entity.entity_type)
    if validator is None:
        return 0.5  # neutral

    try:
        return 1.0 if validator(entity.value) else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# THRESHOLD → ACTION MAPPING
# ─────────────────────────────────────────────

def resolve_action(score: float) -> tuple:
    """
    Maps confidence score to (confidence_level, action).
    Per Taxonomy Section 4 threshold table.
    """
    if score >= 0.90:
        return "Very High", "mask_immediate"
    elif score >= 0.75:
        return "High",      "mask_immediate"
    elif score >= 0.50:
        return "Medium",    "mask_warn"
    elif score >= 0.25:
        return "Low",       "log_suspected"
    else:
        return "Very Low",  "ignore"


# ─────────────────────────────────────────────
# MAIN SCORER
# ─────────────────────────────────────────────

def score_entity(
    entity      : DetectedEntity,
    text        : str,
    all_entities: List[DetectedEntity]
) -> ScoredEntity:
    """
    Compute all 4 factors and return a ScoredEntity with full breakdown.
    """
    # Factor scores (0.0 – 1.0 each, then weighted)
    f1 = score_pattern_strength(entity)
    f2 = score_keyword_proximity(entity, text)
    f3_raw, elevated_sensitivity = score_co_occurrence(entity, all_entities)
    f4 = score_format_validity(entity)

    weighted_score = (
        f1 * W_PATTERN_STRENGTH  +
        f2 * W_KEYWORD_PROXIMITY +
        f3_raw * W_CO_OCCURRENCE +
        f4 * W_FORMAT_VALIDITY
    )
    # Clamp to [0.0, 1.0]
    final_score = round(min(1.0, max(0.0, weighted_score)), 2)

    # Apply co-occurrence sensitivity elevation
    if elevated_sensitivity:
        entity.sensitivity = elevated_sensitivity

    confidence_level, action = resolve_action(final_score)

    return ScoredEntity(
        entity           = entity,
        score            = final_score,
        confidence_level = confidence_level,
        action           = action,
        score_breakdown  = {
            "pattern_strength"   : round(f1 * W_PATTERN_STRENGTH, 3),
            "keyword_proximity"  : round(f2 * W_KEYWORD_PROXIMITY, 3),
            "co_occurrence"      : round(f3_raw * W_CO_OCCURRENCE, 3),
            "format_validity"    : round(f4 * W_FORMAT_VALIDITY, 3),
            "total"              : final_score,
            "elevated_by_cooccur": elevated_sensitivity or "—",
        }
    )


def score_all(
    entities: List[DetectedEntity],
    text    : str
) -> List[ScoredEntity]:
    """
    Score all detected entities in a prompt.
    Passes the full entity list for co-occurrence calculation.
    """
    return [score_entity(e, text, entities) for e in entities]


# ─────────────────────────────────────────────
# QUICK SELF-TEST — NIC Ambiguity Example (Taxonomy Section 4)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from engine.detector import detect
    from engine.normalizer import normalize

    examples = [
        # Ambiguous: 12-digit, keyword 'customer reference' (medium)
        "Customer reference: 200423910321",
        # High confidence: NIC keyword present
        "Please verify NIC 199012345678 for this customer.",
        # Co-occurrence: Name + NIC → CRITICAL
        "Saman Perera, NIC 199012345V, account 001010012345",
        # No context → should be low confidence
        "Serial: 200423910321 logged for inventory.",
    ]

    for text in examples:
        norm    = normalize(text)
        raw     = detect(norm["normalized"], norm["despaced"])
        scored  = score_all(raw, norm["normalized"])

        print(f"\n{'─'*60}")
        print(f"Input : {text}")
        for s in scored:
            print(f"  [{s.entity.entity_type}] score={s.score} | {s.confidence_level} | action={s.action}")
            bd = s.score_breakdown
            print(f"    Pattern={bd['pattern_strength']} | Keyword={bd['keyword_proximity']} | "
                  f"CoOccur={bd['co_occurrence']} | Format={bd['format_validity']}")
            if bd['elevated_by_cooccur'] != "—":
                print(f"    ⚠ Elevated to {bd['elevated_by_cooccur']} by co-occurrence")
