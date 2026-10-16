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

from engine.detector import DetectedEntity, BOOST_KEYWORDS, _in_same_ambiguous_group
from engine.normalizer import get_context_window

# Types ambiguous purely by digit shape — see score_entity() below. A
# 10-digit "071..." number is exactly as plausible as an invoice/reference
# number as it is a phone number, so PHONE_LK/PHONE_INTL belong here too:
# their format validators (valid mobile/landline prefix, digit count) confirm
# structure, not sensitivity, and shouldn't alone push a contextless number
# past the masking threshold.
AMBIGUOUS_DIGIT_TYPES = {
    "NIC_NEW", "NIC_OLD", "TAX_ID", "BANK_ACCOUNT_NO", "PAN",
    "PHONE_LK", "PHONE_INTL",
}


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
    A successful regex match is, by default, full strength (1.0) — per the
    taxonomy's own worked example (Section 4: a bare NIC_NEW match
    contributes the full 40%, with ambiguity handled by the other three
    factors, not by discounting the match itself). A handful of types with
    genuine internal branching in their regex are graded more precisely.
    """
    value, et = entity.value, entity.entity_type

    if et == "PASSPORT":
        letters = re.match(r'^[A-Z]+', value)
        digits  = re.search(r'\d+$', value)
        return 1.0 if (letters and len(letters.group()) == 2
                        and digits and len(digits.group()) == 7) else 0.5

    if et == "SWIFT_BIC":
        return 1.0 if len(value) == 11 else 0.5  # 11-char form includes branch code

    if et == "API_KEY_GENERIC":
        return 1.0 if len(value) >= 40 else 0.5  # longer = exponentially less coincidental

    return 1.0


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

    # A keyword can be glued directly onto the value with no whitespace
    # between them (e.g. "PASSWORD=Bank@2026" or "JWT_SECRET=..." is one
    # single whitespace-delimited token) — get_context_window() tokenizes
    # on whitespace, so it excludes that *entire* token (keyword included)
    # as "the entity's own token", hiding the keyword from context_tokens
    # even though the entity's span (post capture-group extraction, see
    # detector.py) covers only the value, not the keyword. A raw character
    # lookback catches it regardless of tokenization, the same way
    # detector._has_gating_keyword() does for detection-time gating.
    adjacent = text[max(0, entity.start - 25):entity.start].lower()
    combined = f"{context_lower} {entity.value.lower()} {adjacent}"

    for kw in keywords:
        if kw.lower() in combined:
            return 1.0  # exact keyword found

    # Partial: any word in context shares prefix with a keyword
    for kw in keywords:
        kw_root = kw.lower().split()[0][:4]  # first 4 chars of first word
        if kw_root in combined:
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
      - Known dangerous pair (CO_OCCURRENCE_PAIRS) → score 1.0, elevate
      - 3+ distinct OTHER entity types present → score 0.75 (graduated
        boost, NOT an automatic CRITICAL — see note below)
      - Any other co-occurrence → score 0.5
      - No co-occurrence → score 0.0

    NOTE — deliberate deviation from Taxonomy Section 7's literal
    "Any 3+ entities in one prompt → CRITICAL regardless" rule: that rule,
    applied as a blanket count regardless of relatedness, was found to
    actively amplify clusters of false-positive detections into forced
    mask_immediate/CRITICAL status rather than filtering them. Softened
    here to a graduated boost; only the curated CO_OCCURRENCE_PAIRS
    relationships still force CRITICAL. See Model_Regex_Docs.md for the
    rationale write-up.
    """
    other_types = {e.entity_type for e in all_entities if e is not entity}

    if not other_types:
        return 0.0, None

    # Check known dangerous pairs
    my_type = entity.entity_type
    for pair, elevated in CO_OCCURRENCE_PAIRS.items():
        if my_type in pair:
            other_in_pair = pair - {my_type}
            if other_in_pair & other_types:
                return 1.0, elevated

    if len(other_types) >= 3:
        return 0.75, None

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


# Sri Lankan landline area codes (2-digit, after the leading 0), used by
# phone_lk_check alongside the mobile-network prefixes.
_LK_LANDLINE_CODES = {
    "11", "21", "23", "24", "25", "26", "27", "31", "32", "33", "34", "35",
    "36", "37", "38", "41", "45", "47", "51", "52", "54", "55", "57", "63",
    "65", "66", "67", "81", "91",
}
_LK_MOBILE_PREFIXES = {"70", "71", "72", "74", "75", "76", "77", "78"}


def tax_id_check(value: str) -> bool:
    """No public checksum for TIN — reject only obviously-fake runs
    (all-same-digit) rather than validating structure we can't verify."""
    return len(set(value)) > 1


def cvv_check(value: str) -> bool:
    return value not in {"000", "0000"} and len(set(value)) > 1


def bank_account_no_check(value: str) -> bool:
    digits = re.sub(r'\D', '', value)
    if len(set(digits)) <= 1:
        return False
    # reject simple ascending/descending runs (e.g. "1234567890")
    ascending  = "".join(str((int(digits[0]) + i) % 10) for i in range(len(digits)))
    descending = "".join(str((int(digits[0]) - i) % 10) for i in range(len(digits)))
    return digits not in (ascending, descending)


def card_expiry_check(value: str) -> bool:
    m = re.match(r'^(\d{2})/(\d{2}|\d{4})$', value)
    if not m:
        return False
    month = int(m.group(1))
    if not (1 <= month <= 12):
        return False
    year = m.group(2)
    return len(year) == 2 or (2000 <= int(year) <= 2099)


def api_key_generic_check(value: str) -> bool:
    """Character-diversity proxy for entropy — a real key/token uses a
    wide alphabet, unlike a collapsed run of ordinary words."""
    return len(set(value)) >= max(8, len(value) // 4)


def swift_bic_check(value: str) -> bool:
    if len(value) not in (8, 11):
        return False
    return value[4:6].isalpha() and value[6:8] != "00"  # "00" location code is reserved


def passport_check(value: str) -> bool:
    return len(set(re.sub(r'\D', '', value))) > 1


def driving_license_check(value: str) -> bool:
    return len(set(re.sub(r'\D', '', value))) > 1


def phone_lk_check(value: str) -> bool:
    body = re.sub(r'^(?:\+94|0)', '', value)
    return body[:2] in _LK_MOBILE_PREFIXES or body[:2] in _LK_LANDLINE_CODES


def phone_intl_check(value: str) -> bool:
    digits = re.sub(r'\D', '', value)
    return 8 <= len(digits) <= 15


def aws_access_key_check(value: str) -> bool:
    return len(set(value[4:])) > 4  # after the fixed "AKIA" prefix


def aws_secret_key_check(value: str) -> bool:
    secret = re.split(r'[=:]\s*', value)[-1]
    return len(set(secret)) >= 20


def internal_ip_check(value: str) -> bool:
    """The regex already restricts matches to RFC1918 private ranges, but
    each octet's numeric range (0-255) isn't enforced by the pattern itself
    (e.g. '192.168.999.999' still matches \\d{1,3}). Validating that closes
    the gap and gives this structurally unambiguous format (there's no
    legitimate non-IP reading of a valid private-range address) the same
    real confirming evidence other precise types get from their validators."""
    octets = value.split(".")
    if len(octets) != 4:
        return False
    return all(o.isdigit() and 0 <= int(o) <= 255 for o in octets)


def s3_bucket_ref_check(value: str) -> bool:
    bucket = value[5:].split("/")[0] if value.startswith("s3://") else value
    return (3 <= len(bucket) <= 63 and re.fullmatch(r'[a-z0-9.\-]+', bucket) is not None
            and not bucket.startswith("-") and not bucket.endswith("-")
            and ".." not in bucket)


FORMAT_VALIDATORS = {
    "PAN"             : luhn_check,
    "NIC_OLD"         : nic_old_check,
    "NIC_NEW"         : nic_new_check,
    "IBAN"            : iban_check,
    "TAX_ID"          : tax_id_check,
    "CVV"             : cvv_check,
    "BANK_ACCOUNT_NO" : bank_account_no_check,
    "CARD_EXPIRY"     : card_expiry_check,
    "API_KEY_GENERIC" : api_key_generic_check,
    "SWIFT_BIC"       : swift_bic_check,
    "PASSPORT"        : passport_check,
    "DRIVING_LICENSE" : driving_license_check,
    "PHONE_LK"        : phone_lk_check,
    "PHONE_INTL"      : phone_intl_check,
    "AWS_ACCESS_KEY"  : aws_access_key_check,
    "AWS_SECRET_KEY"  : aws_secret_key_check,
    "S3_BUCKET_REF"   : s3_bucket_ref_check,
    "INTERNAL_IP"     : internal_ip_check,
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

    # For types that are ambiguous purely by digit shape (a "valid-looking"
    # NIC/account/tax-ID number is structurally indistinguishable from a
    # coincidentally similar-looking random number), a format check alone
    # is confirming evidence, not standalone justification — per Taxonomy
    # Section 4: "Standalone 12-digit numbers without context score <=0.40
    # and are not masked." Without this, pattern(0.40) + format(0.10) = 0.50
    # would cross into mask_warn for a bare, contextless number that merely
    # happens to pass a lenient structural check (e.g. a random 12-digit
    # serial number whose first 4 digits look like a plausible birth year).
    if entity.entity_type in AMBIGUOUS_DIGIT_TYPES and f2 == 0 and f3_raw == 0:
        f4 = 0.0

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
# OVERLAP / CONFLICT RESOLUTION
# ─────────────────────────────────────────────

def _spans_overlap(a: DetectedEntity, b: DetectedEntity) -> bool:
    return a.start < b.end and b.start < a.end


def resolve_overlapping_entities(scored_entities: List[ScoredEntity]) -> List[ScoredEntity]:
    """
    Run AFTER score_all(). Some entity types are structurally ambiguous
    (detector.AMBIGUOUS_TYPE_GROUPS — e.g. a 12-digit number could be
    NIC_NEW or BANK_ACCOUNT_NO) and detect() deliberately lets both
    candidates through. This picks the higher-scoring candidate per
    overlapping span so context/validators decide the winner, not
    pattern-registration order. Tie-break: longer span (more specific),
    then stable first-seen.
    """
    to_drop = set()
    for i, a in enumerate(scored_entities):
        for b in scored_entities[i + 1:]:
            ta, tb = a.entity.entity_type, b.entity.entity_type
            # Same-type overlapping spans are duplicates of each other
            # (e.g. the same value re-detected via both the normal and
            # despaced passes with slightly different offsets — fixes
            # v.xlsx #9) and are always resolved here too; different types
            # are only resolved when structurally ambiguous per detector.py.
            if ta != tb and not _in_same_ambiguous_group(ta, tb):
                continue
            if not _spans_overlap(a.entity, b.entity):
                continue
            if a.score != b.score:
                loser = a if a.score < b.score else b
            else:
                len_a = a.entity.end - a.entity.start
                len_b = b.entity.end - b.entity.start
                loser = a if len_a < len_b else b
            to_drop.add(id(loser))
    return [s for s in scored_entities if id(s) not in to_drop]


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
        raw     = detect(norm["normalized"], norm["despaced"], norm["despaced_map"])
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
