"""
instruction_generator.py — Instruction Block Generator
R26-CS-012: Context-Aware Masking + Instruction Engine

Generates the instruction payload appended to masked prompts
before transmission to external AI systems (Taxonomy Section 11).

Two template tiers:
  - Standard    : for MEDIUM/LOW risk prompts
  - CRITICAL    : for HIGH/CRITICAL risk prompts (credentials, SWIFT, financial IDs)

WHY INSTRUCTION INJECTION (not just masking):
  - Masking alone does not prevent the LLM from guessing original values
  - Instructions explicitly prohibit reconstruction, inference, and disclosure requests
  - Idempotency note ensures LLM treats <PERSON_1> consistently across turns
  - Required by SWIFT CSP Mandatory Control 6.1 for SWIFT message data
"""

import json
from typing import List
from engine.masker import MaskedResult


# ─────────────────────────────────────────────
# INSTRUCTION TEMPLATES (Taxonomy Section 11)
# ─────────────────────────────────────────────

STANDARD_INSTRUCTIONS = [
    "This prompt contains masked sensitive data represented by tokens (e.g., <APIKEY_1>) or asterisks.",
    "Do not attempt to infer, reconstruct, or request the original values of any masked tokens.",
    "Treat all masked values as placeholders. Do not generate content based on their likely real values.",
    "Do not request the user to reveal masked information.",
    "If a token (e.g., <PERSON_1>) appears in multiple messages, treat it as consistently referring to the same entity.",
]

CRITICAL_INSTRUCTIONS = [
    "CRITICAL: This prompt contained highly sensitive data (credentials/financial identifiers/SWIFT data) that has been masked.",
    "Do not attempt to reconstruct, predict, or suggest the likely values of any masked tokens.",
    "Do not provide any guidance, code, or reasoning that would assist in recovering masked values.",
    "Treat this interaction as security-sensitive. Flag for security review if applicable.",
    "Token references are session-consistent. Each token refers to the same masked value throughout this session.",
]

# Entity types that always trigger CRITICAL instructions regardless of score
CRITICAL_ENTITY_TYPES = {
    "CVV", "PRIVATE_KEY", "SWIFT_MT103", "SWIFT_MT202",
    "PASSWORD", "JWT_TOKEN", "JWT_IN_LOG",
    "AWS_SECRET_KEY", "DB_CONNECTION_STRING",
}


# ─────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────

def generate_instructions(masked_result: MaskedResult) -> dict:
    """
    Determine instruction tier and build the full instruction payload.

    Returns a dict with:
      - tier         : 'standard' | 'critical'
      - instructions : list of instruction strings
      - masked_count : number of masked entities
      - risk_level   : overall risk from MaskedResult
      - token_refs   : list of token labels used (for idempotency note)
    """
    # Collect info from masked entities
    entity_types = {m["entity_type"] for m in masked_result.masked_entities}
    token_refs   = [
        m["replacement"] for m in masked_result.masked_entities
        if m["replacement"].startswith("<")
    ]

    # Determine tier
    is_critical = (
        masked_result.overall_risk in ("CRITICAL", "HIGH")
        or bool(entity_types & CRITICAL_ENTITY_TYPES)
    )

    tier         = "critical" if is_critical else "standard"
    instructions = CRITICAL_INSTRUCTIONS if is_critical else STANDARD_INSTRUCTIONS

    # Add dynamic token idempotency note if tokens were used
    if token_refs:
        token_list = ", ".join(set(token_refs))
        idempotency_note = (
            f"Session tokens present: {token_list}. "
            "Each token consistently refers to the same original value across all prompts in this session."
        )
        instructions = list(instructions) + [idempotency_note]

    return {
        "tier"        : tier,
        "instructions": instructions,
        "masked_count": len(masked_result.masked_entities),
        "risk_level"  : masked_result.overall_risk,
        "token_refs"  : list(set(token_refs)),
    }


def format_for_display(instruction_payload: dict) -> str:
    """
    Format the instruction payload as a readable block
    for terminal output or logging.
    """
    tier  = instruction_payload["tier"].upper()
    risk  = instruction_payload["risk_level"]
    count = instruction_payload["masked_count"]
    lines = [
        f"┌─ INSTRUCTION BLOCK [{tier}] ─ Risk: {risk} ─ Entities masked: {count}",
    ]
    for i, instr in enumerate(instruction_payload["instructions"], 1):
        lines.append(f"│  {i}. {instr}")
    lines.append("└" + "─" * 60)
    return "\n".join(lines)


def format_as_json(instruction_payload: dict) -> str:
    """Export instruction payload as JSON for integration with external AI API calls."""
    return json.dumps({"masking_instructions": instruction_payload["instructions"]}, indent=2)


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from engine.normalizer import normalize
    from engine.detector import detect
    from engine.confidence_scorer import score_all
    from engine.masker import mask
    from engine.token_registry import TokenRegistry

    registry = TokenRegistry()
    registry.next_prompt()

    examples = [
        "Please send report to saman.perera@seylan.lk about the overdue loan.",
        "API key sk-abcdef1234567890abcdef1234567890 is getting a 401 — check.",
        "SWIFT MT103: Saman Perera NIC 199012345V, account 001010012345, LKR 250000.",
    ]

    for text in examples:
        norm    = normalize(text)
        raw     = detect(norm["normalized"], norm["despaced"], norm["despaced_map"])
        scored  = score_all(raw, norm["normalized"])
        result  = mask(norm["normalized"], scored, registry)
        payload = generate_instructions(result)

        print(f"\nOriginal : {text}")
        print(f"Masked   : {result.masked_text}")
        print(format_for_display(payload))
