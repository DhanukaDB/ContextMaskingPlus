"""
main.py — Terminal Entry Point
R26-CS-012: Context-Aware Masking + Instruction Engine

Usage:
  python main.py                    # interactive mode
  python main.py --demo             # run built-in demo cases
  python main.py --text "..."       # process a single prompt
  python main.py --file prompts.txt # process all lines in a file
"""

import sys
import os
import argparse
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.normalizer import normalize
from engine.detector import detect
from engine.confidence_scorer import score_all, resolve_overlapping_entities
from engine.masker import mask
from engine.token_registry import TokenRegistry
from engine.instruction_generator import generate_instructions, format_for_display


# ─────────────────────────────────────────────
# COLOURS (terminal)
# ─────────────────────────────────────────────

class C:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def color_risk(risk):
    colours = {"CRITICAL": C.RED, "HIGH": C.RED, "MEDIUM": C.YELLOW, "LOW": C.GREEN}
    return colours.get(risk, "") + risk + C.RESET


# ─────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────

def process_prompt(text: str, registry: TokenRegistry, verbose: bool = True) -> dict:
    """
    Full pipeline for a single prompt.
    Returns structured result dict.
    """
    # Step 1 — Normalize
    norm = normalize(text)

    # Step 2 — Detect
    raw_entities = detect(norm["normalized"], norm["despaced"])

    # Step 3 — Score
    scored_entities = score_all(raw_entities, norm["normalized"])
    scored_entities = resolve_overlapping_entities(scored_entities)

    # Step 4 — Mask
    masked_result = mask(norm["normalized"], scored_entities, registry)

    # Step 5 — Generate instructions
    instruction_payload = generate_instructions(masked_result)

    if verbose:
        print_result(text, norm, scored_entities, masked_result, instruction_payload)

    return {
        "original"           : text,
        "normalized"         : norm,
        "detected"           : scored_entities,
        "masked_result"      : masked_result,
        "instruction_payload": instruction_payload,
    }


def print_result(text, norm, scored_entities, masked_result, instruction_payload):
    """Pretty-print a single prompt result to terminal."""

    print(f"\n{'═'*65}")
    print(f"{C.BOLD}  INPUT PROMPT{C.RESET}")
    print(f"{'═'*65}")
    print(f"  {C.DIM}{text}{C.RESET}")

    # Preprocessing info
    if norm["transformations"]:
        print(f"\n  {C.CYAN}[PREPROCESSING]{C.RESET}")
        print(f"  Context type   : {norm['context_type']}")
        print(f"  Transformations: {', '.join(norm['transformations'])}")
        print(f"  Normalized to  : {norm['normalized']}")

    # Detection + scoring
    if scored_entities:
        print(f"\n  {C.CYAN}[DETECTED ENTITIES]{C.RESET}")
        print(f"  {'Entity Type':<22} {'Value':<30} {'Score':>6}  {'Level':<10} {'Action'}")
        print(f"  {'─'*22} {'─'*30} {'─'*6}  {'─'*10} {'─'*15}")
        for s in scored_entities:
            e     = s.entity
            val   = (e.value[:27] + "...") if len(e.value) > 30 else e.value
            level = s.confidence_level
            level_col = {
                "Very High": C.RED, "High": C.RED,
                "Medium": C.YELLOW, "Low": C.GREEN, "Very Low": C.DIM
            }.get(level, "")
            print(f"  {e.entity_type:<22} {val:<30} {s.score:>6.2f}  "
                  f"{level_col}{level:<10}{C.RESET} {s.action}")
    else:
        print(f"\n  {C.GREEN}[NO SENSITIVE ENTITIES DETECTED]{C.RESET}")

    # Masking output
    print(f"\n  {C.CYAN}[MASKING RESULT]{C.RESET}")
    print(f"  Risk Level : {color_risk(masked_result.overall_risk)}")
    print(f"  Masked text: {C.BOLD}{masked_result.masked_text}{C.RESET}")

    if masked_result.masked_entities:
        print(f"\n  Masked entities:")
        for m in masked_result.masked_entities:
            orig = (m['original'][:25] + "...") if len(m['original']) > 28 else m['original']
            print(f"    [{m['entity_type']}] '{orig}' → '{m['replacement']}' "
                  f"({m['strategy']}, score={m['score']})")

    if masked_result.skipped_entities:
        print(f"\n  Skipped (not masked):")
        for sk in masked_result.skipped_entities:
            print(f"    [{sk['entity_type']}] '{sk['value']}' — reason: {sk['reason']}")

    # Instructions
    print(f"\n  {format_for_display(instruction_payload)}")


# ─────────────────────────────────────────────
# DEMO CASES
# ─────────────────────────────────────────────

DEMO_CASES = [
    {
        "label": "Normal — Phone + Email",
        "text" : "Please send the statement to saman.perera@seylan.lk and call 0771234567.",
    },
    {
        "label": "Normal — Credit Card (PCI-DSS)",
        "text" : "Card number 4111 1111 1111 1111, CVV 123, expiry 12/27 — is this valid?",
    },
    {
        "label": "Normal — API Key",
        "text" : "Getting 401: api_key=sk-abcdefghijklmnop1234567890abcdef",
    },
    {
        "label": "Normal — DB Connection String",
        "text" : "App cannot connect: postgresql://admin:Secure99@10.0.0.5:5432/corebank_prod",
    },
    {
        "label": "Normal — SWIFT CRITICAL",
        "text" : "MT103 from BCEYLKLX: Saman Perera NIC 199012345V, LKR 250,000 to account 001010012345.",
    },
    {
        "label": "Ambiguity — 12-digit NO context (should NOT mask)",
        "text" : "Serial number for batch: 200423910321 — log for inventory.",
    },
    {
        "label": "Ambiguity — 12-digit WITH context (should mask at MEDIUM)",
        "text" : "Customer id is 200423910321, please verify KYC status.",
    },
    {
        "label": "Co-occurrence — Name + NIC → CRITICAL elevation",
        "text" : "Check account for Saman Perera, NIC 199012345V.",
    },
    {
        "label": "Adversarial — Spaced NIC",
        "text" : "Customer NIC on file: 1 9 9 0 1 2 3 4 5 V — is this valid?",
    },
    {
        "label": "Adversarial — URL-encoded password",
        "text" : "URL param received: password%3DSecure99%40 — is this safe?",
    },
    {
        "label": "Cloud — AWS Keys",
        "text" : "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    },
    {
        "label": "Session idempotency — same API key in second prompt",
        "text" : "Same key again: sk-abcdefghijklmnop1234567890abcdef — should get same token.",
    },
]


def run_demo(registry: TokenRegistry):
    print(f"\n{C.BOLD}{'═'*65}")
    print(f"  R26-CS-012  |  Context-Aware Masking Engine  |  DEMO RUN")
    print(f"{'═'*65}{C.RESET}")
    print(f"  Session: {registry.session_id}")

    for i, case in enumerate(DEMO_CASES, 1):
        print(f"\n\n{C.BOLD}── CASE {i:02d}: {case['label']} ──{C.RESET}")
        registry.next_prompt()
        process_prompt(case["text"], registry, verbose=True)

    print(f"\n\n{'═'*65}")
    print(f"{C.BOLD}  SESSION TOKEN REGISTRY{C.RESET}")
    print(f"{'═'*65}")
    print(registry.summary())
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

def interactive_mode(registry: TokenRegistry):
    print(f"\n{C.BOLD}R26-CS-012 — Masking Engine (Interactive){C.RESET}")
    print(f"Session: {registry.session_id}")
    print(f"Type 'exit' to quit | 'tokens' to view session token map\n")

    prompt_num = 0
    while True:
        try:
            user_input = input(f"{C.CYAN}[Prompt {prompt_num+1}]>{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("\nSession ended. Token registry destroyed.")
            registry.destroy()
            break
        if user_input.lower() == "tokens":
            print(f"\n{registry.summary()}\n")
            continue

        prompt_num += 1
        registry.next_prompt()
        process_prompt(user_input, registry, verbose=True)


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="R26-CS-012 Context-Aware Masking + Instruction Engine"
    )
    parser.add_argument("--demo",  action="store_true", help="Run built-in demo cases")
    parser.add_argument("--text",  type=str,            help="Process a single prompt string")
    parser.add_argument("--file",  type=str,            help="Process each line in a text file")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output (JSON only)")
    args = parser.parse_args()

    registry = TokenRegistry()

    if args.demo:
        run_demo(registry)

    elif args.text:
        registry.next_prompt()
        process_prompt(args.text, registry, verbose=not args.quiet)

    elif args.file:
        if not os.path.isfile(args.file):
            print(f"File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            registry.next_prompt()
            process_prompt(line, registry, verbose=not args.quiet)

    else:
        # Default — interactive
        interactive_mode(registry)


if __name__ == "__main__":
    main()
