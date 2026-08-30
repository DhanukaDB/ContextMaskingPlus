"""
canonical_adapter.py — Canonical Prompt+Context Ingestion Adapter
R26-CS-012: Context-Aware Masking + Instruction Engine

Accepts the "Canonical Prompt and Context Structuring" JSON contract
produced by the upstream RAG/retrieval component — a user `request.prompt`
plus a `context[]` list of code blocks retrieved by semantic similarity
(FAISS score/confidence, source file, line range) — and runs the full
two-layer detection pipeline (Layer 1 regex/NER + Layer 2 ML safety net,
see docs/Comprehensive Technical Documentation.md Section 3) over EVERY
text field that could carry sensitive data.

WHY THE RETRIEVED CONTEXT BLOCKS ARE SCANNED TOO, NOT JUST THE PROMPT:
  Retrieved source code is exactly as likely to carry a hardcoded secret,
  an internal IP, or a real customer ID left in a comment as anything the
  user typed themselves — arguably more likely, since it's pulled from a
  real codebase rather than composed fresh. A masking layer that only
  looks at `request.prompt` and ignores `context[].content` would leave
  the single largest source of pasted-in sensitive text in this contract
  completely unguarded.

WHY EACH BLOCK IS MASKED INDEPENDENTLY, NOT CONCATENATED:
  - Preserves per-block source/location metadata for audit — an incident
    review needs to know WHICH retrieved file leaked WHAT, not just that
    something in the combined context did.
  - Nothing is dropped from the payload's shape; `source`, `location`,
    `retrieval`, and `reason` are untouched so downstream components still
    get the exact contract they expect, just with `content` (and
    `request.prompt`) replaced by their masked text.
"""

import copy
from typing import Any, Dict

from engine.normalizer import normalize
from engine.detector import detect
from engine.confidence_scorer import score_all, resolve_overlapping_entities
from engine.masker import mask
from engine.ml_anomaly import apply_safety_net
from engine.token_registry import TokenRegistry


def _mask_field(text: str, registry: TokenRegistry) -> Dict[str, Any]:
    registry.next_prompt()
    norm = normalize(text)
    raw = detect(norm["normalized"], norm["despaced"], norm["despaced_map"])
    scored = resolve_overlapping_entities(score_all(raw, norm["normalized"]))
    result = mask(norm["normalized"], scored, registry)
    apply_safety_net(norm["normalized"], result)

    ml_flag = next(
        (sk for sk in result.skipped_entities if sk["reason"] == "ml_anomaly_flagged"),
        None,
    )
    return {
        "masked_text": result.masked_text,
        "overall_risk": result.overall_risk,
        "masked_entities": [
            {
                "entity_type": m["entity_type"],
                "strategy": m["strategy"],
                "sensitivity": m["sensitivity"],
            }
            for m in result.masked_entities
        ],
        "ml_flagged": ml_flag is not None,
    }


def process_canonical_request(
    payload: Dict[str, Any],
    registry: TokenRegistry = None,
) -> Dict[str, Any]:
    """
    Returns a deep copy of `payload` with `request.prompt` and every
    `context[i].content` masked in place, each annotated with a `masking`
    block documenting what was found (entity types, strategy, risk level,
    whether the ML safety net flagged it) — everything else in the input
    (source/location/retrieval/reason/metadata) is passed through
    unchanged so the contract the next component expects still holds.

    A single TokenRegistry is shared across the whole request (prompt +
    every context block) so the same secret repeated across the prompt
    and a retrieved snippet resolves to the same token — the idempotency
    guarantee in token_registry.py applies across the whole payload, not
    just within one field.
    """
    if registry is None:
        registry = TokenRegistry()

    out = copy.deepcopy(payload)
    risk_levels = []

    request = out.get("request")
    if isinstance(request, dict):
        prompt = request.get("prompt", "")
        if prompt:
            result = _mask_field(prompt, registry)
            request["prompt"] = result["masked_text"]
            request["masking"] = {
                k: v for k, v in result.items() if k != "masked_text"
            }
            risk_levels.append(result["overall_risk"])

    for block in out.get("context", []) or []:
        if not isinstance(block, dict):
            continue
        content = block.get("content", "")
        if not content:
            continue
        result = _mask_field(content, registry)
        block["content"] = result["masked_text"]
        block["masking"] = {
            k: v for k, v in result.items() if k != "masked_text"
        }
        risk_levels.append(result["overall_risk"])

    risk_priority = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    out.setdefault("masking_summary", {})["overall_risk"] = (
        max(risk_levels, key=lambda r: risk_priority.get(r, 0)) if risk_levels else "LOW"
    )

    meta = out.setdefault("metadata", {})
    meta["masking_component"] = "R26-CS-012 Context-Aware Masking Engine"
    meta["status"] = "masked_ready_for_next_component"

    return out
