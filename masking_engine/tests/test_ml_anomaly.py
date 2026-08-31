"""
test_ml_anomaly.py — Tests for the ML safety-net layer (engine/ml_anomaly.py)
R26-CS-012: Context-Aware Masking + Instruction Engine

Covers the gating logic that must hold regardless of whether scikit-learn
is installed in the test environment: Layer 2 must never override or run
when Layer 1 already masked something, the structural-evidence gate
(added after an evaluated false-positive: the trained classifier alone
scores plain benign sentences ~95%+ "sensitive" — see ml_anomaly.py) must
suppress classifier-only false alarms, and WHAT gets redacted once Layer 2
does act must always trace to the deterministic span-finding rule
(_find_suspicious_spans), never to the model's output alone.
"""

import json
import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.ml_anomaly import (
    apply_safety_net, _has_structural_evidence, _extract_features,
    _find_suspicious_spans, get_ml_flag,
)
from engine.token_registry import TokenRegistry

_PROBE_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research", "ood_probe_set.json",
)


@dataclass
class _FakeMaskedResult:
    masked_entities : List[dict] = field(default_factory=list)
    skipped_entities: List[dict] = field(default_factory=list)
    masked_text     : str        = ""
    overall_risk    : str        = "LOW"


class TestStructuralEvidenceGate(unittest.TestCase):
    """The classifier's raw probability alone is not trusted — see the
    documented false-positive in ml_anomaly.py. This gate is what prevents
    it from firing on ordinary text with none of the 17 features'
    structural secret-markers."""

    def test_plain_sentence_has_no_structural_evidence(self):
        features = _extract_features(
            "Please send me the report by Friday, thanks for the update on the branch meeting."
        )
        self.assertFalse(_has_structural_evidence(features))

    def test_base64_shaped_text_has_structural_evidence(self):
        features = _extract_features(
            "Attach this bearer credential: Q7mZx2vLp9Tn4Rc8Yw1Fh6Ub3Ej0Sg5Aq."
        )
        self.assertTrue(_has_structural_evidence(features))


class TestSpanLocation(unittest.TestCase):
    """WHAT gets redacted must always come from the deterministic regex
    rule, never from the model — these tests exercise that rule in
    isolation from any classifier availability."""

    def test_plain_sentence_has_no_locatable_span(self):
        self.assertEqual(
            _find_suspicious_spans(
                "Please send me the report by Friday, thanks for the update."
            ),
            [],
        )

    def test_locates_long_opaque_token(self):
        text = "Attach this bearer credential: Q7mZx2vLp9Tn4Rc8Yw1Fh6Ub3Ej0Sg5Aq."
        spans = _find_suspicious_spans(text)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(text[start:end], "Q7mZx2vLp9Tn4Rc8Yw1Fh6Ub3Ej0Sg5Aq")

    def test_overlapping_candidate_patterns_merge_into_one_span(self):
        # A long opaque token that ALSO looks base64-shaped must be
        # redacted once, not twice/overlapping.
        text = "token: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        spans = _find_suspicious_spans(text)
        self.assertEqual(len(spans), 1)


class TestApplySafetyNet(unittest.TestCase):
    def setUp(self):
        self.registry = TokenRegistry(session_id="test_ml_anomaly")

    def test_no_op_when_layer1_already_masked_something(self):
        result = _FakeMaskedResult(masked_entities=[{"entity_type": "PAN"}])
        apply_safety_net("irrelevant text", result, self.registry)
        self.assertEqual(result.skipped_entities, [])

    def test_no_flag_and_no_mask_for_ordinary_benign_text(self):
        # Below threshold / no structural evidence at all — Layer 2 must
        # stay completely silent, not just avoid masking.
        result = _FakeMaskedResult(
            masked_text="Please send me the report by Friday, thanks for the update."
        )
        apply_safety_net(
            "Please send me the report by Friday, thanks for the update.",
            result, self.registry,
        )
        self.assertEqual(result.masked_entities, [])
        self.assertEqual(result.skipped_entities, [])

    def test_without_a_registry_falls_back_to_review_only_flag(self):
        from engine.ml_anomaly import is_available
        if not is_available():
            self.skipTest("scikit-learn/joblib not installed in this environment")
        # No registry passed → Layer 2 must never guess at masking; it can
        # only ever fall back to a review flag, even when it's confident
        # and could otherwise have located a span (see test below, which
        # runs the same text WITH a registry and gets it masked instead).
        text = "Bearer credential for the sandbox environment: Zx9kQm2vLp7Tn4Rc8Yw1Fh6Ub3Ej0Sg5Aq2Bn9Cx4Rt3Ol8Wm1Ez7."
        result = _FakeMaskedResult(masked_text=text)
        apply_safety_net(text, result, registry=None)
        self.assertEqual(result.masked_entities, [])
        self.assertEqual(len(result.skipped_entities), 1)
        self.assertIn("no_span", result.skipped_entities[0]["reason"])

    def test_locates_and_masks_span_when_flagged_with_registry(self):
        from engine.ml_anomaly import is_available
        if not is_available():
            self.skipTest("scikit-learn/joblib not installed in this environment")
        # Known Layer-1 miss / Layer-2 catch (verified against the live
        # pipeline): no regex/NER rule matches "bearer" + an opaque token,
        # but the classifier + structural gate both fire on it.
        text = "Bearer credential for the sandbox environment: Zx9kQm2vLp7Tn4Rc8Yw1Fh6Ub3Ej0Sg5Aq2Bn9Cx4Rt3Ol8Wm1Ez7."
        result = _FakeMaskedResult(masked_text=text)
        apply_safety_net(text, result, self.registry)

        self.assertEqual(result.skipped_entities, [])
        self.assertEqual(len(result.masked_entities), 1)
        record = result.masked_entities[0]
        self.assertEqual(record["entity_type"], "ML_FLAGGED_ANOMALY")
        self.assertEqual(record["action"], "ml_flagged_mask")
        self.assertIn(record["original"], text)
        self.assertNotIn(record["original"], result.masked_text)
        self.assertIn(record["replacement"], result.masked_text)
        self.assertEqual(result.overall_risk, "MEDIUM")

        # Same value again in the same session must resolve to the same
        # token — Layer 2 inherits Layer 1's idempotency guarantee.
        result2 = _FakeMaskedResult(masked_text=text)
        apply_safety_net(text, result2, self.registry)
        self.assertEqual(record["replacement"], result2.masked_entities[0]["replacement"])


class TestOodProbeSet(unittest.TestCase):
    """Permanent regression gate: research/ood_probe_set.json is 40 hand-
    authored, genuinely benign sentences NEVER used in training (see
    research/retrain_classifier.py). If a future retrain regresses on
    out-of-distribution generalization (the original bug — see
    docs/Comprehensive Technical Documentation.md Section 8.3), the
    production apply_safety_net() call — model + structural gate together,
    exactly as main.py/web/server.py/evaluate.py call it — must still
    catch it here even if the classifier's raw score alone drifts."""

    @unittest.skipUnless(
        os.path.isfile(_PROBE_SET_PATH), "ood_probe_set.json not present"
    )
    def test_gated_safety_net_never_flags_the_probe_set(self):
        from engine.ml_anomaly import is_available
        if not is_available():
            self.skipTest("scikit-learn/joblib not installed in this environment")

        registry = TokenRegistry(session_id="test_ood_probe")
        probes = json.load(open(_PROBE_SET_PATH))
        false_positives = []
        for p in probes:
            result = _FakeMaskedResult(masked_text=p["prompt"])
            apply_safety_net(p["prompt"], result, registry)
            if get_ml_flag(result) is not None:  # flagged OR masked — both are false alarms here
                false_positives.append(p["prompt"])

        self.assertEqual(
            false_positives, [],
            f"Gated safety net flagged {len(false_positives)} genuinely benign, "
            f"never-trained-on prompt(s) — this is the out-of-distribution "
            f"overfitting regression. Run research/retrain_classifier.py "
            f"(it refuses to save a model that fails this check) rather than "
            f"loosening this test.",
        )


if __name__ == "__main__":
    unittest.main()
