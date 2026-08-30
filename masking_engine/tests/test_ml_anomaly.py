"""
test_ml_anomaly.py — Tests for the ML safety-net layer (engine/ml_anomaly.py)
R26-CS-012: Context-Aware Masking + Instruction Engine

Covers the gating logic that must hold regardless of whether scikit-learn
is installed in the test environment: Layer 2 must never override or run
when Layer 1 already masked something, and the structural-evidence gate
(added after an evaluated false-positive: the trained classifier alone
scores plain benign sentences ~95%+ "sensitive" — see ml_anomaly.py) must
suppress classifier-only false alarms.
"""

import json
import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.ml_anomaly import apply_safety_net, _has_structural_evidence, _extract_features

_PROBE_SET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research", "ood_probe_set.json",
)


@dataclass
class _FakeMaskedResult:
    masked_entities : List[dict] = field(default_factory=list)
    skipped_entities: List[dict] = field(default_factory=list)


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


class TestApplySafetyNet(unittest.TestCase):
    def test_no_op_when_layer1_already_masked_something(self):
        result = _FakeMaskedResult(masked_entities=[{"entity_type": "PAN"}])
        apply_safety_net("irrelevant text", result)
        self.assertEqual(result.skipped_entities, [])

    def test_never_adds_to_masked_entities(self):
        # Layer 2 must only ever be able to add a skipped/flagged record —
        # it must never be capable of masking text itself.
        result = _FakeMaskedResult()
        apply_safety_net(
            "Please send me the report by Friday, thanks for the update.", result
        )
        self.assertEqual(result.masked_entities, [])


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

        probes = json.load(open(_PROBE_SET_PATH))
        false_positives = []
        for p in probes:
            result = _FakeMaskedResult()
            apply_safety_net(p["prompt"], result)
            if result.skipped_entities:
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
