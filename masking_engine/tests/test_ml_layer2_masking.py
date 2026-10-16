"""
test_ml_layer2_masking.py — End-to-end Layer-2-only masking scenarios
R26-CS-012: Context-Aware Masking + Instruction Engine

Each prompt below carries an opaque 16–31 char secret that no Layer-1
regex/NER rule matches (too short for API_KEY_GENERIC's 32+, no known
vendor prefix), so ONLY the ML safety net can catch it. For every one,
the pipeline must:
  - report no Layer-1 entities,
  - mask exactly the secret (rest of the sentence untouched) as a
    readable <ML_SECRET_N> token, and
  - surface the catch in the entities list as a Layer-2 entity, so the
    web demo doesn't say "No sensitive entities detected" for it.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.ml_anomaly import is_available

LAYER2_ONLY_CASES = [
    ("here is the deploy token: q8Zr2LxVp9NwT4bKe7Hs1MdYc6Ju for staging", "q8Zr2LxVp9NwT4bKe7Hs1MdYc6Ju"),
    ("slack bot token xoxb7Tq2Lm9Vr4Kp8Zw3Nc6Hy1Bd please keep private", "xoxb7Tq2Lm9Vr4Kp8Zw3Nc6Hy1Bd"),
    ("stripe live token rkLive9Xq2Wm7Tz4Kp8Vn3Rb6Hy1Jc for billing service", "rkLive9Xq2Wm7Tz4Kp8Vn3Rb6Hy1Jc"),
    ("session cookie value a9F2kL7mQ3xZ8vR1tB6nW4yH0cJ5pD2s", "a9F2kL7mQ3xZ8vR1tB6nW4yH0cJ5pD2s"),
    ("the access token is M2xQ9vT7kR4zL8pW3nB6hY1cJ5dF0sG", "M2xQ9vT7kR4zL8pW3nB6hY1cJ5dF0sG"),
    ("my secret token Hk3Lp9Qz2Wm8Rt5Vx7Nb4Jc6Yd1Fs is used by the mobile app", "Hk3Lp9Qz2Wm8Rt5Vx7Nb4Jc6Yd1Fs"),
    ("api credential for partner: Tz8Wq3Lm7Kp2Xv9Rn4Hb6Jc1Yd5", "Tz8Wq3Lm7Kp2Xv9Rn4Hb6Jc1Yd5"),
    ("internal deploy key Rb6Hy1JcXq2Wm7Tz4Kp8Vn3Lm9Qs please rotate", "Rb6Hy1JcXq2Wm7Tz4Kp8Vn3Lm9Qs"),
    ("backup encryption token N4bJ8cY2dF6sG0hK3lM7qR1tV5wX9zA", "N4bJ8cY2dF6sG0hK3lM7qR1tV5wX9zA"),
    ("webhook token: Zp4Qw8Rt2Yv6Nb3Mk9Lx7Hc5Ja1Ds0", "Zp4Qw8Rt2Yv6Nb3Mk9Lx7Hc5Ja1Ds0"),
    ("token=Lq8Wz3Xv7Rn2Kp9Tm4Hb6Jc1Yd5Fs", "Lq8Wz3Xv7Rn2Kp9Tm4Hb6Jc1Yd5Fs"),
    ("refresh token 7Hs1MdYc6Juq8Zr2LxVp9NwT4bKe for the oauth client", "7Hs1MdYc6Juq8Zr2LxVp9NwT4bKe"),
]


@unittest.skipUnless(is_available(), "scikit-learn/joblib or trained model not available")
class TestLayer2OnlyMasking(unittest.TestCase):

    def setUp(self):
        from web.server import run_pipeline
        self.run_pipeline = run_pipeline

    def test_secret_masked_and_reported_as_layer2_entity(self):
        for text, secret in LAYER2_ONLY_CASES:
            with self.subTest(text=text):
                r = self.run_pipeline(text)

                layer1 = [e for e in r["entities"] if e["layer"] == 1]
                layer2 = [e for e in r["entities"] if e["layer"] == 2]
                self.assertEqual(layer1, [], "Layer 1 should not catch this case")
                self.assertEqual(len(layer2), 1)

                ml = layer2[0]
                self.assertEqual(ml["entity_type"], "ML_FLAGGED_ANOMALY")
                self.assertEqual(ml["value"], secret)
                self.assertEqual(text[ml["start"]:ml["end"]], secret)
                self.assertRegex(ml["replacement"], r"^<ML_SECRET_\d+>$")
                self.assertEqual(ml["sensitivity"], "MEDIUM")
                self.assertEqual(ml["action"], "ml_flagged_mask")
                self.assertGreaterEqual(ml["score"], 0.75)

                # Only the secret is replaced — everything else is unchanged.
                self.assertEqual(r["masked_text"], text.replace(secret, ml["replacement"]))
                self.assertEqual(r["overall_risk"], "MEDIUM")
                self.assertTrue(r["ml_safety_net"]["masked"])

    def test_layer1_entities_are_tagged_layer1(self):
        r = self.run_pipeline("Call 0771234567 to confirm.")
        self.assertTrue(r["entities"])
        self.assertTrue(all(e["layer"] == 1 for e in r["entities"]))

    def test_plain_text_still_has_no_entities(self):
        r = self.run_pipeline("Please summarise the meeting notes about quarterly sales targets")
        self.assertEqual(r["entities"], [])


class TestTokenLabel(unittest.TestCase):

    def test_registry_uses_explicit_label(self):
        from engine.token_registry import TokenRegistry
        reg = TokenRegistry(session_id="test_label")
        self.assertEqual(reg.get_or_create_token("abc", "ML_FLAGGED_ANOMALY", label="ML_SECRET"), "<ML_SECRET_1>")
        self.assertEqual(reg.get_or_create_token("def", "ML_FLAGGED_ANOMALY", label="ML_SECRET"), "<ML_SECRET_2>")
        # Idempotent for a repeated value.
        self.assertEqual(reg.get_or_create_token("abc", "ML_FLAGGED_ANOMALY", label="ML_SECRET"), "<ML_SECRET_1>")

    def test_default_label_unchanged(self):
        from engine.token_registry import TokenRegistry
        reg = TokenRegistry(session_id="test_label_default")
        self.assertEqual(reg.get_or_create_token("1234567890", "BANK_ACCOUNT_NO"), "<BANKACCOUN_1>")


if __name__ == "__main__":
    unittest.main()
