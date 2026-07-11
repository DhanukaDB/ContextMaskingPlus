"""
test_detector.py — Regression tests for engine/detector.py + confidence_scorer.py
R26-CS-012: Context-Aware Masking + Instruction Engine

Run with:
    python -m unittest discover -s tests -v
or:
    python tests/test_detector.py

Covers the specific false-positive/false-negative bugs found during the
accuracy audit (see masking_engine_architecture.md history / plan notes):
  - Sri Lankan phone number detection (the original bug report)
  - CVV/TAX_ID/BANK_ACCOUNT_NO/API_KEY_GENERIC keyword-gating
  - CARD_EXPIRY vs. DATE_OF_BIRTH collision
  - PAN vs. BANK_ACCOUNT_NO vs. NIC_NEW ambiguity resolution
  - API_KEY_GENERIC false-firing on ordinary sentences (despacing bug)
  - FULL_NAME recall beyond the original 10-surname hardcoded list
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.normalizer import normalize
from engine.detector import detect
from engine.confidence_scorer import score_all, resolve_overlapping_entities


def run(text: str):
    """Full pipeline up to (but not including) masking; returns list of
    (entity_type, value, score, action) tuples sorted by start position."""
    norm = normalize(text)
    raw = detect(norm["normalized"], norm["despaced"])
    scored = resolve_overlapping_entities(score_all(raw, norm["normalized"]))
    return [(s.entity.entity_type, s.entity.value, s.score, s.action) for s in scored]


def types_detected(text: str):
    return {t for t, _, _, _ in run(text)}


class TestPhoneNumberDetection(unittest.TestCase):
    """The user's original bug report: valid Sri Lankan numbers were being
    missed while embedded digit substrings were being falsely flagged —
    'seems like only checking the number limit'."""

    def test_valid_lk_mobile_numbers_detected(self):
        for number in ["0771234567", "0712264567", "0752345678", "0781234567"]:
            with self.subTest(number=number):
                self.assertIn("PHONE_LK", types_detected(number))

    def test_valid_lk_landline_detected(self):
        # 011 = Colombo landline area code
        self.assertIn("PHONE_LK", types_detected("0112345678"))

    def test_plus94_form_detected_as_phone_lk_not_phone_intl(self):
        types = types_detected("+94771234567")
        self.assertIn("PHONE_LK", types)
        self.assertNotIn("PHONE_INTL", types)

    def test_number_without_valid_lk_prefix_not_detected_as_phone(self):
        # 10 digits, doesn't start with 0 or +94 — must NOT be PHONE_LK.
        types = types_detected("9922121234")
        self.assertNotIn("PHONE_LK", types)
        self.assertNotIn("PHONE_INTL", types)

    def test_phone_keyword_context_raises_confidence(self):
        bare = run("0771234567")
        with_context = run("Call 0771234567 to confirm the appointment.")
        bare_score = next(s for t, v, s, a in bare if t == "PHONE_LK")
        ctx_score = next(s for t, v, s, a in with_context if t == "PHONE_LK")
        self.assertGreater(ctx_score, bare_score)

    def test_phone_not_falsely_detected_inside_longer_account_number(self):
        # Regression: PHONE_LK regex used to have no left boundary, so a
        # "0" + 9 digits embedded inside a longer digit run (e.g. a bank
        # account number) would falsely match.
        text = "Customer's DFCC Bank account number is 38690776552781 — link to loan LN516362."
        self.assertNotIn("PHONE_LK", types_detected(text))

    def test_phone_not_falsely_detected_inside_nic(self):
        text = "National identity 200011388437 — KYC pending."
        self.assertNotIn("PHONE_LK", types_detected(text))

    def test_international_number_detected_and_not_confused_with_lk(self):
        types = types_detected("International client reachable at +447911123456 for queries.")
        self.assertIn("PHONE_INTL", types)
        self.assertNotIn("PHONE_LK", types)


class TestKeywordGatingAtDetection(unittest.TestCase):
    """CVV/TAX_ID/BANK_ACCOUNT_NO/API_KEY_GENERIC must not fire on bare
    numbers/strings with no supporting keyword nearby."""

    def test_bare_three_digit_number_not_cvv(self):
        # A house number, not a CVV.
        types = types_detected("Deliver the card to 227, Bauddhaloka Mawatha.")
        self.assertNotIn("CVV", types)

    def test_cvv_with_keyword_detected(self):
        types = types_detected("Card number 4111 1111 1111 1111, CVV 123, expiry 12/27.")
        self.assertIn("CVV", types)

    def test_cvv_not_falsely_split_from_spaced_pan(self):
        # Regression: a spaced PAN's own 4-digit groups (4111, 1111...)
        # used to each independently match the CVV regex once any "CVV"
        # keyword appeared anywhere in the same sentence.
        result = run("Card number 4111 1111 1111 1111, CVV 123, expiry 12/27 — is this Luhn valid?")
        cvv_values = [v for t, v, s, a in result if t == "CVV"]
        self.assertEqual(cvv_values, ["123"])

    def test_amount_fragments_not_detected_as_cvv(self):
        text = "Debit LKR 1,047,961 from account 08196570298891 at National Development Bank."
        self.assertNotIn("CVV", types_detected(text))

    def test_bare_nine_digit_number_not_tax_id(self):
        self.assertNotIn("TAX_ID", types_detected("Reference 123456789 attached for filing."))

    def test_tax_id_with_keyword_detected(self):
        types = types_detected("Please confirm the TIN 123456789 for this corporate account.")
        self.assertIn("TAX_ID", types)

    def test_bare_long_digit_run_not_bank_account(self):
        self.assertNotIn(
            "BANK_ACCOUNT_NO",
            types_detected("Serial number for batch: 200423910321 — log for inventory."),
        )

    def test_ordinary_sentence_not_flagged_as_api_key(self):
        # Regression: remove_adversarial_spacing() used to collapse entire
        # ordinary sentences into long blobs that matched API_KEY_GENERIC.
        text = "Update the profile for Kamal Ranaweerasinghe, DOB 20/09/2001, address 227, Bauddhaloka Mawatha, Kurunegala."
        self.assertNotIn("API_KEY_GENERIC", types_detected(text))

    def test_real_api_key_with_keyword_detected(self):
        text = "I'm getting a 401 error when using api_key=abcdEFGH12345678ijklMNOP90qrstuvWXYZ12 against the gateway."
        self.assertIn("API_KEY_GENERIC", types_detected(text))


class TestCardExpiryVsDateOfBirth(unittest.TestCase):
    def test_dob_not_flagged_as_card_expiry(self):
        text = "KYC file for Nimal Perera: born 20/09/2001, residing at 12 Galle Road."
        self.assertNotIn("CARD_EXPIRY", types_detected(text))

    def test_genuine_expiry_still_detected(self):
        self.assertIn("CARD_EXPIRY", types_detected("Card expiring 12/27 — trigger renewal workflow."))


class TestAmbiguousDigitRunResolution(unittest.TestCase):
    """PAN / BANK_ACCOUNT_NO / NIC_NEW / NIC_OLD / TAX_ID all structurally
    overlap on plain digit runs — resolve_overlapping_entities() should let
    keyword context pick the right one instead of pattern-registration order."""

    def test_account_context_wins_over_pan_default(self):
        text = "Customer's DFCC Bank account number is 38690776552781 — link to loan LN516362."
        result = run(text)
        types = {t for t, v, s, a in result}
        self.assertIn("BANK_ACCOUNT_NO", types)
        self.assertNotIn("PAN", types)

    def test_taxonomy_worked_example_lands_medium_confidence(self):
        # Taxonomy Section 4's own worked example: "Customer reference:
        # 200423910321" is expected to land around 0.65 / Medium.
        result = run("Customer reference: 200423910321")
        nic = next((v for t, v, s, a in result if t == "NIC_NEW"), None)
        self.assertIsNotNone(nic)
        score = next(s for t, v, s, a in result if t == "NIC_NEW")
        self.assertGreaterEqual(score, 0.5)
        self.assertLess(score, 0.9)

    def test_no_context_ambiguous_number_stays_low_confidence(self):
        result = run("Serial: 200423910321 logged for inventory.")
        for t, v, s, a in result:
            self.assertNotEqual(a, "mask_immediate")

    def test_no_context_ambiguous_number_not_masked_at_all(self):
        # Taxonomy Section 4: "Standalone 12-digit numbers without context
        # score <=0.40 and are not masked." mask_warn still masks — only
        # log_suspected/ignore genuinely leave the value untouched.
        result = run("Serial number for batch: 200423910321 — log for inventory.")
        for t, v, s, a in result:
            self.assertIn(a, ("log_suspected", "ignore"))
            self.assertLessEqual(s, 0.40)

    def test_name_plus_nic_elevates_to_critical(self):
        result = run("Check account for Saman Perera, NIC 199012345V.")
        nic_action = next(a for t, v, s, a in result if t == "NIC_OLD")
        self.assertEqual(nic_action, "mask_immediate")


class TestJwtDetection(unittest.TestCase):
    """Regression: the base64 decoder used to unconditionally decode any
    20+-char base64 run, including a JWT's own eyJ-prefixed header/payload
    segments, destroying the dot-separated structure JWT_TOKEN needs to
    match — JWT_TOKEN had 0% recall as a result."""

    def _make_jwt(self):
        import base64
        header = base64.b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip("=")
        payload = base64.b64encode(b'{"sub":"12345","role":"teller"}').decode().rstrip("=")
        sig = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN0123456789_-XYZ")[:43]
        return f"{header}.{payload}.{sig}"

    def test_jwt_detected_after_base64_normalization(self):
        jwt = self._make_jwt()
        text = f"Authorization: Bearer {jwt} — why is the token rejected?"
        self.assertIn("JWT_TOKEN", types_detected(text))

    def test_jwt_segments_not_separately_flagged_as_api_key(self):
        jwt = self._make_jwt()
        text = f"Authorization: Bearer {jwt} — why is the token rejected?"
        self.assertNotIn("API_KEY_GENERIC", types_detected(text))


class TestAwsSecretKeyDetection(unittest.TestCase):
    def test_aws_secret_key_detected_alongside_access_key(self):
        text = ("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE "
                "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        types = types_detected(text)
        self.assertIn("AWS_ACCESS_KEY", types)
        self.assertIn("AWS_SECRET_KEY", types)


class TestFullNameRecall(unittest.TestCase):
    """Regression: detector used to only recognize 10 hardcoded surnames
    while the dataset generator's pool has 75."""

    def test_non_top10_surname_detected(self):
        # "Ranaweerasinghe" was NOT in the original 10-surname list.
        types = types_detected("Update the profile for Kamal Ranaweerasinghe, address 12 Galle Road.")
        self.assertIn("FULL_NAME", types)

    def test_another_non_top10_surname_detected(self):
        types = types_detected("Mortgage applicant: Nadeesha Karunaratne, currently at 5 Kandy Road.")
        self.assertIn("FULL_NAME", types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
