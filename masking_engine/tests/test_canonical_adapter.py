"""
test_canonical_adapter.py — Tests for engine/canonical_adapter.py
R26-CS-012: Context-Aware Masking + Instruction Engine

Uses the exact "Canonical Prompt and Context Structuring" payload shape
the upstream retrieval component produces (request.prompt + context[]
code blocks retrieved via FAISS), to confirm the adapter masks every text
field while leaving the rest of the contract (source/location/retrieval/
reason/metadata) untouched.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.canonical_adapter import process_canonical_request


SAMPLE_PAYLOAD = {
    "request": {
        "prompt": "Why is the login failing after the user clicks submit?"
    },
    "context": [
        {
            "source": {"file": "authentication.js", "language": "javascript"},
            "location": {"start_line": 12, "end_line": 26},
            "content": (
                "async function submitLoginForm(formData){\n\n"
                "    const result = validateLogin(\n"
                "        formData.username,\n"
                "        formData.password\n"
                "    );\n\n\n"
                "    if(!result){\n"
                "        return \"Login failed\";\n"
                "    }\n\n\n"
                "    return \"Success\";\n"
                "}"
            ),
            "retrieval": {
                "method": "FAISS semantic similarity",
                "score": 0.545,
                "confidence": "medium",
            },
            "reason": "Relevant code block selected based on semantic similarity",
        },
        {
            "source": {"file": "authentication.js", "language": "javascript"},
            "location": {"start_line": 1, "end_line": 8},
            "content": (
                "function validateLogin(username, password) {\n\n"
                "    if(!username || !password){\n"
                "        return false;\n"
                "    }\n\n"
                "    return true;\n"
                "}"
            ),
            "retrieval": {
                "method": "FAISS semantic similarity",
                "score": 0.4778,
                "confidence": "low",
            },
            "reason": "Relevant code block selected based on semantic similarity",
        },
    ],
    "metadata": {
        "component": "Canonical Prompt and Context Structuring",
        "version": "1.0",
        "status": "ready_for_next_component",
    },
}


class TestCanonicalAdapter(unittest.TestCase):
    def test_clean_payload_passes_through_with_no_false_positives(self):
        # Neither the prompt nor either code block contains real sensitive
        # data — formData.password is a property reference, not an
        # assignment, and must not trip the PASSWORD pattern.
        out = process_canonical_request(SAMPLE_PAYLOAD)
        self.assertEqual(out["request"]["prompt"], SAMPLE_PAYLOAD["request"]["prompt"])
        self.assertEqual(out["request"]["masking"]["masked_entities"], [])
        for block in out["context"]:
            self.assertEqual(block["masking"]["masked_entities"], [])
            self.assertIn("content", block)

    def test_non_content_fields_are_preserved_exactly(self):
        out = process_canonical_request(SAMPLE_PAYLOAD)
        self.assertEqual(out["context"][0]["source"], SAMPLE_PAYLOAD["context"][0]["source"])
        self.assertEqual(out["context"][0]["location"], SAMPLE_PAYLOAD["context"][0]["location"])
        self.assertEqual(out["context"][0]["retrieval"], SAMPLE_PAYLOAD["context"][0]["retrieval"])
        self.assertEqual(out["context"][1]["reason"], SAMPLE_PAYLOAD["context"][1]["reason"])

    def test_original_payload_not_mutated(self):
        import copy
        original = copy.deepcopy(SAMPLE_PAYLOAD)
        process_canonical_request(SAMPLE_PAYLOAD)
        self.assertEqual(SAMPLE_PAYLOAD, original)

    def test_metadata_annotated_with_masking_status(self):
        out = process_canonical_request(SAMPLE_PAYLOAD)
        self.assertEqual(out["metadata"]["status"], "masked_ready_for_next_component")
        # Upstream metadata fields survive alongside the new ones.
        self.assertEqual(out["metadata"]["component"], "Canonical Prompt and Context Structuring")

    def test_sensitive_content_in_a_context_block_is_masked(self):
        payload = {
            "request": {"prompt": "Why is this connection failing?"},
            "context": [
                {
                    "source": {"file": "db.js", "language": "javascript"},
                    "location": {"start_line": 1, "end_line": 1},
                    "content": "const DB_PASSWORD = 'Secure99@'; // TODO rotate",
                    "retrieval": {"method": "FAISS semantic similarity", "score": 0.9, "confidence": "high"},
                    "reason": "top match",
                }
            ],
            "metadata": {"component": "Canonical Prompt and Context Structuring", "version": "1.0"},
        }
        out = process_canonical_request(payload)
        self.assertNotIn("Secure99@", out["context"][0]["content"])
        self.assertIn("PASSWORD", [m["entity_type"] for m in out["context"][0]["masking"]["masked_entities"]])


if __name__ == "__main__":
    unittest.main()
