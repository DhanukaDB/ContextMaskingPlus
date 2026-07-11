# 🛡️ Context-Aware Masking Engine: Technical Architecture

This document provides a deep dive into the multi-layered data protection approach implemented in the Masking Engine. Our strategy avoids single points of failure by layering **Regex, Statistical ML, and Sequence Tagging (CRF)**.

## 1. The Regex Layer (Pattern Matching)
**Purpose:** High-speed deterministic detection for known, rigid formats.

Regex operates as the first line of defense. We define regular expressions for well-known structured formats such as:
- **Payment Card Industry (PCI):** 16-digit credit card numbers (`PAN`), 3-digit `CVV`.
- **Contact Information:** Standard Email formats, phone numbers.
- **Secrets/Credentials:** Formats with known prefixes (e.g., `AKIA...` for AWS, `sk-...` for OpenAI).

### How it works
The `mask_text()` function scans the input against predefined regex patterns. If a match is found, it applies a redaction strategy:
- **Secrets:** Replaced entirely with a token (e.g., `<SECRET_REDACTED>`).
- **Cards/Emails:** Partially masked (e.g., `41************44`) to preserve structural utility for downstream debugging while obscuring the sensitive payload.

### Limitations of Regex
Regex is rigid. It fails when formats change, when users add unexpected whitespace, or when detecting context-dependent data (like a generic API key or a highly random JWT without a specific signature). This is where ML comes in.

---

## 2. The Statistical ML Classifier
**Purpose:** Identifying if a prompt *contains* sensitive information based on structural and statistical signals, avoiding data leakage.

Instead of analyzing raw text tokens (which leads to memorization and data leakage), the classifier extracts **17 character-level and entropy-based features**.

### Key Features Extracted:
1. **Shannon Entropy (`mean_token_entropy`, `max_token_entropy`):** High entropy indicates randomness, a strong signal for keys, tokens, and hashes.
2. **Length & Counts (`long_token_ratio`):** Sensitive tokens (like JWTs or connection strings) tend to be unusually long compared to natural language words.
3. **Statistical Patterns (`n_digit_runs`, `n_special_chars`):** Counts of continuous numbers or symbols help identify financial data and code snippets.
4. **Pattern Flags (`has_base64_pattern`, `has_hex_run`):** Binary flags for structural identifiers common in cryptography and session tokens.

### The Models
We implemented a stratified cross-validation (5-Fold) to prevent data leakage and evaluated multiple models (Logistic Regression, Random Forest, Gradient Boosting, Neural Networks, SVM). The **Gradient Boosting** and **Neural Network (MLP with 50 epochs)** models proved highly effective, balancing precision and recall.

The ML Classifier outputs a binary decision: **SAFE** or **SENSITIVE**. If flagged sensitive, the prompt requires deeper token-level inspection.

---

## 3. The CRF NER Model (Token-Level Tagging)
**Purpose:** If the Classifier flags a prompt as sensitive, the Conditional Random Field (CRF) pinpoints *exactly which tokens* within the sentence are the sensitive ones.

### How it works
CRF evaluates sequences. It doesn't just look at a token in isolation; it looks at the **context** (the token before and after it). 

For each word, we extract features such as:
- Is it uppercase?
- Does it contain digits or special characters?
- What is its individual Shannon Entropy?
- What were the properties of the word preceding it (`-1:w`) and following it (`+1:w`)?

The CRF outputs a label for every single word (e.g., `O` for safe, `B-SENSITIVE` for a detected secret). This allows the system to dynamically redact random, highly-entropic tokens (like unknown passwords) that Regex would miss.

---

## 4. Integration: The Pipeline
1. **Input Generation:** User submits a prompt.
2. **Regex Scan:** Immediate deterministic masking applied.
3. **Classifier Gateway:** Remaining prompt features extracted. Classifier decides if hidden risks remain.
4. **Deep Scan (CRF):** If flagged by the classifier, CRF isolates high-entropy / sensitive tokens based on contextual sequence and tags them for further redaction.
5. **Output:** A sanitized prompt, safe for logging or LLM consumption.

> **Current status:** steps 3–4 (the ML classifier/CRF gateway) exist as a
> trained research artifact in `research/`, but are **not** wired into the
> live pipeline described here — `main.py`/`evaluate.py` run steps 1–2 and
> 5 only (`normalize → detect → score_all → resolve_overlapping_entities →
> mask → generate_instructions`). Real ML integration (deriving span-level
> training labels, retraining on them, and adding the gateway/deep-scan
> steps to the live pipeline) is a deliberately separate, later phase of
> work — see the accuracy-fix notes below for what *is* live today.

---

## 5. Accuracy Fixes — Documented Deviations From the Taxonomy Spec

A round of false-positive/false-negative fixes to `engine/detector.py`,
`engine/normalizer.py`, and `engine/confidence_scorer.py` (verified via
`tests/test_detector.py` and a full dataset regeneration + `evaluate.py`
run, which raised overall F1 from ~0.46 to ~0.91) made a few deliberate,
empirically-justified departures from the taxonomy document's literal
wording. Recorded here so they read as intentional refinements, not
unexplained spec drift:

- **Co-occurrence "3+ entities → CRITICAL" rule (Taxonomy Section 7).**
  Applied as a blanket entity-count trigger regardless of whether the
  entities were actually related, this rule was found to amplify clusters
  of false positives into forced `mask_immediate`/CRITICAL status rather
  than filtering them (e.g. three coincidental noisy digit-run matches in
  one prompt would auto-elevate all three). `score_co_occurrence()` now
  applies a graduated boost (0.75, not an automatic 1.0/CRITICAL) for 3+
  *unrelated* entities, and reserves the hard CRITICAL elevation for the
  curated `CO_OCCURRENCE_PAIRS` relationships (Name+NIC, Email+Password,
  etc.), which are still forced to CRITICAL as originally specified.
- **Format-validity credit for ambiguous digit-shaped types.** The
  taxonomy states "standalone 12-digit numbers without context score
  ≤0.40 and are not masked" (Section 4), but a literal reading of the
  4-factor formula (pattern 0.40 + a lenient format check passing by
  coincidence) could still reach 0.50. For `NIC_NEW`/`NIC_OLD`/`TAX_ID`/
  `BANK_ACCOUNT_NO`/`PAN`, format-validity is now only counted when there's
  *some* other corroborating signal (a keyword or co-occurring entity) —
  a structural check alone on an inherently ambiguous shape isn't treated
  as standalone justification to mask.
- **Detection-time keyword gating.** `CVV`, `TAX_ID`, `BANK_ACCOUNT_NO`,
  and `API_KEY_GENERIC` previously matched unconditionally at the regex
  level (the taxonomy already specifies these as "Regex + keyword," but
  the keyword requirement was only ever enforced in scoring, not
  detection). They now require a `BOOST_KEYWORDS` match within ±5 tokens
  (or immediately adjacent, to handle glued patterns like `api_key=VALUE`)
  to be registered as a candidate detection at all — a bare match with no
  keyword still scores in the "log_suspected" band under the old scoring
  formula, which `evaluate.py` counts as a false positive since it's a
  real detection, just not a masked one.

## 6. Known Limitations (Not Bugs)

- **`FULL_NAME` contextual masking.** `masker.py`'s existing (unmodified)
  logic only masks/logs contextual-severity types (`FULL_NAME`, `GENDER`,
  `RACE_ETHNICITY`) when a HIGH+ severity entity co-occurs in the same
  prompt — matching the taxonomy's own "Name alone → pass" design
  (Section 6). This means aggregate `FULL_NAME` recall in `evaluate.py`
  looks lower than the detector's actual entity-level recognition rate
  (verified separately via `tests/test_detector.py`) — most of the gap is
  names correctly *not* masked in isolation, not missed detections.
- **Keyword-proximity heuristics can't resolve genuine language ambiguity**
  (e.g. "Debit LKR 50,000 from account X" — "Debit" as a verb still
  matches the `PAN` boost keyword list, since "debit card" is a legitimate
  PAN-adjacent phrase). Real disambiguation here needs POS-aware NLP, out
  of scope for the current regex+keyword design.
- **ML classifier generalization.** The `research/` notebook's classifier
  scores ~98% accuracy on its own held-out synthetic test set, but is
  trained on entropy/structural features only (deliberately, to avoid
  memorizing synthetic template text) — it doesn't reliably generalize to
  hand-written prompts whose sensitive content is keyword/format-based
  (credit card numbers, DB connection strings) rather than high-entropy
  (API keys, tokens). Expected given the feature design; would need
  keyword/contextual features (with a leakage-aware dataset split) to
  address, which reopens the memorization risk Section 2's anti-leakage
  design was built to avoid.
