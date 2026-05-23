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
