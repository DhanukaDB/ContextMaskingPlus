# ContextMaskingPlus: Architecture & Logic Explained
### R26-CS-012 — Context-Aware Masking + Instruction Engine

The Context-Aware Masking Engine is a multi-layered security pipeline designed to intercept, detect, and mask sensitive data (PII, PCI, Secrets, Cloud credentials) in text prompts *before* they are sent to Large Language Models (LLMs).

It combines the speed, determinism, and auditability of a rule-based detection layer with a genuinely live, second-layer Machine Learning safety net — not ML "explored in a research notebook," but a trained classifier wired directly into `main.py`, `web/server.py`, and `evaluate.py` on every request.

---

## 0. Why the Architecture Looks Like This (Panel-Driven Decisions)

Two pieces of review feedback shaped the current design and are worth stating up front, because they explain *why* the pipeline is split the way it is below:

1. **"Focus more on cyber security concepts, not just ML."** — Detection logic (Section 2) was hardened against 15 concrete panel-flagged bugs (business-reference numbers being misread as NICs, DOB false-positives on transaction dates, capture-group span errors on SWIFT/PAN fields, password values losing their labels, etc.) rather than adding ML surface area. These are now locked in as regression tests (`tests/test_detector.py::TestPanelReviewFixes`, 13 tests).
2. **"Without a model, only using regex, is not accepted."** — A live ML layer was required, but it could not be allowed to *replace* the auditable regex/NER layer for a banking compliance product, where every masking decision must trace to a named, explainable rule. The resolution is the **Hybrid Safety-Net Layer** architecture (Section 1, Step 4.5): ML runs as a strictly additive second opinion on Layer 1's misses. The model's probability score only ever decides **whether** Layer 2 acts — it can never itself decide **what** to redact. A separate, deterministic rule (`_find_suspicious_spans`, the same regex shapes the structural-evidence gate already checks) locates the exact secret-shaped substring, which is then masked and tokenized exactly like a Layer-1 value. If no such span can be located, Layer 2 falls back to a review-only flag rather than guessing or blanking the whole prompt.

The safety-net's own justification draws directly on a real-world case study: the 2022 **Optus data breach** exposed ~9.8M customer records (including passport/driver's-licence numbers) through a single unauthenticated API path — a single detection/access-control layer with one gap *was* the whole security boundary. The lesson applied here is defense-in-depth: no one layer's blind spot should be the whole story. Layer 1 covers every *known* sensitive-data shape; Layer 2 exists specifically to catch the "unknown unknowns" that shape does not cover — a novel secret format, an unlisted keyword, or content shaped like a leak even though it matches no named rule.

---

## 1. How It Works: The Core Pipeline Logic

The engine processes incoming text sequentially through six stages (as wired in `main.py::process_prompt`):

1. **Normalizer (`normalizer.py`)**
   - **Logic:** Cleans the text to defeat adversarial obfuscation. For example, malicious users might type `4 1 1 1` or `password%3DSecure99%40` to bypass scanners.
   - **Action:** Removes spaces, normalizes case, and URL-decodes strings so the detector has clean data to work with.
2. **Detector (`detector.py`)** — Layer 1, primary/only masking authority
   - **Logic:** Scans the clean text for candidate entities using ~30 named regex patterns plus rule-based NER, each scoped to avoid the panel-flagged false-positive classes (e.g. `NON_SENSITIVE_ID_PHRASES` suppresses business references like "transaction ID"/"order reference" from being misread as national IDs; word-boundary-safe address matching; DOB-gating keywords).
   - **Action:** Emits candidate entities with source spans. This is the *only* stage that can produce a `mask_immediate` decision — every masked value in the product traces to one named rule here, which is the compliance property that ruled out ML as the primary detector.
3. **Confidence Scorer (`confidence_scorer.py`)**
   - **Logic:** Determines the risk of a detected entity using a 4-factor weighted score: pattern strength (40%), keyword proximity (30%, looks at surrounding words like "Account", "ID", "NIC"), co-occurrence with other entities (20%, e.g. a name next to a NIC elevates both), and format validity (10%). A 12-digit number could be a sensitive ID or a random part number — this stage is what tells them apart.
   - **Action:** Assigns a confidence score → action threshold: `mask_immediate` (≥0.90 Very High / ≥0.75 High), `mask_warn` (≥0.50 Medium), `log_suspected` (≥0.25 Low), `ignore` (<0.25).
4. **Masker (`masker.py`)**
   - **Logic:** Applies a masking strategy based on the entity type and risk level.
   - **Action:** Replaces sensitive data with safe placeholders.
     - *Tokenization*: `sk-12345` → `<API_KEY_1>` (lets the LLM maintain referential integrity without seeing the real key, via `token_registry.py`).
     - *Partial Masking*: `4111222233334444` → `4111********4444`.
     - *Full Redaction*: `123` → `***`.
5. **ML Safety Net (`ml_anomaly.py`) — Layer 2, deterministic span, model-gated**
   - **Logic:** Runs **only if Step 4 masked nothing at all**. A trained classifier scores how secret-shaped the prompt looks structurally (entropy, token shape, secret-adjacent keywords — never raw text). Its probability alone is **not** trusted — see Section 3 — so a score is only ever acted on if an independent evidence check also agrees.
   - **Action:** If both the model AND the structural gate agree, `_find_suspicious_spans()` — a deterministic regex rule, not the model — locates the specific secret-shaped substring(s) (base64/hex runs, `sk-`/`AKIA`-prefixed strings, long opaque tokens). Those spans are masked via `TokenRegistry` (same idempotency guarantee as Layer 1: the same value in the same session always gets the same `<ML_FLAGGED_ANOMALY_N>` token) and appended to `masked_entities` with `action: "ml_flagged_mask"`. **If the gate passed but no span could be located**, Layer 2 falls back to a `skipped_entities` record (`reason: "ml_anomaly_flagged_no_span"`) for human review instead of masking blindly. Either way, the ML score only ever decided *whether* to act — *what* got redacted still traces to a named rule, the same auditability property Layer 1 has. Covered by `tests/test_ml_anomaly.py::TestApplySafetyNet` and `TestSpanLocation`.
   - **Availability:** Optional at runtime. If `scikit-learn`/`joblib` or the trained model file (`research/models_rf_classifier.pkl`) aren't present, this step silently no-ops and the engine runs Layer 1 only — a security control should not hard-fail because an *enhancement* is missing.
6. **Instruction Generator (`instruction_generator.py`)**
   - **Logic:** LLMs need to know how to handle the masked data (and, when present, the Layer-2 review flag).
   - **Action:** Appends a strict system prompt instructing the LLM not to attempt to guess or reconstruct masked values, enforcing a security boundary.

---

## 2. Algorithms Used & Why

### A. Heuristic & Rule-Based Algorithms (Layer 1 — `detector.py`, `confidence_scorer.py`)
- **Regular Expressions (Regex) + Rule-Based NER**
  - **Why:** Fast, computationally cheap, and highly accurate for data that follows a structural format (IBANs, credit cards, NICs, IP addresses) — and, critically for a banking compliance product, every decision is *explainable*: a reviewer can point at the exact rule that fired.
  - **Panel-driven refinements this cycle:** capture-group span extraction scoped to an explicit opt-in set (`VALUE_GROUP_TYPES`) so a deliberate group (PASSWORD's value) doesn't accidentally hijack an unrelated pattern's incidental group (e.g. CARD_EXPIRY's month capture); phrase-level (not word-level) suppression of business-reference language so "Customer reference: ..." still masks while "transaction ID: ..." doesn't; an `AMBIGUOUS_TYPE_GROUPS` mechanism so structurally-identical spans (e.g. a narrowed PASSWORD value vs. a bare API_KEY_GENERIC match) resolve to the more specific type instead of silently losing the label.
- **Shannon Entropy**
  - **Where:** Both `detector.py` (as a keyword-boosted heuristic) and `engine/ml_anomaly.py` (as a first-class classifier feature).
  - **Why:** API keys and passwords don't have predictable formats, but they do have high mathematical randomness (entropy) — a signal that generalizes past any single regex.

### B. Machine Learning — Live in Production (Layer 2 — `engine/ml_anomaly.py`)
This is the section that changed most substantially this cycle. ML is no longer confined to the research notebook — a trained model is loaded and scored on every request that reaches `main.py`, `web/server.py`, or `evaluate.py`.

1. **Gradient Boosting Classifier (scikit-learn) — the production Layer 2 model**
   - **Where:** Trained by `research/retrain_classifier.py`, loaded and scored by `engine/ml_anomaly.py`, shipped as `research/models_rf_classifier.pkl`.
   - **What:** Binary sensitive/not-sensitive classification based on structural/entropy signals — not raw text, so it can't just memorize dataset templates.
   - **Why Gradient Boosting over a single tree/logistic model:** sequential boosting on structural features handled the mixed continuous (entropy, ratios) and count-based (digit runs, symbol counts) feature types well in evaluation, and `StandardScaler` normalization kept it stable across very different feature magnitudes (text length vs. binary flags).
2. **TF-IDF + Classifiers, CRF, DistilBERT**
   - **Where:** `research/Colab_Masking_Engine_Lab.ipynb` — exploratory comparison work, not shipped.
   - **Why kept as research, not production:** these were evaluated as alternative approaches (TF-IDF+LogReg/RF/SVM on word frequencies; CRF for sequence-labeling NER; DistilBERT for semantic understanding) but the structural-feature Gradient Boosting model was chosen for production because it needs no text embeddings/tokenizer at inference time, keeps `engine/` close to dependency-free, and its signals stay directly inspectable — important for the same auditability requirement that keeps Layer 1 in charge of actual masking.

---

## 3. The Out-of-Distribution Problem — Why Layer 2 Isn't Just "the classifier"

**The failure, found by hand:** the first trained classifier scored ordinary, unrelated sentences ("The weather in Colombo today is sunny...") as 95%+ "sensitive." Root cause: its negative (non-sensitive) training examples came *only* from the dataset's own narrow "should-not-mask" templates (serial numbers, tracking references) — the model learned to recognize *that template family*, not genuine non-sensitivity.

**What did NOT fix it:** retraining on a larger dataset from the same template family (scaling `synthetic_dataset.json` toward the panel's 5,000–5,500 target) left the false-positive rate essentially unchanged — a negative result that was diagnosed rather than papered over, because it correctly pointed at *diversity* of negative examples, not *volume*, as the real gap.

**What did fix it — the process now permanently in place:**
1. **`research/generate_negative_corpus.py`** — generates 2,000 topically/structurally diverse, genuinely open-domain benign prompts (greetings, scheduling, HR chat, general tech how-to, and banking-*adjacent*-but-safe process questions like branch hours or transfer timelines) that carry zero PII or secrets. Deliberately kept separate from `synthetic_dataset.json` — merging it in would blow past the panel's dataset-size requirement for no Layer-1 benefit.
2. **`research/ood_probe_set.json`** — 40 hand-authored sentences, permanently held out of *all* training, used as a repeatable generalization check (as opposed to one-off manual testing).
3. **`research/retrain_classifier.py`** — a **gated retrain pipeline**: trains on `synthetic_dataset.json` + `negative_corpus.json`, then evaluates against the OOD probe set and **refuses to overwrite `models_rf_classifier.pkl` if the probe false-positive rate exceeds 15%** (`OOD_FP_THRESHOLD`). A bad retrain now fails loudly at training time instead of shipping silently.
4. **`tests/test_ml_anomaly.py::TestOodProbeSet`** — the same probe set, run against the *production* `apply_safety_net()` call path (model + structural gate together), as a permanent CI-style regression gate.

**Verified result:** the weather-sentence example dropped from 98.1% → 3.9% "sensitive"; a second example ("report by Friday...") dropped 93.0% → 10.6%; the genuine catch this layer exists for (a bearer-style credential string with no matching Layer-1 rule) remained correctly flagged at 96.9%. Gated probe false-positive rate: **0.0%**.

---

## 4. Two Ingestion Paths

The engine now serves two separate request contracts:

- **Direct prompt masking** — `POST /api/detect` (`web/server.py`) or `main.py --text "..."`. A single string in, a `MaskedResult` + Layer-2 flag out.
- **Canonical RAG-contract masking** — `POST /api/detect-canonical`, backed by `engine/canonical_adapter.py`. Accepts the upstream retrieval component's `{"request": {"prompt": ...}, "context": [{"source":..., "location":..., "content":..., "retrieval":..., "reason":...}], "metadata": {...}}` payload and runs the **full two-layer pipeline** (not just Layer 1) over `request.prompt` **and every `context[i].content` block** — retrieved source snippets are at least as likely to carry a hardcoded secret or internal IP as anything the user typed. Each field is masked independently (preserving per-block source/location for audit) but shares one `TokenRegistry`, so a secret repeated across the prompt and a retrieved snippet resolves to the same token. Output preserves the input's shape, annotating each masked field with a `masking` block and setting `metadata.status = "masked_ready_for_next_component"` so the contract downstream components expect still holds.

---

## 5. Current Verified Metrics (`data/eval_metrics.json`, 5,012-prompt dataset)

| Metric | Value |
|---|---|
| Precision / Recall / F1 (overall) | 95.1% / 93.2% / 94.1% |
| Adversarial detection rate | 95.9% |
| Edge-case (should-not-mask) false-positive rate | 0.0% |
| Layer 2 flags raised | 123 (103 span located & masked, 20 review-only) |
| Layer 2 false alarms | 0 (all 123 were true Layer-1 misses) |

Per-entity precision/recall breakdowns and the full ML safety-net report are regenerated by `python evaluate.py` — see that script's printed report for the current numbers, since the dataset and model are periodically regenerated by `data/generate_dataset.py` and `research/retrain_classifier.py`.

---

## 6. How to Implement This in Production

Deploy as a **Middleware Proxy** or **API Gateway Integration**:

1. **Intercept:** The user types a prompt (or an upstream RAG component assembles a prompt + retrieved context). Instead of going straight to the LLM provider, the request is routed to the Masking Engine API — `/api/detect` for a bare prompt, `/api/detect-canonical` for the full retrieval-contract payload.
2. **Process:** The engine runs the six-stage pipeline (Section 1), including the Layer-2 safety net if available.
3. **Forward:** The engine sends the *masked text + instruction block* (and, on the canonical path, the whole masked payload) to the LLM.
4. **Layer 2 outcomes:** a Layer-2 catch shows up one of two ways — as a `masked_entities` record (`entity_type: "ML_FLAGGED_ANOMALY"`, `action: "ml_flagged_mask"`) when a specific span was located and redacted, already safe to forward; or as a `skipped_entities` record (`reason: "ml_anomaly_flagged_no_span"`) when the gate fired but no span could be pinned down — route only this second case to a reviewer queue, since nothing was masked for it.
5. **Receive & Unmask (optional):** When the LLM replies (e.g. "I have processed `<API_KEY_1>`"), the application can intercept the response, look up `<API_KEY_1>` in the `TokenRegistry`, and swap the real value back in before showing it to the user.

By using this proxy architecture, sensitive data never leaves the secure infrastructure, and every masking decision remains traceable to a named rule — supporting compliance with data-privacy regulations (GDPR, PCI-DSS) and internal audit requirements.
