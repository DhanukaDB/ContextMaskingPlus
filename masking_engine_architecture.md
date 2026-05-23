# ContextMaskingPlus: Architecture & Logic Explained

The Context-Aware Masking Engine is a multi-layered security pipeline designed to intercept, detect, and mask sensitive data (PII, PCI, Secrets, Cloud credentials) in text prompts *before* they are sent to Large Language Models (LLMs). 

It balances the speed and determinism of rule-based systems with the contextual understanding of Machine Learning algorithms.

---

## 1. How It Works: The Core Pipeline Logic
The engine processes incoming text sequentially through five stages (as seen in `main.py`):

1. **Normalizer (`normalizer.py`)**
   - **Logic:** Cleans the text to defeat adversarial obfuscation. For example, malicious users might type `4 1 1 1` or `password%3DSecure99%40` to bypass scanners.
   - **Action:** Removes spaces, normalizes case, and URL-decodes strings so the detector has clean data to work with.
2. **Detector (`detector.py`)**
   - **Logic:** Scans the clean text for candidate entities.
   - **Action:** Uses predefined Regex patterns for structured data (Emails, Credit Cards, National IDs) and integrates tools like TruffleHog for unstructured secrets.
3. **Confidence Scorer (`confidence_scorer.py`)**
   - **Logic:** Determines the risk of a detected entity. A 12-digit number could be a sensitive ID or just a random part number. The scorer looks at the surrounding context (e.g., words like "Account", "ID", "User") to increase or decrease confidence.
   - **Action:** Assigns a risk level (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
4. **Masker (`masker.py`)**
   - **Logic:** Applies a masking strategy based on the entity type and risk level.
   - **Action:** Replaces sensitive data with safe placeholders.
     - *Tokenization*: `sk-12345` → `<API_KEY_1>` (Allows the LLM to maintain referential integrity without seeing the real key).
     - *Partial Masking*: `4111222233334444` → `4111********4444`.
     - *Full Redaction*: `123` → `***`.
5. **Instruction Generator (`instruction_generator.py`)**
   - **Logic:** LLMs need to know how to handle the masked data.
   - **Action:** Appends a strict system prompt instructing the LLM not to attempt to guess or reconstruct the masked values, enforcing a security boundary.

---

## 2. Algorithms Used & Why

The engine uses a hybrid approach, combining heuristics with Machine Learning (evaluated in your Jupyter Research Lab).

### A. Heuristic & Rule-Based Algorithms (Core Engine)
- **Regular Expressions (Regex)**
  - **Where:** `detector.py`
  - **Why:** Regex is incredibly fast, computationally cheap, and highly accurate for data that follows strict structural formats (like IBANs, Credit Cards, or IP addresses).
- **Shannon Entropy & TruffleHog**
  - **Where:** `detector.py` and `Colab_Masking_Engine_Lab.py`
  - **Why:** API keys and passwords don't have predictable formats, but they do have high mathematical randomness (entropy). TruffleHog is a specialized algorithm for detecting high-entropy strings.

### B. Machine Learning Algorithms (Research Lab)
Because Regex fails when context is ambiguous, the engine explores ML models to understand the *meaning* of the text.

1. **TF-IDF + Classifiers (Logistic Regression, Random Forest, SVM)**
   - **Where:** Task A in the Notebook.
   - **What:** Classifies if a whole prompt contains sensitive data based on word frequencies (TF-IDF) combined with custom features (special character counts, token counts).
   - **Why:** 
     - *Random Forest* prevents overfitting by averaging multiple decision trees.
     - *SVM (Support Vector Machines)* is highly effective at drawing boundaries in high-dimensional spaces (like word vectors).
2. **Conditional Random Fields (CRF)**
   - **Where:** Task B in the Notebook.
   - **What:** A statistical modeling algorithm used for Named Entity Recognition (NER).
   - **Why:** CRFs are designed for *sequence prediction*. Instead of looking at a word in isolation, a CRF looks at the previous word and the next word (e.g., `word[i-1]` and `word[i+1]`) to determine if the current word is sensitive. If the previous word is "Password:", the current word is highly likely to be a secret.
3. **Transformers (DistilBERT)**
   - **Where:** Task C in the Notebook.
   - **What:** A lightweight, pre-trained Large Language Model.
   - **Why:** Transformers possess deep semantic understanding. While Regex checks *syntax*, BERT checks *semantics*, allowing it to flag sensitive intents even if the data doesn't match a known format.

---

## 3. How to Implement This in Production

To implement this engine in a real-world application, it should be deployed as a **Middleware Proxy** or an **API Gateway Integration**.

**The Flow:**
1. **Intercept:** The user types a prompt into your UI. Instead of going straight to OpenAI/Anthropic, the request is routed to the Masking Engine API.
2. **Process:** The engine runs `process_prompt(text)`.
3. **Forward:** The engine sends the *Masked Text + Instruction Block* to the LLM.
4. **Receive & Unmask (Optional):** When the LLM replies (e.g., "I have processed `<API_KEY_1>`"), your application can intercept the response, look up `<API_KEY_1>` in the `TokenRegistry`, and swap the real value back in before showing it to the user.

By using this proxy architecture, the sensitive data never leaves your secure infrastructure, ensuring compliance with data privacy regulations (GDPR, PCI-DSS).
