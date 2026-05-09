# R26-CS-012 — Context-Aware Masking + Instruction Engine

**Author:** Abayathilake S.S — IT22193186  
**Domain:** Banking Sector (Sri Lanka + International)

---

## Project Structure

```
masking_engine/
├── data/
│   ├── generate_dataset.py       # Synthetic dataset generator (250 prompts)
│   ├── synthetic_dataset.csv     # Generated dataset (CSV)
│   └── synthetic_dataset.json    # Generated dataset (JSON)
├── engine/
│   ├── normalizer.py             # Preprocessing: decode, despacing, context classify
│   ├── detector.py               # Regex + rule-based NER entity detection
│   ├── confidence_scorer.py      # 4-factor weighted confidence scoring
│   ├── masker.py                 # Masking strategy application
│   ├── token_registry.py         # Idempotent session token map
│   └── instruction_generator.py  # Standard / CRITICAL instruction block output
├── main.py                       # Terminal entry point
├── evaluate.py                   # Precision / Recall / F1 evaluation
└── README.md
```

---

## Setup

No external dependencies required. Pure Python 3.8+.

```bash
cd masking_engine
```

---

## Usage

### 1. Generate the synthetic dataset (run first)
```bash
python data/generate_dataset.py
```

### 2. Run built-in demo (12 cases covering all taxonomy categories)
```bash
python main.py --demo
```

### 3. Process a single prompt
```bash
python main.py --text "Customer NIC 199012345V, card 4111 1111 1111 1111"
```

### 4. Interactive mode (multi-turn session)
```bash
python main.py
```
Type prompts and see live masking. Type `tokens` to view the session token registry.

### 5. Run evaluation (precision / recall / F1)
```bash
python evaluate.py
```

---

## Algorithms Used

| Component | Algorithm | Justification |
|---|---|---|
| Entity Detection | Compiled Regex | Deterministic, auditable, PCI-DSS/GDPR compliant |
| NER (names, addresses) | Rule-based keyword patterns | No external deps, interpretable, runs air-gapped |
| Context Classification | Weighted pattern scoring | Transparent, per-regulation customisable |
| Confidence Scoring | Weighted linear combination (4 factors) | Fully traceable — each factor maps to a compliance concern |
| Masking Selection | Decision tree logic | Deterministic per entity, no probabilistic guesswork |
| Token Mapping | In-memory hash map | O(1) lookup, session-scoped, idempotent |
| Format Validation | Luhn (PAN), digit-sum (NIC) | Domain-standard validators, no training data needed |

---

## Taxonomy Coverage

| Category | Entities Covered |
|---|---|
| 1A — Government IDs | NIC_OLD, NIC_NEW, PASSPORT, DRIVING_LICENSE, TAX_ID |
| 1B — Contact | PHONE_LK, PHONE_INTL, EMAIL, HOME_ADDRESS |
| 1C — Demographics | FULL_NAME, DATE_OF_BIRTH |
| 2A — Card (PCI-DSS) | PAN, CVV, CARD_EXPIRY |
| 2B — Bank | BANK_ACCOUNT_NO, IBAN, SWIFT_BIC |
| 2C — Transactions | SWIFT_MT103, SWIFT_MT202 |
| 3A — API/Tokens | API_KEY_OPENAI, API_KEY_GENERIC, JWT_TOKEN |
| 3B — Passwords | PASSWORD |
| 3C — Private Keys | PRIVATE_KEY |
| 4B — DB | DB_CONNECTION_STRING |
| 4C — Network | INTERNAL_IP |
| 7A — Cloud | AWS_ACCESS_KEY, AWS_SECRET_KEY |
| 7B — Storage | S3_BUCKET_REF |
| 7C — Encoded | JWT_IN_LOG (Base64/hex decoded by normalizer) |
