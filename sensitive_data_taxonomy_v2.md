# Sensitive Data Taxonomy
## Context-Aware Masking + Instruction Engine

**Project:** R26-CS-012 — AI-Safe Data Masking and Leakage Prevention Framework  
**Component:** Context-Aware Masking + Instruction Engine  
**Author:** Abayathilake S.S — IT22193186  
**Domain Focus:** Banking Sector (Sri Lanka + International)  
**Version:** 2.0  
**Date:** May 2026  
**Changes from v1.0:** Supervisor review integration — Cloud infrastructure patterns, confidence scoring, SWIFT CSP compliance, idempotent token mapping, adversarial dataset strategy, Sample Interaction Map added.

---

## 1. Purpose

This taxonomy defines all categories of sensitive data that the Context-Aware Masking + Instruction Engine must detect, classify, and mask before user prompts are transmitted to external AI systems (e.g., ChatGPT, GitHub Copilot, Gemini, Claude).

It serves as the **ground truth reference** for:
- Building regex detection patterns
- Training the NLP/ML detection model
- Defining masking strategies and confidence thresholds per entity type
- Aligning with regulatory compliance frameworks
- Generating idempotent, session-coherent masking instructions

---

## 2. Compliance Framework Alignment

All data categories in this taxonomy are mapped to one or more of the following regulatory standards:

| Standard | Jurisdiction | Scope |
|---|---|---|
| **GDPR** | European Union | Personal data of EU residents |
| **PCI-DSS v4.0** | International | Cardholder and payment data |
| **CBSL** (Central Bank of Sri Lanka) | Sri Lanka | Banking and financial data |
| **SWIFT CSP** (Customer Security Programme) | International | Interbank transaction security — *added v2.0* |
| **HIPAA** | USA | Health-related personal data |
| **ISO/IEC 27001** | International | Information security management |

> **v2.0 Addition — SWIFT CSP:**  
> SWIFT CSP Mandatory Control 6.1 requires that all systems exchanging SWIFT messages implement controls to prevent unauthorized data disclosure. Any SWIFT transaction data (MT103, MT202, field references) detected in a prompt must be treated as CRITICAL and trigger immediate masking + alert regardless of context.

---

## 3. Taxonomy Overview

The taxonomy is organized into **7 top-level categories** (Category 7 added in v2.0):

```
SENSITIVE DATA
├── Category 1: Personally Identifiable Information (PII)
│   ├── 1A: National & Government Identifiers
│   ├── 1B: Contact Information
│   └── 1C: Personal Demographics
│
├── Category 2: Financial & Payment Data
│   ├── 2A: Card Data (PCI-DSS)
│   ├── 2B: Bank Account Data
│   └── 2C: Transaction Data
│
├── Category 3: Credentials & Authentication Secrets
│   ├── 3A: API Keys & Tokens
│   ├── 3B: Passwords & Passphrases
│   └── 3C: Private Keys & Certificates
│
├── Category 4: Configuration & Infrastructure Secrets
│   ├── 4A: Environment Variables
│   ├── 4B: Database Connection Strings
│   └── 4C: Internal Network & Infrastructure Data
│
├── Category 5: Banking-Specific Organizational Data
│   ├── 5A: Core Banking Identifiers
│   ├── 5B: Customer Account Intelligence
│   └── 5C: Regulatory & Audit Data (incl. SWIFT CSP)
│
├── Category 6: Contextual & Implicit Sensitive Data
│   ├── 6A: Sensitive Code Comments
│   └── 6B: Internal Document References
│
└── Category 7: Cloud Infrastructure & Architecture Secrets [NEW v2.0]
    ├── 7A: Cloud Resource Identifiers
    ├── 7B: Cloud Storage References
    └── 7C: Encoded & Obfuscated Secrets
```

---

## 4. Confidence Score Framework *(New — v2.0)*

> **Supervisor Feedback Addressed:** *"The number `200423910321` could be an NIC or a random serial number. How will your engine decide? Add a Confidence Score attribute."*

Every detected entity is assigned a **Confidence Score (0.00 – 1.00)** based on the combination of:

| Factor | Weight | Description |
|---|---|---|
| **Pattern Match Strength** | 40% | Regex match quality — exact vs. partial match |
| **Context Keyword Proximity** | 30% | Distance to keywords like `nic`, `password`, `api_key` within a window of ±5 tokens |
| **Co-occurrence Boost** | 20% | Presence of other detected entities in the same prompt |
| **Format Validity** | 10% | Structural validation (e.g., Luhn check for PAN, NIC digit sum rule) |

### Confidence Thresholds & Actions:

| Score Range | Confidence Level | Engine Action |
|---|---|---|
| 0.90 – 1.00 | ✅ Very High | Mask immediately, no user prompt |
| 0.75 – 0.89 | ✅ High | Mask immediately |
| 0.50 – 0.74 | ⚠️ Medium | Mask with warning log; flag for review |
| 0.25 – 0.49 | ❓ Low | Do not mask; log as suspected entity; alert security dashboard |
| 0.00 – 0.24 | ❌ Very Low | Ignore; treat as false positive |

### Ambiguity Resolution Example:

**Input:** `"Customer reference: 200423910321"`

| Check | Result | Score Contribution |
|---|---|---|
| Matches `NIC_NEW` pattern (`\b\d{12}\b`) | Yes | +0.40 |
| Keyword `customer reference` near entity | Partial match (not `nic`, `id`) | +0.15 |
| No co-occurring PII entities | None | +0.00 |
| Format validity (NIC digit-sum check) | Pass | +0.10 |
| **Total Confidence Score** | | **0.65 → ⚠️ Medium** |

**Decision:** Mask with warning; log as `NIC_NEW (suspected)`. Security reviewer confirms or dismisses.

---

## 5. Detailed Category Definitions

---

### CATEGORY 1: Personally Identifiable Information (PII)

> **Definition:** Any data that can be used to identify, contact, or locate a specific individual.

**Compliance:** GDPR Article 4, CBSL Data Protection Guidelines

---

#### 1A — National & Government Identifiers

| Entity Type | Description | Example | Regex Pattern | Confidence Boost Keywords | Sensitivity | Masking Strategy |
|---|---|---|---|---|---|---|
| `NIC_OLD` | Old Sri Lankan NIC | `199012345V` | `\b\d{9}[vVxX]\b` | `nic`, `id`, `identity`, `national` | 🔴 HIGH | Full Mask |
| `NIC_NEW` | New Sri Lankan NIC (12 digits) | `199012345678` | `\b\d{12}\b` | `nic`, `id`, `customer id` | 🔴 HIGH | Full Mask |
| `PASSPORT` | Passport number | `N1234567` | `\b[A-Z]{1,2}\d{6,7}\b` | `passport`, `travel document` | 🔴 HIGH | Full Mask |
| `DRIVING_LICENSE` | Driving license | `B1234567` | `\b[A-Z]\d{7}\b` | `license`, `driving` | 🔴 HIGH | Full Mask |
| `TAX_ID` | Tax Identification Number | `123456789` | `\b\d{9}\b` | `tin`, `tax`, `vat` | 🔴 HIGH | Full Mask |

**Ambiguity Note — NIC_NEW vs. Random Number:**  
`\b\d{12}\b` matches many 12-digit values. The engine **must** require at least one confidence keyword within ±5 tokens OR co-occurrence with another PII entity before masking. Standalone 12-digit numbers without context score ≤ 0.40 and are not masked.

---

#### 1B — Contact Information

| Entity Type | Description | Example | Regex Pattern | Sensitivity | Masking Strategy |
|---|---|---|---|---|---|
| `PHONE_LK` | Sri Lankan phone | `0771234567` | `(\+94\|0)[0-9]{9}` | 🟡 MEDIUM | Partial Mask: `077****567` |
| `PHONE_INTL` | International phone | `+1-800-555-0199` | `\+[1-9]\d{6,14}` | 🟡 MEDIUM | Partial Mask |
| `EMAIL` | Email address | `john.doe@seylan.lk` | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | 🟡 MEDIUM | Partial Mask: `j***@seylan.lk` |
| `HOME_ADDRESS` | Physical address | `12, Galle Road, Colombo 03` | NLP NER (GPE + street) | 🟡 MEDIUM | Full Mask |

---

#### 1C — Personal Demographics

| Entity Type | Description | Example | Detection Method | Sensitivity | Masking Strategy |
|---|---|---|---|---|---|
| `FULL_NAME` | Full person name | `Saman Perera` | NLP NER (PERSON) | 🟢 LOW | Contextual — mask if co-occurring with HIGH |
| `DATE_OF_BIRTH` | Date of birth | `1990-03-15` | Regex + NLP (`born`, `dob`, `date of birth`) | 🟡 MEDIUM | Full Mask |
| `GENDER` | Gender linked to identity | `Male` | NLP context | 🟢 LOW | Contextual |
| `RACE_ETHNICITY` | Ethnicity linked to identity | — | NLP context | 🔴 HIGH | Full Mask |

---

### CATEGORY 2: Financial & Payment Data

**Compliance:** PCI-DSS v4.0, CBSL Payment System Regulations, SWIFT CSP

---

#### 2A — Card Data (PCI-DSS)

| Entity Type | Description | Example | Regex Pattern | Sensitivity | Masking Strategy |
|---|---|---|---|---|---|
| `PAN` | Primary Account Number | `4111 1111 1111 1111` | `\b(?:\d[ -]?){13,16}\b` + Luhn | 🔴 HIGH | Partial: `4111 **** **** 1111` |
| `CVV` | Card Verification Value | `123` | Regex + keyword (`cvv`, `cvc`) | 🔴 CRITICAL | Full Mask — always |
| `CARD_EXPIRY` | Card expiry | `12/27` | `\b(0[1-9]\|1[0-2])\/(\d{2}\|\d{4})\b` | 🔴 HIGH | Full Mask |
| `CARD_HOLDER_NAME` | Name on card | `SAMAN PERERA` | NLP + context | 🟡 MEDIUM | Full Mask |
| `CARD_TRACK_DATA` | Magnetic stripe data | `%B4111...^PERERA^` | `%B\d{13,19}\^` | 🔴 CRITICAL | Full Mask |

---

#### 2B — Bank Account Data

| Entity Type | Description | Example | Regex Pattern | Sensitivity | Masking Strategy |
|---|---|---|---|---|---|
| `BANK_ACCOUNT_NO` | Bank account number | `001010012345` | `\b\d{8,16}\b` + context | 🔴 HIGH | Partial Mask |
| `IBAN` | International Bank Account Number | `GB29NWBK60161331926819` | `\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b` | 🔴 HIGH | Full Mask |
| `SWIFT_BIC` | SWIFT/BIC code | `BCEYLKLX` | `\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b` | 🟡 MEDIUM | Tokenization |
| `SORT_CODE` | UK Sort Code | `60-16-13` | `\b\d{2}-\d{2}-\d{2}\b` | 🟡 MEDIUM | Partial Mask |
| `ROUTING_NO` | US ABA Routing | `021000021` | `\b\d{9}\b` + context | 🟡 MEDIUM | Partial Mask |

---

#### 2C — Transaction Data

| Entity Type | Description | Example | Detection Method | Sensitivity | Masking Strategy |
|---|---|---|---|---|---|
| `TRANSACTION_ID` | Transaction reference | `TXN20240315001234` | Regex + prefix | 🟡 MEDIUM | Tokenization |
| `SWIFT_MT_REF` | SWIFT message field reference | `20:F01BCEYLKLX0001` | Regex + SWIFT field tags (`20:`, `32A:`) | 🔴 CRITICAL | Tokenization + Alert |
| `TRANSACTION_AMOUNT` | Amount linked to identity | `LKR 250,000.00` | NLP + currency | 🟢 LOW–MEDIUM | Contextual |
| `MERCHANT_ID` | Merchant ID | `MID123456789` | Regex + context | 🟡 MEDIUM | Tokenization |

---

### CATEGORY 3: Credentials & Authentication Secrets

> **Rule:** ALL credential types are HIGH or CRITICAL regardless of context. No exceptions.

---

#### 3A — API Keys & Tokens

| Entity Type | Example / Pattern | Regex Pattern | Sensitivity | Masking |
|---|---|---|---|---|
| `API_KEY_GENERIC` | `sk-abc123...` (20+ chars) | `[a-zA-Z0-9_\-]{20,}` + keyword | 🔴 HIGH | Tokenization |
| `JWT_TOKEN` | `eyJhbGci...` | `eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+` | 🔴 HIGH | Tokenization |
| `BEARER_TOKEN` | `Bearer abc123xyz` | `Bearer\s+[a-zA-Z0-9_\-\.]+` | 🔴 HIGH | Tokenization |
| `OAUTH_TOKEN` | `ya29.A0ARrd...` | Known prefix patterns | 🔴 HIGH | Tokenization |
| `OPENAI_KEY` | `sk-proj-...` | `sk-[a-zA-Z0-9]{48}` | 🔴 HIGH | Tokenization |
| `STRIPE_KEY` | `sk_live_...` | `sk_live_[a-zA-Z0-9]{24}` | 🔴 HIGH | Tokenization |
| `PAYMENT_GW_KEY` | PayHere / WebXPay / iPay keys | `merchant_secret=\S+` | 🔴 HIGH | Tokenization |

**JWT Note (v2.0):**  
JWTs are Base64-encoded. The engine must detect the three-part dot-separated structure `xxxxx.yyyyy.zzzzz` and flag even when embedded inside log output or curl commands. The payload may contain sensitive claims (sub, email, role) even if the token itself is expired.

---

#### 3B — Passwords & Passphrases

| Entity Type | Detection Method | Sensitivity | Masking |
|---|---|---|---|
| `PASSWORD_INLINE` | `password\s*[:=]\s*\S+` | 🔴 HIGH | Full Mask |
| `PASSWORD_ENV` | `(PASSWORD\|PASS\|PWD)=\S+` | 🔴 HIGH | Full Mask |
| `SECRET_KEY` | `SECRET_KEY\s*=\s*['"][^'"]+['"]` | 🔴 HIGH | Full Mask |
| `DATABASE_PASSWORD` | Regex + NLP context | 🔴 HIGH | Full Mask |

---

#### 3C — Private Keys & Certificates

| Entity Type | Pattern | Sensitivity | Masking |
|---|---|---|---|
| `RSA_PRIVATE_KEY` | `-----BEGIN RSA PRIVATE KEY-----` | 🔴 CRITICAL | Full Mask |
| `EC_PRIVATE_KEY` | `-----BEGIN EC PRIVATE KEY-----` | 🔴 CRITICAL | Full Mask |
| `PRIVATE_KEY_GENERIC` | `-----BEGIN PRIVATE KEY-----` | 🔴 CRITICAL | Full Mask |
| `SSH_PRIVATE_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----` | 🔴 CRITICAL | Full Mask |
| `PGP_PRIVATE_KEY` | `-----BEGIN PGP PRIVATE KEY BLOCK-----` | 🔴 CRITICAL | Full Mask |

**Rule:** Any `-----BEGIN ... KEY-----` block = CRITICAL, full mask, no exceptions.

---

### CATEGORY 4: Configuration & Infrastructure Secrets

---

#### 4A — Environment Variables

| Entity Type | Example | Detection | Sensitivity | Masking |
|---|---|---|---|---|
| `ENV_SECRET` | `DB_PASSWORD=P@ssw0rd` | `[A-Z_]+(SECRET\|KEY\|TOKEN\|PASS\|PWD)=\S+` | 🔴 HIGH | Full Mask |
| `ENV_CONNECTION` | `DATABASE_URL=postgres://user:pass@host` | Regex + URI | 🔴 HIGH | Tokenization |
| `ENV_API_KEY` | `STRIPE_SECRET_KEY=sk_live_...` | Known prefix patterns | 🔴 HIGH | Tokenization |

---

#### 4B — Database Connection Strings

| Entity Type | Example | Regex Pattern | Sensitivity | Masking |
|---|---|---|---|---|
| `DB_CONN_POSTGRES` | `postgres://user:pass@host:5432/db` | `postgres(ql)?:\/\/[^:]+:[^@]+@` | 🔴 HIGH | Tokenization |
| `DB_CONN_MYSQL` | `mysql://user:pass@host/db` | `mysql:\/\/[^:]+:[^@]+@` | 🔴 HIGH | Tokenization |
| `DB_CONN_MONGODB` | `mongodb+srv://user:pass@cluster` | `mongodb(\+srv)?:\/\/[^:]+:[^@]+@` | 🔴 HIGH | Tokenization |
| `DB_CONN_MSSQL` | `Server=host;User Id=sa;Password=pass` | Keyword + structure | 🔴 HIGH | Tokenization |
| `DB_CONN_ORACLE` | `Data Source=host;User Id=sys;Password=pass` | Keyword + structure | 🔴 HIGH | Tokenization |

**Banking Context:** Core banking systems (T24, Finacle, Flexcube) use Oracle or MSSQL. Connection strings must never reach external AI.

---

#### 4C — Internal Network & Infrastructure Data

| Entity Type | Example | Detection | Sensitivity | Masking |
|---|---|---|---|---|
| `INTERNAL_IP` | `192.168.1.100`, `10.0.0.5` | RFC 1918 regex | 🟡 MEDIUM | Full Mask |
| `INTERNAL_HOSTNAME` | `prod-db-01.internal.bank.lk` | Regex: `\w+\.(internal\|corp\|local)\b` | 🟡 MEDIUM | Tokenization |
| `INTERNAL_URL` | `http://internal.seylan.lk/api` | Regex + `internal` keyword | 🟡 MEDIUM | Tokenization |
| `MAC_ADDRESS` | `00:1A:2B:3C:4D:5E` | `([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}` | 🟡 MEDIUM | Full Mask |
| `AWS_ACCESS_KEY` | `AKIAIOSFODNN7EXAMPLE` | `AKIA[0-9A-Z]{16}` | 🔴 HIGH | Tokenization |
| `AWS_SECRET_KEY` | `wJalrXUtnFEMI/K7MDENG` | Context + keyword | 🔴 HIGH | Tokenization |
| `AZURE_KEY` | Azure connection/access key | NLP + known patterns | 🔴 HIGH | Tokenization |

---

### CATEGORY 5: Banking-Specific Organizational Data

**Compliance:** CBSL Banking Act No. 30 of 1988, PCI-DSS, SWIFT CSP

---

#### 5A — Core Banking Identifiers

| Entity Type | Example | Detection | Sensitivity | Masking |
|---|---|---|---|---|
| `CUSTOMER_ID` | `CUS-2024-001234` | Regex + prefix | 🔴 HIGH | Tokenization |
| `LOAN_ACCOUNT_NO` | `LN-2024-00056789` | Regex + prefix | 🔴 HIGH | Tokenization |
| `CHEQUE_NO` | `001234` | Regex + context | 🟡 MEDIUM | Partial Mask |
| `BRANCH_CODE` | `001` (BOC Colombo Main) | Regex + bank lookup | 🟢 LOW | Contextual |
| `SWIFT_MSG_REF` | `20:F01BCEYLKLX0001` | SWIFT field tag regex | 🔴 CRITICAL | Tokenization + Alert |
| `LC_NUMBER` | `LC/2024/001234` | Regex + context | 🔴 HIGH | Tokenization |

---

#### 5B — Customer Account Intelligence

| Entity Type | Detection | Sensitivity | Masking |
|---|---|---|---|
| `CREDIT_SCORE` | NLP + (`credit score`, `CRIB`, `rating`) | 🔴 HIGH | Full Mask |
| `ACCOUNT_BALANCE` | NLP + currency + identity co-occurrence | 🔴 HIGH | Full Mask |
| `SALARY_INFO` | NLP + (`salary`, `income`, `earnings`) | 🔴 HIGH | Full Mask |
| `LOAN_AMOUNT` | NLP + context | 🔴 HIGH | Full Mask |
| `CRIB_REPORT` | NLP + keyword `CRIB` | 🔴 HIGH | Full Mask |

---

#### 5C — Regulatory & Audit Data

| Entity Type | Detection | Sensitivity | Masking | Compliance |
|---|---|---|---|---|
| `SAR_DATA` | NLP + (`suspicious activity`, `SAR`) | 🔴 CRITICAL | Full Mask | CBSL, AML |
| `AML_FLAG` | NLP + (`AML`, `anti-money laundering`) | 🔴 CRITICAL | Full Mask | CBSL |
| `KYC_DATA` | NLP + (`KYC`, `know your customer`) | 🔴 HIGH | Full Mask | CBSL |
| `SWIFT_TRANSACTION` | SWIFT field tags (`32A:`, `57A:`, `70:`) | 🔴 CRITICAL | Tokenization + Alert | SWIFT CSP |
| `AUDIT_LOG_ENTRY` | NLP + log structure | 🟡 MEDIUM | Tokenization | ISO 27001 |
| `COMPLIANCE_REPORT` | NLP + keyword | 🟡 MEDIUM | Tokenization | CBSL |

---

### CATEGORY 6: Contextual & Implicit Sensitive Data

---

#### 6A — Sensitive Code Comments

| Pattern Type | Example | Detection | Sensitivity | Masking |
|---|---|---|---|---|
| Hardcoded credential comment | `// TODO: remove password before commit` | NLP + keyword | 🔴 HIGH | Full Mask of value |
| Credential in comment | `# Production API key: sk-abc123` | Regex + context | 🔴 HIGH | Tokenization |
| Internal system reference | `// Points to CBS prod endpoint` | NLP context | 🟡 MEDIUM | Tokenization |

---

#### 6B — Internal Document References

| Pattern Type | Example | Detection | Sensitivity | Masking |
|---|---|---|---|---|
| Internal system name + credentials | `FlexCube password: admin123` | NLP keyword | 🔴 HIGH | Full Mask |
| Internal file path | `/etc/bank/config/prod.env` | Regex path + keyword | 🟡 MEDIUM | Tokenization |
| Internal API endpoint | `POST /internal/api/v2/cbs/accounts` | Regex + `internal` | 🟡 MEDIUM | Tokenization |

---

### CATEGORY 7: Cloud Infrastructure & Architecture Secrets *(NEW — v2.0)*

> **Supervisor Feedback Addressed:** *"Are you detecting AWS ARNs, S3 bucket names, Azure Resource IDs? These leak architectural details even if they aren't credentials."*

---

#### 7A — Cloud Resource Identifiers

| Entity Type | Description | Example | Regex Pattern | Sensitivity | Masking |
|---|---|---|---|---|---|
| `AWS_ARN` | AWS Resource Name | `arn:aws:s3:::prod-bank-backups` | `arn:aws:[a-z0-9\-]+:[a-z0-9\-]*:\d{12}:[^\s]+` | 🟡 MEDIUM | Tokenization |
| `AWS_ACCOUNT_ID` | 12-digit AWS Account ID | `123456789012` | `\b\d{12}\b` + ARN context | 🟡 MEDIUM | Tokenization |
| `AZURE_RESOURCE_ID` | Azure Resource ID | `/subscriptions/xxxx/resourceGroups/prod-rg` | `/subscriptions\/[a-f0-9\-]{36}` | 🟡 MEDIUM | Tokenization |
| `GCP_PROJECT_ID` | Google Cloud Project ID | `bank-prod-20240101` | Context + GCP keyword | 🟢 LOW | Contextual |
| `AZURE_TENANT_ID` | Azure Tenant/Directory ID | `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` | UUID pattern + Azure context | 🟡 MEDIUM | Tokenization |

---

#### 7B — Cloud Storage References

| Entity Type | Description | Example | Detection | Sensitivity | Masking |
|---|---|---|---|---|---|
| `S3_BUCKET_NAME` | AWS S3 bucket name | `s3://prod-bank-customer-docs` | `s3:\/\/[a-z0-9\-\.]+` | 🟡 MEDIUM | Tokenization |
| `S3_BUCKET_URL` | S3 public/presigned URL | `https://prod-bank.s3.amazonaws.com/...` | Regex: `s3\.amazonaws\.com` | 🟡 MEDIUM | Tokenization |
| `AZURE_BLOB_URL` | Azure Blob Storage URL | `https://bankstore.blob.core.windows.net/...` | Regex: `blob\.core\.windows\.net` | 🟡 MEDIUM | Tokenization |
| `GCS_BUCKET` | Google Cloud Storage | `gs://bank-audit-logs-prod` | `gs:\/\/[a-z0-9\-\_]+` | 🟡 MEDIUM | Tokenization |

**Banking Architecture Risk:**  
S3 bucket names and Azure Blob URLs reveal the cloud architecture of the bank's storage systems. Attackers can use these to identify target resources even without credentials. All storage references are MEDIUM sensitivity minimum.

---

#### 7C — Encoded & Obfuscated Secrets *(NEW — v2.0)*

> **Supervisor Feedback Addressed:** *"Developers often paste logs containing JWT tokens. Ensure your regex/NLP patterns account for encoded strings that decode into sensitive data."*

| Entity Type | Description | Example | Detection | Sensitivity | Masking |
|---|---|---|---|---|---|
| `BASE64_ENCODED_SECRET` | Base64 string that decodes to sensitive data | `c2stMTIzNDU2...` | Base64 detection + decode-and-check | 🔴 HIGH | Tokenization |
| `JWT_IN_LOG` | JWT embedded inside log output | `Authorization: Bearer eyJhbG...` | JWT regex within log structure | 🔴 HIGH | Tokenization |
| `URL_ENCODED_SECRET` | URL-encoded credential | `password%3DP%40ssw0rd` | URL decode + keyword | 🔴 HIGH | Full Mask |
| `HEX_ENCODED_SECRET` | Hex-encoded sensitive value | `736b2d313233343536...` | Hex detection + decode-and-check | 🟡 MEDIUM | Tokenization |
| `OBFUSCATED_NIC` | NIC with spaces/dashes (adversarial) | `199 012 345V`, `199-012-345-V` | Normalized regex after whitespace/dash removal | 🔴 HIGH | Full Mask |

**Adversarial Detection Note:**  
The engine must apply **input normalization** before pattern matching:
1. Strip whitespace between digit groups
2. Remove dashes, underscores within potential entity values
3. Decode Base64, URL encoding, and hex before pattern matching
4. Then run full detection pipeline on normalized input

---

## 6. Sensitivity Scoring Matrix

| Sensitivity Level | Score | Examples | Default Masking |
|---|---|---|---|
| 🔴 **CRITICAL** | 10 | CVV, SWIFT messages, Private Keys, SAR data | Full Mask / Tokenization + Immediate Alert |
| 🔴 **HIGH** | 7–9 | API keys, NIC, IBAN, JWT, passwords, cloud keys | Full Mask / Tokenization |
| 🟡 **MEDIUM** | 4–6 | Phone, email, internal IP, S3 bucket, hostname | Partial Mask / Tokenization |
| 🟢 **LOW** | 1–3 | Names alone, currency, GCP project IDs | Contextual — mask only if co-occurring |

---

## 7. Co-occurrence Risk Elevation Rules

| Combination | Individual Levels | Elevated Level |
|---|---|---|
| Name + NIC | LOW + HIGH | CRITICAL |
| Name + Account Number | LOW + HIGH | CRITICAL |
| Email + Password | MEDIUM + HIGH | CRITICAL |
| Account No + Balance + Name | HIGH + HIGH + LOW | CRITICAL |
| Internal IP + DB Password | MEDIUM + HIGH | CRITICAL |
| AWS ARN + AWS Secret Key | MEDIUM + HIGH | CRITICAL |
| Any 3+ entities in one prompt | — | CRITICAL regardless |

---

## 8. Context Type Classification

| Context Type | Indicators | Detection Method | Masking Priority |
|---|---|---|---|
| `natural_language` | Conversational tone, no code syntax | NLP classifier | PII-first |
| `source_code` | Function defs, imports, assignments | Language detection | Credentials-first |
| `config_file` | KEY=VALUE, JSON/YAML, `.env` structure | Regex structure | Secrets-first |
| `log_output` | Timestamps, INFO/ERROR/DEBUG, stack traces | Regex + NLP | PII + credentials |
| `mixed` | Combination of above | Ensemble | Highest sensitivity first |

---

## 9. Masking Strategy Reference

| Strategy | Symbol | When to Apply | Example |
|---|---|---|---|
| **Full Mask** | `************` | CRITICAL/HIGH PII, CVV, passwords, SAR | `P@ssw0rd` → `************` |
| **Partial Mask** | `1234 **** 5678` | MEDIUM — phone, email, PAN | `0771234567` → `077****567` |
| **Tokenization** | `<API_KEY_1>` | Credentials, IDs, cloud resources needing traceability | `sk-123...` → `<API_KEY_1>` |
| **Contextual** | Pass / Mask | LOW — mask only if co-occurring with HIGH | Name alone → pass |

---

## 10. Idempotent Token Mapping *(Updated — v2.0)*

> **Supervisor Feedback Addressed:** *"If the LLM sees `[PERSON_1]` in two different prompts within the same session, it should refer to the same entity to maintain conversational coherence."*

### Session-Scoped Token Registry

The engine maintains a **session-scoped token map** — a dictionary that persists across multiple prompts within the same user session:

```json
{
  "session_id": "sess_20260501_abc123",
  "token_map": {
    "<API_KEY_1>": {
      "original": "sk-123456...",
      "entity_type": "API_KEY_GENERIC",
      "first_seen_prompt": 1,
      "last_seen_prompt": 3
    },
    "<PERSON_1>": {
      "original": "Saman Perera",
      "entity_type": "FULL_NAME",
      "first_seen_prompt": 1,
      "last_seen_prompt": 2
    },
    "<ACCOUNT_1>": {
      "original": "001010012345",
      "entity_type": "BANK_ACCOUNT_NO",
      "first_seen_prompt": 1,
      "last_seen_prompt": 1
    }
  }
}
```

### Idempotency Rules:

1. **Same value, same session → same token.** If `sk-123456` appears in Prompt 1 and Prompt 3, both are replaced with `<API_KEY_1>`, not `<API_KEY_1>` and `<API_KEY_2>`.
2. **Different value, same type → incremented token.** A second different API key becomes `<API_KEY_2>`.
3. **Session ends → token map is destroyed.** Tokens do not persist across sessions to minimize re-identification risk.
4. **Token map is stored locally only.** It is never transmitted to the external AI system.

---

## 11. Instruction Generation Templates *(Updated — v2.0)*

### Standard Instructions:
```json
{
  "instructions": [
    "This prompt contains masked sensitive data represented by tokens (e.g., <API_KEY_1>) or asterisks.",
    "Do not attempt to infer, reconstruct, or request the original values of any masked tokens.",
    "Treat all masked values as placeholders. Do not generate content based on their likely real values.",
    "Do not request the user to reveal masked information.",
    "If a token (e.g., <PERSON_1>) appears in multiple messages, treat it as consistently referring to the same entity."
  ]
}
```

### High-Risk / CRITICAL Instructions:
```json
{
  "instructions": [
    "CRITICAL: This prompt contained highly sensitive data (credentials/financial identifiers/SWIFT data) that has been masked.",
    "Do not attempt to reconstruct, predict, or suggest the likely values of any masked tokens.",
    "Do not provide any guidance, code, or reasoning that would assist in recovering masked values.",
    "Treat this interaction as security-sensitive. Flag for security review.",
    "Token references are session-consistent. <API_KEY_1> refers to the same masked value throughout this session."
  ]
}
```

---

## 12. Dataset Strategy *(Updated — v2.0)*

> **Supervisor Feedback Addressed:** *"Faker often lacks the noise of real-world prompts. Supplement with adversarial examples."*

| Dataset | Purpose | Adversarial Variant |
|---|---|---|
| TruffleHog Regex Patterns | Regex detection baseline | N/A |
| SecretBench (15,084 secrets) | Real-world credential examples | N/A |
| Faker (Python) synthetic PII | Normal PII generation | Add typos, mixed formats |
| Custom SL NIC + phone generator | Sri Lankan PII | Obfuscated: `199 012 345V`, `0771-234567` |
| Custom banking prompt generator | Banking scenario prompts | Mixed Sinhala-English, noisy logs |
| **Adversarial Dataset (NEW)** | Engine evasion testing | Spaced NICs, Base64 keys, hex-encoded passwords |

### Adversarial Test Case Types:
- **Spacing attacks:** `sk - 1 2 3 4 5 6` (spaces inserted into API key)
- **Mixed case:** `Sk-123456` (capitalization variation)
- **Encoded secrets:** Base64, URL-encoded, hex-encoded sensitive values
- **Partial disclosure:** `my password starts with P@ss...` (incomplete but inferable)
- **Indirect reference:** `same credentials as last time` (cross-prompt reference)

---

## 13. Out of Scope (v2.0)

- Sinhala/Tamil language NER detection
- OCR-based extraction from images/documents
- Audio/video prompt analysis
- Inbound AI response filtering
- Real-time external threat intelligence feed integration
- SWIFT network-level interception (out of scope — prompt-level only)

---

## 14. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0 | May 2026 | Initial taxonomy — Banking sector, SL + International compliance |
| 2.0 | May 2026 | Supervisor review integration: SWIFT CSP, Category 7 (Cloud + Encoded Secrets), Confidence Score framework, Idempotent token mapping, Adversarial dataset strategy |

---

*This document is version-controlled at `/taxonomy/sensitive_data_taxonomy.md` in the project GitHub repository.*
