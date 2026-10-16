# R26-CS-012 — Context-Aware Masking + Instruction Engine

**Author:** Abayathilake S.S — IT22193186
**Domain:** Banking Sector (Sri Lanka + International)

A multi-layered pipeline that detects and masks sensitive data (PII, PCI, credentials, SWIFT/banking identifiers) in text prompts before they reach an external LLM.

## Where things live

- **`masking_engine/`** — the actual project. See [`masking_engine/README.md`](masking_engine/README.md) for the production pipeline (`engine/`, `main.py`, `evaluate.py`), the dataset generator, the test suite, and the `research/` notebook (ML classifier + CRF NER training, kept separate from the production regex/rule-based pipeline).
- **`masking_engine_architecture.md`** — pipeline design walkthrough (Normalizer → Detector → Confidence Scorer → Masker → Instruction Generator).
- **`sensitive_data_taxonomy_v2.md`** — the full sensitive-data taxonomy spec (entity categories, confidence-score framework, co-occurrence rules) this project implements against.
- **`masking_engine/Model_Regex_Docs.md`** — technical deep-dive, including documented, deliberate deviations from the taxonomy spec where empirical testing found the literal rule caused problems (e.g. the co-occurrence "3+ entities → CRITICAL" rule).
- **`archive/`** — superseded one-off scripts (notebook patch scripts, early scratch drafts) kept for reference, not part of the active project.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is only needed for `masking_engine/research/` (the Jupyter notebook). The production pipeline (`masking_engine/engine/`, `main.py`, `evaluate.py`) has zero external dependencies — pure Python 3.8+.

## Quick start

```bash
cd masking_engine
python main.py --demo              # run the built-in demo cases
python -m unittest discover -s tests -v   # run the regression test suite
python evaluate.py                 # precision/recall/F1 against the synthetic dataset
```
