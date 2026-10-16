"""
retrain_classifier.py — Gated Retrain Pipeline for the Layer-2 ML Safety Net
R26-CS-012: Context-Aware Masking + Instruction Engine

WHY THIS SCRIPT EXISTS (see docs/Comprehensive Technical Documentation.md
Section 8.3): the first version of this classifier scored ordinary,
unrelated sentences as 95%+ "sensitive" because its negative-class
training examples were all narrow, taxonomy-specific templates. That was
found by hand, once, by manually testing a couple of sentences after the
model was already trained and shipped. That is not a repeatable process.

This script makes the check structural instead of manual:
  1. Trains on data/synthetic_dataset.json (taxonomy-labeled) PLUS
     research/negative_corpus.json (2,000 open-domain negative examples —
     see generate_negative_corpus.py for why that file exists).
  2. Evaluates on a genuine held-out split of THAT combined data.
  3. ALSO evaluates against research/ood_probe_set.json — 40 hand-authored
     sentences NEVER included in training, at all, in any split. This is
     the only number in this script that actually tests generalization to
     text the model has never structurally seen the shape of before.
  4. REFUSES to overwrite research/models_rf_classifier.pkl if the probe
     set's false-positive rate exceeds OOD_FP_THRESHOLD — a bad retrain
     fails loudly here instead of silently shipping to production.

Run with:
    cd masking_engine
    python research/retrain_classifier.py
(needs scikit-learn/joblib — the same optional dependency engine/ml_anomaly.py
already treats as soft; this script is a research/ tool, not part of engine/.)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from engine.ml_anomaly import _extract_features, _has_structural_evidence

_BASE = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_BASE, "..", "data", "synthetic_dataset.json")
_NEGATIVE_CORPUS_PATH = os.path.join(_BASE, "negative_corpus.json")
_PROBE_SET_PATH = os.path.join(_BASE, "ood_probe_set.json")
_MODEL_OUT_PATH = os.path.join(_BASE, "models_rf_classifier.pkl")

# A retrain that flags more than this fraction of the (all-benign) OOD
# probe set is rejected outright — this is a low bar deliberately: the
# probe set exists specifically to catch the "flags ordinary sentences"
# failure mode, so any nontrivial rate there means the same root cause
# (narrow/non-generalizing negative class) is still present.
OOD_FP_THRESHOLD = 0.15

CATEGORY_MAP = {
    'NIC_OLD': 'PII', 'NIC_NEW': 'PII', 'PASSPORT': 'PII', 'DRIVING_LICENSE': 'PII', 'TAX_ID': 'PII',
    'FULL_NAME': 'PII', 'DATE_OF_BIRTH': 'PII', 'HOME_ADDRESS': 'PII',
    'PHONE_LK': 'Contact', 'PHONE_INTL': 'Contact', 'EMAIL': 'Contact',
    'PAN': 'PCI', 'CVV': 'PCI', 'CARD_EXPIRY': 'PCI',
    'BANK_ACCOUNT_NO': 'Financial', 'IBAN': 'Financial', 'SWIFT_BIC': 'Financial',
    'SWIFT_MT103': 'Financial', 'SWIFT_MT202': 'Financial',
    'API_KEY_OPENAI': 'Secret', 'API_KEY_GENERIC': 'Secret', 'JWT_TOKEN': 'Secret',
    'PASSWORD': 'Secret', 'PRIVATE_KEY': 'Secret', 'JWT_IN_LOG': 'Secret',
    'AWS_ACCESS_KEY': 'Cloud', 'AWS_SECRET_KEY': 'Cloud', 'S3_BUCKET_REF': 'Cloud',
    'DB_CONNECTION_STRING': 'Infra', 'INTERNAL_IP': 'Infra',
}


def _label_for(entities):
    cats = {CATEGORY_MAP.get(e, 'Other') for e in entities}
    return 1 if any(c in ('Secret', 'PCI', 'PII', 'Cloud') for c in cats) else 0


def load_training_data():
    taxonomy_rows = json.load(open(_DATASET_PATH))
    negative_rows = json.load(open(_NEGATIVE_CORPUS_PATH))

    X, y = [], []
    for row in taxonomy_rows:
        X.append(list(_extract_features(row["prompt"]).values()))
        y.append(_label_for(row["entities"]))
    for row in negative_rows:
        X.append(list(_extract_features(row["prompt"]).values()))
        y.append(0)  # negative_corpus.json is open-domain, always non-sensitive

    return np.array(X), np.array(y), len(taxonomy_rows), len(negative_rows)


def evaluate_ood_probe(model) -> float:
    """Returns the false-positive RATE on the probe set (all label 0) —
    what fraction of genuinely benign, never-trained-on sentences the
    model alone (no structural gate) would flag."""
    probe_rows = json.load(open(_PROBE_SET_PATH))
    X_probe = np.array([list(_extract_features(r["prompt"]).values()) for r in probe_rows])
    preds = model.predict(X_probe)
    fp_rate = float(np.mean(preds == 1))

    gated_flags = 0
    for r, pred in zip(probe_rows, preds):
        if pred == 1 and _has_structural_evidence(_extract_features(r["prompt"])):
            gated_flags += 1

    print(f"\n  OOD PROBE SET ({len(probe_rows)} never-trained-on sentences, all genuinely benign)")
    print(f"    Classifier-alone false-positive rate : {fp_rate:.1%}")
    print(f"    Would-flag WITH structural gate       : {gated_flags}/{len(probe_rows)} "
          f"({gated_flags/len(probe_rows):.1%}) — this is what production actually does")
    return fp_rate


def main():
    X, y, n_taxonomy, n_negative = load_training_data()
    print(f"Training rows: {len(y)} ({n_taxonomy} from synthetic_dataset.json, "
          f"{n_negative} from negative_corpus.json)")
    print(f"Class balance: positive={int(y.sum())} negative={int(len(y) - y.sum())}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    print(f"Split: train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(max_depth=4, random_state=42)),
    ])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"5-fold CV accuracy on train split: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    pipe.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, pipe.predict(X_val))
    print(f"Validation accuracy: {val_acc:.4f}")

    test_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Held-out TEST accuracy: {test_acc:.4f}")
    print(classification_report(y_test, test_pred, target_names=['not_sensitive', 'sensitive']))

    ood_fp_rate = evaluate_ood_probe(pipe)

    print(f"\n{'='*70}")
    if ood_fp_rate > OOD_FP_THRESHOLD:
        print(f"  REJECTED — OOD false-positive rate {ood_fp_rate:.1%} exceeds "
              f"the {OOD_FP_THRESHOLD:.0%} threshold.")
        print(f"  models_rf_classifier.pkl was NOT overwritten.")
        print(f"  Fix: add more diverse examples to negative_corpus.json "
              f"(different topics/structures, not more of the same), then retry.")
        print(f"{'='*70}\n")
        sys.exit(1)

    joblib.dump(pipe, _MODEL_OUT_PATH)
    print(f"  ACCEPTED — OOD false-positive rate {ood_fp_rate:.1%} is within "
          f"the {OOD_FP_THRESHOLD:.0%} threshold.")
    print(f"  Saved: {_MODEL_OUT_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
