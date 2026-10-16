# Install required dependencies

# Install TruffleHog binary
# !curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b /usr/local/bin

# Verify installation
# !trufflehog --version

import pandas as pd
import plotly.express as px
import ast
import os

DATA_PATH = 'data/synthetic_dataset.json'

if os.path.exists(DATA_PATH):
    df = pd.read_json(DATA_PATH)
    # JSON handles lists natively
    print("Dataset loaded successfully.")
else:
    print(f"File {DATA_PATH} not found. Using a mock dataset for demonstration.")
    df = pd.DataFrame({
        'prompt': ['My card is 4111222233334444 and my key is sk-abcdef1234567890'],
        'entities': [['PAN', 'API_KEY_OPENAI']],
        'type': ['normal']
    })

# Show class distribution
all_entities = [ent for sublist in df['entities'] for ent in sublist]
if all_entities:
    entity_counts = pd.Series(all_entities).value_counts().reset_index()
    entity_counts.columns = ['Entity Type', 'Count']
    fig = px.bar(entity_counts, x='Entity Type', y='Count', title='Entity Type Distribution', color='Count', color_continuous_scale='Viridis')
    pass # 
# Map to broader categories
category_map = {
    "NIC_OLD": "PII", "NIC_NEW": "PII", "PASSPORT": "PII", "DRIVING_LICENSE": "PII", "TAX_ID": "PII",
    "PHONE_LK": "Contact", "PHONE_INTL": "Contact", "EMAIL": "Contact",
    "PAN": "PCI", "CVV": "PCI", "CARD_EXPIRY": "PCI",
    "BANK_ACCOUNT_NO": "Financial", "IBAN": "Financial", "SWIFT_BIC": "Financial", "SWIFT_MT103": "Financial", "SWIFT_MT202": "Financial",
    "API_KEY_OPENAI": "Secret", "API_KEY_GENERIC": "Secret", "JWT_TOKEN": "Secret", "PASSWORD": "Secret", "PRIVATE_KEY": "Secret",
    "AWS_ACCESS_KEY": "Cloud", "AWS_SECRET_KEY": "Cloud", "S3_BUCKET_REF": "Cloud",
    "DB_CONNECTION_STRING": "Infra", "INTERNAL_IP": "Infra", "JWT_IN_LOG": "Secret"
}

df['broad_categories'] = df['entities'].apply(lambda ents: list(set(category_map.get(e, "Other") for e in ents)))
broad_counts = pd.Series([cat for sublist in df['broad_categories'] for cat in sublist]).value_counts().reset_index()
broad_counts.columns = ['Category', 'Count']
if not broad_counts.empty:
    fig2 = px.pie(broad_counts, names='Category', values='Count', title='Broad Category Distribution', hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma)
    pass # 
import re
import base64
import urllib.parse

def classify_context(text: str) -> str:
    CONTEXT_INDICATORS = {
        "source_code": [r"\bdef\s+\w+\s*\(", r"\bimport\s+\w+", r"=>", r"\breturn\b", r"class\s+\w+"],
        "config_file": [r"^[A-Z_]+=.+", r'"[^"]+"\s*:\s*"[^"]+"', r"\.env", r"\w+:\s+\S+"],
        "log_output": [r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", r"\b(INFO|ERROR|DEBUG|WARN|CRITICAL)\b", r"\[[\d\-:T]+\]"],
        "natural_language": [r"\b(please|can you|help me|I need)\b", r"\b(customer|account|transfer)\b"]
    }
    scores = {ctx: 0 for ctx in CONTEXT_INDICATORS}
    for ctx, patterns in CONTEXT_INDICATORS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                scores[ctx] += 1
    max_score = max(scores.values())
    if max_score == 0: return "natural_language"
    winners = [ctx for ctx, s in scores.items() if s == max_score]
    return "mixed" if len(winners) > 1 else winners[0]

def tokenize(text: str):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]

def extract_custom_features(text: str):
    # Custom Feature Flags for ML Models
    tokens = tokenize(text)
    return {
        "is_high_entropy": len(set(text)) / len(text) > 0.7 if len(text) > 0 else False,
        "has_special_chars": bool(re.search(r'[^a-zA-Z0-9\s]', text)),
        "token_count": len(tokens),
        "context_type": classify_context(text),
        "has_base64_like": bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text)),
        "has_hex_like": bool(re.search(r'\b[0-9a-fA-F]{10,}\b', text))
    }

# Demo
sample_text = "Please check this customer log. error token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
print(f"Sample: {sample_text}")
print(f"Features: {extract_custom_features(sample_text)}")

import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import numpy as np

# Define binary target: 1 if contains Secret, PCI, PII, Cloud; else 0
df['contains_sensitive'] = df['broad_categories'].apply(lambda x: 1 if any(c in ['Secret', 'PCI', 'PII', 'Cloud'] for c in x) else 0)

X_text = df['prompt'].fillna('').to_numpy(dtype=object)
y = df['contains_sensitive'].to_numpy(dtype=int)

if len(X_text) > 1:
    # Calculate custom features
    custom_features = pd.DataFrame([extract_custom_features(text) for text in X_text])
    custom_features['context_encoded'] = LabelEncoder().fit_transform(custom_features['context_type'])
    X_custom = custom_features[['is_high_entropy', 'has_special_chars', 'token_count', 'has_base64_like', 'has_hex_like', 'context_encoded']].astype(float).to_numpy(dtype=float)

    X_train_txt, X_test_txt, X_train_cust, X_test_cust, y_train, y_test = train_test_split(X_text, X_custom, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train_txt)
    X_test_tfidf = vectorizer.transform(X_test_txt)

    # Combine TF-IDF with custom features
    X_train_combined = hstack([X_train_tfidf, X_train_cust])
    X_test_combined = hstack([X_test_tfidf, X_test_cust])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear')
    }

    results = []
    for name, model in models.items():
        # if there's only one class in y_train (like in demo), this could fail, wrap in try-except
        try:
            model.fit(X_train_combined, y_train)
            score = model.score(X_test_combined, y_test)
            results.append({'Model': name, 'Accuracy': score})
            print(f"--- {name} ---")
            print(classification_report(y_test, model.predict(X_test_combined)))
        except ValueError as e:
            print(f"Could not train {name} due to: {e}")

    # Save the best model (e.g. Random Forest)
    if 'Random Forest' in models:
        joblib.dump(models['Random Forest'], 'models_rf_classifier.pkl')
        joblib.dump(vectorizer, 'models_tfidf_vectorizer.pkl')
        print('Models saved to models_rf_classifier.pkl and models_tfidf_vectorizer.pkl')
    if results:
        fig3 = px.bar(pd.DataFrame(results), x='Model', y='Accuracy', title='Classifier Comparison (TF-IDF + Custom Features)', color='Accuracy', color_continuous_scale='Blues')
        pass # else:
    print("Not enough data to train classifiers.")

import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import random

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.length()': len(word),
        'word.suffix_3': word[-3:],
        'word.prefix_3': word[:3],
        'word.is_numeric': bool(re.match(r'^\d+$', word)),
        'word.has_special_chars': bool(re.search(r'[^a-zA-Z0-9\s]', word))
    }
    if i > 0:
        word1 = sent[i-1][0][0]
        features.update({'-1:word.lower()': word1.lower(), '-1:word.istitle()': word1.istitle()})
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0][0]
        features.update({'+1:word.lower()': word1.lower(), '+1:word.istitle()': word1.istitle()})
    else:
        features['EOS'] = True
    return features

def sent2features(sent): return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent): return [label for token, label in sent]

# Mocking data for CRF training
crf_data = []
for idx, row in df.head(500).iterrows():
    tokens = [t[0] for t in tokenize(row['prompt'])]
    labels = ['O'] * len(tokens)
    # For demo, assign B-SENSITIVE based on contains_sensitive flag randomly to simulate entities
    if row['contains_sensitive'] == 1 and len(tokens) > 0:
        labels[random.randint(0, len(tokens)-1)] = 'B-SENSITIVE'
    crf_data.append(list(zip(tokens, labels)))

X_crf = [sent2features(s) for s in crf_data]
y_crf = [sent2labels(s) for s in crf_data]

if len(X_crf) > 1:
    X_train_crf, X_test_crf, y_train_crf, y_test_crf = train_test_split(X_crf, y_crf, test_size=0.2, random_state=42)
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=50,
        all_possible_transitions=True
    )
    try:
        crf.fit(X_train_crf, y_train_crf)
        y_pred_crf = crf.predict(X_test_crf)
        labels = list(crf.classes_)
        if 'O' in labels: labels.remove('O')
        joblib.dump(crf, 'models_crf_ner.pkl')
        print('CRF Model saved to models_crf_ner.pkl')
        print("CRF Evaluation:")
        if labels:
            print(metrics.flat_classification_report(y_test_crf, y_pred_crf, labels=labels))
        else:
            print("No sensitive labels found to evaluate.")
    except Exception as e:
        print(f"CRF Error: {e}")

import torch
from transformers import pipeline
import subprocess
import json

print("Loading DistilBERT pipeline for prompt classification...")
try:
    import torch
    classifier = pipeline("text-classification", model="distilbert-base-uncased")
except Exception as e:
    print(f"DistilBERT load failed: {e}")
    classifier = None

def run_trufflehog(text: str):
    with open("temp_scan.txt", "w") as f:
        f.write(text)
    try:
        result = subprocess.run(
            ["trufflehog", "filesystem", "temp_scan.txt", "--json"], 
            capture_output=True, text=True
        )
        findings = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    findings.append(json.loads(line))
                except:
                    pass
        return findings
    except FileNotFoundError:
        return "Trufflehog binary not found."

def security_audit(text: str):
    print(f"\n--- Security Audit ---")
    print(f"Input: {text}\n")
    transformer_pred = classifier(text) if classifier else [{"label": "N/A", "score": 0.0}]
    print(f"Transformer Prediction: {transformer_pred}\n")
    
    th_findings = run_trufflehog(text)
    if isinstance(th_findings, list) and len(th_findings) > 0:
        print(f"TruffleHog Detected: {len(th_findings)} secret(s)")
        for f in th_findings:
            print(f" - Type: {f.get('DetectorName', 'Unknown')} | Value: {f.get('Raw', 'Masked')}")
    else:
        print("TruffleHog: No secrets found or binary missing.")

test_secret_text = "Here is an AWS key for you to use: AKIAIOSFODNN7EXAMPLE and secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
security_audit(test_secret_text)

import plotly.graph_objects as go
from IPython.display import display, HTML
import ipywidgets as widgets

# Mock Engine Evaluation Dashboard (Based on evaluate.py outputs)
eval_data = {
    'Entity Type': ['API_KEY_OPENAI', 'PAN', 'EMAIL', 'NIC_OLD', 'AWS_ACCESS_KEY'],
    'Precision': [0.99, 0.95, 1.00, 0.92, 0.98],
    'Recall': [0.98, 0.91, 0.99, 0.88, 0.99],
    'F1 Score': [0.98, 0.93, 0.99, 0.90, 0.98]
}
eval_df = pd.DataFrame(eval_data)
fig_eval = go.Figure(data=[go.Table(
    header=dict(values=list(eval_df.columns), fill_color='paleturquoise', align='left'),
    cells=dict(values=[eval_df['Entity Type'], eval_df['Precision'], eval_df['Recall'], eval_df['F1 Score']], fill_color='lavender', align='left'))
])
fig_eval.update_layout(title='Engine Evaluation: Precision/Recall Table', margin=dict(l=0, r=0, t=30, b=0), height=300)
pass # 
def partial_mask(value: str) -> str:
    if len(value) > 6:
        return value[:2] + ("*" * (len(value) - 4)) + value[-2:]
    return "*" * len(value)

def apply_masking_strategy(text):
    masked_text = text
    # Dummy regex detector
    spans = []
    for m in re.finditer(r'\b4\d{12}(?:\d{3})?\b', text): spans.append((m.start(), m.end(), 'PAN'))
    for m in re.finditer(r'\b\d{3,4}\b', text): spans.append((m.start(), m.end(), 'CVV'))
    for m in re.finditer(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text): spans.append((m.start(), m.end(), 'EMAIL'))
    for m in re.finditer(r'\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16})\b', text): spans.append((m.start(), m.end(), 'SECRET'))

    for start, end, label in sorted(spans, key=lambda x: x[0], reverse=True):
        val = text[start:end]
        if label == 'SECRET':
            replacement = f"<{label}_TOKEN>"
        else:
            replacement = partial_mask(val)
        masked_text = masked_text[:start] + replacement + masked_text[end:]
    return masked_text


# Non-interactive demo (widgets not available in nbconvert)
demo_text = "Refund requested for card 4111222233334444 CVV 123. Contact: saman.p@example.com. API key sk-abcdef12345678901234."
print("Input:", demo_text)
print("Masked:", apply_masking_strategy(demo_text))
