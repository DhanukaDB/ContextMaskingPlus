
import pandas as pd
import plotly.express as px
import json, os, re, math

DATA_PATH = 'data/synthetic_dataset.json'
category_map = {
    'NIC_OLD':'PII','NIC_NEW':'PII','PASSPORT':'PII','DRIVING_LICENSE':'PII','TAX_ID':'PII',
    'FULL_NAME':'PII','DATE_OF_BIRTH':'PII','HOME_ADDRESS':'PII',
    'PHONE_LK':'Contact','PHONE_INTL':'Contact','EMAIL':'Contact',
    'PAN':'PCI','CVV':'PCI','CARD_EXPIRY':'PCI',
    'BANK_ACCOUNT_NO':'Financial','IBAN':'Financial','SWIFT_BIC':'Financial',
    'SWIFT_MT103':'Financial','SWIFT_MT202':'Financial',
    'API_KEY_OPENAI':'Secret','API_KEY_GENERIC':'Secret','JWT_TOKEN':'Secret',
    'PASSWORD':'Secret','PRIVATE_KEY':'Secret','JWT_IN_LOG':'Secret',
    'AWS_ACCESS_KEY':'Cloud','AWS_SECRET_KEY':'Cloud','S3_BUCKET_REF':'Cloud',
    'DB_CONNECTION_STRING':'Infra','INTERNAL_IP':'Infra'
}

if os.path.exists(DATA_PATH):
    df = pd.read_json(DATA_PATH)
    print(f'Loaded {len(df)} records')
else:
    raise FileNotFoundError(f'{DATA_PATH} not found. Run from masking_engine/ directory.')

df['broad_categories'] = df['entities'].apply(
    lambda ents: list(set(category_map.get(e, 'Other') for e in ents))
)
df['contains_sensitive'] = df['broad_categories'].apply(
    lambda x: 1 if any(c in ['Secret','PCI','PII','Cloud'] for c in x) else 0
)

print('Class balance:', df['contains_sensitive'].value_counts().to_dict())
print('Prompt types:', df['type'].value_counts().to_dict())

# EDA chart
all_ents = [e for row in df['entities'] for e in row]
ec = pd.Series(all_ents).value_counts().reset_index()
ec.columns = ['Entity','Count']
fig = px.bar(ec, x='Entity', y='Count', title='Entity Distribution', color='Count',
             color_continuous_scale='Viridis')
pass
import numpy as np

def shannon_entropy(text: str) -> float:
    if not text: return 0.0
    freq = {c: text.count(c)/len(text) for c in set(text)}
    return -sum(p * math.log2(p) for p in freq.values() if p > 0)

def extract_features(text: str) -> dict:
    tokens = text.split()
    entropies = [shannon_entropy(t) for t in tokens] if tokens else [0]
    long_tokens = [t for t in tokens if len(t) > 15]
    return {
        # Entropy signals
        'mean_token_entropy':  float(np.mean(entropies)),
        'max_token_entropy':   float(np.max(entropies)),
        'high_entropy_ratio':  sum(1 for e in entropies if e > 3.5) / max(len(entropies), 1),
        # Length / structure signals
        'text_length':         len(text),
        'token_count':         len(tokens),
        'long_token_ratio':    len(long_tokens) / max(len(tokens), 1),
        'avg_token_length':    float(np.mean([len(t) for t in tokens])) if tokens else 0,
        # Pattern signals (regex match counts, NOT the matched value)
        'n_digit_runs':        len(re.findall(r'\d{6,}', text)),
        'n_special_chars':     len(re.findall(r'[^a-zA-Z0-9\s]', text)),
        'n_at_symbols':        text.count('@'),
        'n_slashes':           text.count('/'),
        'n_equals':            text.count('='),
        'has_base64_pattern':  int(bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text))),
        'has_hex_run':         int(bool(re.search(r'[0-9a-fA-F]{12,}', text))),
        'has_bearer_keyword':  int(bool(re.search(r'\b(token|key|secret|password|bearer|jwt|api)\b', text, re.I))),
        'has_akia_prefix':     int('AKIA' in text),
        'has_sk_prefix':       int(bool(re.search(r'\bsk-[A-Za-z0-9]', text))),
    }

print('Feature count:', len(extract_features('test')))
print('Sample features for a known sensitive prompt:')
test_p = df[df['contains_sensitive']==1]['prompt'].iloc[0]
print(' Prompt:', test_p[:80])
print(' Features:', extract_features(test_p))
from sklearn.model_selection import train_test_split

# Build feature matrix
X_features = np.array([list(extract_features(p).values()) for p in df['prompt']])
feature_names = list(extract_features(df['prompt'].iloc[0]).keys())
y = df['contains_sensitive'].to_numpy()

# Stratify on combined label to keep type balance
strat_key = df['contains_sensitive'].astype(str) + '_' + df['type']

X_temp, X_test, y_temp, y_test, strat_temp, _ = train_test_split(
    X_features, y, strat_key, test_size=0.20, random_state=42, stratify=strat_key
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=strat_temp  # 0.25 x 0.8 = 0.20
)

print(f'Train:      {len(X_train):>4} samples  sensitive={y_train.sum()} / non={len(y_train)-y_train.sum()}')
print(f'Validation: {len(X_val):>4} samples  sensitive={y_val.sum()} / non={len(y_val)-y_val.sum()}')
print(f'Test:       {len(X_test):>4} samples  sensitive={y_test.sum()} / non={len(y_test)-y_test.sum()}')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.5, max_iter=500, class_weight='balanced'))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=8,
                                        min_samples_leaf=5, random_state=42,
                                        class_weight='balanced'))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                            learning_rate=0.1, random_state=42))
    ]),
    'Neural Network (50 Epochs)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, early_stopping=False, random_state=42))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True))
    ]),
}

cv_results = []
trained_models = {}

for name, pipe in models.items():
    print(f'\n--- {name} - {N_FOLDS}-Fold CV ---')
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        pipe.fit(X_train[tr_idx], y_train[tr_idx])
        acc = pipe.score(X_train[va_idx], y_train[va_idx])
        fold_scores.append(acc)
        print(f'  Epoch/Fold {fold}/{N_FOLDS}  accuracy = {acc:.4f}')

    mean_cv = float(np.mean(fold_scores))
    std_cv  = float(np.std(fold_scores))
    print(f'  CV mean = {mean_cv:.4f}  ±  {std_cv:.4f}')

    # Retrain on full train set, evaluate on val
    pipe.fit(X_train, y_train)
    val_acc = pipe.score(X_val, y_val)
    try:
        val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:,1])
    except Exception:
        val_auc = float('nan')
    print(f'  Val accuracy = {val_acc:.4f}  |  Val AUC = {val_auc:.4f}')

    cv_results.append({'Model': name, 'CV Mean Acc': round(mean_cv,4),
                        'CV Std': round(std_cv,4), 'Val Acc': round(val_acc,4),
                        'Val AUC': round(val_auc,4)})
    trained_models[name] = pipe

cv_df = pd.DataFrame(cv_results)
print('\n--- Summary Table ---')
print(cv_df.to_string(index=False))
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go

best_model_name = cv_df.sort_values('Val AUC', ascending=False).iloc[0]['Model']
best_pipe = trained_models[best_model_name]

print(f'Best model (by Val AUC): {best_model_name}')
y_pred_test = best_pipe.predict(X_test)
print('\nClassification Report (Test Set):')
print(classification_report(y_test, y_pred_test,
      target_names=['Non-Sensitive', 'Sensitive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
fig_cm = go.Figure(go.Heatmap(
    z=cm, x=['Pred Non-Sensitive','Pred Sensitive'],
    y=['True Non-Sensitive','True Sensitive'],
    colorscale='Blues', showscale=True,
    text=cm.astype(str), texttemplate='%{text}'
))
fig_cm.update_layout(title=f'Confusion Matrix — {best_model_name} (Test Set)')
pass

# CV comparison chart
fig_cv = go.Figure()
fig_cv.add_trace(go.Bar(name='CV Mean Acc', x=cv_df['Model'], y=cv_df['CV Mean Acc'],
                         error_y=dict(type='data', array=cv_df['CV Std'].tolist())))
fig_cv.add_trace(go.Bar(name='Val AUC',     x=cv_df['Model'], y=cv_df['Val AUC']))
fig_cv.update_layout(title='Classifier Comparison — CV Accuracy vs Val AUC',
                      barmode='group', yaxis=dict(range=[0,1]))
pass
import joblib
joblib.dump(best_pipe, 'models_rf_classifier.pkl')
print(f'Saved: models_rf_classifier.pkl  ({best_model_name})')
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

def word_features(tokens, i):
    w = tokens[i]
    feat = {
        'bias': 1.0,
        'w.lower': w.lower(),
        'w.isupper': w.isupper(),
        'w.isdigit': w.isdigit(),
        'w.len': len(w),
        'w.entropy': round(shannon_entropy(w), 2),
        'w.has_digit': any(c.isdigit() for c in w),
        'w.has_special': bool(re.search(r'[^a-zA-Z0-9]', w)),
        'w.suffix3': w[-3:],
        'w.prefix3': w[:3],
    }
    if i > 0:
        feat['-1:w'] = tokens[i-1].lower()
    else:
        feat['BOS'] = True
    if i < len(tokens)-1:
        feat['+1:w'] = tokens[i+1].lower()
    else:
        feat['EOS'] = True
    return feat

SENSITIVE_ENTITIES = {
    'API_KEY_OPENAI','PAN','AWS_ACCESS_KEY','AWS_SECRET_KEY',
    'PASSWORD','PRIVATE_KEY','JWT_TOKEN','API_KEY_GENERIC'
}

def build_crf_sequence(row):
    tokens = row['prompt'].split()
    is_sensitive = any(e in SENSITIVE_ENTITIES for e in row['entities'])
    labels = ['O'] * len(tokens)
    if is_sensitive:
        # Mark high-entropy tokens as B-SENSITIVE
        for i, t in enumerate(tokens):
            if shannon_entropy(t) > 3.5 and len(t) > 8:
                labels[i] = 'B-SENSITIVE'
    return ([word_features(tokens, i) for i in range(len(tokens))], labels)

crf_samples = [build_crf_sequence(r) for _, r in df.iterrows() if len(r['prompt'].split()) > 1]
X_crf = [s[0] for s in crf_samples]
y_crf = [s[1] for s in crf_samples]

X_ctr, X_cte, y_ctr, y_cte = train_test_split(
    X_crf, y_crf, test_size=0.20, random_state=42
)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', c1=0.1, c2=0.1,
    max_iterations=50, all_possible_transitions=True
)
crf.fit(X_ctr, y_ctr)
y_crf_pred = crf.predict(X_cte)

active_labels = [l for l in crf.classes_ if l != 'O']
if active_labels:
    print('CRF NER Evaluation (test split):')
    print(crf_metrics.flat_classification_report(y_cte, y_crf_pred, labels=active_labels))
else:
    print('No B-SENSITIVE tokens detected in test set.')

joblib.dump(crf, 'models_crf_ner.pkl')
print('Saved: models_crf_ner.pkl')
PATTERNS = [
    (r'\b4\d{12}(?:\d{3})?\b',                            'PAN'),
    (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',   'EMAIL'),
    (r'\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16})\b',      'SECRET'),
    (r'\b\d{9,16}\b',                                         'NUMBER'),
]

def partial_mask(v):
    return v[:2] + '*'*(len(v)-4) + v[-2:] if len(v)>6 else '*'*len(v)

def mask_text(text):
    spans = []
    for pat, label in PATTERNS:
        for m in re.finditer(pat, text):
            spans.append((m.start(), m.end(), label, m.group()))
    result = text
    for start, end, label, val in sorted(spans, key=lambda x: x[0], reverse=True):
        rep = f'<{label}_REDACTED>' if label=='SECRET' else partial_mask(val)
        result = result[:start] + rep + result[end:]
    return result

demos = [
    'Card 4111222233334444 CVV 123. Email: saman@example.com. Key sk-abcdef12345678901234.',
    'IAM access key AKIAIOSFODNN7EXAMPLE has full S3 access.',
    'Account transfer from 07044404463297 approved.',
]
for d in demos:
    print('IN: ', d)
    print('OUT:', mask_text(d))
    print()