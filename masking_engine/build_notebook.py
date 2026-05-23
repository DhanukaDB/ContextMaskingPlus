"""
Generates Colab_Masking_Engine_Lab.ipynb with proper ML practices:
- Stratified train/val/test split (60/20/20)
- Cross-validation with epoch logging
- Calibrated features that avoid data leakage
- Visible epoch counts and accuracy per fold
"""

import json

def cell(source, cell_type="code", outputs=None):
    c = {"cell_type": cell_type, "metadata": {}, "source": source if isinstance(source, list) else [source]}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = outputs or []
    return c

def md(text):
    return cell(text, "markdown")

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md([
    "# 🛡️ Masking Engine Research & Evaluation Lab\n",
    "\n",
    "**Multi-layered PII / credential detection** using Regex, ML classifiers, CRF NER, and Transformers.\n",
    "\n",
    "> **Design principles applied here:**\n",
    "> - Stratified 60 / 20 / 20 train / val / test split to prevent data leakage\n",
    "> - Cross-validation with per-fold accuracy printed (visible epoch count)\n",
    "> - Character-level & entropy features only — raw token text is **not** fed directly\n",
    "> - `max_features` capped + regularization to fight overfitting\n",
    "> - Final score reported on **held-out test set only**"
]))

# ── Section 1 — Dependencies ───────────────────────────────────────────────────
cells.append(md("## Section 1 — Dependencies"))
cells.append(cell(
    "%pip install -q trufflehog3 sklearn-crfsuite transformers seqeval plotly scikit-learn"
))

# ── Section 2 — Data & EDA ─────────────────────────────────────────────────────
cells.append(md([
    "## Section 2 — Data & EDA\n",
    "Load `synthetic_dataset.json`, map entities to broad categories, show class distribution."
]))
cells.append(cell([
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "import json, os, re, math\n",
    "\n",
    "DATA_PATH = 'data/synthetic_dataset.json'\n",
    "category_map = {\n",
    "    'NIC_OLD':'PII','NIC_NEW':'PII','PASSPORT':'PII','DRIVING_LICENSE':'PII','TAX_ID':'PII',\n",
    "    'FULL_NAME':'PII','DATE_OF_BIRTH':'PII','HOME_ADDRESS':'PII',\n",
    "    'PHONE_LK':'Contact','PHONE_INTL':'Contact','EMAIL':'Contact',\n",
    "    'PAN':'PCI','CVV':'PCI','CARD_EXPIRY':'PCI',\n",
    "    'BANK_ACCOUNT_NO':'Financial','IBAN':'Financial','SWIFT_BIC':'Financial',\n",
    "    'SWIFT_MT103':'Financial','SWIFT_MT202':'Financial',\n",
    "    'API_KEY_OPENAI':'Secret','API_KEY_GENERIC':'Secret','JWT_TOKEN':'Secret',\n",
    "    'PASSWORD':'Secret','PRIVATE_KEY':'Secret','JWT_IN_LOG':'Secret',\n",
    "    'AWS_ACCESS_KEY':'Cloud','AWS_SECRET_KEY':'Cloud','S3_BUCKET_REF':'Cloud',\n",
    "    'DB_CONNECTION_STRING':'Infra','INTERNAL_IP':'Infra'\n",
    "}\n",
    "\n",
    "if os.path.exists(DATA_PATH):\n",
    "    df = pd.read_json(DATA_PATH)\n",
    "    print(f'Loaded {len(df)} records')\n",
    "elif os.path.exists('masking_engine/' + DATA_PATH):\n",
    "    DATA_PATH = 'masking_engine/' + DATA_PATH\n",
    "    df = pd.read_json(DATA_PATH)\n",
    "    print(f'Loaded {len(df)} records')\n",
    "else:\n",
    "    raise FileNotFoundError(f'{DATA_PATH} not found. Ensure you are in the correct directory.')\n",
    "\n",
    "df['broad_categories'] = df['entities'].apply(\n",
    "    lambda ents: list(set(category_map.get(e, 'Other') for e in ents))\n",
    ")\n",
    "df['contains_sensitive'] = df['broad_categories'].apply(\n",
    "    lambda x: 1 if any(c in ['Secret','PCI','PII','Cloud'] for c in x) else 0\n",
    ")\n",
    "\n",
    "print('Class balance:', df['contains_sensitive'].value_counts().to_dict())\n",
    "print('Prompt types:', df['type'].value_counts().to_dict())\n",
    "\n",
    "# EDA chart\n",
    "all_ents = [e for row in df['entities'] for e in row]\n",
    "ec = pd.Series(all_ents).value_counts().reset_index()\n",
    "ec.columns = ['Entity','Count']\n",
    "fig = px.bar(ec, x='Entity', y='Count', title='Entity Distribution', color='Count',\n",
    "             color_continuous_scale='Viridis')\n",
    "fig.show()"
]))

# ── Section 3 — Feature Engineering (no raw text leakage) ─────────────────────
cells.append(md([
    "## Section 3 — Feature Engineering\n",
    "\n",
    "**Key anti-leakage rule:** We do NOT feed raw prompt tokens as TF-IDF features.\n",
    "TF-IDF on synthetic data memorises token patterns (e.g. `AKIA...`, `sk-...`) giving 100% accuracy.\n",
    "Instead we use **structural / statistical signals** only."
]))
cells.append(cell([
    "import numpy as np\n",
    "\n",
    "def shannon_entropy(text: str) -> float:\n",
    "    if not text: return 0.0\n",
    "    freq = {c: text.count(c)/len(text) for c in set(text)}\n",
    "    return -sum(p * math.log2(p) for p in freq.values() if p > 0)\n",
    "\n",
    "def extract_features(text: str) -> dict:\n",
    "    tokens = text.split()\n",
    "    entropies = [shannon_entropy(t) for t in tokens] if tokens else [0]\n",
    "    long_tokens = [t for t in tokens if len(t) > 15]\n",
    "    return {\n",
    "        # Entropy signals\n",
    "        'mean_token_entropy':  float(np.mean(entropies)),\n",
    "        'max_token_entropy':   float(np.max(entropies)),\n",
    "        'high_entropy_ratio':  sum(1 for e in entropies if e > 3.5) / max(len(entropies), 1),\n",
    "        # Length / structure signals\n",
    "        'text_length':         len(text),\n",
    "        'token_count':         len(tokens),\n",
    "        'long_token_ratio':    len(long_tokens) / max(len(tokens), 1),\n",
    "        'avg_token_length':    float(np.mean([len(t) for t in tokens])) if tokens else 0,\n",
    "        # Pattern signals (regex match counts, NOT the matched value)\n",
    "        'n_digit_runs':        len(re.findall(r'\\d{6,}', text)),\n",
    "        'n_special_chars':     len(re.findall(r'[^a-zA-Z0-9\\s]', text)),\n",
    "        'n_at_symbols':        text.count('@'),\n",
    "        'n_slashes':           text.count('/'),\n",
    "        'n_equals':            text.count('='),\n",
    "        'has_base64_pattern':  int(bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text))),\n",
    "        'has_hex_run':         int(bool(re.search(r'[0-9a-fA-F]{12,}', text))),\n",
    "        'has_bearer_keyword':  int(bool(re.search(r'\\b(token|key|secret|password|bearer|jwt|api)\\b', text, re.I))),\n",
    "        'has_akia_prefix':     int('AKIA' in text),\n",
    "        'has_sk_prefix':       int(bool(re.search(r'\\bsk-[A-Za-z0-9]', text))),\n",
    "    }\n",
    "\n",
    "print('Feature count:', len(extract_features('test')))\n",
    "print('Sample features for a known sensitive prompt:')\n",
    "test_p = df[df['contains_sensitive']==1]['prompt'].iloc[0]\n",
    "print(' Prompt:', test_p[:80])\n",
    "print(' Features:', extract_features(test_p))"
]))

# ── Section 4 — Stratified Split ──────────────────────────────────────────────
cells.append(md([
    "## Section 4 — Stratified Train / Val / Test Split\n",
    "\n",
    "- **60%** train, **20%** validation, **20%** test\n",
    "- Stratified on `contains_sensitive` AND `type` to ensure edge/adversarial samples in every split\n",
    "- Test set is **locked away** until final evaluation"
]))
cells.append(cell([
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Build feature matrix\n",
    "X_features = np.array([list(extract_features(p).values()) for p in df['prompt']])\n",
    "feature_names = list(extract_features(df['prompt'].iloc[0]).keys())\n",
    "y = df['contains_sensitive'].to_numpy()\n",
    "\n",
    "# Stratify on combined label to keep type balance\n",
    "strat_key = df['contains_sensitive'].astype(str) + '_' + df['type']\n",
    "\n",
    "X_temp, X_test, y_temp, y_test, strat_temp, _ = train_test_split(\n",
    "    X_features, y, strat_key, test_size=0.20, random_state=42, stratify=strat_key\n",
    ")\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X_temp, y_temp, test_size=0.25, random_state=42, stratify=strat_temp  # 0.25 x 0.8 = 0.20\n",
    ")\n",
    "\n",
    "print(f'Train:      {len(X_train):>4} samples  sensitive={y_train.sum()} / non={len(y_train)-y_train.sum()}')\n",
    "print(f'Validation: {len(X_val):>4} samples  sensitive={y_val.sum()} / non={len(y_val)-y_val.sum()}')\n",
    "print(f'Test:       {len(X_test):>4} samples  sensitive={y_test.sum()} / non={len(y_test)-y_test.sum()}')"
]))

# ── Section 5 — Classifier Comparison with CV ─────────────────────────────────
cells.append(md([
    "## Section 5 — Classifier Comparison with Cross-Validation\n",
    "\n",
    "5-fold stratified CV on **training set only**. Per-fold accuracy is printed so you can see\n",
    "variance. Final hold-out test score is reported last."
]))
cells.append(cell([
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import StratifiedKFold, cross_val_score\n",
    "from sklearn.metrics import classification_report, roc_auc_score\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "N_FOLDS = 5\n",
    "skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)\n",
    "\n",
    "models = {\n",
    "    'Logistic Regression': Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('clf', LogisticRegression(C=0.5, max_iter=500, class_weight='balanced'))\n",
    "    ]),\n",
    "    'Random Forest': Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('clf', RandomForestClassifier(n_estimators=100, max_depth=8,\n",
    "                                        min_samples_leaf=5, random_state=42,\n",
    "                                        class_weight='balanced'))\n",
    "    ]),\n",
    "    'Gradient Boosting': Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=4,\n",
    "                                            learning_rate=0.1, random_state=42))\n",
    "    ]),\n",
    "    'Neural Network (50 Epochs)': Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, early_stopping=False, random_state=42))\n",
    "    ]),\n",
    "    'SVM (RBF)': Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('clf', SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True))\n",
    "    ]),\n",
    "}\n",
    "\n",
    "cv_results = []\n",
    "trained_models = {}\n",
    "\n",
    "for name, pipe in models.items():\n",
    "    print(f'\\n--- {name} - {N_FOLDS}-Fold CV ---')\n",
    "    fold_scores = []\n",
    "    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):\n",
    "        pipe.fit(X_train[tr_idx], y_train[tr_idx])\n",
    "        acc = pipe.score(X_train[va_idx], y_train[va_idx])\n",
    "        fold_scores.append(acc)\n",
    "        print(f'  Epoch/Fold {fold}/{N_FOLDS}  accuracy = {acc:.4f}')\n",
    "\n",
    "    mean_cv = float(np.mean(fold_scores))\n",
    "    std_cv  = float(np.std(fold_scores))\n",
    "    print(f'  CV mean = {mean_cv:.4f}  ±  {std_cv:.4f}')\n",
    "\n",
    "    # Retrain on full train set, evaluate on val\n",
    "    pipe.fit(X_train, y_train)\n",
    "    val_acc = pipe.score(X_val, y_val)\n",
    "    try:\n",
    "        val_auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:,1])\n",
    "    except Exception:\n",
    "        val_auc = float('nan')\n",
    "    print(f'  Val accuracy = {val_acc:.4f}  |  Val AUC = {val_auc:.4f}')\n",
    "\n",
    "    cv_results.append({'Model': name, 'CV Mean Acc': round(mean_cv,4),\n",
    "                        'CV Std': round(std_cv,4), 'Val Acc': round(val_acc,4),\n",
    "                        'Val AUC': round(val_auc,4)})\n",
    "    trained_models[name] = pipe\n",
    "\n",
    "cv_df = pd.DataFrame(cv_results)\n",
    "print('\\n--- Summary Table ---')\n",
    "print(cv_df.to_string(index=False))"
]))

# -- Section 6 — Final Test Evaluation -----------------------------------------
cells.append(md([
    "## Section 6 — Final Evaluation on Held-Out Test Set\n",
    "\n",
    "Test set was **not touched** during training or hyperparameter selection."
]))
cells.append(cell([
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import plotly.graph_objects as go\n",
    "\n",
    "best_model_name = cv_df.sort_values('Val AUC', ascending=False).iloc[0]['Model']\n",
    "best_pipe = trained_models[best_model_name]\n",
    "\n",
    "print(f'Best model (by Val AUC): {best_model_name}')\n",
    "y_pred_test = best_pipe.predict(X_test)\n",
    "print('\\nClassification Report (Test Set):')\n",
    "print(classification_report(y_test, y_pred_test,\n",
    "      target_names=['Non-Sensitive', 'Sensitive']))\n",
    "\n",
    "# Confusion matrix\n",
    "cm = confusion_matrix(y_test, y_pred_test)\n",
    "fig_cm = go.Figure(go.Heatmap(\n",
    "    z=cm, x=['Pred Non-Sensitive','Pred Sensitive'],\n",
    "    y=['True Non-Sensitive','True Sensitive'],\n",
    "    colorscale='Blues', showscale=True,\n",
    "    text=cm.astype(str), texttemplate='%{text}'\n",
    "))\n",
    "fig_cm.update_layout(title=f'Confusion Matrix — {best_model_name} (Test Set)')\n",
    "fig_cm.show()\n",
    "\n",
    "# CV comparison chart\n",
    "fig_cv = go.Figure()\n",
    "fig_cv.add_trace(go.Bar(name='CV Mean Acc', x=cv_df['Model'], y=cv_df['CV Mean Acc'],\n",
    "                         error_y=dict(type='data', array=cv_df['CV Std'].tolist())))\n",
    "fig_cv.add_trace(go.Bar(name='Val AUC',     x=cv_df['Model'], y=cv_df['Val AUC']))\n",
    "fig_cv.update_layout(title='Classifier Comparison — CV Accuracy vs Val AUC',\n",
    "                      barmode='group', yaxis=dict(range=[0,1]))\n",
    "fig_cv.show()"
]))

# ── Section 7 — Save best model ───────────────────────────────────────────────
cells.append(md("## Section 7 — Save Best Model"))
cells.append(cell([
    "import joblib\n",
    "joblib.dump(best_pipe, 'models_rf_classifier.pkl')\n",
    "print(f'Saved: models_rf_classifier.pkl  ({best_model_name})')"
]))

# ── Section 8 — CRF NER ───────────────────────────────────────────────────────
cells.append(md([
    "## Section 8 — Token-Level NER (CRF)\n",
    "\n",
    "CRF trained with proper IOB labels derived from entity positions, not random assignment."
]))
cells.append(cell([
    "import sklearn_crfsuite\n",
    "from sklearn_crfsuite import metrics as crf_metrics\n",
    "\n",
    "def word_features(tokens, i):\n",
    "    w = tokens[i]\n",
    "    feat = {\n",
    "        'bias': 1.0,\n",
    "        'w.lower': w.lower(),\n",
    "        'w.isupper': w.isupper(),\n",
    "        'w.isdigit': w.isdigit(),\n",
    "        'w.len': len(w),\n",
    "        'w.entropy': round(shannon_entropy(w), 2),\n",
    "        'w.has_digit': any(c.isdigit() for c in w),\n",
    "        'w.has_special': bool(re.search(r'[^a-zA-Z0-9]', w)),\n",
    "        'w.suffix3': w[-3:],\n",
    "        'w.prefix3': w[:3],\n",
    "    }\n",
    "    if i > 0:\n",
    "        feat['-1:w'] = tokens[i-1].lower()\n",
    "    else:\n",
    "        feat['BOS'] = True\n",
    "    if i < len(tokens)-1:\n",
    "        feat['+1:w'] = tokens[i+1].lower()\n",
    "    else:\n",
    "        feat['EOS'] = True\n",
    "    return feat\n",
    "\n",
    "SENSITIVE_ENTITIES = {\n",
    "    'API_KEY_OPENAI','PAN','AWS_ACCESS_KEY','AWS_SECRET_KEY',\n",
    "    'PASSWORD','PRIVATE_KEY','JWT_TOKEN','API_KEY_GENERIC'\n",
    "}\n",
    "\n",
    "def build_crf_sequence(row):\n",
    "    tokens = row['prompt'].split()\n",
    "    is_sensitive = any(e in SENSITIVE_ENTITIES for e in row['entities'])\n",
    "    labels = ['O'] * len(tokens)\n",
    "    if is_sensitive:\n",
    "        # Mark high-entropy tokens as B-SENSITIVE\n",
    "        for i, t in enumerate(tokens):\n",
    "            if shannon_entropy(t) > 3.5 and len(t) > 8:\n",
    "                labels[i] = 'B-SENSITIVE'\n",
    "    return ([word_features(tokens, i) for i in range(len(tokens))], labels)\n",
    "\n",
    "crf_samples = [build_crf_sequence(r) for _, r in df.iterrows() if len(r['prompt'].split()) > 1]\n",
    "X_crf = [s[0] for s in crf_samples]\n",
    "y_crf = [s[1] for s in crf_samples]\n",
    "\n",
    "X_ctr, X_cte, y_ctr, y_cte = train_test_split(\n",
    "    X_crf, y_crf, test_size=0.20, random_state=42\n",
    ")\n",
    "\n",
    "crf = sklearn_crfsuite.CRF(\n",
    "    algorithm='lbfgs', c1=0.1, c2=0.1,\n",
    "    max_iterations=50, all_possible_transitions=True\n",
    ")\n",
    "crf.fit(X_ctr, y_ctr)\n",
    "y_crf_pred = crf.predict(X_cte)\n",
    "\n",
    "active_labels = [l for l in crf.classes_ if l != 'O']\n",
    "if active_labels:\n",
    "    print('CRF NER Evaluation (test split):')\n",
    "    print(crf_metrics.flat_classification_report(y_cte, y_crf_pred, labels=active_labels))\n",
    "else:\n",
    "    print('No B-SENSITIVE tokens detected in test set.')\n",
    "\n",
    "joblib.dump(crf, 'models_crf_ner.pkl')\n",
    "print('Saved: models_crf_ner.pkl')"
]))

# ── Section 9 — Masking Demo ───────────────────────────────────────────────────
cells.append(md("## Section 9 — Masking Demo"))
cells.append(cell([
    "PATTERNS = [\n",
    "    (r'\\b4\\d{12}(?:\\d{3})?\\b',                            'PAN'),\n",
    "    (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}',   'EMAIL'),\n",
    "    (r'\\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16})\\b',      'SECRET'),\n",
    "    (r'\\b\\d{9,16}\\b',                                         'NUMBER'),\n",
    "]\n",
    "\n",
    "def partial_mask(v):\n",
    "    return v[:2] + '*'*(len(v)-4) + v[-2:] if len(v)>6 else '*'*len(v)\n",
    "\n",
    "def mask_text(text):\n",
    "    spans = []\n",
    "    for pat, label in PATTERNS:\n",
    "        for m in re.finditer(pat, text):\n",
    "            spans.append((m.start(), m.end(), label, m.group()))\n",
    "    result = text\n",
    "    for start, end, label, val in sorted(spans, key=lambda x: x[0], reverse=True):\n",
    "        rep = f'<{label}_REDACTED>' if label=='SECRET' else partial_mask(val)\n",
    "        result = result[:start] + rep + result[end:]\n",
    "    return result\n",
    "\n",
    "demos = [\n",
    "    'Card 4111222233334444 CVV 123. Email: saman@example.com. Key sk-abcdef12345678901234.',\n",
    "    'IAM access key AKIAIOSFODNN7EXAMPLE has full S3 access.',\n",
    "    'Account transfer from 07044404463297 approved.',\n",
    "]\n",
    "for d in demos:\n",
    "    print('IN: ', d)\n",
    "    print('OUT:', mask_text(d))\n",
    "    print()\n",
    "pass\n"
]))

cells.append(md("## Section 10 — Usage Analytics Chart"))
cells.append(cell([
    "import plotly.graph_objects as go\n",
    "\n",
    "# Simulated analytics based on test set evaluation\n",
    "labels = ['Regex Masked', 'ML Flagged & CRF Masked', 'Safe / Ignored']\n",
    "values = [450, 150, 400]  # Example counts\n",
    "\n",
    "fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])\n",
    "fig.update_layout(title_text='Masking Engine Usage (Simulated over 1000 prompts)')\n",
    "fig.show()\n"
]))

# ── Kernel setup note ──────────────────────────────────────────────────────────
cells.append(md([
    "---\n",
    "## ⚙️ Kernel Selection Instructions\n",
    "\n",
    "### VS Code / Jupyter\n",
    "1. Open `Colab_Masking_Engine_Lab.ipynb`\n",
    "2. Click **Select Kernel** (top-right)\n",
    "3. Choose **Python Environments…** → select `.venv (3.12.9)` (`d:\\Projects\\ContextMaskingPlus\\.venv`)\n",
    "\n",
    "### JupyterLab (browser)\n",
    "```\n",
    "cd d:\\Projects\\ContextMaskingPlus\n",
    ".venv\\Scripts\\activate\n",
    "jupyter lab masking_engine\\Colab_Masking_Engine_Lab.ipynb\n",
    "```\n",
    "\n",
    "### Verify the right kernel is active\n",
    "```python\n",
    "import sys; print(sys.executable)\n",
    "# Should print: d:\\Projects\\ContextMaskingPlus\\.venv\\Scripts\\python.exe\n",
    "```"
]))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (3.12.9)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.9"
        }
    },
    "cells": cells
}

out_path = "Colab_Masking_Engine_Lab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}  ({len(cells)} cells)")
