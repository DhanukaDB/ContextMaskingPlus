import json

notebook_path = r"d:\Projects\ContextMaskingPlus\masking_engine\Colab_Masking_Engine_Lab.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        
        # Add model saving for Task A
        if any("fig3.show()" in line for line in source):
            # Check if joblib import is there
            has_joblib = any("import joblib" in line for line in source)
            if not has_joblib:
                # Add at the top
                source.insert(0, "import joblib\n")
            
            # Find the end of the models loop or if results section
            for i, line in enumerate(source):
                if "if results:" in line:
                    # insert model saving just before this
                    source.insert(i, "    # Save the best model (e.g. Random Forest)\n")
                    source.insert(i+1, "    if 'Random Forest' in models:\n")
                    source.insert(i+2, "        joblib.dump(models['Random Forest'], 'models_rf_classifier.pkl')\n")
                    source.insert(i+3, "        joblib.dump(vectorizer, 'models_tfidf_vectorizer.pkl')\n")
                    source.insert(i+4, "        print('Models saved to models_rf_classifier.pkl and models_tfidf_vectorizer.pkl')\n")
                    break

        # Add model saving for Task B (CRF)
        if any("print(\"CRF Evaluation:\")" in line for line in source):
            has_joblib = any("import joblib" in line for line in source)
            if not has_joblib:
                source.insert(0, "import joblib\n")
            for i, line in enumerate(source):
                if "print(\"CRF Evaluation:\")" in line:
                    source.insert(i, "        joblib.dump(crf, 'models_crf_ner.pkl')\n")
                    source.insert(i+1, "        print('CRF Model saved to models_crf_ner.pkl')\n")
                    break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated with model saving.")
