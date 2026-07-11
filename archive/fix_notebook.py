import json

notebook_path = r"d:\Projects\ContextMaskingPlus\masking_engine\Colab_Masking_Engine_Lab.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "DATA_PATH = 'data/synthetic_dataset.csv'" in line:
                source[i] = "DATA_PATH = 'data/synthetic_dataset.json'\n"
            if "df = pd.read_csv(DATA_PATH)" in line:
                source[i] = "    df = pd.read_json(DATA_PATH)\n"
            if "df['entities'] = df['entities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])" in line:
                source[i] = "    # JSON handles lists natively\n"
        
        # also fix trufflehog installation
        for i, line in enumerate(source):
            if "!curl -sSfL" in line:
                source[i] = "# !curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b /usr/local/bin\n"
            if "!trufflehog --version" in line:
                source[i] = "# !trufflehog --version\n"

        # mock trufflehog run to bypass the error on windows with attrs conflict
        if "def run_trufflehog(text: str):" in source[0] if source else False:
            # wait, run_trufflehog is inside a cell.
            for i, line in enumerate(source):
                if '["trufflehog", "filesystem", "temp_scan.txt", "--json"]' in line:
                    source[i] = '            ["python", "-m", "trufflehog3", "-f", "JSON", "temp_scan.txt"],\n'

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook updated.")
