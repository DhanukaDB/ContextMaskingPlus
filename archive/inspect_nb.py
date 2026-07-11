import json
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

nb_path = 'd:/Projects/ContextMaskingPlus/masking_engine/Colab_Masking_Engine_Lab.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== NOTEBOOK EXECUTION RESULTS ===\n")
for i, cell in enumerate(data['cells']):
    if cell['cell_type'] != 'code':
        continue
    outputs = cell.get('outputs', [])
    if not outputs:
        continue
    print(f"--- Cell {i} ---")
    for out in outputs:
        otype = out.get('output_type', '')
        if otype == 'stream':
            text = ''.join(out.get('text', []))
            print(text[:500])
        elif otype == 'error':
            print(f"ERROR: {out.get('ename')}: {out.get('evalue')}")
    print()
