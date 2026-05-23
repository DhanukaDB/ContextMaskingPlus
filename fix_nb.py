import json

nb_path = 'd:/Projects/ContextMaskingPlus/masking_engine/Colab_Masking_Engine_Lab.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fix cell 14: replace fig_eval.show() with pass
for cell in data['cells']:
    src = ''.join(cell.get('source', []))
    if 'fig_eval' in src and 'fig_eval.show()' in src:
        cell['source'] = [
            l.replace('fig_eval.show()', 'pass # fig_eval.show()')
            for l in cell['source']
        ]
        print("Fixed fig_eval.show() in cell 14")

# Fix cell 16: replace ipywidgets display() calls with a plain print demo
for cell in data['cells']:
    src = ''.join(cell.get('source', []))
    if 'widgets.Textarea' in src or 'text_area.value' in src:
        # Replace the widget-based interactive section with a direct demo call
        new_src_lines = []
        skip_widget_block = False
        for line in cell['source']:
            if 'text_area = widgets.Textarea' in line:
                skip_widget_block = True
                # Insert a direct, non-widget demo call instead
                new_src_lines.append(
                    '\n# Non-interactive demo (widgets not available in nbconvert)\n'
                    'demo_text = "Refund requested for card 4111222233334444 CVV 123. Contact: saman.p@example.com. API key sk-abcdef12345678901234."\n'
                    'print("Input:", demo_text)\n'
                    'print("Masked:", apply_masking_strategy(demo_text))\n'
                )
            if skip_widget_block:
                continue
            new_src_lines.append(line)
        cell['source'] = new_src_lines
        print("Fixed widgets cell 16 to plain demo")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Done patching cells 14 and 16.")
