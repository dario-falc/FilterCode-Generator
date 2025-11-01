import json
import os

# === PERCORSI FILE ===
base_dir = os.path.join(os.getcwd(), "data")  # percorso alla cartella data
input_path = os.path.join(base_dir, "data.json")
output_path = os.path.join(base_dir, "new_data.json")

# === LETTURA FILE ORIGINALE ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === ESTRARRE SOLO LE APPLET DA 150 A 299 ===
keys = list(data.keys())
subset = {k: data[k] for i, k in enumerate(keys) if 150 <= i < 300}

# === SALVATAGGIO NUOVO FILE ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=4, ensure_ascii=False)

print(f"✅ File '{output_path}' creato con {len(subset)} applet (da 150 a 299).")
