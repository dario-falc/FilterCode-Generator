import json
import re

# === Percorsi dei file ===
input_path = "./data/generated_filtercodes_final.jsonl"
output_path = "./data/generated_filtercodes_cleaned.json"

# === Funzione per pulire il testo ===
def extract_javascript(text):
    """
    Estrae solo il codice JavaScript da un testo generato dal modello.
    Rimuove descrizioni, esempi, markdown e commenti superflui.
    """
    # Rimuove blocchi markdown tipo ```javascript ... ```
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Elimina intestazioni tipo 'Example:', 'Description:', ecc.
    text = re.sub(r"(?i)(example|description|goal|rules|available|write|output).*", "", text)
    # Mantiene solo righe che sembrano codice JS
    code_lines = [
        line for line in text.splitlines()
        if re.match(r"^\s*(var|let|const|if|else|for|while|function|{|}|;|\)|\()", line.strip())
    ]
    cleaned = "\n".join(code_lines).strip()
    return cleaned

# === Pulizia ===
cleaned_data = []

with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            obj = json.loads(line)
            applet_name = obj.get("applet", "")
            raw_code = obj.get("filter_code", "")
            js_code = extract_javascript(raw_code)
            if js_code:
                cleaned_data.append({
                    "applet": applet_name,
                    "filter_code": js_code
                })
        except json.JSONDecodeError:
            continue

# === Salvataggio nel formato finale ===
with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(cleaned_data, outfile, ensure_ascii=False, indent=2)

print(f"✅ File pulito salvato in: {output_path}")
print(f"Applet pulite: {len(cleaned_data)}")
