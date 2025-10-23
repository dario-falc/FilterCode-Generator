from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# === MODELLO ===
model_name = "bigcode/starcoderbase-1b"

print(f"Caricamento modello: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Modello caricato ✅")

# === LETTURA FILE ===
dataset_path = "./data/data_cleaned.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # carica tutto come un dizionario

# ottieni le chiavi (applet_0, applet_1, ecc.)
applet_keys = list(data.keys())
print(f"Trovati {len(applet_keys)} applet nel dataset.\n")

# scegli manualmente l'indice (es. applet 0)
index = 0
applet_key = applet_keys[index]
applet = data[applet_key]

print(f"Generazione per l’applet n° {index} ({applet_key})...\n")


# === DATI PRINCIPALI ===
description = applet.get("original_description", "")
intent = applet.get("intent", "")

trigger_vars = []
for item in applet.get("trigger_details", []):
    details = item.get("details", {})
    if "Filter code" in details:
        trigger_vars.append(details["Filter code"])

action_methods = []
action_info = applet.get("action_developer_info", {})
main_method = action_info.get("Filter code method")
if main_method:
    action_methods.append(main_method)

for item in applet.get("action_details", []):
    details = item.get("details", {})
    if "Filter code method" in details:
        action_methods.append(details["Filter code method"])

# includiamo anche eventuali "Runtime method"
runtime_method = action_info.get("Runtime method")
if runtime_method:
    action_methods.append(runtime_method)

action_methods = list(set(action_methods))


# === PROMPT MIGLIORATO ===
content = f"""
Generate only valid JavaScript filter code for the following IFTTT applet.

Description: {description}
Goal: {intent}

You can use these trigger variables: {', '.join(trigger_vars) if trigger_vars else 'None'}
Available action methods: {', '.join(action_methods) if action_methods else 'None'}

Rules:
- Output only JavaScript code (no comments, examples, explanations, or markdown).
- The code must follow this logic:
    → If trigger conditions are met, the action runs normally.
    → Otherwise, use the .skip("reason") method to stop the action.
- Do not call skip() when the condition is true.
- Use parseFloat() when comparing numeric values.
- Keep the syntax simple and human-like.
Start directly with the JavaScript code.
"""




inputs = tokenizer(content, return_tensors="pt").to(model.device)

print("Generazione del filter code in corso... (attendere 1-2 minuti su CPU)\n")

outputs = model.generate(**inputs, max_new_tokens=200)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("=== Filter Code ===")
print(result)
