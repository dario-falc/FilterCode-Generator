import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "bigcode/starcoderbase-1b"
prompt_template = """
Ignore any previous context.

Write ONLY the JavaScript filter code for this IFTTT applet.

Description: {description}
Goal: {intent}

Trigger variables:
{triggers}

Action methods:
{actions}, {skip}

Rules:
- Begin with 'var' or 'let'.
- Use parseFloat() for numeric comparisons.
- If goal condition is TRUE → run the action.
- If FALSE → call .skip("reason").
- Output only valid JavaScript code.

"""

with open("./data/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")

output_path = "./data/generated_filtercodes_final.jsonl"
batch_size = 3

with open(output_path, "w", encoding="utf-8") as out_file:
    keys = list(data.keys())
    for start in range(0, len(keys), batch_size):
        for key in keys[start:start+batch_size]:
            applet = data[key]
            desc = applet.get("original_description", "")
            intent = applet.get("intent", "")
            triggers = ", ".join(applet.get("triggers", []))
            actions = ", ".join(applet.get("actions", []))
            skip = applet.get("skip", "")

            prompt = prompt_template.format(
                description=desc,
                intent=intent,
                triggers=triggers,
                actions=actions,
                skip=skip
            )

            inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.25,
                    top_p=0.9,
                    do_sample=True
                )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # pulizia
            lines = [l for l in result.splitlines()
                     if l.strip() and not l.lower().startswith(("description", "goal", "rules", "ignore"))]
            cleaned = "\n".join(lines).strip()
            out_file.write(json.dumps({
                "applet": key,
                "filter_code": cleaned
            }, ensure_ascii=False) + "\n")

            # micro-reset
            torch.manual_seed(torch.initial_seed())
            # (possibilità di reinizializzare modello se necessario)

print("✅ Generazione completata.")
