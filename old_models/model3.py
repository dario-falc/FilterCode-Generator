from transformers import AutoModelForCausalLM, GemmaTokenizer
import json

# === MODELLO ===
model_name = "google/gemma-2-2b-it"

if __name__ == "__main__":
    tokenizer = GemmaTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    with open("./data/data.json", "r", encoding="utf-8") as f:
        applets = json.load(f)

    i = 1
    for key, applet in applets.items():
        print(f"Analyzing applet n°{i}: {key}")

        if model_name in applet.keys():
            i += 1
            continue

        original_description = applet.get("original_description", "")
        intent = applet.get("intent", "")
        triggers = applet.get("triggers", [])
        actions = applet.get("actions", [])
        skip = applet.get("skip", "")

        print(f"Original description: {original_description}")
        print(f"Intent: {intent}")

        # === PROMPT ===
        prompt = f"""
        Generate only valid JavaScript filter code for the following IFTTT applet.

        Description: {original_description}
        Goal: {intent}

        Trigger variables: {', '.join(triggers) if triggers else 'None'}
        Action methods: {', '.join(actions) if actions else 'None'}, {skip or 'None'}

        Rules:
        - Output only JavaScript code (no comments or markdown).
        - The code must follow this logic:
            → If trigger conditions are met, the action runs normally.
            → Otherwise, use the .skip("reason") method.
        - Only call skip() inside the "else" clause.
        - Use parseFloat() for numeric comparisons.
        - Start directly with 'var'.
        """

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.4,
            top_p=0.9,
            do_sample=True
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Pulizia semplice
        if "var " in result:
            result = "var " + result.split("var ", 1)[1]
        result = result.split("```")[0].strip()

        print("=== Filter Code ===")
        print(result)
        applet[model_name] = result

        # Salvataggio progressivo
        with open("./data/data.json", "w", encoding="utf-8") as f:
            json.dump(applets, f, indent=3, ensure_ascii=False)

        i += 1

    print("\nTutte le applet elaborate!")
