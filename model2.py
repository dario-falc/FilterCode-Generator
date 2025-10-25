import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen/Qwen2.5-Coder-1.5B-Instruct

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    
    with open("data\\data_test.json", "r") as f:
        applets = json.load(f)

    i=1
    for applet in applets.values():
        print(f"Analyzing applet n°{i}")
        if model_name in applet.keys():
            continue
        
        original_description = applet["original_description"]
        intent = applet["intent"]
        triggers = applet["triggers"]
        actions = applet["actions"]
        skip = applet["skip"]
    
        print(f"Original description: {original_description}")
        print(f"Intent: {intent}")
        # print(f"Triggers: {triggers}")
        # print(f"Actions: {actions}")
        # print(f"Skip: {skip}")
        
    
        content = f"""
        Generate only valid JavaScript filter code for the following IFTTT applet.

        Description: {original_description}
        Goal: {intent}

        You can use these trigger variables: {', '.join(triggers) if triggers else 'None'}
        Available action methods: {', '.join(actions) if actions else 'None'}
        The skip method for the action service is {skip}

        Rules:
        - Output only JavaScript code (no comments, examples, explanations, or markdown).
        - The code must follow this logic:
            → If trigger conditions are met, the action runs normally.
            → Otherwise, use the .skip("reason") method to stop the action.
        - Only call skip() in the "else" clause of the if statement.
        - Use parseFloat() when comparing numeric values.
        - Keep the syntax simple and human-like.
        - Assign the trigger variables to JavaScript variables to simplify the syntax.
        Start directly with the JavaScript code."""
    
    
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(res)
        
        applet[model_name] = res
        # applet.update({model_name: res})

        with open("data\\data.json", "w") as f:
            json.dump(applets, f, indent=3, separators=(',', ': '))
        
        i+=1