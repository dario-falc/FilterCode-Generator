from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# deepseek-ai/deepseek-coder-1.3b-instruct

if __name__ == "__main__":
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)#, torch_dtype=torch.bfloat16).cuda()
    
    
    with open("data\\data.json", "r") as f:
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
        
        
        prompt = f"""
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
        
        messages=[
            { 'role': 'user', 'content': prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        res = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        print(res)
        
        applet[model_name] = res
        # applet.update({model_name: res})

        with open("data\\data.json", "w") as f:
            json.dump(applets, f, indent=3, separators=(',', ': '))
        
        i+=1