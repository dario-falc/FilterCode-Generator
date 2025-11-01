from openai import OpenAI
import json

if __name__ == "__main__":
    # model_name = "qwen2.5-coder-7b-instruct"
    # model_name = "deepseek-coder-6.7b-instruct"
    # model_name = "gemma-2-9b-it"
    model_name = "yi-coder-9b-chat"
    
    client = OpenAI(base_url="http://127.0.0.1:1234/v1/", api_key="lm-studio")

    with open("data\\new_data.json", "r", encoding="utf-8") as f:
        applets = json.load(f)
    
    i=1
    for applet in applets.values():
        print(f"Analyzing applet n°{i}")
        if model_name in applet.keys():
            i+=1
            continue

        original_description = applet["original_description"]
        intent = applet["intent"]
        triggers = applet["triggers"]
        actions = applet["actions"]
        skip = applet["skip"]
        
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

        r = client.chat.completions.create(
            model = model_name,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens = 512,
            seed = 978652
        )
        
        res = r.choices[0].message.content.strip()
        
        applet[model_name] = res
        
        with open("data\\new_data.json", "w") as f:
            json.dump(applets, f, indent=3, separators=(',', ': '))
        
        i+=1