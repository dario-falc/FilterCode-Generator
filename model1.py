from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# deepseek-ai/deepseek-coder-1.3b-instruct

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)#, torch_dtype=torch.bfloat16).cuda()
    
    
    with open("data\\Json200_cleaned.json", "r") as f:
        applets = json.load(f)
    
    # print(f"applets: {applet}, type: {type(applet)}")
    
    for applet in applets.values():

        original_description = applet["original_description"]
        intent = applet["intent"]
        triggers = ", ".join(applet["triggers"])
        action = applet["action"]
    
        print(f"Original description: {original_description}")
        print(f"Intent: {intent}")
    
        prompt = f"""I have an IFTTT applet who's description is: '{original_description}'.
        The applet is already built and working: when the trigger event happens, the action event is executed.
        I want to add JavaScript filter code to implent the following functionality: '{intent}'.
        Trigger variables contain useful information provided by the trigger service for your custom applets.
        The most likely variables from the trigger service that you are going to need to use are the following: {triggers}; keep in mind that some of these might be useless and others might be missing.
        The applet that does what's described in the original description already runs by default.
        The skip action method allows you to specify when the applet should NOT be ran, so use it properly so that the generated code matches the original description.
        The skip action method for this applet is {action} and it's the only one you can use.
        Generate me the correct working JavaScript filter code without using placeholders and by only using the information that i gave you, so that I can copy and paste into IFTTT without making any changes."""
        
        messages=[
            { 'role': 'user', 'content': prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
