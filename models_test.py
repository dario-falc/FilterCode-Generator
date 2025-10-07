from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)#, torch_dtype=torch.bfloat16).cuda()
    
    original_description = "When your Android device's battery drops below 15%, you will get a text notification as a reminder to plug it in."
    intent = "When the Android device's battery drops below 15%, send a text notification to remind the user to plug it in."
    
    variables = "AndroidBattery.batteryLow.BatteryPercentage, AndroidBattery.batteryLow.DeviceName, AndroidBattery.batteryLow.OccurredAt"
    action = "AndroidMessages.sendAMessage.skip(string?: reason)"
    
    content = f"""I have an IFTTT applet who's description is: '{original_description}'.
    I want to add JavaScript filter code to implent the following functionality: '{intent}'.
    The most likely variables from the trigger service that you have to use are the following {variables} (some might be useless and others might be missing) while the action method is {action}.
    Generate me the correct JavaScript filter code"""
    
    messages=[
        { 'role': 'user', 'content': content}
    ]
    
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # tokenizer.eos_token_id is the id of <|EOT|> token
    outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
