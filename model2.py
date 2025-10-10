from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen/Qwen2.5-Coder-1.5B-Instruct

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_description = "This applet will initiate sync for a data.world dataset or data project at a specific time every day."
    intent = "This IFTTT applet initiates a sync for a data.world dataset or data project at a specific time every day, based on user input."
    variables = "DateAndTime.everyDayAt.CheckTime"
    action = "Datadotworld.sync.skip(string?: reason)"
    
    content = f"""I have an IFTTT applet who's description is: '{original_description}'.
    I want to add JavaScript filter code to implent the following functionality: '{intent}'.
    The most likely variables from the trigger service that you have to use are the following {variables} (some might be useless and others might be missing) while the action method is {action}.
    Generate me the correct JavaScript filter code"""
    
    
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

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)