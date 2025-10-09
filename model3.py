from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "infly/OpenCoder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)


# === INPUT ===
description = "When your Android device's battery drops below 15%, you will get a text notification as a reminder to plug it in."
intent = "When the Android device's battery drops below 15%, send a text notification to remind the user to plug it in."
variables = [
    "AndroidBattery.batteryLow.BatteryPercentage",
    "AndroidBattery.batteryLow.DeviceName",
    "AndroidBattery.batteryLow.OccurredAt"
]
action = "AndroidMessages.sendAMessage.skip(string?: reason)"

# === PROMPT ===
prompt = f"""IFTTT Filter Code
Description: {description}
Intent: {intent}
Variables: {', '.join(variables)}
Action: {action}
Write ONLY valid JavaScript that fulfills the intent using the given variables and action.
Do not include explanations, examples, placeholders, or test cases.
Start directly with code.
End your answer with ### END.
"""

# === GENERAZIONE ===
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,  # più bassa = meno verboso
        top_p=0.9,
        do_sample=True
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# === POST-PROCESSING ===
# 1. Togli righe
cleaned = "\n".join([line for line in result.splitlines() if not line.strip().startswith("//")])
# 2. Taglia tutto dopo "### END"
cleaned = cleaned.split("### END")[0].strip()

print("=== Filter Code ===")
print(result.strip())
