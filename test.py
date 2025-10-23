from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = "bigcode/starcoderbase-1b"
device = "cpu"  # o "cuda" se hai una GPU

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
