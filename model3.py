from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    # CARICAMENTO MODELLO
    model_name = "infly/OpenCoder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # più leggero
        trust_remote_code=True
    ).to("cpu")

    # === ESEMPIO INPUT ===
    original_description = "When your car drives away from a location you specify, your garage door will automatically close."
    intent = "When Dad's Honda departs from Home, close the Front Garage Door."
    variables = "Zubie.departures.EventDetails, Zubie.departures.EventTime, Zubie.departures.Place, Zubie.departures.Vehicle"
    action = "Garageio.closeGarageDoor.skip(string?: reason)"

    # PROMPT SEMPLICE 
    content = f"""
    I have an IFTTT applet whose description is: '{original_description}'.
    I want to add JavaScript filter code to implement the following functionality: '{intent}'.
    The most likely variables from the trigger service that you have to use are the following {variables} (some might be useless and others might be missing),
    while the action method is {action}.
    Generate the correct JavaScript filter code.

    Start now:
    ```javascript
    """

    # TOKENIZZAZIONE 
    inputs = tokenizer(content, return_tensors="pt").to(model.device)

    # GENERAZIONE 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,  # da 200 → 60
            do_sample=False
        )


    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # STAMPA RISULTATO 
    print("=== Filter Code ===")
    print(result)
