import json
import os

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as json_file:
        json_list = list(json_file)

    i=0
    data = {}
    for json_str in json_list:
        result = json.loads(json_str)
        data[i] = result
        
        i+=1
    
    return data


def create_triggers_json():
    if not os.path.exists(".//data//triggers.json"):
        print("Extracting trigger variables (dict)...")
        in_filename = ".//data//generated_filtercode_from_intent_1-547_300_qwen3 1.jsonl"    
        data = load_jsonl(in_filename)
        
        trigger_filter_codes = []
        
        for i in range(len(data)):
            trigger_details = data[i]["trigger_details"]
        
            for elem in trigger_details:
                if elem["section"] == "Ingredients":
                    trigger_filter_codes.append(elem["details"]["Filter code"])

        trigger_filter_codes = sorted(list(set(trigger_filter_codes)))
        
        triggers_json = {}
        
        for elem in trigger_filter_codes:
            app, functionality, variable = elem.split(".")
            
            triggers_json[app] = {}
        
        
        for elem in trigger_filter_codes:
            app, functionality, variable = elem.split(".")

            triggers_json[app][functionality] = []
        
        
        for elem in trigger_filter_codes:
            app, functionality, variable = elem.split(".")

            triggers_json[app][functionality].append(variable)
        
        out_filename = ".//data//triggers.json"
        with open(out_filename, "w") as f:
            json.dump(triggers_json, f, indent=3, separators=(',', ': '))
        
        print("Trigger variables (dict) extracted.")

    else:
        print("Trigger variables (dict) already extracted.")


def create_actions_json():
    if not os.path.exists(".//data//actions.json"):
        print("Extracting action methods...")
        in_filename = ".//data//generated_filtercode_from_intent_1-547_300_qwen3 1.jsonl"    
        data = load_jsonl(in_filename)
    
    
        actions_json = {}
        for i in range(len(data)):
            elem = data[i]["action_developer_info"]["Filter code method"]
            
            actions_json[elem.split(".")[0]] = elem
        
        actions_json = dict(sorted(actions_json.items()))

        out_filename = ".//data//actions.json"
        with open(out_filename, "w") as f:
            json.dump(actions_json, f, indent=3)
        
        print("Action methods extracted.")

    else:
        print("Action methods already extracted.")


def create_triggers_list():
    if not os.path.exists(".//data//triggers_list.json"):
        print("Extracting trigger variables (list)...")
        in_filename = ".//data//generated_filtercode_from_intent_1-547_300_qwen3 1.jsonl"    
        data = load_jsonl(in_filename)
        
        trigger_filter_codes = []
        
        for i in range(len(data)):
            trigger_details = data[i]["trigger_details"]
        
            for elem in trigger_details:
                if elem["section"] == "Ingredients":
                    trigger_filter_codes.append(elem["details"]["Filter code"])

        trigger_filter_codes = sorted(list(set(trigger_filter_codes)))

        out_filename = ".//data//triggers_list.json"
        with open(out_filename, "w") as f:
            json.dump(trigger_filter_codes, f, indent=3, separators=(',', ': '))
            
        print("Trigger variables (list) extracted.")

    else:
        print("Trigger variables (list) already extracted.")


def create_actions_list():
    if not os.path.exists(".//data//actions_list.json"):
        print("Extracting action methods...")
        in_filename = ".//data//generated_filtercode_from_intent_1-547_300_qwen3 1.jsonl"    
        data = load_jsonl(in_filename)
    
    
        actions_list = []
        for i in range(len(data)):
            elem = data[i]["action_developer_info"]["Filter code method"]
            
            actions_list.append(elem)
        
        actions_list = sorted(list(set(actions_list)))

        out_filename = ".//data//actions_list.json"
        with open(out_filename, "w") as f:
            json.dump(actions_list, f, indent=3)
        
        print("Action methods extracted.")

    else:
        print("Action methods already extracted.")


if __name__ == "__main__":
    # create_triggers_json()
    # create_actions_json()
    # create_triggers_list()
    # create_actions_list()
    pass