import json

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


def create_triggers_json(data):
    trigger_filter_codes = []
    
    for i in range(len(data)):
        trigger_details = data[i]["trigger_details"]
    
        for elem in trigger_details:
            if elem["section"] == "Ingredients":
                trigger_filter_codes.append(elem["details"]["Filter code"])

    # Remove duplicates
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
    
    filename = ".//data//triggers.json"
    with open(filename, "w") as f:
        json.dump(triggers_json, f, indent=3, separators=(',', ': '))


def create_actions_json(data):
    actions_json = {}
    for i in range(len(data)):
        elem = data[i]["action_developer_info"]["Filter code method"]
        
        actions_json[elem.split(".")[0]] = elem
    
    actions_json = dict(sorted(actions_json.items()))

    filename = ".//data//actions.json"
    with open(filename, "w") as f:
        json.dump(actions_json, f, indent=3)#, separators=(',', ': '))




if __name__ == "__main__":
    filename = ".//data//generated_filtercode_from_intent_1-547_300_qwen3 1.jsonl"    

    data = load_jsonl(filename)

    create_triggers_json(data)
    create_actions_json(data)
    