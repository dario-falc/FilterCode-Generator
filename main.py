import json
from statistics import median
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_applet_triggers(applet):
    trigger_filter_codes = []
    
    trigger_details = applet["trigger_details"]
    for elem in trigger_details:
        if elem["section"] == "Ingredients":
            trigger_filter_codes.append(elem["details"]["Filter code"])
    
    trigger_filter_codes = sorted(list(set(trigger_filter_codes)))
    return trigger_filter_codes


def get_applet_action(applet):
    return applet["action_developer_info"]["Filter code method"]


def match(model, intent, triggers):
    intent_embedding = model.encode([intent])
    trigger_embeddings = model.encode(triggers)

    similarities = cosine_similarity(intent_embedding, trigger_embeddings)[0]
    
    scores = {}
    for i in range(len(triggers)):
        if similarities[i] > 0:
            scores[triggers[i]] = similarities[i]
    
    if len(scores) <= 3:
        return scores
    else:    
        # Keep entry only if greater than median value
        median_value = median(scores.values())
        # print(f"Median value: {median_value}")

        filtered_scores = {k: v for k, v in scores.items() if v > median_value}
        filtered_scores = dict(sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True))
        
        return filtered_scores


if __name__ == "__main__":
    # with open("data/generated_filtercode_from_intent.jsonl", "r", encoding="utf-8") as f:
    #     data = f.readlines()

    with open("data/Json200.jsonl", "r", encoding="utf-8") as f:
        data = f.readlines()


    # For testing
    start = 0
    n_to_insert = 20
    
    applets = []
    n_inserted = 0
    i=0

    while n_inserted!=n_to_insert:
        entry = json.loads(data[start+i])
    
        # Keep only "tipologia = Corretto"
        if entry["tipologia"] == "Corretto":
            applets.append(entry)
            n_inserted+=1
        i+=1
    
    # for i in range(start, start+n_to_insert):
    #     applets.append(json.loads(data[i]))

    to_json = {}

    i=1
    for applet in applets:
        print(f"Original description: {applet["original_description"]}")
        print(f"Intent: {applet["intent"]}\n")
        
        # Triggers
        triggers = get_applet_triggers(applet)
        
        # Action
        action = get_applet_action(applet)
        
        # Intent-variables matching
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        res = match(model, applet["intent"], triggers)
        
        # for key, value in res.items():
        #     print(f"{key}: {value}")
        
        print("Variable names:")
        for key in res.keys():
            print(key)
        
        print(f"Action method: {action}")
        print("="*60)
                
        to_json[f"applet_{i}"] = {}
        to_json[f"applet_{i}"]["original_description"] = applet["original_description"]
        to_json[f"applet_{i}"]["intent"] = applet["intent"]
        to_json[f"applet_{i}"]["triggers"] = list(res.keys())
        to_json[f"applet_{i}"]["action"] = action
        to_json[f"applet_{i}"]["filter_code"] = applet["filter_code"]

        i+=1

    with open("data\\Json200_cleaned.json", "w") as f:
        json.dump(to_json, f, indent=3, separators=(',', ': '))
    
    print("Data saved to file.")
