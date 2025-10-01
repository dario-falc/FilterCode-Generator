import json
import math
import numpy as np
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def match(model, intent, triggers, alpha=0.5):
    intent_embedding = model.encode([intent])
    trigger_embeddings = model.encode(triggers)
 
    similarities = cosine_similarity(intent_embedding, trigger_embeddings)[0]
    
    scores = {}
    for i in range(len(triggers)):
        if similarities[i] > 0:
            # Normalize score
            scores[triggers[i]] = sigmoid(similarities[i])
    
    if len(scores) <= 3:
        return scores
    else:    
        # Keep top alpha%
        n = math.ceil(len(scores) * alpha)
        top_alpha_perc_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n])
        return top_alpha_perc_scores


if __name__ == "__main__":
    with open("data/generated_filtercode_from_intent.jsonl", "r", encoding="utf-8") as f:
        data = f.readlines()

    # For testing
    start = 129
    n = 10
    
    applets = []
    for i in range(start, start+n):
        applets.append(json.loads(data[i]))


    for applet in applets:
        print(f"Intent: {applet["intent"]}\n")
        
        # Triggers
        triggers = get_applet_triggers(applet)
        
        # Action
        action = get_applet_action(applet)
        
        # Intent-variables matching
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        res = match(model, applet["intent"], triggers)
        
        for key, value in res.items():
            print(f"{key}: {value}")
        print("="*60)
