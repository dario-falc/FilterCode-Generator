import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# carica il modello una volta sola
model = SentenceTransformer("all-MiniLM-L6-v2")

def match(intent, triggers, actions):
    intent_emb = model.encode([intent])
    trig_ris, act_ris = [], []

    if triggers:
        trig_emb = model.encode(triggers)
        trig_scores = cosine_similarity(intent_emb, trig_emb)[0]
        trig_scores = (trig_scores + 1) / 2  # porta da [-1,1] a [0,1]
        best = trig_scores.max()
        cutoff = best * 0.7
        trig_ris = [(t, s) for t, s in zip(triggers, trig_scores) if s >= cutoff]
        if not trig_ris:  # fallback
            best_idx = trig_scores.argmax()
            trig_ris = [(triggers[best_idx], trig_scores[best_idx])]
        trig_ris = sorted(trig_ris, key=lambda x: x[1], reverse=True)[:3]

    if actions:
        act_emb = model.encode(actions)
        act_scores = cosine_similarity(intent_emb, act_emb)[0]
        act_scores = (act_scores + 1) / 2
        best = act_scores.max()
        cutoff = best * 0.7
        act_ris = [(a, s) for a, s in zip(actions, act_scores) if s >= cutoff]
        if not act_ris:  # fallback
            best_idx = act_scores.argmax()
            act_ris = [(actions[best_idx], act_scores[best_idx])]
        act_ris = sorted(act_ris, key=lambda x: x[1], reverse=True)[:3]

    return trig_ris, act_ris


# leggi le ultime 10 righe dal file .jsonl
with open("data/generated_filtercode_from_intent.jsonl", "r", encoding="utf-8") as f:
   
    righe = f.readlines()[:10]
    for line in righe:
        riga = json.loads(line)
        intent = riga.get("intent", "")

        # estrai trigger
        triggers = [d["details"]["Filter code"] for d in riga.get("trigger_details", [])
                    if "details" in d and "Filter code" in d["details"]]

        # estrai action
        actions = []
        if "action_developer_info" in riga and "Filter code method" in riga["action_developer_info"]:
            actions.append(riga["action_developer_info"]["Filter code method"])

        # calcola i match
        trig_ris, act_ris = match(intent, triggers, actions)

        # stampa
        print("="*60)
        print("Intent:", intent)
        print("\nTrigger trovati:")
        for t, s in trig_ris:
            print(f" - {t} (score {s:.3f})")

        print("\nAction trovate:")
        for a, s in act_ris:
            print(f" - {a} (score {s:.3f})")
