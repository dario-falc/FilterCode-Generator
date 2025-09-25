import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- helpers ----------
_CAMEL_RE = re.compile(r'([a-z])([A-Z])')

def preprocess(s: str) -> str:
    if not s:
        return ""
    s = _CAMEL_RE.sub(r'\1 \2', s)            # split camelCase / PascalCase
    s = s.replace('.', ' ').replace('_', ' ') # separa . e _
    s = re.sub(r'[^a-z0-9\s]+', ' ', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def uniq(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# ========== 1) Carica i dati ==========
with open("data/triggers.json", "r") as f:
    TRIGGERS = json.load(f)

with open("data/actions.json", "r") as f:
    ACTIONS = json.load(f)

# ========== 2) Index: trigger-level (App.Funzione) e actions ==========
# Costruisci corpus per TRIGGER a livello di funzione (non variabile)
TRIGGER_FUNCS = []          # [{"id": "App.Func", "app": "App", "func": "Func", "vars": [...], "text": "..."}]
TRIGGER_FUNCS_BY_APP = defaultdict(list)

for app, funcs in TRIGGERS.items():
    for func, vars_ in funcs.items():
        # testo di rappresentazione = app + func + (tutte le variabili come segnali)
        parts = [app, func] + list(set(vars_))
        text = preprocess(" ".join(parts))
        item = {"id": f"{app}.{func}", "app": app, "func": func, "vars": vars_, "text": text}
        TRIGGER_FUNCS.append(item)
        TRIGGER_FUNCS_BY_APP[app].append(item)

# Corpus ACTIONS (uno o più metodi per app, ma li trattiamo come voci autonome globali)
ACTION_ITEMS = []           # [{"id": "App.Method", "app": "App", "method": "...", "text": "..."}]
for app, method in ACTIONS.items():
    # pulizia firma metodo
    method_core = method.split("(")[0]  # "App.methodName"
    text_repr = preprocess(app + " " + method_core)  # es: "Email email send", "GoogleSheets append to google spreadsheet"
    ACTION_ITEMS.append({"id": method_core, "app": app, "method": method, "text": text_repr})

# ========== 3) App mention detector (solo per filtrare trigger se utile) ==========
APP_ALIASES = {app.lower(): app for app in TRIGGERS.keys()}
# aggiungi alias comuni
APP_ALIASES.update({
    "google sheets": "GoogleSheets",
    "sheet": "GoogleSheets",
    "spreadsheet": "GoogleSheets",
    "gmail": "Email",
    "email": "Email",
    "inbox": "Email",
})

def detect_apps(prompt: str):
    pl = prompt.lower()
    detected = []
    for alias, canon in APP_ALIASES.items():
        if alias in pl:
            detected.append(canon)
    return uniq(detected)

# ========== 4) Ranker generico TF-IDF + blend fuzzy ==========
def rank_items(prompt_text: str, items_text: list[str], topk=5):
    if not items_text:
        return []
    q = preprocess(prompt_text)
    docs = [q] + items_text
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=4000, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()

    # sblocco leggero con fuzzy se tutto molto basso
    # backup se TF-IDF non trova nulla (sblocca ranking anche su somiglianze deboli)
    if np.all(sims < 0.05):
        fuzzy = np.array([fuzzy_ratio(q, t) for t in items_text], dtype=float)
        sims = 0.6 * sims + 0.4 * fuzzy

    order = np.argsort(sims)[::-1]
    return [(i, float(sims[i])) for i in order[:topk] if sims[i] > 0]

# ========== 5) Pipeline ==========
def process_prompt(prompt: str, topk_triggers=5):
    print(f"\n Prompt: {prompt}\n")

    # 5.1 — Se troviamo app menzionate, filtriamo i trigger su quelle app. Altrimenti corpus globale.
    apps_mentioned = detect_apps(prompt)
    if apps_mentioned:
        print("App menzionate nel prompt:", ", ".join(apps_mentioned))
        trig_pool = []
        for a in apps_mentioned:
            trig_pool.extend(TRIGGER_FUNCS_BY_APP.get(a, []))
        pool_desc = f"trigger delle app menzionate ({len(trig_pool)} funzioni)"
    else:
        trig_pool = TRIGGER_FUNCS
        pool_desc = f"tutte le funzioni trigger ({len(trig_pool)})"

    # 5.2 — Rank TRIGGER (a livello App.Funzione)
    trig_texts = [it["text"] for it in trig_pool]
    trig_rank = rank_items(prompt, trig_texts, topk=topk_triggers)

    print(f"\n Trigger candidate (ranking su {pool_desc}):")
    if not trig_rank:
        print("  (nessun trigger trovato)")
    else:
        for idx, score in trig_rank:
            it = trig_pool[idx]
            print(f"  - {it['id']}  (score {score:.3f})")

    # 5.3 — Rank ACTION (globale su tutte le action di tutte le app)
    act_texts = [it["text"] for it in ACTION_ITEMS]
    act_rank = rank_items(prompt, act_texts, topk=3)

    print("\n Action candidate (global ranking):")
    if not act_rank:
        print("  (nessuna action trovata)")
    else:
        for idx, score in act_rank:
            it = ACTION_ITEMS[idx]
            print(f"  - {it['id']}  (score {score:.3f})  → {it['method']}")

# ========== 6) Test ==========
if __name__ == "__main__":
    # test_prompt = "Record the air quality data from my uHoo sensor whenever the PM2.5 level exceeds 15 ug/m3, and append this data to a Google Sheets spreadsheet."
    # test_prompt = "Send me a Telegram message if the light is on" # non va non capisce il concetto di light on
    # test_prompt = "Send me a Telegram message when a new file is added to my Dropbox folder and the message contains urgent"
    # test_prompt = "Add a row in Google Sheets when I save a Spotify track after 8 PM"
    # test_prompt = "When the \"Front Door\" is opened, as detected by the abode sensor, record a 30-second video clip using the \"Living Room Camera\" on Arlo."
    test_prompt = "If I star an email in Gmail, create a new task in Todoist." # non va
    # test_prompt = "If I add a note tagged with '#followup' to Evernote, create a Trello card."
    # test_prompt = "Share my new YouTube video on both Facebook and X (Twitter)." # non va
    # test_prompt = "Automatically create a draft blog post in WordPress from a new row in Airtable." # non va
    process_prompt(test_prompt, topk_triggers=5)
