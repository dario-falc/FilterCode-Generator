import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------- helpers ----------
def preprocess(s: str) -> str:
    """Pulisce e normalizza una stringa (per TF-IDF)."""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)  # split camelCase
    s = s.replace('.', ' ').replace('_', ' ')
    s = re.sub(r'[^a-z0-9\s]+', ' ', s, flags=re.I)
    return re.sub(r'\s+', ' ', s).strip().lower()

def uniq(seq):
    """Mantiene l'ordine ed elimina duplicati."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ========== 1) Carica i dati ==========
with open("data/triggers.json", "r") as f:
    TRIGGERS = json.load(f)

with open("data/actions.json", "r") as f:
    ACTIONS = json.load(f)

# Prepara i trigger
TRIGGER_FUNCS = []
for app, funcs in TRIGGERS.items():
    for func, vars_ in funcs.items():
        parts = [app, func] + vars_
        text = preprocess(" ".join(parts))
        TRIGGER_FUNCS.append({"id": f"{app}.{func}", "text": text})

# Prepara le action
ACTION_ITEMS = []
for app, method in ACTIONS.items():
    method_core = method.split("(")[0]
    text = preprocess(app + " " + method_core)
    ACTION_ITEMS.append({"id": method_core, "method": method, "text": text})

# ========== 2) App mention detector ==========
APP_ALIASES = {app.lower(): app for app in TRIGGERS.keys()}
APP_ALIASES.update({
    "google sheets": "GoogleSheets",
    "gmail": "Email",
    "email": "Email",
    "inbox": "Email",
})

def detect_apps(prompt: str):
    pl = prompt.lower()
    return uniq([canon for alias, canon in APP_ALIASES.items() if alias in pl])

# ========== 3) Ranker semplice ==========
def rank_items(prompt: str, items: list):
    """Ritorna i top-3 item più simili al prompt."""
    if not items:
        return []
    q = preprocess(prompt)
    docs = [q] + [it["text"] for it in items]
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vec.fit_transform(docs)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    order = np.argsort(sims)[::-1]
    return [(items[i], float(sims[i])) for i in order[:3] if sims[i] > 0]

# ========== 4) Pipeline ==========
def process_prompt(prompt: str):
    print(f"\nPrompt: {prompt}\n")

    # Trigger candidates
    trig_candidates = rank_items(prompt, TRIGGER_FUNCS)
    print("Trigger candidate:")
    if not trig_candidates:
        print("  (nessun trigger trovato)")
    for it, score in trig_candidates:
        print(f"  - {it['id']} (score {score:.3f})")

    # Action candidates
    act_candidates = rank_items(prompt, ACTION_ITEMS)
    print("\nAction candidate:")
    if not act_candidates:
        print("  (nessuna action trovata)")
    for it, score in act_candidates:
        print(f"  - {it['id']} (score {score:.3f}) → {it['method']}")

# ========== 5) Test ==========
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
    process_prompt(test_prompt)
