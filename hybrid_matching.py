import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import SequenceMatcher

# --- helpers ---
_CAMEL_RE = re.compile(r'([a-z])([A-Z])')

def preprocess(s: str) -> str:
    if not s:
        return ""
    s = _CAMEL_RE.sub(r'\1 \2', s)            # split camelCase
    s = s.replace('.', ' ').replace('_', ' ') # separa puntini/underscore
    s = re.sub(r'[^a-z0-9\s]+', ' ', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def fuzzy_ratio(a: str, b: str) -> float:
    # normalized [0,1]
    return SequenceMatcher(None, a, b).ratio()

# === 1. Carica i dati ===
with open("data/triggers.json", "r") as f:
    triggers = json.load(f)

with open("data/actions.json", "r") as f:
    actions = json.load(f)

# === 2. Vocabolario manuale per mappare i nomi delle app dal prompt ===
APP_ALIASES = {
    "spotify": "Spotify",
    "google sheets": "GoogleSheets",
    "google drive": "GoogleDrive",
    "google calendar": "GoogleCalendar",
    "gmail": "Email",
    "soundcloud": "Soundcloud",
    "asana": "Asana",
    "trello": "Trello",
    "todoist": "Todoist",
    "discord": "Discord",
    "telegram": "Telegram",
    "dropbox": "Dropbox",
    "reddit": "Reddit",
    "twitter": "Twitter",
    "instagram": "Instagram",
    "facebook": "FacebookPages",
    "evernote": "Evernote",
    "slack": "Slack",
    "github": "Github",
    "notion": "Notebook", 
    # aggiungerne altre se necessario
}

def detect_apps(prompt: str):
    prompt_lower = prompt.lower()
    detected = []
    for alias, app_name in APP_ALIASES.items():
        if alias in prompt_lower:
            detected.append(app_name)
    return detected

# === 3. Estrai tutti i trigger di una app ===
def get_triggers_for_app(app_name: str):
    if app_name not in triggers:
        return []
    result = []
    for func, variables in triggers[app_name].items():
        for var in variables:
            result.append(f"{app_name}.{func}.{var}")
    return result

# === 4. TF-IDF su lista di trigger filtrata ===
def tfidf_match(prompt, lista_elementi, n_risultati=5):
    if not lista_elementi:
        return []

    # preprocess prompt + elementi
    pre_prompt = preprocess(prompt)
    pre_elems = [preprocess(e) for e in lista_elementi]

    # fit/transform su corpus preprocessato
    corpus = [pre_prompt] + pre_elems
    vectorizer = TfidfVectorizer(
        stop_words=None,
        ngram_range=(1, 2),
        max_features=2000,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # cosine tra query (idx 0) e candidati (idx 1..)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # fallback: se tutte molto basse, usa un blend con fuzzy
    if np.all(similarities < 0.05):
        fuzzy_scores = np.array([fuzzy_ratio(pre_prompt, pe) for pe in pre_elems], dtype=float)
        # blend leggero: privilegia ancora cosine ma sblocca ranking
        similarities = 0.6 * similarities + 0.4 * fuzzy_scores

    indici_ordinati = np.argsort(similarities)[::-1]

    risultati = []
    count = 0
    for i in indici_ordinati:
        if similarities[i] > 0 and count < n_risultati:
            risultati.append((lista_elementi[i], float(similarities[i])))
            count += 1
        if count >= n_risultati:
            break
    return risultati

# === 5. Trova l’action per una app (se esiste) ===
def get_action_for_app(app_name: str):
    return actions.get(app_name, None)

# === 6. Pipeline completa ===
def process_prompt(prompt: str):
    print(f"\n Prompt: {prompt}\n")

    detected_apps = detect_apps(prompt)
    if not detected_apps:
        print("Nessuna app riconosciuta.")
        return

    for app in detected_apps:
        print(f"App rilevata: {app}")

        # 1. prendi trigger dell'app
        app_triggers = get_triggers_for_app(app)

        # 2. fai TF-IDF solo su questi trigger
        risultati = tfidf_match(prompt, app_triggers)

        print("\n Trigger trovati:")
        for r in risultati:
            print(f"  - {r[0]} (similarità {r[1]:.3f})") # cosine similarity

        # 3. trova un'action collegata
        action = get_action_for_app(app)
        if action:
            print(f"\n Azione disponibile: {action}")
        else:
            print("\n Nessuna azione trovata per questa app.")

# === 7. Test ===
if __name__ == "__main__":
    #test_prompt = "Add a row in Google Sheets when I save a Spotify track after 8 PM"
    test_prompt = "Send a daily email digest at 08:00 to my inbox with the title \"New Podcasts\" and a message containing the show name, description, and URL, including a link to the show on Spotify, for all new shows found on Spotify with the keyword \"technology\" or topic \"innovation\", including shows like \"Darknet Diaries\" or \"How I Built This\""
    process_prompt(test_prompt)
