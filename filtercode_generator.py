import json
import re
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ================================
# Filter Code Generator (Upgraded)
# - naming JS pulito + dedup
# - fallback cross-app quando mancano trigger
# - blocchi separati per app (multi-app)
# - commenti esplicativi
# - conditions: time/day/numeric/text/boolean/state (già gestite)
# - text condition prova a creare un alias 'message' se trova una var messaggio
# ================================

# === 1) Caricamento dati ===
with open("data/triggers.json", "r") as f:
    TRIGGERS = json.load(f)

with open("data/actions.json", "r") as f:
    ACTIONS = json.load(f)

# Per il fallback globale
try:
    with open("data/triggers_list.json", "r") as f:
        GLOBAL_TRIGGER_LIST = json.load(f)
except FileNotFoundError:
    GLOBAL_TRIGGER_LIST = []

# === 2) Alias app (amplia/aggiorna se serve) ===
APP_ALIASES = {
    # Google & Office
    "google sheets": "GoogleSheets",
    "google drive": "GoogleDrive",
    "google calendar": "GoogleCalendar",
    "gmail": "Email",
    "microsoft outlook": "Outlook",
    "onenote": "Onenote",

    # Social / messaging
    "spotify": "Spotify",
    "soundcloud": "Soundcloud",
    "telegram": "Telegram",
    "discord": "Discord",
    "slack": "Slack",              # potrebbe non esistere nel dataset
    "twitter": "Twitter",
    "facebook": "FacebookPages",
    "instagram": "Instagram",
    "reddit": "Reddit",

    # Productivity
    "trello": "Trello",
    "todoist": "Todoist",
    "asana": "Asana",
    "evernote": "Evernote",
    "notion": "Notebook",          # Notion può essere come "Notebook"
    "github": "Github",

    # Cloud storage
    "dropbox": "Dropbox",
    "box": "Box",
    "onedrive": "Onedrive",

    # Smart home / IoT (esempi presenti nel dataset)
    "uhoo": "Uhoo",
    "smartthings": "SmartthingsV2",
    "netatmo": "Netatmo",
    "switchbot": "Switchbot",
    "hue": "Hue",
    "lifx": "Lifx",
    "irobot": "Irobot",
    "wemo": "WemoSwitch"
}

# === 3) Utilità ===
def detect_apps(prompt: str):
    pl = prompt.lower()
    return [app for alias, app in APP_ALIASES.items() if alias in pl]

def get_triggers_for_app(app_name: str):
    if app_name not in TRIGGERS:
        return []
    out = []
    for func, variables in TRIGGERS[app_name].items():
        for var in variables:
            out.append(f"{app_name}.{func}.{var}")
    return out

def tfidf_rank(prompt, items, topk=5):
    if not items:
        return []
    corpus = [prompt] + items
    vec = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=2000)
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    order = np.argsort(sims)[::-1]
    ranked = [(items[i], float(sims[i])) for i in order if sims[i] > 0]
    return ranked[:topk]

def nice_var_name(var: str, used: set):
    """
    Converte NomeVariabile in camelCase (TrackName -> trackName, DataPm25 -> dataPm25)
    + deduplica se già usata (trackName, trackName2, ...)
    """
    if not var:
        base = "value"
    else:
        base = var[0].lower() + var[1:]
    name = base
    i = 2
    while name in used:
        name = f"{base}{i}"
        i += 1
    used.add(name)
    return name

# === 4) Estrazione condizioni (time/day/numeric/text/boolean/state) ===
def extract_conditions(prompt: str):
    cond = {}

    # between HH:MM and HH:MM
    m = re.search(r"between (\d{1,2}):(\d{2}) and (\d{1,2}):(\d{2})", prompt, re.IGNORECASE)
    if m:
        cond["time_between"] = (int(m.group(1)), int(m.group(3)))

    # after HH:MM
    m = re.search(r"after (\d{1,2}):(\d{2})", prompt, re.IGNORECASE)
    if m:
        cond["time_after"] = int(m.group(1))

    # weekday specific / range
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    pl = prompt.lower()
    for i, day in enumerate(weekdays):
        if f"on {day}" in pl:
            cond["weekday"] = i+1 if i < 6 else 0
    if "on weekdays" in pl:
        cond["weekday_range"] = (1,5)
    if "on weekends" in pl:
        cond["weekday_range"] = (6,0)

    # numeric: temperature/humidity/pm2.5/co2 above/below N
    m = re.search(r"(temperature|humidity|pm2\.5|co2).*?(exceeds|above|greater than|below|less than|under)\s?(\d+)", prompt, re.IGNORECASE)
    if m:
        var = m.group(1).lower().replace(".", "")
        word = m.group(2).lower()
        val = int(m.group(3))
        comp = ">" if word in ["exceeds","above","greater than"] else "<"
        cond["numeric"] = (var, comp, val)

    # text contains
    m = re.search(r"message contains ['\"]?([\w\- ]+)['\"]?", prompt, re.IGNORECASE)
    if m:
        cond["text"] = m.group(1).strip()

    # boolean/state
    if "light is on" in pl:
        cond["boolean"] = ("light", True)
    if "light is off" in pl:
        cond["boolean"] = ("light", False)
    if "door is opened" in pl or "door is open" in pl:
        cond["state"] = ("door", "opened")
    if "door is closed" in pl:
        cond["state"] = ("door", "closed")

    return cond

def build_condition_string(conditions, message_alias="message"):
    parts = []

    if "time_between" in conditions:
        s, e = conditions["time_between"]
        parts.append(f"(Meta.currentUserTime.hour >= {s} && Meta.currentUserTime.hour < {e})")
    if "time_after" in conditions:
        parts.append(f"(Meta.currentUserTime.hour >= {conditions['time_after']})")

    if "weekday" in conditions:
        parts.append(f"(Meta.currentUserTime.dayOfWeek == {conditions['weekday']})")
    if "weekday_range" in conditions:
        s, e = conditions["weekday_range"]
        if s < e:
            parts.append(f"(Meta.currentUserTime.dayOfWeek >= {s} && Meta.currentUserTime.dayOfWeek <= {e})")
        else:
            parts.append(f"(Meta.currentUserTime.dayOfWeek == 6 || Meta.currentUserTime.dayOfWeek == 0)")

    if "numeric" in conditions:
        v, cmp_, val = conditions["numeric"]
        parts.append(f"({v} {cmp_} {val})")

    if "text" in conditions:
        kw = conditions["text"].replace('"', '\\"')
        parts.append(f"({message_alias}.includes(\"{kw}\"))")

    if "boolean" in conditions:
        var, state = conditions["boolean"]
        parts.append(f"({var.lower()} == {'true' if state else 'false'})")

    if "state" in conditions:
        var, state = conditions["state"]
        parts.append(f"({var.lower()} == \"{state}\")")

    return " && ".join(parts)

# === 5) Generazione Filter Code ===
def generate_filter_code(prompt: str):
    detected = detect_apps(prompt)

    # Raccoglitore: per app -> {func, vars}
    triggers_by_app = defaultdict(list)  # app -> list of (funcName, [vars])
    actions_by_app = {}

    # 5.1 Per ogni app riconosciuta prova a prendere il trigger "migliore"
    for app in detected:
        app_triggers = get_triggers_for_app(app)
        if not app_triggers:
            continue
        # rank su tutte le (app.func.var), poi scegliamo la func dominante del top
        ranked = tfidf_rank(prompt, app_triggers, topk=5)
        if not ranked:
            continue
        top = ranked[0][0]  # "App.Func.Var"
        app_name, func, _ = top.split(".")
        all_vars = TRIGGERS[app_name][func]
        triggers_by_app[app_name].append((func, all_vars))
        # action collegata se presente
        if app_name in ACTIONS:
            actions_by_app[app_name] = ACTIONS[app_name]

    # 5.2 Fallback se non abbiamo trovato nulla per nessuna app
    used_fallback = False
    if not triggers_by_app and GLOBAL_TRIGGER_LIST:
        used_fallback = True
        ranked = tfidf_rank(prompt, GLOBAL_TRIGGER_LIST, topk=3)
        for trig, _score in ranked:
            try:
                app_name, func, var = trig.split(".")
                all_vars = TRIGGERS[app_name][func]
                triggers_by_app[app_name].append((func, all_vars))
                if app_name in ACTIONS:
                    actions_by_app[app_name] = ACTIONS[app_name]
            except Exception:
                continue

    # 5.3 Costruzione codice
    code = []
    code.append("// === Auto-generated Filter Code ===")
    code.append("// DO NOT EDIT: generated by filtercode_generator.py")
    code.append(f"// Prompt: {prompt}")
    if used_fallback:
        code.append("// NOTE: no direct app match found → used global TF-IDF fallback")
    code.append("")

    # Prepariamo condizioni (costruiremo alias 'message' se possibile)
    conditions = extract_conditions(prompt)
    message_alias = "message"  # default
    will_use_condition_block = bool(conditions)

    # Indent se c'è un if
    indent = "    " if will_use_condition_block else ""

    # Prima generiamo i blocchi trigger, cercando se esiste una var 'messaggio' a cui agganciare l'alias
    trigger_lines = []
    action_lines = []

    # Per evitare nomi JS duplicati
    used_varnames = set()

    # euristica per trovare una var "message-like"
    MESSAGE_CANDIDATE_NAMES = {"Text", "Message", "NotificationMessage", "Body", "BodyHTML", "Content"}

    # ---- TRIGGER BLOCKS (per app) ----
    for app_name, funcs in triggers_by_app.items():
        trigger_lines.append(f"{indent}// Trigger variables ({app_name})")
        for func, vars_ in funcs:
            for var in vars_:
                js_name = nice_var_name(var, used_varnames)
                trigger_lines.append(f"{indent}let {js_name} = {app_name}.{func}.{var};")
                # setta alias per condizione testuale, se utile e non ancora impostato
                if "text" in conditions and message_alias == "message" and var in MESSAGE_CANDIDATE_NAMES:
                    # definiamo message come alias della variabile testuale disponibile
                    trigger_lines.append(f'{indent}let message = {app_name}.{func}.{var};')
                    message_alias = "message"
        trigger_lines.append("")

    # ---- ACTION BLOCKS (per app) ----
    for app_name in detected:
        if app_name in actions_by_app:
            action_lines.append(f"{indent}// Actions ({app_name})")
            action_lines.append(f"{indent}{actions_by_app[app_name]};")
            action_lines.append("")
    # Se detected vuoto o alcune app non avevano action ma i fallback sì, aggiungi anche le azioni dai fallback
    for app_name, action in actions_by_app.items():
        if app_name not in detected:
            action_lines.append(f"{indent}// Actions ({app_name})")
            action_lines.append(f"{indent}{action};")
            action_lines.append("")

    # Ora che sappiamo se esiste un alias 'message' reale, costruiamo la riga di condizione
    condition_str = build_condition_string(conditions, message_alias=message_alias) if will_use_condition_block else ""

    if will_use_condition_block and condition_str:
        code.append(f"if ({condition_str}) " + "{")

    code.extend(trigger_lines)
    code.extend(action_lines)

    if will_use_condition_block and condition_str:
        code.append("}")

    return "\n".join(code)

# === 6) Esecuzione di test rapido ===
if __name__ == "__main__":
    # Prompt precedente:
    # test_prompt = "Add a row in Google Sheets when I save a Spotify track after 8 PM on weekdays"
    # test_prompt = "Record the air quality data from my uHoo sensor whenever the PM2.5 level exceeds 15 ug/m3, and append this data to a Google Sheets spreadsheet."
    # test_prompt = "Send me a Telegram message if the light is on"
    # test_prompt = "Send me a Telegram message when a new file is added to my Dropbox folder and the message contains urgent"

    # Prompt nuovo:
    test_prompt = "Turn on the temperature control zone with serial number 'SN12345' and set to 'Heating' mode when the measured temperature from device 'Living Room Sensor' exceeds 22°C."
    print(generate_filter_code(test_prompt))
