import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def st_match(intent, trigger_variables, action_methods, n_res=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    intent_embedding = model.encode([intent])
    variable_embeddings = model.encode(trigger_variables)
    action_methods_embeddings = model.encode(action_methods)

    trigger_variables_similarities = cosine_similarity(intent_embedding, variable_embeddings)[0]
    action_methods_similarities = cosine_similarity(intent_embedding, action_methods_embeddings)[0]
    
    trigger_variables_ordered = np.argsort(trigger_variables_similarities)[::-1]
    action_methods_ordered = np.argsort(action_methods_similarities)[::-1]
    
    variables = []
    for i in trigger_variables_ordered[:n_res]:
        if trigger_variables_similarities[i] > 0:
            variables.append((trigger_variables[i], trigger_variables_similarities[i]))
    
    methods = []
    for i in action_methods_ordered[:n_res]:
        if action_methods_similarities[i] > 0:
            methods.append((action_methods[i], action_methods_similarities[i]))
    
    return variables, methods



if __name__ == "__main__":

    # intent = "When a new track is found on Soundcloud with the search query \"electronic music\" and tags including \"killer, noise\", add it to my Spotify playlist named \"New Music\" if the track is available, using the song title from the Soundcloud track's title and including the artist name \"Daft Punk\" in the search query."
    # intent = "Record the air quality data from my uHoo sensor named \"Living Room\" whenever the PM2.5 level exceeds 15 ug/m3, and append this data to a Google Sheets spreadsheet named \"Air Quality Log\" with a formatted row including the timestamp, sensor name, and sensor value, in a folder path \"Air Quality Records/2024\"."
    # intent = "When the button widget is pressed at a location with latitude 37.8267 and longitude -122.4230, and the press occurs between 07:00 and 19:00, turn on the living room fan using the SwitchBot device."
    intent = "Turn off the Living Room Lamp when Apilio receives the \"run_only_at_nighttime\" event between 20:00 and 06:00 with a custom value of \"Green\""
    
    with open(".//data//triggers_list.json", "r") as f:
        trigger_variables = json.load(f)

    with open(".//data//actions_list.json", "r") as f:
        action_methods = json.load(f)

    variables, actions = st_match(intent, trigger_variables, action_methods)
    
    
    variable_names = [elem[0] for elem in variables]
    action_names = [elem[0] for elem in actions]
    
    print("Variables:")
    [print(v[0]) for v in variables]
    print()
    
    print("Actions:")
    [print(a[0]) for a in actions]
    print()