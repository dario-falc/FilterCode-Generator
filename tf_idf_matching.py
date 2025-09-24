from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def tfidf_match(intent, trigger_variables, n_res=5):
    corpus = [intent] + trigger_variables
    
    vectorizer = TfidfVectorizer(
        stop_words=None,
        ngram_range=(1, 2),
        max_features=1000
    )
    
    # Calcola TF-IDF
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Cosine similarity tra il prompt (primo elemento) e tutti gli altri
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Ordina per similarità decrescente
    ordered_idx = np.argsort(similarities)[::-1]
    
    # Restituisci i migliori
    res = []
    for i in ordered_idx[:n_res]:
        if similarities[i] > 0:
            res.append((trigger_variables[i], similarities[i]))
    
    return res


if __name__ == "__main__":
    with open(".//data//triggers_list.json", "r") as f:
        trigger_variables = json.load(f)
    
    ## Tests
    intent = "When a new track is found on Soundcloud with the search query \"electronic music\" and tags including \"killer, noise\", add it to my Spotify playlist named \"New Music\" if the track is available, using the song title from the Soundcloud track's title and including the artist name \"Daft Punk\" in the search query."
    # intent = "Record the air quality data from my uHoo sensor named \"Living Room\" whenever the PM2.5 level exceeds 15 ug/m3, and append this data to a Google Sheets spreadsheet named \"Air Quality Log\" with a formatted row including the timestamp, sensor name, and sensor value, in a folder path \"Air Quality Records/2024\"."
    # intent = "When the button widget is pressed at a location with latitude 37.8267 and longitude -122.4230, and the press occurs between 07:00 and 19:00, turn on the living room fan using the SwitchBot device."

    res = tfidf_match(intent, trigger_variables)
    
    print(res)