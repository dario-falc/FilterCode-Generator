from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def tfidf_match(prompt, lista_elementi, n_risultati=5):
    corpus = [prompt] + lista_elementi
    
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
    indici_ordinati = np.argsort(similarities)[::-1]
    
    # Restituisci i migliori
    risultati = []
    for i in indici_ordinati[:n_risultati]:
        if similarities[i] > 0:
            risultati.append((lista_elementi[i], similarities[i]))
    
    return risultati


if __name__ == "__main__":
    with open(".//data//triggers_list.json", "r") as f:
        lista_elementi = json.load(f)
    
    ## Tests
    # prompt = "When a new track is found on Soundcloud with the search query \"electronic music\" and tags including \"killer, noise\", add it to my Spotify playlist named \"New Music\" if the track is available, using the song title from the Soundcloud track's title and including the artist name \"Daft Punk\" in the search query."
    # prompt = "Record the air quality data from my uHoo sensor named \"Living Room\" whenever the PM2.5 level exceeds 15 ug/m3, and append this data to a Google Sheets spreadsheet named \"Air Quality Log\" with a formatted row including the timestamp, sensor name, and sensor value, in a folder path \"Air Quality Records/2024\"."
    prompt = "When the button widget is pressed at a location with latitude 37.8267 and longitude -122.4230, and the press occurs between 07:00 and 19:00, turn on the living room fan using the SwitchBot device."

    res = tfidf_match(prompt, lista_elementi)
    
    print(res)