import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import requests

# Configura le tue credenziali qui
load_dotenv()
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = 'user-library-read'  # Cambia il scope in base alle tue esigenze

# Autenticazione
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=os.getenv("CLIENT_ID"),
                                               client_secret=os.getenv("CLIENT_SECRET"),
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE))


# Esempio di utilizzo: ottenere le canzoni salvate dell'utente
def get_saved_tracks():
    results = sp.current_user_saved_tracks()
    for idx, item in enumerate(results['items']):
        track = item['track']
        print(f"{idx + 1}: {track['name']} di {track['artists'][0]['name']}")


def get_mbrz_data():
    track_name = "Time"
    artist_name = "Pink Floyd"

    # Costruisci l'URL della richiesta
    url = f"https://musicbrainz.org/ws/2/recording/?query=track:{track_name}+artist:{artist_name}&fmt=json"
    response = requests.get(url)
    data = response.json()

    # Estrai l'MBID del primo risultato
    if data['recordings']:
        mbid = data['recordings'][0]['id']
        print("MBID:", mbid)
    else:
        print("Nessun risultato trovato.")

    low_level_data = requests.get(f"https://acousticbrainz.org/{mbid}/low-level").json()
    print(low_level_data)
    print("\n")
    high_level_data = requests.get(f"https://acousticbrainz.org/{mbid}/high-level").json()
    print(high_level_data)



if __name__ == "__main__":
    get_saved_tracks()
    get_mbrz_data()
