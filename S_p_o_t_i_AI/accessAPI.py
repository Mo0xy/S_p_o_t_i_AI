import os
import requests
import base64
import dataset
import spotipy
from dotenv import load_dotenv
from spotipy import SpotifyOAuth
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SpotifyClient:
    def __init__(self):
        # Carica le variabili d'ambiente e inizializza l'access token
        load_dotenv()
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.access_token = self.get_spotify_token()

    def get_spotify_token(self):
        url = "https://accounts.spotify.com/api/token"
        auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode("utf-8")
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials"
        }
        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            raise Exception(f"Errore {response.status_code}: {response.json()}")

    def search_track(self, track_name, artist_name):

        url = "https://api.spotify.com/v1/search"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        params = {
            "q": f"track:{track_name} artist:{artist_name}",
            "type": "track",
            "limit": 1
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            results = response.json()
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                return {
                    "track_name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "album": track['album']['name'],
                    "url": track['external_urls']['spotify'],
                    "id": track["id"]
                }
            else:
                return "No track found."
        else:
            raise Exception(f"Error {response.status_code}: {response.json()}")

    def get_saved_tracks(self):

        REDIRECT_URI = 'http://localhost:8888/callback'
        SCOPE = 'user-library-read'  # Cambia il scope in base alle tue esigenze

        # Autenticazione
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=os.getenv("CLIENT_ID"),
                                                       client_secret=os.getenv("CLIENT_SECRET"),
                                                       redirect_uri=REDIRECT_URI,
                                                       scope=SCOPE))
        results = sp.current_user_saved_tracks()
        for idx, item in enumerate(results['items']):
            track = item['track']
            print(f"{idx + 1}: {track['name']} di {track['artists'][0]['name']}")

    def get_track_metadata(self, track_id):

        url = f"https://api.spotify.com/v1/audio-features/{track_id}"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            track = response.json()
            return {
                "valence": track['valence'],
                "acousticness": track['acousticness'],
                "danceability": track['danceability'],
                "duration_ms": track['duration_ms'],
                "energy": track['energy'],
                "instrumentalness": track['instrumentalness'],
                "key": track['key'],
                "liveness": track['liveness'],
                "loudness": track['loudness'],
                "tempo": track['tempo'],
            }
        else:
            raise Exception(f"Error {response.status_code}: {response.json()}")

    def create_training_example(self, track_name, artist_name):
        track = self.search_track(track_name, artist_name)
        track_metadata = self.get_track_metadata(track['id'])

        return tuple(track_metadata.values())

    def getTrackId(self, track_name, artist_name):
        track = self.search_track(track_name, artist_name)
        return track['id']

    def normalize_tuple(self, tup):
        # Crea un'istanza di MinMaxScaler
        scaler = MinMaxScaler()

        # Riscrive la tupla come array 2D per il metodo fit_transform
        array = np.array(tup).reshape(-1, 1)

        # Adatta e trasforma la tupla
        normalized_array = scaler.fit_transform(array)

        # Restituisce la tupla normalizzata

        tempTuple = tuple(normalized_array.flatten())

        return tempTuple

