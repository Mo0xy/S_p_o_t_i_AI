import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import dataset

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


if __name__ == "__main__":
    get_saved_tracks()
    print("\n\n\n")
    dataByArtist = dataset.Dataset("data_by_artist.csv")
    print(dataByArtist.getDataset())
    print("\n\n\n")
    print(dataByArtist.getDataFrame(["artists", "popularity"]))
    dataByArtist.normalizeColumn("popularity")
    print(dataByArtist.getDataset())
    print("\n\n\n")
    dataByArtist.EDA()


