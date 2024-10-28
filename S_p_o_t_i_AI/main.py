import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Configura le tue credenziali qui
CLIENT_ID = 'cc6fead3f5504f4d8e88802f8f1184be'
CLIENT_SECRET = 'd2ffd3f1fdc1499dbbd4f04667ffd7c7'
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = 'user-library-read'  # Cambia il scope in base alle tue esigenze

# Autenticazione
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
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
