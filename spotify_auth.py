"""
A simple script to handle the callback from Spotify's authorization flow
"""
from os import environ
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth

CLIENT_ID = environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = environ.get("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = environ.get("SPOTIFY_CALLBACK_URI")
SCOPE = "user-library-read"

sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
access_token = sp_oauth.get_access_token(as_dict=False)

file_name = "spotify_access_token.json"
with open(file_name, "w") as token_file:
	json.dump(access_token, token_file)

print(f"Success! Access token file generated at {file_name}")