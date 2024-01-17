"""
Script to generate the prediction dataset for the Spotify genre predictor classifier. The dataset
contains all the tracks in the playlist, with only the audio features used to train the classifier.

This script takes a Spotify user and playlist name.
"""
import argparse
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("SPOTIFY_CALLBACK_URI")
SCOPE = "user-library-read"

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
)
access_token = sp_oauth.get_access_token(as_dict=False)
spotify = spotipy.Spotify(auth=access_token)

def generate_predict_dataset(user: str, playlist_name: str, config_file_name: str) -> None:
	"""
	Generate the prediction dataset
	"""
    with open(config_file_name, encoding="utf-8") as config_file:
        config = json.load(config_file)
	playlists = spotify.user_playlists(user)
    for playlist in playlists["items"]:
        if playlist["name"] == playlist_name:
            playlist_id = playlist["id"]
            break
    playlist_data = spotify.playlist_tracks(playlist_id)

    data_list = []
    for item in playlist_data.get("items"):
        track = item.get("track")
        track_name = track.get("name")
        track_id = track.get("id")
        audio_features = spotify.audio_features(track_id)
        additional_info = spotify.track(track_id)

        data_list.append({
            "track_name": track_name,
            **audio_features[0],  # Include all audio features in the DataFrame
            "artist_name": additional_info["artists"][0][
                "name"
            ],  # First artist's name
            "album_name": additional_info["album"]["name"],
            "album_release_date": additional_info["album"]["release_date"],
            "track_popularity": additional_info["popularity"],
            "duration_ms": additional_info["duration_ms"],
            "explicit_content": additional_info["explicit"],
        })

    data = pd.DataFrame(data_list)
    data = data[config.get("features")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spotify-user-name",
        dest="user",
        type=str,
        help="Spotify username",
    )
    parser.add_argument(
        "--playlist-name",
        dest="playlist_name",
        type=str,
        help="Name of the playlist to predict the genre of",
    )
    parser.add_argument(
        "--config-file-name",
        dest="config_file_name",
        type=str,
        help="Config file for the classifier",
    )

    kwargs = vars(parser.parse_args())
    generate_predict_dataset(**kwargs)
