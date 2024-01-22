"""
Script to generate the prediction dataset for the Spotify genre predictor classifier.
The dataset contains all the tracks in a playlist, with audio features used to train the classifier.

This script takes a Spotify user, playlist name, and path to config file as inputs.
"""

from __future__ import annotations
import os
import argparse
import json
from typing import List, Dict, Any
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


def get_spotify_client() -> spotipy.Spotify:
    """
    Create and return a Spotify client using OAuth credentials.

    Returns:
        Spotify client object with authorized access token.
    """
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.environ.get("SPOTIFY_CALLBACK_URI")
    scope = "user-library-read"

    sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
    )
    access_token = sp_oauth.get_access_token(as_dict=False)
    return spotipy.Spotify(auth=access_token)


def get_playlist_id(
    spotify_client: spotipy.Spotify, user: str, playlist_name: str
) -> str:
    """
    Retrieve the ID of a playlist by its name for a given user.

    Args:
        spotify_client: Authenticated Spotify client.
        user: Spotify username of the playlist owner.
        playlist_name: Name of the playlist.

    Returns:
        The Spotify ID of the playlist.

    Raises:
        ValueError: If the playlist is not found.
    """
    playlists = spotify_client.user_playlists(user)
    for playlist in playlists["items"]:
        if playlist["name"] == playlist_name:
            return playlist["id"]
    raise ValueError(f"Playlist '{playlist_name}' not found for user '{user}'.")


def fetch_playlist_data(
    spotify_client: spotipy.Spotify, playlist_id: str
) -> List[Dict[str, Any]]:
    """
    Fetch track data from a Spotify playlist using its ID.

    Args:
        spotify_client: Authenticated Spotify client.
        playlist_id: Spotify ID of the playlist.

    Returns:
        A list of track items from the playlist.
    """
    playlist_data = spotify_client.playlist_tracks(playlist_id)
    return playlist_data.get("items", [])


def extract_track_data(
    spotify_client: spotipy.Spotify, track_item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract relevant data from a track item.

    Args:
        spotify_client: Authenticated Spotify client.
        track_item: A single track item.

    Returns:
        A dictionary containing extracted data of the track.
    """
    track = track_item.get("track")
    track_id = track.get("id")
    audio_features = spotify_client.audio_features(track_id)[0]
    additional_info = spotify_client.track(track_id)

    return {
        "track_name": track.get("name"),
        **audio_features,
        "artist_name": additional_info["artists"][0]["name"],
        "album_name": additional_info["album"]["name"],
        "album_release_date": additional_info["album"]["release_date"],
        "track_popularity": additional_info["popularity"],
        "duration_ms": additional_info["duration_ms"],
        "explicit_content": additional_info["explicit"],
    }


def generate_predict_dataset(
    user: str,
    playlist_name: str,
    spotify_client: spotipy.Spotify,
    config: Dict[str, Any],
) -> None:
    """
    Generate a prediction dataset for a specific Spotify playlist.

    Args:
        user: Spotify username of the playlist owner.
        playlist_name: Name of the playlist.
        spotify_client: Authenticated Spotify client.
        config: Configuration dictionary with classifier columns.

    Saves a CSV file with the dataset.
    """
    playlist_id = get_playlist_id(spotify_client, user, playlist_name)
    track_items = fetch_playlist_data(spotify_client, playlist_id)

    data_list = [extract_track_data(spotify_client, item) for item in track_items]
    data = pd.DataFrame(data_list)

    classifier_columns = config["features"]
    data = data[classifier_columns]

    dir_name = os.path.join(
        "classifiers", "spotify_genre_predictor", "input", "predict_datasets", user
    )
    file_name = os.path.join(dir_name, f"{playlist_name}.csv")
    os.makedirs(dir_name, exist_ok=True)
    data.to_csv(file_name, index=False)

    print(f"Success! File saved to: {file_name}.")


def main() -> None:
    """
    Main function to parse arguments and initiate dataset generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spotify-user-name", required=True, type=str, help="Spotify username"
    )
    parser.add_argument(
        "--playlist-name",
        required=True,
        type=str,
        help="Name of the playlist to predict the genre of",
    )
    parser.add_argument(
        "--config-file", required=True, type=str, help="Path to the configuration file"
    )

    args = parser.parse_args()
    spotify_client = get_spotify_client()
    config = load_config(args.config_file)

    generate_predict_dataset(
        args.spotify_user_name, args.playlist_name, spotify_client, config
    )


if __name__ == "__main__":
    main()
