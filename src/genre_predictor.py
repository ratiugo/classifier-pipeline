"""
This module utilizes the FMA dataset to construct an XGBoost model for genre prediction based on 
Spotify's available track features. While the dataset comprises a vast number of tracks, only 
approximately 10,000 contain data for the specific audio features available in Spotify. This limited 
subset might affect the model's accuracy.

The current implementation focuses on predicting genre using Spotify-exposed audio features. 
Future adaptations aim to extend support for predicting genres from actual song files. This 
evolution would enable more comprehensive feature engineering and training on a larger song 
dataset, potentially enhancing model performance.
"""

from __future__ import annotations
import json
import os
import argparse
from abc import ABC, abstractmethod
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import fma_utils
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


class GenrePredictor(ABC):
    """
    Parent class to build a genre predicting model.
    """

    def __init__(self, **kwargs):
        """
        Arguments:
        * `config`: python object containing the configuration for building a model
        """
        self.config = kwargs.get("config")
        self.tracks = fma_utils.load("data/fma_metadata/tracks.csv")
        self.audio_features = fma_utils.load("data/fma_metadata/echonest.csv")
        self.output_dir = self.config.get("output_dir")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.data = None
        self.train_test_data = None
        self.predictor = None

    def run(self) -> GenrePredictor:
        """
        Run the pipeline to create the predictor.
        """
        (
            self.process_fma_data()
            .merge_tracks_and_audio_features()
            .train_test_split()
            .build_predictor()
            .evaluate_predictor()
            .save_predictor()
        )
        return self

    def process_fma_data(self) -> GenrePredictor:
        """
        Tracks in FMA are store in the 'tracks.csv' dataset, and features are stored in both the
        'features.csv' and 'echonest.csv' datasets. The echonest dataset contains data for features
        which Spotify exposes. The common audio features between the FMA dataset, and Spotify's
        web API are:

        acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, and
        valence.

        The FMA datasets use the index of the dataframe as the track ID. This function finds all
        the common tracks between the tracks and echonest dataset.

        This unfortunately reduces the FMA dataset quite drastically, but there are no other
        features I could use that I could then predict spotify songs with.
        """
        common_indices = self.tracks.index.intersection(self.audio_features.index)
        self.tracks = self.tracks.loc[common_indices]
        self.audio_features = self.audio_features.loc[
            self.tracks.index.intersection(self.audio_features.index)
        ]

        return self

    def merge_tracks_and_audio_features(self) -> GenrePredictor:
        """
        Flatten the track and audio features dataframes, merge the datasets, and clean them.
        """
        self.tracks.columns = self.tracks.columns.map("_".join)
        self.audio_features.columns = self.audio_features.columns.map("_".join)

        self.data = self.tracks.merge(
            self.audio_features, left_index=True, right_index=True
        )
        audio_cols = self.audio_features.columns
        columns_to_drop = [
            col
            for col in audio_cols
            if "echonest" in col and "echonest_audio_features" not in col
        ]
        self.data = self.data.drop(columns=columns_to_drop)
        columns_to_rename = [
            col for col in self.data.columns if "echonest_audio_features" in col
        ]
        new_column_names = {col: col.split("_")[-1] for col in columns_to_rename}
        self.data = self.data.rename(columns=new_column_names)
        self.data = self.data.dropna(
            subset=["artist_name", "track_title", "track_genre_top"]
        )

        return self

    def train_test_split(self) -> GenrePredictor:
        """
        Split the data into train/test sets, and store the prepared data as a class attribute
        """

        features = self.data[self.config.get("audio_features")]
        genre = self.data["track_genre_top"]
        # pylint: disable=invalid-name
        X_train, X_test, y_train, y_test = train_test_split(
            features, genre, test_size=0.2, random_state=42
        )
        self.train_test_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        return self

    @abstractmethod
    def build_predictor(self) -> GenrePredictor:
        """
        Train a model with the dataset generated
        """

    @abstractmethod
    def evaluate_predictor(self) -> GenrePredictor:
        """
        Run a classification report to see the predictor's performance.
        """

    def save_predictor(self) -> GenrePredictor:
        """
        Save file to output location specified in config file
        """
        predictor_file = self.output_dir + "/predictor.joblib"
        joblib.dump(self.predictor, predictor_file)

        print(f"Predictor saved to: {predictor_file}.\n")

        return self


class XGBoostPredictor(GenrePredictor):
    """
    Build the predictor with the XGBoost algorithm
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_encoder = LabelEncoder()

    def build_predictor(self) -> XGBoostPredictor:
        """
        Build XGBoost Predictor
        """
        y_train = self.label_encoder.fit_transform(self.train_test_data["y_train"])
        self.predictor = xgb.XGBClassifier().fit(
            self.train_test_data["X_train"], y_train
        )

        return self

    def evaluate_predictor(self) -> XGBoostPredictor:
        """
        Evaluate XGBoost predictor
        """
        playlists = spotify.user_playlists("ratiugo")
        for playlist in playlists["items"]:
            if playlist["name"] == "Key Glock Radio":
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
        data = data[
            [
                "acousticness",
                "danceability",
                "energy",
                "instrumentalness",
                "liveness",
                "speechiness",
                "tempo",
                "valence",
            ]
        ]
        # print(self.train_test_data["X_test"].columns)

        prediction = self.predictor.predict(data)
        y_pred = self.label_encoder.inverse_transform(prediction)
        print(y_pred)
        raise x
        y_pred = self.label_encoder.inverse_transform(prediction)
        report = classification_report(self.train_test_data["y_test"], y_pred)

        print(f"Performance report: \n\n{report}")

        return self


class RandomForestPredictor(GenrePredictor):
    """
    Build the predictor with the random forest algorithm
    """

    def build_predictor(self) -> RandomForestPredictor:
        """
        Build random forest predictor
        """
        self.predictor = RandomForestClassifier().fit(
            self.train_test_data["X_train"], self.train_test_data["y_train"]
        )

        return self

    def evaluate_predictor(self) -> RandomForestPredictor:
        """
        Evaluate random forest predictor
        """
        prediction = self.predictor.predict(self.train_test_data["X_test"])
        report = classification_report(self.train_test_data["y_test"], prediction)

        print(f"Performance report: \n\n{report}")

        return self


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file-name", dest="config_file")
    args = parser.parse_args()

    with open(args.config_file, encoding="utf-8") as config_file:
        config = json.load(config_file)

    if config["model"] == "xgboost":
        predictor = XGBoostPredictor(config=config)
    elif config["model"] == "random_forest":
        predictor = RandomForestPredictor(config=config)
    else:
        raise ValueError("Invalid model specified in the config file")

    predictor.run()
