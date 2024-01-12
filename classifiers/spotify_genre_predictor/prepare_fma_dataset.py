"""
Script to prepare the FMA dataset for creating an XGBoost predictor.
"""
from classifiers.spotify_genre_predictor import fma_utils
import pandas as pd

tracks = fma_utils.load("data/fma/tracks.csv")
echonest_audio_features = fma_utils.load("data/fma/echonest.csv")
common_indices = tracks.index.intersection(echonest_audio_features.index)
tracks = tracks.loc[common_indices]
echonest_audio_features = echonest_audio_features.loc[
    tracks.index.intersection(echonest_audio_features.index)
]

tracks.columns = tracks.columns.map("_".join)
echonest_audio_features.columns = echonest_audio_features.columns.map("_".join)

data = tracks.merge(echonest_audio_features, left_index=True, right_index=True)
audio_cols = echonest_audio_features.columns
columns_to_drop = [
    col
    for col in audio_cols
    if "echonest" in col and "echonest_audio_features" not in col
]
data = data.drop(columns=columns_to_drop)
columns_to_rename = [
    col for col in data.columns if "echonest_audio_features" in col
]
new_column_names = {col: col.split("_")[-1] for col in columns_to_rename}
data = data.rename(columns=new_column_names)
data = data.dropna(
    subset=["artist_name", "track_title", "track_genre_top"]
)
data.reset_index(inplace=True)
data = data[["acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "tempo", "valence", "track_genre_top"]]

data.to_csv("classifiers/spotify_genre_predictor/input/data.csv", index=False)

