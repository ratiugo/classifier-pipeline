"""
Script to prepare the FMA dataset for creating an XGBoost predictor.
"""
from classifiers.spotify_genre_predictor import fma_utils
import json
import pandas as pd

def prepare_fma_dataset(config_file_name: str):
    """
    Prepare the FMA dataset to train a classifier for genre based on Spotify (Echo Nest) audio 
    features
    """
    with open(config_file_name, encoding="utf-8") as config_file:
        config = json.load(config_file)
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
    data = data[config.get("features")]

    data.to_csv("classifiers/spotify_genre_predictor/input/data.csv", index=False)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file-name",
        dest="config_file_name",
        type=str,
        help="Config file for the classifier",
    )

    kwargs = vars(parser.parse_args())
    prepare_fma_dataset(**kwargs)

