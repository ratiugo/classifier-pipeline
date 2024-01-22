"""
Script to prepare the FMA dataset for creating an XGBoost predictor.
"""

import argparse
import json
import pandas as pd
from typing import Dict, Any
from classifiers.spotify_genre_predictor import fma_utils


def load_config(config_file_name: str) -> Dict[str, Any]:
    """
    Load the configuration from a JSON file.

    Args:
        config_file_name: Path to the JSON configuration file.

    Returns:
        Configuration settings as a dictionary.
    """
    with open(config_file_name, encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


def load_and_process_data(tracks_file: str, echonest_file: str) -> pd.DataFrame:
    """
    Load and process the FMA tracks and Echo Nest audio features datasets.

    Args:
        tracks_file: Path to the FMA tracks CSV file.
        echonest_file: Path to the Echo Nest audio features CSV file.

    Returns:
        A processed pandas DataFrame of the merged datasets.
    """
    tracks = fma_utils.load(tracks_file)
    echonest_audio_features = fma_utils.load(echonest_file)
    tracks.columns = tracks.columns.map("_".join)
    echonest_audio_features.columns = echonest_audio_features.columns.map("_".join)

    merged_data = pd.merge(
        tracks, echonest_audio_features, left_index=True, right_index=True, how="inner"
    )

    return merged_data


def clean_and_select_features(
    data: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Clean the merged dataset and select specific features.

    Args:
        data: Merged dataset from FMA tracks and Echo Nest audio features.
        config: Configuration settings including selected features.

    Returns:
        Cleaned and feature-selected pandas DataFrame.
    """
    columns_to_drop = [
        col
        for col in data.columns
        if "echonest" in col and "echonest_audio_features" not in col
    ]
    data = data.drop(columns=columns_to_drop)
    columns_to_rename = [
        col for col in data.columns if "echonest_audio_features" in col
    ]
    new_column_names = {col: col.split("_")[-1] for col in columns_to_rename}
    data = data.rename(columns=new_column_names)
    data = data.dropna(subset=["track_genre_top"])
    data.reset_index(inplace=True, drop=True)

    return data[config.get("features") + ["track_genre_top"]]


def save_dataset(data: pd.DataFrame, output_path: str) -> None:
    """
    Save the dataset to a CSV file.

    Args:
        data: Dataset to be saved.
        output_path: Path where the dataset CSV file will be saved.
    """
    data.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


def prepare_fma_dataset(config_file_name: str) -> None:
    """
    Main function to orchestrate the preparation of the FMA dataset for classifier training.

    Args:
        config_file_name: Path to the configuration file.
    """
    config = load_config(config_file_name)
    processed_data = load_and_process_data(
        "data/fma/tracks.csv", "data/fma/echonest.csv"
    )
    cleaned_data = clean_and_select_features(processed_data, config)
    save_dataset(cleaned_data, "classifiers/spotify_genre_predictor/input/data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file-name",
        required=True,
        type=str,
        help="Config file for the classifier",
    )
    args = parser.parse_args()
    prepare_fma_dataset(args.config_file_name)
