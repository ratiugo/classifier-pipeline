"""
Train an XGBoost classifier
"""

from __future__ import annotations
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.classifier import Classifier


class TrainClassifier:
    """
    Pipeline for training an XGBoost classifier
    """

    def __init__(self, data_file_name: str, config_file_name: str) -> None:
        self.dataset = pd.read_csv(data_file_name)
        with open(config_file_name, encoding="utf-8") as config_file:
            self.config = json.load(config_file)

        self.params = self.config.get("training_params", {})
        self.label_encoded = False
        self.data = {
            "X_data": None,
            "y_data": None,
            "X_train": None,
            "y_train": None,
            "X_test": None,
            "y_test": None,
        }

        self.classifier: Classifier = None

    def run(self) -> TrainClassifier:
        """
        Run the pipeline
        """
        (
            self.create_classifier()
            .split_data_x_y()
            .create_label_encoder()
            .create_target_encoder()
            .create_train_test_split()
            .train_classifier()
            .get_feature_importance()
        )

        return self

    def create_classifier(self) -> TrainClassifier:
        """
        Create an empty classifier instance
        """
        self.classifier = Classifier(self.config)

        return self

    def split_data_x_y(self) -> TrainClassifier:
        """
        Split the data into X (features) and y (target) dataframes
        """
        self.data["X_data"] = self.dataset.drop(
            self.config.get("target_column"), axis=1
        )
        self.data["y_data"] = self.dataset.loc[:, self.config.get("target_column")]

        return self

    def create_label_encoder(self) -> TrainClassifier:
        """
        Create a label encoder if necessary
        """
        if self.data["y_data"].dtype.name in ["object", "category"]:
            self.label_encoded = True
            self.classifier.create_label_encoder(self.data["y_data"])
            self.data["y_data"] = self.classifier.label_encoder.transform(
                self.data["y_data"]
            )

        return self

    def create_target_encoder(self) -> TrainClassifier:
        """
        Create a target encoder if specified in the config
        """
        if self.config.get("create_target_encoder"):
            self.classifier.create_target_encoder(
                self.data["X_data"], self.data["y_data"]
            )
            self.data["X_data"] = self.classifier.target_encoder.transform(
                self.data["X_data"]
            )

        return self

    def create_train_test_split(self) -> TrainClassifier:
        """
        Split the data into training and testing sets, based on the test size specified in the
        config
        """
        kwargs_ = {"test_size": self.config.get("test_size", 0.2), "random_state": 123}
        values = train_test_split(self.data["X_data"], self.data["y_data"], **kwargs_)
        keys = ["X_train", "X_test", "y_train", "y_test"]
        self.data.update(dict(zip(keys, values)))

        return self

    def train_classifier(self) -> TrainClassifier:
        """
        Train the XGBoost classifier
        """
        self.classifier.train_classifier(
            self.data["X_train"], self.data["y_train"], self.params
        )
        print(self.classifier.metrics)

        return self

    def get_feature_importance(self) -> TrainClassifier:
        """
        Get the classifier metrics
        """
        self.classifier.get_feature_importance()
        print(self.classifier.metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier")
    parser.add_argument(
        "--data-file-name",
        dest="data_file_name",
        type=str,
        help="File name of the input dataset to be used to train a classifier",
    )
    parser.add_argument(
        "--config-file-name",
        dest="config_file_name",
        type=str,
        help="File name of the configuration JSON file pertaining to the process",
    )
    kwargs = vars(parser.parse_args())
    TrainClassifier(**kwargs).run()
