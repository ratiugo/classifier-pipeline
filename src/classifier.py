"""
Contains logic and metrics associated to an XGBoost classifier

"""

from __future__ import annotations
import os
from xgboost.sklearn import XGBClassifier
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder


# pylint: disable=invalid-name
class Classifier:
    """
    XGBoost classifier
    """

    def __init__(self, output_dir) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.classifier: XGBClassifier = None
        self.target_encoder: TargetEncoder = None
        self.label_encoder: LabelEncoder = None
        self.train_columns: list = None
        self.metrics = {}

    def create_label_encoder(self, y_data: pd.Series) -> Classifier:
        """
        Create a label encoder for the target variable
        """
        self.label_encoder = LabelEncoder().fit(y_data)

        print(f"Label encoding column: {y_data.name}")

        return self

    def create_target_encoder(
        self, X_data: pd.DataFrame, y_data: pd.DataFrame
    ) -> Classifier:
        """
        Create a target encoder for the dataset
        """
        type_groups = X_data.columns.to_series().groupby(X_data.dtypes).groups
        type_groups = {key.name: value.tolist() for key, value in type_groups.items()}

        print(f"Categorical columns being target encoded: {type_groups['object']}")

        self.target_encoder = TargetEncoder(cols=type_groups["object"]).fit(
            X_data, y_data
        )

        return self

    def train_classifier(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict
    ) -> Classifier:
        """
        Train the XGBoost classifier
        """
        self.train_columns = X_train.columns
        self.classifier = XGBClassifier(**params)
        self.classifier.fit(X_train, y_train)

        return self

    def get_feature_importance(self) -> Classifier:
        """
        Extract the feature importance from the XGBoost classifier
        """
        importance = self.classifier.feature_importances_
        feat_importances = pd.Series(importance, index=self.train_columns)
        self.metrics["feature_importance"] = feat_importances

        return self

    def save_classifier(self) -> Classifier:
        """
        Save the classifier to a file.
        """
        classifier_file_name = f"{self.output_dir}/classifier.joblib"
        label_encoder_file_name = f"{self.output_dir}/label_encoder.joblib"
        target_encoder_file_name = f"{self.output_dir}/target_encoder.joblib"

        if self.classifier is None:
            raise ValueError(
                "Classifier not trained yet. Train the classifier before saving."
            )
        joblib.dump(self.classifier, classifier_file_name)

        message = f"Classifier saved to {classifier_file_name}"

        if self.label_encoder:
            joblib.dump(self.label_encoder, label_encoder_file_name)
            message += f", Label encoder saved to {label_encoder_file_name}"

        if self.target_encoder:
            joblib.dump(self.target_encoder, target_encoder_file_name)
            message += f", Target encoder saved to {target_encoder_file_name}"

        print(message)
        return self

    def load_classifier(self) -> Classifier:
        """
        Load the classifier from a file
        """
        classifier_file_name = f"{self.output_dir}/classifier.joblib"
        label_encoder_file_name = f"{self.output_dir}/label_encoder.joblib"
        target_encoder_file_name = f"{self.output_dir}/target_encoder.joblib"

        if not os.path.exists(classifier_file_name):
            raise FileNotFoundError(
                f"Model file '{classifier_file_name}' not found. Make sure to save the"
                " model first."
            )

        self.classifier = joblib.load(classifier_file_name)

        message = f"Classifier loaded from {classifier_file_name}"

        if os.path.exists(label_encoder_file_name):
            self.label_encoder = joblib.load(label_encoder_file_name)
            message += f", Label encoder loaded from {label_encoder_file_name}"

        if os.path.exists(target_encoder_file_name):
            self.target_encoder = joblib.load(target_encoder_file_name)
            message += f", Target encoder loaded from {target_encoder_file_name}"

        print(message)
        return self
