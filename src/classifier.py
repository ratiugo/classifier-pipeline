"""
Contains logic and metrics associated to an XGBoost classifier
"""

from __future__ import annotations
import json
import os
import argparse
from abc import ABC, abstractmethod
from xgboost.sklearn import XGBClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.metrics import classification_report


class Classifier:
    """
    XGBoost classifier
    """

    def __init__(self) -> None:
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

    def create_target_encoder(self, X_data: pd.DataFrame, y_data: pd.DataFrame) -> Classifier:
        """
        Create a target encoder for the dataset
        """
        type_groups = X_data.columns.to_series().groupby(X_data.dtypes).groups
        type_groups = {key.name: value.tolist() for key, value in type_groups.items()}

        print(f"Categorical columns being target encoded: {type_groups['object']}")

        self.target_encoder = TargetEncoder(cols=type_groups["object"]).fit(X_data, y_data)

        return self

    def train_classifier(self, X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict) -> Classifier:
        """
        Train the XGBoost classifier
        """
        self.train_columns = X_train.columns
        self.classifier = XGBClassifier(**params)
        self.classifier.fit(X_train, y_train)

        return self

    def run_metrics(self) -> Classifier:
        """
        Run the metrics pipeline
        """
        self.feature_importance()

        return self

    def feature_importance(self) -> Classifier:
        """
        Extract the feature importance from the XGBoost classifier
        """
        importance = self.classifier.feature_importances_
        feat_importances = pd.Series(importance, index=self.train_columns)
        self.metrics["feature_importance"] = feat_importances
        
        return self



