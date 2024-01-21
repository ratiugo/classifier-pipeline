"""
Make predictions with a trained classifier model
"""

from __future__ import annotations
import logging
import argparse
import pickle
from abc import ABC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


class Predict:
    """
    Prediction pipeline takes a model and config file and generates predictions.
    """

    def __init__(self, config_file_name) -> None:
        with open(config_file_name, encoding="utf-8") as config_file:
            self.config = json.load(config_file)

    def run(self) -> Predict:
        """
        Run the prediction pipeline
        """
