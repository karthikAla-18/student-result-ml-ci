import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    rng = np.random.default_rng(42)
    number_of_students = 300

    data = pd.DataFrame({
        "attendance": rng.integers(50, 101, number_of_students),
