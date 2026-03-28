import os
import sys

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC


def train_CuT(dataset_name, model_name, save_to="CuT.joblib"):
    """Train a classifier under test (CuT) on the specified training dataset."""
    
    # Load training data
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = f"{root_dir}/datasets_prepared/train/{dataset_name}_train.csv"
    train_df = pd.read_csv(train_path)

    X = train_df.iloc[:, :-1].values
    y = train_df.iloc[:, -1].values

    # Initialize and train the selected model
    if model_name == "SVM":
        CuT = LinearSVC(penalty="l2", dual="auto")
    elif model_name == "MLP":
        CuT = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16, 8, 4, 2),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=5000,
            early_stopping=True,
        )
    elif model_name == "RanForest":
        CuT = RandomForestClassifier(n_estimators=50, criterion="gini")
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Expected one of: 'SVM', 'MLP', 'RanForest'."
        )

    CuT.fit(X, y)

    # Save the trained model if an output path is provided
    if save_to is not None:
        dump(CuT, save_to)

    return CuT

    
model_name = sys.argv[1]
dataset_name = sys.argv[2]

current_dir = os.path.dirname(os.path.abspath(__file__))
model_trained_path = f"{current_dir}/{model_name}_{dataset_name}.joblib"

train_CuT(dataset_name, model_name, model_trained_path)
