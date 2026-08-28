
import os
import joblib
import pandas as pd
from one_client import save_client_outputs
from helpers import add_features, handle_missing_values, load_and_split_data


# Add or remove entries here to choose which clients to explain.
CLIENTS = [
    {"id": 310536, "split": "train", "description": "train_correct"},
    {"id": 365516, "split": "train", "description": "train_wrong"},
    {"id": 396899, "split": "test", "description": "test_client"},
]



model_data = joblib.load("../results/model/xgboost.pkl")
print("Model loaded.")

X_train, X_test, y_train, y_test = load_and_split_data("../data/application_train.csv")
X_train = add_features(X_train)
X_test = add_features(X_test)
X_train, X_test = handle_missing_values(X_train, X_test)

save_client_outputs(CLIENTS[0], X_train, y_train, model_data)
save_client_outputs(CLIENTS[1], X_train, y_train, model_data)
save_client_outputs(CLIENTS[2], X_test, y_test, model_data)