
import os
import joblib
import pandas as pd
from one_client import save_client_outputs
from helpers import add_features, handle_missing_values, load_and_split_data


# 1) a correctly predicted client from the train set
# 2) a misclassified client from the train set
# 3) one client from the test set
CLIENTS = [
    {"id": 310536, "split": "train", "description": "train_correct"},
    {"id": 242055, "split": "train", "description": "train_wrong"},
    {"id": 396899, "split": "test", "description": "test_client"},
]



model_data = joblib.load("../results/model/xgboost.pkl")
print("Model loaded.")



X_train = pd.read_csv("../data/X_train_processed.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()
y_test = pd.read_csv("../data/y_test.csv").squeeze()
X_test = pd.read_csv("../data/X_test_processed.csv")

save_client_outputs(CLIENTS[0], X_train, y_train, model_data)
save_client_outputs(CLIENTS[1], X_train, y_train, model_data)
save_client_outputs(CLIENTS[2], X_test, y_test, model_data)