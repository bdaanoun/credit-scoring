
import os
import joblib
import pandas as pd
from one_client import save_client_outputs


# Add or remove entries here to choose which clients to explain.
CLIENTS = [
    {"id": 310536, "split": "train", "description": "train_correct"},
    {"id": 365516, "split": "train", "description": "train_wrong"},
    {"id": 396899, "split": "test", "description": "test_client"},
]


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "results", "model", "xgboost.pkl")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "clients_outputs")




model_data = joblib.load(MODEL_PATH)
print("Model loaded.")

datasets = {}
for split in {client["split"] for client in CLIENTS}:
    X_path = os.path.join(PROJECT_DIR, "data", f"X_{split}_processed.csv")
    y_path = os.path.join(PROJECT_DIR, "data", f"y_{split}.csv")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze("columns")
    datasets[split] = (X, y)
    print(f"{split.capitalize()} data loaded: {X.shape}")


for client in CLIENTS:
    X, y = datasets[client["split"]]
    save_client_outputs(client, X, y, model_data)


