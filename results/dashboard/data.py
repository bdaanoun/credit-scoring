import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split as _tts

from config import (
    MODEL_PATH,
    PROJECT_DIR,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)

print("Loading model...")
model_data = joblib.load(MODEL_PATH)
preprocessor = model_data["preprocessor"]
classifier = model_data["model"]
print("Model loaded.")

print("Loading data...")
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

if "SK_ID_CURR" not in X_train.columns:
    _raw = pd.read_csv(os.path.join(PROJECT_DIR, "data", "application_train.csv"), usecols=["SK_ID_CURR", "TARGET"])
    _ids_train, _ids_test = _tts(
        _raw["SK_ID_CURR"],
        test_size=0.20,
        stratify=_raw["TARGET"],
        random_state=42,
    )
    X_train.insert(0, "SK_ID_CURR", _ids_train.values)
    X_test.insert(0, "SK_ID_CURR", _ids_test.values)

y_train_s = y_train.reset_index(drop=True)
y_test_s = y_test.reset_index(drop=True)

if len(X_train) != len(y_train_s):
    raise ValueError(f"X_train/y_train length mismatch: {len(X_train)} vs {len(y_train_s)}")
if len(X_test) != len(y_test_s):
    raise ValueError(f"X_test/y_test length mismatch: {len(X_test)} vs {len(y_test_s)}")

X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train_s, y_test_s], ignore_index=True)

ALL_IDS = sorted(X_all["SK_ID_CURR"].unique().tolist())
print(f"Dataset ready — {len(ALL_IDS):,} customers.")
print(f"Train default rate: {y_train_s.mean():.2%}")
print(f"Test default rate: {y_test_s.mean():.2%}")

_POP_FEATURES = X_all.drop(columns=["SK_ID_CURR"])
_POP_SAMPLE = _POP_FEATURES.sample(min(1000, len(_POP_FEATURES)), random_state=42)
_POP_TRANSFORMED = preprocessor.transform(_POP_SAMPLE)


def _clean_names(names):
    return [n.replace("remainder__", "").replace("onehotencoder__", "") for n in names]


FEATURE_NAMES = _clean_names(preprocessor.get_feature_names_out())
