import sys
sys.path.append("..")
import joblib
import pandas as pd

from sklearn.metrics import roc_auc_score
from feature_engineering.build_dataset import build_dataset


model_data = joblib.load("../results/model/xgboost.pkl")

preprocessor = model_data["preprocessor"]
model = model_data["model"]
print("Model and preprocessor loaded.")



X_test = pd.read_csv("../data/X_test_processed.csv")
X_test= X_test.drop(columns=["SK_ID_CURR"])
y_test = pd.read_csv("../data/y_test.csv").squeeze()
print(f"X_test shape: {X_test.shape}")

# Predict
X_test_encoded = preprocessor.transform(X_test)
y_pred = model.predict(X_test_encoded)
y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]


# 4. Evaluate
roc_auc = roc_auc_score(y_test,y_pred_proba)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"ROC AUC: {roc_auc:.4f}")



# Load Kaggle test data
test_df = build_dataset("../data/application_test.csv")
print(f"test dataset shape: {test_df.shape}")

ids = test_df["SK_ID_CURR"].copy()
X_kaggle = test_df.drop(columns=["SK_ID_CURR"])


# transform and predict
X_kaggle_enc = preprocessor.transform(X_kaggle)
predictions = model.predict_proba(X_kaggle_enc)[:, 1]


# Create Kaggle submission
submission = pd.DataFrame({
    "SK_ID_CURR": ids,
    "TARGET": predictions
})

submission.to_csv("../results/prediction.csv", index=False)

print("\nSubmission saved to:")
print("../results/prediction.csv")