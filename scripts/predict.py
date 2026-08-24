import joblib
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report


pipeline = joblib.load("../results/model/random_forest.pkl")
print("Model pipeline loaded.")


X_test = pd.read_csv("../data/X_test_processed.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

print(f"X_test shape: {X_test.shape}")


# 3. Predict
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]


# 4. Evaluate
roc_auc = roc_auc_score(y_test,y_pred_proba)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"ROC AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))