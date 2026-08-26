import joblib
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report

from helpers import add_features, handle_missing_values


model_data = joblib.load("../results/model/xgboost.pkl")

preprocessor = model_data["preprocessor"]
model = model_data["model"]
print("Model and preprocessor loaded.")



X_test = pd.read_csv("../data/X_test_processed.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

print(f"X_test shape: {X_test.shape}")


# 3. Predict
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
test_df = pd.read_csv("../data/application_test.csv")
ids = test_df["SK_ID_CURR"]
# X_kaggle = test_df.drop(columns=["SK_ID_CURR"])
X_kaggle = test_df.copy()

train_df = pd.read_csv("../data/application_train.csv")
X_train = train_df.drop(columns=["TARGET"])



# Feature engineering
X_train = add_features(X_train)
X_kaggle = add_features(X_kaggle)
X_train, X_kaggle = handle_missing_values(X_train,X_kaggle)
print("\nKaggle test data prepared.")


# transform and predict
X_kaggle = preprocessor.transform(X_kaggle)
predictions = model.predict_proba(X_kaggle)[:, 1]


# Create Kaggle submission
submission = pd.DataFrame({
    "SK_ID_CURR": ids,
    "TARGET": predictions
})

submission.to_csv("../results/prediction.csv", index=False)

print("\nSubmission saved to:")
print("../results/prediction.csv")