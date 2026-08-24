import joblib
import pandas as pd

from helpers import add_features, handle_missing_values


model = joblib.load("../results/model/random_forest.pkl")
print("Model loaded.")


# Load training data
train_df = pd.read_csv("../data/application_train.csv")
X_train = train_df.drop(columns=["TARGET"])


# Load Kaggle test data
test_df = pd.read_csv("../data/application_test.csv")

ids = test_df["SK_ID_CURR"]
X_test = test_df.copy()


# Same preprocessing as training
X_train = add_features(X_train)
X_test = add_features(X_test)

X_train, X_test = handle_missing_values(X_train, X_test)

print("Test data prepared.")


# Predict
predictions = model.predict_proba(X_test)[:, 1]


# Kaggle submission
submission = pd.DataFrame({
    "SK_ID_CURR": ids,
    "TARGET": predictions
})

submission.to_csv("../results/prediction.csv", index=False)

print("\nSubmission saved to:")
print("../results/prediction.csv")