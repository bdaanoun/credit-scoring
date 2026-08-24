import os
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

X_train = pd.read_csv("../data/X_train_processed.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()

print(f"X_train shape: {X_train.shape}")



pipeline = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"),make_column_selector(dtype_include=["object", "category"])),
        remainder="passthrough"
    ),
    RandomForestClassifier(
        n_estimators=50, max_depth=5, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
)


print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed.")



# Save model + feature columns
os.makedirs("../results/model", exist_ok=True)

joblib.dump(pipeline,"../results/model/random_forest.pkl")

print("Model saved.")