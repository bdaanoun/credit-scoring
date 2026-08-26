import os
import joblib
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from generate_L_curves import  plot_learning_curve
from feature_importance import plot_feature_importance

X_train = pd.read_csv("../data/X_train_processed.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()

print(f"X_train shape: {X_train.shape}")


X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42
)



preprocessor = make_column_transformer(
    (
        OneHotEncoder(handle_unknown="ignore"),
        make_column_selector(dtype_include=["object", "category"])
    ),
    remainder="passthrough"
)


X_train_encoded = preprocessor.fit_transform(X_train)
X_val_encoded = preprocessor.transform(X_val)


model = XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)



model.fit(
    X_train_encoded,
    y_train,
    eval_set=[(X_train_encoded, y_train), (X_val_encoded, y_val)],
    verbose=True
)

print("Training completed.")
os.makedirs("../results/model", exist_ok=True)

plot_feature_importance(
    model,
    preprocessor,
    "../results/model/feature_importance.png"
)

plot_learning_curve(
    model,
    "../results/model/learning_curve.png"
)


# Save model + feature columns

model_data = {"preprocessor": preprocessor,"model": model}
joblib.dump(model_data,"../results/model/xgboost.pkl")


print("Model saved.")