import os
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from generate_L_curves import  plot_learning_curve
from sklearn.metrics import roc_auc_score,average_precision_score

X_train = pd.read_csv("../data/X_train_processed.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()

X_test = pd.read_csv("../data/X_test_processed.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


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

#evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

#feature importance
model = model.named_steps["xgbclassifier"]

feature_names = model.named_steps["columntransformer"].get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_})

importance = importance.sort_values("importance",ascending=False)
os.makedirs("../results/model", exist_ok=True)
importance.to_csv("../results/model/feature_importance.csv",index=False)

print(importance.head(30))

plot_learning_curve(
    model,
    "../results/model/learning_curve.png"
)


# Save model + feature columns
os.makedirs("../results/model", exist_ok=True)

model_data = {"preprocessor": preprocessor,"model": model}
joblib.dump(model_data,"../results/model/xgboost.pkl")


print("Model saved.")