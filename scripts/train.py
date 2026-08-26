import os
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from generate_L_curves import generate_learning_curve
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,average_precision_score


X_train = pd.read_csv("../data/X_train_processed.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()

X_test = pd.read_csv("../data/X_test_processed.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")



pipeline = make_pipeline(make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"),
         make_column_selector(dtype_include=["object", "category"])),
        remainder="passthrough"),
    XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
))

DROP_COLUMNS = ["SK_ID_CURR"]

X_train = X_train.drop(columns=DROP_COLUMNS,errors="ignore")
X_test = X_test.drop(columns=DROP_COLUMNS,errors="ignore")


print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("Training completed.")

#evaluate
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

#feature importance
model = pipeline.named_steps["xgbclassifier"]

feature_names = pipeline.named_steps["columntransformer"].get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_})

importance = importance.sort_values("importance",ascending=False)
os.makedirs("../results/model", exist_ok=True)
importance.to_csv("../results/model/feature_importance.csv",index=False)

print(importance.head(30))

# Save model + feature columns
generate_learning_curve(pipeline,X_train,y_train,"../results/model/learning_curve.png")
joblib.dump(pipeline,"../results/model/xgboost.pkl")

print("Model saved.")