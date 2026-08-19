import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def load_and_split_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def handle_missing_values(X_train, X_test, missing_threshold=0.65):

    X_train = X_train.copy()
    X_test = X_test.copy()

    # 365243 kat3ni DAYS_EMPLOYED is unknown
    if "DAYS_EMPLOYED" in X_train.columns:
        train_anomaly = X_train["DAYS_EMPLOYED"] == 365243
        test_anomaly = X_test["DAYS_EMPLOYED"] == 365243
        
        X_train["DAYS_EMPLOYED_ANOM"] = train_anomaly.astype(int)
        X_test["DAYS_EMPLOYED_ANOM"] = test_anomaly.astype(int)

        X_train.loc[train_anomaly, "DAYS_EMPLOYED"] = np.nan
        X_test.loc[test_anomaly, "DAYS_EMPLOYED"] = np.nan



    missing_ratio = X_train.isnull().mean()
    dropped_columns = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    X_train = X_train.drop(columns=dropped_columns)
    X_test = X_test.drop(columns=dropped_columns)

    numerical_columns = X_train.select_dtypes(include=["number"]).columns
    categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns

    #Numerical columns
    for column in numerical_columns:
        if X_train[column].isnull().any():
            X_train[f"{column}_MISSING"] = (X_train[column].isnull().astype(int))
            X_test[f"{column}_MISSING"] = (X_test[column].isnull().astype(int))

        median = X_train[column].median()

        X_train[column] = X_train[column].fillna(median)
        X_test[column] = X_test[column].fillna(median)

    #Categorical columns
    for column in categorical_columns:
        X_train[column] = X_train[column].fillna("Missing")
        X_test[column] = X_test[column].fillna("Missing")

    return X_train, X_test, dropped_columns