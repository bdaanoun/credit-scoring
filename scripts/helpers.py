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


def handle_missing_values(X_train, X_test):

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

    return X_train, X_test

def add_features(df: pd.DataFrame) -> pd.DataFrame:

        # 1. CREDIT / INCOME
            df["CREDIT_INCOME_RATIO"] = (df["AMT_CREDIT"]/ df["AMT_INCOME_TOTAL"].replace(0, np.nan))
            df["ANNUITY_INCOME_RATIO"] = (df["AMT_ANNUITY"]/ df["AMT_INCOME_TOTAL"].replace(0, np.nan))
            df["ANNUITY_CREDIT_RATIO"] = (df["AMT_ANNUITY"]/ df["AMT_CREDIT"].replace(0, np.nan))
            
            
        # 2. AGE / EMPLOYMENT
            df["employed_to_age_ratio"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)

        # 3. EXTERNAL SOURCE AGGREGATION
            ext_source_columns = [column for column in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"] if column in df.columns]
            df["EXT_SOURCE_AGG"] = df[ext_source_columns].mean(axis=1)
            
            # how well t sources are aligned ....
            if len(ext_source_columns) >= 2:
                df["EXT_SOURCE_STD"] = df[ext_source_columns].std(axis=1)


        # 4. SOCIAL DEFAULT RATE
            total_observations = (df["OBS_30_CNT_SOCIAL_CIRCLE"]+ df["OBS_60_CNT_SOCIAL_CIRCLE"])
            total_defaults = (df["DEF_30_CNT_SOCIAL_CIRCLE"]+ df["DEF_60_CNT_SOCIAL_CIRCLE"])
            df["SOCIAL_DEFAULT_RATE"] = (total_defaults/ total_observations.replace(0, np.nan))


        # 5. CREDIT BUREAU ACTIVITY
            recent_columns = [
                "AMT_REQ_CREDIT_BUREAU_HOUR",
                "AMT_REQ_CREDIT_BUREAU_DAY",
                "AMT_REQ_CREDIT_BUREAU_WEEK",
                "AMT_REQ_CREDIT_BUREAU_MON",
            ]

            annual_columns = [
                "AMT_REQ_CREDIT_BUREAU_QRT",
                "AMT_REQ_CREDIT_BUREAU_YEAR",
            ]

            df["CREDIT_BUREAU_RECENT"] = df[recent_columns].sum(axis=1,min_count=1)
            df["CREDIT_BUREAU_ANNUAL"] = df[annual_columns].sum(axis=1,min_count=1)
            all_bureau_columns = recent_columns + annual_columns
            df["CREDIT_BUREAU_ACTIVITY"] = df[all_bureau_columns].sum(axis=1, min_count=1)

        # 6. DOCUMENT ACTIVITY
            document_columns = [column for column in df.columns if column.startswith("FLAG_DOCUMENT_")]
            df["DOCUMENT_ACTIVITY"] = df[document_columns].sum(axis=1)

            return df
     