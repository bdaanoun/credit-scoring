import pandas as pd
import numpy as np
def previous_apps_features():
    prev_apps = pd.read_csv("../data/previous_application.csv")

    features = prev_apps.groupby("SK_ID_CURR")["SK_ID_PREV"].count().to_frame("PREV_APP_COUNT")
    
    # prev apps status
    status_counts = pd.crosstab(prev_apps["SK_ID_CURR"],prev_apps["NAME_CONTRACT_STATUS"])

    status_counts.columns = [f"PREV_STATUS_{col.upper()}_COUNT"
        for col in status_counts.columns]

    features = features.join(status_counts)
    #status ratio
    features["PREV_APPROVAL_RATIO"] = (features["PREV_STATUS_APPROVED_COUNT"]/ features["PREV_APP_COUNT"])
    features["PREV_REFUSAL_RATIO"] = (features["PREV_STATUS_REFUSED_COUNT"]/ features["PREV_APP_COUNT"])

    #num features
    numerical_cols = [
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "CNT_PAYMENT",
        "DAYS_DECISION",
    ]
    agg = {}
    for col in numerical_cols:
        if col in prev_apps.columns:
            agg[col] = ["mean", "max", "min"]

    numerical_features = prev_apps.groupby("SK_ID_CURR").agg(agg)
    
    # Flatten MultiIndex columns
    numerical_features.columns = [
        f"PREV_{col.upper()}_{stat.upper()}"
        for col, stat in numerical_features.columns
    ]

    features = features.join(numerical_features)
    prev_apps["CREDIT_APPLICATION_RATIO"] = (prev_apps["AMT_CREDIT"]/ prev_apps["AMT_APPLICATION"].replace(0, np.nan))

    ratio_features = (
        prev_apps.groupby("SK_ID_CURR")["CREDIT_APPLICATION_RATIO"]
        .agg(["mean", "max", "min"]))

    ratio_features.columns = [
            f"PREV_CREDIT_APPLICATION_RATIO_{stat.upper()}"
            for stat in ratio_features.columns]
    features = features.join(ratio_features)

    #days since prev decision
    prev_apps["DAYS_SINCE_PREV_DECISION"] = -prev_apps["DAYS_DECISION"]

    decision_features = (
        prev_apps.groupby("SK_ID_CURR")["DAYS_SINCE_PREV_DECISION"]
        .agg(["mean", "min", "max"]))

    decision_features.columns = [
        f"PREV_DAYS_SINCE_DECISION_{stat.upper()}"
        for stat in decision_features.columns]

    features = features.join(decision_features)

    #Insurance
    insurance = (
            prev_apps.groupby("SK_ID_CURR")["NFLAG_INSURED_ON_APPROVAL"]
            .agg(["sum", "mean"]))

    insurance.columns = ["PREV_INSURED_COUNT","PREV_INSURED_RATIO"]
    features = features.join(insurance)
    
    features = features.reset_index()

    return features
