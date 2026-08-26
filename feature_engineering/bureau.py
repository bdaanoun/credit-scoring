import pandas as pd

def bureau_features():
    bureau = pd.read_csv("../data/bureau.csv")
    
    # categorical columns encoding
    bureau_encoded = pd.get_dummies(bureau,columns=["CREDIT_ACTIVE","CREDIT_TYPE"],dtype=int)
    
    #aggregation
    bureau_features = bureau_encoded.groupby("SK_ID_CURR").agg(
        # Number of previous credits
        bureau_credit_count=("SK_ID_BUREAU", "count"),

        # Credit amount
        bureau_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_credit_mean=("AMT_CREDIT_SUM", "mean"),
        bureau_credit_max=("AMT_CREDIT_SUM", "max"),

        # Debt
        bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
        bureau_debt_max=("AMT_CREDIT_SUM_DEBT", "max"),

        # Credit limit
        bureau_limit_sum=("AMT_CREDIT_SUM_LIMIT", "sum"),
        bureau_limit_mean=("AMT_CREDIT_SUM_LIMIT", "mean"),
        bureau_limit_max=("AMT_CREDIT_SUM_LIMIT", "max"),

        # Overdue amount
        bureau_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_overdue_mean=("AMT_CREDIT_SUM_OVERDUE", "mean"),
        bureau_overdue_max=("AMT_CREDIT_SUM_OVERDUE", "max"),

        # Days overdue
        bureau_days_overdue_mean=("CREDIT_DAY_OVERDUE", "mean"),
        bureau_days_overdue_max=("CREDIT_DAY_OVERDUE", "max"),

        # Maximum overdue amount
        bureau_max_overdue_mean=("AMT_CREDIT_MAX_OVERDUE", "mean"),
        bureau_max_overdue_max=("AMT_CREDIT_MAX_OVERDUE", "max"),

        # Credit prolongation
        bureau_prolong_sum=("CNT_CREDIT_PROLONG", "sum"),
        bureau_prolong_mean=("CNT_CREDIT_PROLONG", "mean"),
        bureau_prolong_max=("CNT_CREDIT_PROLONG", "max"),

        # Credit history
        bureau_days_credit_min=("DAYS_CREDIT", "min"),
        bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
        bureau_days_credit_max=("DAYS_CREDIT", "max"),

        # Expected end date
        bureau_enddate_min=("DAYS_CREDIT_ENDDATE", "min"),
        bureau_enddate_mean=("DAYS_CREDIT_ENDDATE", "mean"),
        bureau_enddate_max=("DAYS_CREDIT_ENDDATE", "max"),

        # Actual end date
        bureau_fact_enddate_min=("DAYS_ENDDATE_FACT", "min"),
        bureau_fact_enddate_mean=("DAYS_ENDDATE_FACT", "mean"),
        bureau_fact_enddate_max=("DAYS_ENDDATE_FACT", "max"),

        # Last update
        bureau_update_min=("DAYS_CREDIT_UPDATE", "min"),
        bureau_update_mean=("DAYS_CREDIT_UPDATE", "mean"),
        bureau_update_max=("DAYS_CREDIT_UPDATE", "max"),

        # Annuity
        bureau_annuity_sum=("AMT_ANNUITY", "sum"),
        bureau_annuity_mean=("AMT_ANNUITY", "mean"),
        bureau_annuity_max=("AMT_ANNUITY", "max"),
    )

    # Aggregate CREDIT_TYPE dummy columns
    credit_type_columns = [
        col for col in bureau_encoded.columns
        if col.startswith("CREDIT_ACTIVE_")
        or col.startswith("CREDIT_TYPE_")]

    credit_type_features = (bureau_encoded.groupby("SK_ID_CURR")[credit_type_columns].sum())

    # Merge aggregated credit types
    bureau_features = bureau_features.join(credit_type_features)

    return bureau_features