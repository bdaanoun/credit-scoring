import pandas as pd

def installments_features():
    installments = pd.read_csv("../data/installments_payments.csv")

    installments["PAYMENT_DELAY"] = (installments["DAYS_ENTRY_PAYMENT"]- installments["DAYS_INSTALMENT"])
    installments["LATE_PAYMENT"] = (installments["PAYMENT_DELAY"] > 0).astype(int)
    installments["PAYMENT_RATIO"] = (installments["AMT_PAYMENT"]/ installments["AMT_INSTALMENT"].replace(0, pd.NA))
    installments["UNDERPAYMENT"] = (installments["AMT_PAYMENT"] < installments["AMT_INSTALMENT"]).astype(int)

    features = installments.groupby("SK_ID_CURR").agg(
        INSTALLMENT_COUNT=("SK_ID_PREV", "count"),

        PAYMENT_DELAY_MEAN=("PAYMENT_DELAY", "mean"),
        PAYMENT_DELAY_MAX=("PAYMENT_DELAY", "max"),

        LATE_PAYMENT_COUNT=("LATE_PAYMENT", "sum"),
        LATE_PAYMENT_RATIO=("LATE_PAYMENT", "mean"),

        PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        PAYMENT_RATIO_MIN=("PAYMENT_RATIO", "min"),
        PAYMENT_RATIO_MAX=("PAYMENT_RATIO", "max"),

        UNDERPAYMENT_COUNT=("UNDERPAYMENT", "sum"),
        UNDERPAYMENT_RATIO=("UNDERPAYMENT", "mean"),

        AMT_INSTALMENT_MEAN=("AMT_INSTALMENT", "mean"),
        AMT_PAYMENT_MEAN=("AMT_PAYMENT", "mean"),

        AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
        AMT_INSTALMENT_SUM=("AMT_INSTALMENT", "sum"),
    )

    features = features.reset_index()

    return features