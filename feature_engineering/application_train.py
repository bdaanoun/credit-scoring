import pandas as pd
import numpy as np

def application_features(path = "../data/application_train.csv"):
    df = pd.read_csv(path)
    df = df.copy()

    # DAYS_BIRTH is negative: convert to years
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25

    # Employment duration in years
    df["EMPLOYED_YEARS"] = np.where(df["DAYS_EMPLOYED"] == 365243, np.nan , (-df["DAYS_EMPLOYED"]) / 365.25)

    # Flag the special DAYS_EMPLOYED value
    df["FLAG_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")

    # Convert other DAYS variables to positive years
    df["ID_PUBLISH_YEARS"] = (-df["DAYS_ID_PUBLISH"]) / 365.25
    df["REGISTRATION_YEARS"] = (-df["DAYS_REGISTRATION"]) / 365.25
    df["LAST_PHONE_CHANGE_YEARS"] = ((-df["DAYS_LAST_PHONE_CHANGE"]) / 365.25)

    # credit / income features

    #compare the requested credit with the person's income
    df["CREDIT_INCOME_RATIO"] = (df["AMT_CREDIT"] /df["AMT_INCOME_TOTAL"])

    # Annuity relative to income
    df["ANNUITY_INCOME_RATIO"] = (df["AMT_ANNUITY"] /df["AMT_INCOME_TOTAL"])

    # Goods price relative to income
    df["GOODS_INCOME_RATIO"] = (df["AMT_GOODS_PRICE"] /df["AMT_INCOME_TOTAL"])

    # Credit relative to goods price
    df["CREDIT_GOODS_RATIO"] = (df["AMT_CREDIT"] /df["AMT_GOODS_PRICE"])

    # Difference between credit and goods price
    df["CREDIT_GOODS_DIFF"] = (df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"])

    # Income minus annuity
    df["INCOME_AFTER_ANNUITY"] = (df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"])

    # Credit minus income
    df["CREDIT_INCOME_DIFF"] = (df["AMT_CREDIT"] - df["AMT_INCOME_TOTAL"])

    # Approximate number of monthly payments
    df["CREDIT_TERM_MONTHS"] = (df["AMT_CREDIT"] /df["AMT_ANNUITY"])

    # Income per family member
    df["INCOME_PER_FAMILY_MEMBER"] = (df["AMT_INCOME_TOTAL"] /df["CNT_FAM_MEMBERS"].replace(0, np.nan))

    # Income per child
    df["INCOME_PER_CHILD"] = (df["AMT_INCOME_TOTAL"] /(df["CNT_CHILDREN"] + 1))
    
    # Using positive years makes interpretation easier
    df["EMPLOYED_AGE_RATIO"] = (df["EMPLOYED_YEARS"] /df["AGE_YEARS"])#.replace(0, np.nan)

    # Registration relative to age
    df["REGISTRATION_AGE_RATIO"] = (df["REGISTRATION_YEARS"] /df["AGE_YEARS"].replace(0, np.nan))

    # ID publication relative to age
    df["ID_PUBLISH_AGE_RATIO"] = (df["ID_PUBLISH_YEARS"] /df["AGE_YEARS"].replace(0, np.nan))


    # 4. EXT_SOURCE FEATURES

    ext_cols = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    existing_ext = [
        col for col in ext_cols
        if col in df.columns
    ]

    if existing_ext:
        # Mean of available external sources
        df["EXT_SOURCE_MEAN"] = df[existing_ext].mean(axis=1)

        # Minimum
        df["EXT_SOURCE_MIN"] = df[existing_ext].min(axis=1)

        # Maximum
        df["EXT_SOURCE_MAX"] = df[existing_ext].max(axis=1)

        # Standard deviation
        df["EXT_SOURCE_STD"] = df[existing_ext].std(axis=1)

        # Number of available external sources
        df["EXT_SOURCE_COUNT"] = df[existing_ext].notna().sum(axis=1)

        # Missing count
        df["EXT_SOURCE_MISSING_COUNT"] = (df[existing_ext].isna().sum(axis=1))


    # FAMILY FEATURES
    df["CHILDREN_RATIO"] = (df["CNT_CHILDREN"] /(df["CNT_FAM_MEMBERS"].replace(0, np.nan)))

    df["ADULTS_COUNT"] = (df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"])

    df["FAMILY_SIZE_PER_INCOME"] = (df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"])

    df["INCOME_PER_ADULT"] = (df["AMT_INCOME_TOTAL"] / df["ADULTS_COUNT"].replace(0, np.nan))

    # PHONE / CONTACT FEATURES
    phone_flags = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]

    existing_phone = [
        col for col in phone_flags
        if col in df.columns]

    if existing_phone:
        df["TOTAL_CONTACT_FLAGS"] = (df[existing_phone].sum(axis=1))


    # document features

    document_cols = [
        col for col in df.columns
        if col.startswith("FLAG_DOCUMENT_")
    ]

    if document_cols:
        # Number of documents provided
        df["DOCUMENTS_PROVIDED_COUNT"] = (df[document_cols].sum(axis=1))


    # CAR FEATURES
    if "FLAG_OWN_CAR" in df.columns:

        df["CAR_OWNER"] = (df["FLAG_OWN_CAR"] == "Y").astype("int8")

        if "OWN_CAR_AGE" in df.columns:
            # Car age relative to applicant age
            df["CAR_AGE_TO_PERSON_AGE"] = (df["OWN_CAR_AGE"] / df["AGE_YEARS"].replace(0, np.nan))


    # Living area relative to total area
    df["LIVING_TO_TOTAL_AREA"] = (df["LIVINGAREA_AVG"] /df["TOTALAREA_MODE"].replace(0, np.nan))


    # Living apartments relative to apartments
    df["LIVING_APARTMENTS_RATIO"] = (df["LIVINGAPARTMENTS_AVG"] /df["APARTMENTS_AVG"].replace(0, np.nan))


    # loan / annuity features
    if {"AMT_ANNUITY","AMT_CREDIT"}.issubset(df.columns):
        # Monthly payment as fraction of loan
        df["ANNUITY_CREDIT_RATIO"] = (df["AMT_ANNUITY"] /df["AMT_CREDIT"].replace(0, np.nan))


    # log transformations

    # Useful for heavily skewed financial variables.
    log_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ]

    for col in log_cols:
        if col in df.columns:
            df[f"{col}_LOG"] = np.log1p(df[col].clip(lower=0))


    # clean infinite values
    df.replace([np.inf, -np.inf],np.nan,inplace=True)
    return df     