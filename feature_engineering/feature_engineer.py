#feature engineering to add 

import numpy as np
import pandas as pd


def feature_engineering(df):
    """
    Designed for tree-based models such as XGBoost/LightGBM.
    """

    df = df.copy()

    # 1. DAYS_* -> AGE / TIME FEATURES

    # DAYS_BIRTH is negative: convert to years
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25

    # Employment duration in years
    # 365243 is an anomalous value used for missing employment info
    df["EMPLOYED_YEARS"] = np.where(df["DAYS_EMPLOYED"] == 365243,np.nan,(-df["DAYS_EMPLOYED"]) / 365.25)

    # Flag the special DAYS_EMPLOYED value
    df["FLAG_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")

    # Convert other DAYS variables to positive years
    df["ID_PUBLISH_YEARS"] = (-df["DAYS_ID_PUBLISH"]) / 365.25
    df["REGISTRATION_YEARS"] = (-df["DAYS_REGISTRATION"]) / 365.25
    df["LAST_PHONE_CHANGE_YEARS"] = (
        (-df["DAYS_LAST_PHONE_CHANGE"]) / 365.25
    )


    # 2. CREDIT / INCOME FEATURES

    # Credit amount relative to income
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    # Annuity relative to income
    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    # Goods price relative to income
    df["GOODS_INCOME_RATIO"] = (
        df["AMT_GOODS_PRICE"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    # Credit relative to goods price
    df["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_GOODS_PRICE"].replace(0, np.nan)
    )

    # Difference between credit and goods price
    df["CREDIT_GOODS_DIFF"] = (
        df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    )

    # Income minus annuity
    df["INCOME_AFTER_ANNUITY"] = (
        df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]
    )

    # Credit minus income
    df["CREDIT_INCOME_DIFF"] = (
        df["AMT_CREDIT"] - df["AMT_INCOME_TOTAL"]
    )

    # Approximate number of monthly payments
    df["CREDIT_TERM_MONTHS"] = (
        df["AMT_CREDIT"] /
        df["AMT_ANNUITY"].replace(0, np.nan)
    )

    # Income per family member
    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"] /
        df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    )

    # Income per child
    df["INCOME_PER_CHILD"] = (
        df["AMT_INCOME_TOTAL"] /
        (df["CNT_CHILDREN"] + 1)
    )


    # 3. AGE / EMPLOYMENT RELATIONSHIPS

    # Fraction of person's life spent employed
    df["EMPLOYED_AGE_RATIO"] = (
        df["DAYS_EMPLOYED"].replace(365243, np.nan) /
        df["DAYS_BIRTH"]
    )

    # Using positive years makes interpretation easier
    df["EMPLOYED_AGE_RATIO_2"] = (
        df["EMPLOYED_YEARS"] /
        df["AGE_YEARS"].replace(0, np.nan)
    )

    # Registration relative to age
    df["REGISTRATION_AGE_RATIO"] = (
        df["REGISTRATION_YEARS"] /
        df["AGE_YEARS"].replace(0, np.nan)
    )

    # ID publication relative to age
    df["ID_PUBLISH_AGE_RATIO"] = (
        df["ID_PUBLISH_YEARS"] /
        df["AGE_YEARS"].replace(0, np.nan)
    )


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
        df["EXT_SOURCE_MISSING_COUNT"] = (
            df[existing_ext].isna().sum(axis=1)
        )

    # Pairwise interactions
    if {"EXT_SOURCE_1", "EXT_SOURCE_2"}.issubset(df.columns):
        df["EXT_SOURCE_1_2"] = (
            df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        )

    if {"EXT_SOURCE_1", "EXT_SOURCE_3"}.issubset(df.columns):
        df["EXT_SOURCE_1_3"] = (
            df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
        )

    if {"EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(df.columns):
        df["EXT_SOURCE_2_3"] = (
            df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        )

    if {
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    }.issubset(df.columns):

        df["EXT_SOURCE_PRODUCT"] = (
            df["EXT_SOURCE_1"] *
            df["EXT_SOURCE_2"] *
            df["EXT_SOURCE_3"]
        )


    # 5. FAMILY FEATURES

    df["CHILDREN_RATIO"] = (
        df["CNT_CHILDREN"] /
        (df["CNT_FAM_MEMBERS"].replace(0, np.nan))
    )

    df["ADULTS_COUNT"] = (
        df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
    )

    df["FAMILY_SIZE_PER_INCOME"] = (
        df["CNT_FAM_MEMBERS"] /
        df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["INCOME_PER_ADULT"] = (
        df["AMT_INCOME_TOTAL"] /
        df["ADULTS_COUNT"].replace(0, np.nan)
    )


    # 6. CREDIT BUREAU REQUEST FEATURES

    bureau_request_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]

    existing_bureau = [
        col for col in bureau_request_cols
        if col in df.columns
    ]

    if existing_bureau:

        # Total number of bureau inquiries
        df["AMT_REQ_CREDIT_BUREAU_TOTAL"] = (
            df[existing_bureau].sum(axis=1, min_count=1)
        )

        # Number of periods with available information
        df["BUREAU_REQUEST_PERIOD_COUNT"] = (
            df[existing_bureau].notna().sum(axis=1)
        )


    # 7. SOCIAL CIRCLE FEATURES

    social_cols = [
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    ]

    existing_social = [
        col for col in social_cols
        if col in df.columns
    ]

    if existing_social:

        # Total observed social-circle people
        df["SOCIAL_CIRCLE_OBSERVED_TOTAL"] = (
            df[
                [
                    c for c in [
                        "OBS_30_CNT_SOCIAL_CIRCLE",
                        "OBS_60_CNT_SOCIAL_CIRCLE"
                    ]
                    if c in df.columns
                ]
            ].sum(axis=1, min_count=1)
        )

        # Total defaults in social circle
        df["SOCIAL_CIRCLE_DEFAULT_TOTAL"] = (
            df[
                [
                    c for c in [
                        "DEF_30_CNT_SOCIAL_CIRCLE",
                        "DEF_60_CNT_SOCIAL_CIRCLE"
                    ]
                    if c in df.columns
                ]
            ].sum(axis=1, min_count=1)
        )

        # Default ratio among observed social-circle records
        df["SOCIAL_CIRCLE_DEFAULT_RATIO"] = (
            df["SOCIAL_CIRCLE_DEFAULT_TOTAL"] /
            df["SOCIAL_CIRCLE_OBSERVED_TOTAL"].replace(0, np.nan)
        )


    # 8. PHONE / CONTACT FEATURES

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
        if col in df.columns
    ]

    if existing_phone:
        df["TOTAL_CONTACT_FLAGS"] = (
            df[existing_phone].sum(axis=1)
        )


    # 9. DOCUMENT FEATURES

    document_cols = [
        col for col in df.columns
        if col.startswith("FLAG_DOCUMENT_")
    ]

    if document_cols:

        # Number of documents provided
        df["DOCUMENTS_PROVIDED_COUNT"] = (
            df[document_cols].sum(axis=1)
        )

        # Whether at least one document was provided
        df["HAS_DOCUMENT"] = (
            df["DOCUMENTS_PROVIDED_COUNT"] > 0
        ).astype("int8")


    # 10. CAR FEATURES

    if "FLAG_OWN_CAR" in df.columns:

        df["CAR_OWNER"] = (
            df["FLAG_OWN_CAR"] == "Y"
        ).astype("int8")

        if "OWN_CAR_AGE" in df.columns:

            # Only the unusual case:
            # owns car but age is unknown
            df["CAR_AGE_UNKNOWN"] = (
                (df["FLAG_OWN_CAR"] == "Y") &
                (df["OWN_CAR_AGE"].isna())
            ).astype("int8")

            # Car age relative to applicant age
            df["CAR_AGE_TO_PERSON_AGE"] = (
                df["OWN_CAR_AGE"] /
                df["AGE_YEARS"].replace(0, np.nan)
            )


    # 11. REAL ESTATE / PROPERTY FEATURES

    # These columns describe different aspects of the building/property.
    building_avg_cols = [
        col for col in [
            "APARTMENTS_AVG",
            "BASEMENTAREA_AVG",
            "YEARS_BEGINEXPLUATATION_AVG",
            "YEARS_BUILD_AVG",
            "COMMONAREA_AVG",
            "ELEVATORS_AVG",
            "ENTRANCES_AVG",
            "FLOORSMAX_AVG",
            "FLOORSMIN_AVG",
            "LANDAREA_AVG",
            "LIVINGAPARTMENTS_AVG",
            "LIVINGAREA_AVG",
            "NONLIVINGAPARTMENTS_AVG",
            "NONLIVINGAREA_AVG",
            "TOTALAREA_MODE",
        ]
        if col in df.columns
    ]

    if building_avg_cols:

        df["BUILDING_AVG_MEAN"] = (
            df[building_avg_cols].mean(axis=1)
        )

        df["BUILDING_AVG_MISSING_COUNT"] = (
            df[building_avg_cols].isna().sum(axis=1)
        )


    # Living area relative to total area
    if {
        "LIVINGAREA_AVG",
        "TOTALAREA_MODE"
    }.issubset(df.columns):

        df["LIVING_TO_TOTAL_AREA"] = (
            df["LIVINGAREA_AVG"] /
            df["TOTALAREA_MODE"].replace(0, np.nan)
        )


    # Living apartments relative to apartments
    if {
        "LIVINGAPARTMENTS_AVG",
        "APARTMENTS_AVG"
    }.issubset(df.columns):

        df["LIVING_APARTMENTS_RATIO"] = (
            df["LIVINGAPARTMENTS_AVG"] /
            df["APARTMENTS_AVG"].replace(0, np.nan)
        )


    # 12. INCOME TYPE / FAMILY INTERACTIONS

    if {
        "AMT_INCOME_TOTAL",
        "CNT_CHILDREN"
    }.issubset(df.columns):

        df["INCOME_PER_CHILD_ADJUSTED"] = (
            df["AMT_INCOME_TOTAL"] /
            (df["CNT_CHILDREN"] + 1)
        )


    # 13. LOAN / ANNUITY FEATURES

    if {
        "AMT_ANNUITY",
        "AMT_CREDIT"
    }.issubset(df.columns):

        # Monthly payment as fraction of loan
        df["ANNUITY_CREDIT_RATIO"] = (
            df["AMT_ANNUITY"] /
            df["AMT_CREDIT"].replace(0, np.nan)
        )

        # Remaining amount after one annuity payment
        df["CREDIT_MINUS_ANNUITY"] = (
            df["AMT_CREDIT"] - df["AMT_ANNUITY"]
        )


    # 14. LOG TRANSFORMATIONS

    # Useful for heavily skewed financial variables.
    log_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
    ]

    for col in log_cols:
        if col in df.columns:
            df[f"{col}_LOG"] = np.log1p(
                df[col].clip(lower=0)
            )


    # 15. MISSINGNESS COUNT

    # Instead of creating one missing flag for every column,
    # create a general measure of how incomplete an application is.

    feature_cols = [
        col for col in df.columns
        if col != "TARGET"
    ]

    df["TOTAL_MISSING_COUNT"] = (
        df[feature_cols].isna().sum(axis=1)
    )

    df["TOTAL_MISSING_RATIO"] = (
        df["TOTAL_MISSING_COUNT"] /
        len(feature_cols)
    )


    # 16. CLEAN INFINITE VALUES

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    return df


"""
Very high missingness

COMMONAREA_* > ~70%
NONLIVINGAPARTMENTS_* > ~69%
OWN_CAR_AGE > ~66%
LANDAREA_* > ~59%
BASEMENTAREA_* > ~59%
EXT_SOURCE_1 > ~56%
ELEVATORS_* > ~53%
WALLSMATERIAL_MODE > ~51%
APARTMENTS_* > ~51%





Medium missingness;
OCCUPATION_TYPE > 31%
EXT_SOURCE_3 > 20%
AMT_REQ_CREDIT_BUREAU_* > 13%
EMERGENCYSTATE_MODE > 47%

the type of work if not exist, fill it with unknown
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")
"""