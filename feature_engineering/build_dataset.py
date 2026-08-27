from feature_engineering.previous_application import previous_apps_features
from feature_engineering.installments import installments_features
from feature_engineering.bureau import bureau_features
from feature_engineering.application_train import application_features
def build_dataset():
    application = application_features()
    bureau = bureau_features()
    previous = previous_apps_features()
    installments = installments_features()
    # pos_cash = create_pos_cash_features(...)
    # credit_card = create_credit_card_features(...)

    df = application

    df = df.merge(bureau, on="SK_ID_CURR", how="left")
    df = df.merge(previous, on="SK_ID_CURR", how="left")
    df = df.merge(installments, on="SK_ID_CURR", how="left")
    # df = df.merge(pos_cash, on="SK_ID_CURR", how="left")
    # df = df.merge(credit_card, on="SK_ID_CURR", how="left")
    return df