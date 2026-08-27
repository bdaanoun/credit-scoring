from feature_engineering.previous_application import previous_apps_features
from feature_engineering.installments import installments_features
from feature_engineering.bureau import bureau_features
from feature_engineering.application_train import application_features

def build_dataset(application_path="../data/application_train.csv"):

    application = application_features(application_path)
    bureau = bureau_features()
    previous = previous_apps_features()
    installments = installments_features()

    df = application

    df = df.merge(bureau, on="SK_ID_CURR", how="left")
    df = df.merge(previous, on="SK_ID_CURR", how="left")
    df = df.merge(installments, on="SK_ID_CURR", how="left")

    return df