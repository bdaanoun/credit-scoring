import sys
sys.path.append("..")
from helpers import  handle_missing_values, load_and_split_data ,  add_features
from feature_engineering.build_dataset import build_dataset

# X_train, X_test, y_train, y_test = load_and_split_data("../data/application_train.csv")
df = build_dataset()

X_train, X_test, y_train, y_test = load_and_split_data(df)

X_train, X_test = handle_missing_values(X_train, X_test)
# X_train = add_features(X_train)
# X_test = add_features(X_test)
# X_train, X_test = handle_missing_values(X_train,X_test)
# for col in X_train.columns:
#     print(f"{col}")

X_train.to_csv("../data/X_train_processed.csv", index=False)
X_test.to_csv("../data/X_test_processed.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)


print (X_train.head())
print("\nRemaining missing values:")
print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())