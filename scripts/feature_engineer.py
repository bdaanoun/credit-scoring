
from preprocessing import handle_missing_values, load_and_split_data


X_train, X_test, y_train, y_test = load_and_split_data("data/application_train.csv")

X_train, X_test, dropped_columns = handle_missing_values(X_train,X_test)

print("Dropped columns:")
print(dropped_columns)

print("\nRemaining missing values:")
print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())