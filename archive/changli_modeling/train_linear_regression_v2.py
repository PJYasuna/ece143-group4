"""
train_linear_regression_v2.py

Author: Changli Liu
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# loading data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_time_features_normalized.csv"))
print("Data shape:", df.shape)

# creating the input feature vector and label(time_taken)
TARGET = "Time_taken(min)"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# split trainset and test(validation set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}")
print(f"Test size:  {X_test.shape[0]}")

#training
model = LinearRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
train_mse  = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae  = mean_absolute_error(y_train, y_train_pred)
print("\n=== Train Loss ===")
print(f"Train MSE:  {train_mse:.4f}")
print(f"Train RMSE: {train_rmse:.4f} min")
print(f"Train MAE:  {train_mae:.4f} min")


y_pred = model.predict(X_test)
test_mae  = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2   = r2_score(y_test, y_pred)

print("\n=== Test Set Evaluation ===")
print(f"MAE:  {test_mae:.4f} min")
print(f"RMSE: {test_rmse:.4f} min")
print(f"R2:   {test_r2:.4f}")


results_df = X_test.copy()
results_df["actual"]        = y_test.values
results_df["predicted"]     = y_pred.round(4)
results_df["error"]         = (y_pred - y_test.values).round(4)
results_df["abs_error_MAE"] = np.abs(y_pred - y_test.values).round(4)
results_df["sq_error_MSE"]  = ((y_pred - y_test.values) ** 2).round(4)

output_path = os.path.join(os.path.dirname(__file__), "results_linear_regression_test.csv")
results_df.to_csv(output_path, index=False)
print(f"\nSaved test results to: {output_path} ({len(results_df)} rows)")


train_results_df = X_train.copy()
train_results_df["actual"]        = y_train.values
train_results_df["predicted"]     = y_train_pred.round(4)
train_results_df["error"]         = (y_train_pred - y_train.values).round(4)
train_results_df["abs_error_MAE"] = np.abs(y_train_pred - y_train.values).round(4)
train_results_df["sq_error_MSE"]  = ((y_train_pred - y_train.values) ** 2).round(4)

train_output_path = os.path.join(os.path.dirname(__file__), "results_linear_regression_train.csv")
train_results_df.to_csv(train_output_path, index=False)
print(f"Saved train results to: {train_output_path} ({len(train_results_df)} rows)")


# coef_df = pd.DataFrame({
#     "Feature": X.columns,
#     "Coefficient": model.coef_
# }).sort_values("Coefficient", key=lambda x: x.abs(), ascending=False)
# print("\n=== Coefficients (sorted by abs value) ===")
# print(coef_df.to_string(index=False))
# print(f"Intercept: {model.intercept_:.4f}")
