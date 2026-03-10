"""
Author: Changli Liu
"""

import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv(os.path.join(os.path.dirname(__file__), "time_pre_input_data.csv"))
print("Data shape:", df.shape)

TARGET = "Time_taken(min)"
X = df.drop(columns=[TARGET])
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print(f"Trainsize:{X_train.shape[0]}")
# print(f"Test size:{X_test.shape[0]}")

model = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    max_iter=300,          
    tol=0.0,               
    n_iter_no_change=300, 
    random_state=42,
    verbose=True           
)

model.fit(X_train, y_train)
# print(f"实际迭代轮数: {model.n_iter_}")

loss_df = pd.DataFrame({
    "epoch": range(1, len(model.loss_curve_) + 1),
    "train_loss(MSE)": model.loss_curve_
})
loss_path = os.path.join(os.path.dirname(__file__), "time_pre_loss_curve_mlp.csv")
loss_df.to_csv(loss_path, index=False)
# print(f"Saved loss curve to:{loss_path} ({len(loss_df)} epochs)")
# print(f"Initial_Loss:{model.loss_curve_[0]:.6f}")
# print(f"Final  Loss: {model.loss_curve_[-1]:.6f}")
y_pred = model.predict(X_test)
test_mae  = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2   = r2_score(y_test, y_pred)
# print("Test Set Evaluation")
# print(f"MAE:{test_mae:.4f} min")
# print(f"RMSE: {test_rmse:.4f} min")
# print(f"R2: {test_r2:.4f}")

# results_df = X_test.copy()
# results_df["actual"]        = y_test.values
# results_df["predicted"]     = y_pred.round(4)
# results_df["error"]         = (y_pred - y_test.values).round(4)
# results_df["abs_error_MAE"] = np.abs(y_pred - y_test.values).round(4)
# results_df["sq_error_MSE"]  = ((y_pred - y_test.values) ** 2).round(4)

# output_path = os.path.join(os.path.dirname(__file__), "time_pre_results_mlp_test.csv")
# results_df.to_csv(output_path, index=False)
# # print(f"savedtest results to: {output_path} ({len(results_df)} rows)")

# y_train_pred = model.predict(X_train)
# train_results_df = X_train.copy()
# train_results_df["actual"]        = y_train.values
# train_results_df["predicted"]     = y_train_pred.round(4)
# train_results_df["error"]         = (y_train_pred - y_train.values).round(4)
# train_results_df["abs_error_MAE"] = np.abs(y_train_pred - y_train.values).round(4)
# train_results_df["sq_error_MSE"]  = ((y_train_pred - y_train.values) ** 2).round(4)

# train_output_path = os.path.join(os.path.dirname(__file__), "time_pre_results_mlp_train.csv")
# train_results_df.to_csv(train_output_path, index=False)
# # print(f"saved train results to: {train_output_path} ({len(train_results_df)} rows)")
