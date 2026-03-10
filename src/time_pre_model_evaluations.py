"""
evaluate_models.py
changli Liu

"""

import os
import pandas as pd
import numpy as np

_DIR = os.path.dirname(__file__)

model_files = {
    "Linear Regression": os.path.join(_DIR, "results_linear_regression_test.csv"),
    "MLP":               os.path.join(_DIR, "results_mlp_test.csv"),
}


records = []

for model_name, filepath in model_files.items():
    try:
        df = pd.read_csv(filepath)

        mean_abs_error = df["abs_error_MAE"].mean()
        mean_sq_error  = df["sq_error_MSE"].mean()
        rmse           = np.sqrt(mean_sq_error)
        mean_error     = df["error"].mean()          

        # R² = 1 - SS_res / SS_tot
        ss_res = df["sq_error_MSE"].sum()
        ss_tot = ((df["actual"] - df["actual"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        records.append({
            "Model":          model_name,
            "Mean Error":     round(mean_error, 4),
            "MAE":            round(mean_abs_error, 4),
            "MSE":            round(mean_sq_error, 4),
            "RMSE":           round(rmse, 4),
            "R2":             round(r2, 4),
            "Num Samples":    len(df),
        })

        print(f"[OK] {model_name} loaded: {len(df)} samples from {filepath}")

    except FileNotFoundError:
        print(f"[SKIP] {filepath} not found, skipping {model_name}.")


summary_df = pd.DataFrame(records)

print("\n" + "=" * 65)
print("              Model Evaluation Summary (Test Set)")
print("=" * 65)
print(summary_df.to_string(index=False))
print("=" * 65)
print("\nNote:")
print("  Mean Error  : average signed error (positive = over-predict)")
print("  MAE         : mean absolute error (minutes)")
print("  MSE         : mean squared error")
print("  RMSE        : root mean squared error (minutes)")
print("  R2          : coefficient of determination (1.0 = perfect)")


output_path = os.path.join(_DIR, "evaluation_summary.csv")
summary_df.to_csv(output_path, index=False)
print(f"\nSaved summary to: {output_path}")
