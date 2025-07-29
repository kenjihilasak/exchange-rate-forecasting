from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_rmse_mae_per_path(df_paths, df_test, rate_col='rate'):
    """
    Evaluate RMSE and MAE for each simulated path against the real test data.

    Parameters:
    - df_paths: DataFrame with simulated forecast paths (n_paths x horizon)
                columns = dates, rows = simulated paths
    - df_test: DataFrame with actual test data (must include rate_col and matching dates)
    - rate_col: column name in df_test with actual exchange rate

    Returns:
    - rmse_values: np.array of RMSE values (length = n_paths)
    - expected_rmse: float, mean RMSE across all paths (Monte Carlo expectation)
    - mae_values: np.array of MAE values (length = n_paths)
    - expected_mae: float, mean MAE across all paths (Monte Carlo expectation)
    """

    # Drop rows in df_test where rate is NaN
    df_eval = df_test[[rate_col]].dropna().copy()

    # Find overlapping dates between df_test and forecast paths
    eval_dates = df_eval.index.intersection(df_paths.columns)

    # Extract true values only for valid dates
    y_true = df_eval.loc[eval_dates, rate_col].values

    # Evaluate RMSE and MAE per path
    rmse_values = []
    mae_values = []
    for i in range(df_paths.shape[0]):
        y_pred = df_paths.loc[i, eval_dates].values
        rmse_values.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_values.append(mean_absolute_error(y_true, y_pred))

    rmse_values = np.array(rmse_values)
    mae_values = np.array(mae_values)

    expected_rmse = rmse_values.mean()
    expected_mae = mae_values.mean()

    return rmse_values, expected_rmse, mae_values, expected_mae
