import numpy as np
import pandas as pd

def pack_metrics(metrics: dict, model_name: str, horizon: int | str) -> dict:
    """
    Normalize an evaluation dict into a single summary row.
    Works with outputs from evaluate_simulations / evaluate_structural_sims / rmse.py.

    Expected keys it will try to read:
      - 'expected_rmse', 'expected_mae' (required for the summary)
      - 'rmse_list', 'mae_list', 'n_paths' (optional but useful)
      - Optional ensemble keys if present:
          'ensemble_mean_rmse', 'ensemble_mean_mae',
          'ensemble_median_rmse', 'ensemble_median_mae'
    """
    row = {
        "model": model_name,
        "horizon": horizon,
        "expected_rmse": metrics.get("expected_rmse", np.nan),
        "expected_mae": metrics.get("expected_mae", np.nan),
    }
    # Include ensemble metrics if they exist (harmless if not)
    for k in ["ensemble_mean_rmse","ensemble_mean_mae","ensemble_median_rmse","ensemble_median_mae"]:
        if k in metrics:
            row[k] = metrics[k]
    return row

def append_result(summary_df: pd.DataFrame, metrics: dict, model_name: str, horizon: int | str) -> pd.DataFrame:
    """Append one normalized row to the running summary table."""
    row = pack_metrics(metrics, model_name, horizon)
    return pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

def finalize_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add handy ranking columns and a consistent sort.
    """
    df = summary_df.copy()
    # Rank per horizon (lower is better)
    df["rank_rmse"] = df.groupby("horizon")["expected_rmse"].rank(method="min")
    df["rank_mae"]  = df.groupby("horizon")["expected_mae"].rank(method="min")
    # Nice ordering
    cols_first = ["horizon","model","n_paths","expected_rmse","expected_mae","rank_rmse","rank_mae"]
    other_cols = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + other_cols]
    # Optional: round for display
    num_cols = ["expected_rmse","expected_mae","rmse_mean_over_paths","rmse_std_over_paths",
                "mae_mean_over_paths","mae_std_over_paths",
                "ensemble_mean_rmse","ensemble_mean_mae","ensemble_median_rmse","ensemble_median_mae"]
    for c in num_cols:
        if c in df:
            df[c] = df[c].astype(float).round(6)
    return df
