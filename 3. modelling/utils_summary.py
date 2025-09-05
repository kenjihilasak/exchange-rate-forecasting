import numpy as np
import pandas as pd

def pack_metrics(metrics: dict, model_name: str, horizon: int | str) -> dict:
    """
    Normalize an evaluation dict into a single summary row.

    Expected keys it will try to read:
      - 'rmse(%)', 'mae(%)' (required for the summary)
    """
    row = {
        "model_name": model_name,
        "horizon": horizon,
        "rmse(%)": metrics.get("rmse(%)", np.nan),
        "mae(%)": metrics.get("mae(%)", np.nan),
    }
    # Include ensemble metrics if they exist (harmless if not)
    for k in ["ensemble_mean_rmse","ensemble_mean_mae","ensemble_median_rmse","ensemble_median_mae"]:
        if k in metrics:
            row[k] = metrics[k]
    return row

def append_result(summary_df: pd.DataFrame, metrics: dict, model_name: str, horizon: int | str) -> pd.DataFrame:
    """Append or update one normalized row in the running summary table."""
    row = pack_metrics(metrics, model_name, horizon)

    # Asegurar que model_name y horizon están en el DataFrame
    if summary_df.empty or not {"model_name", "horizon"}.issubset(summary_df.columns):
        return pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

    # Convertir tipos para comparación segura
    summary_df["model_name"] = summary_df["model_name"].astype(str)
    summary_df["horizon"] = summary_df["horizon"].astype(str)
    model_name = str(model_name)
    horizon = str(horizon)

    # Buscar coincidencias
    mask = (summary_df["model_name"] == model_name) & (summary_df["horizon"] == horizon)

    if mask.any():
        # Reemplazar métricas existentes
        for key, value in row.items():
            summary_df.loc[mask, key] = value
    else:
        # Añadir nueva fila
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)

    return summary_df

def finalize_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add handy ranking columns and a consistent sort.
    """
    df = summary_df.copy()
    # Rank per horizon (lower is better)
    df["rank_rmse"] = df.groupby("horizon")["rmse(%)"].rank(method="min")
    df["rank_mae"]  = df.groupby("horizon")["mae(%)"].rank(method="min")
    # Nice ordering
    cols_first = ["horizon","model","n_paths","rmse(%)","mae(%)","rank_rmse","rank_mae"]
    other_cols = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + other_cols]
    # Optional: round for display
    num_cols = ["rmse(%)","mae(%)","rmse_mean_over_paths","rmse_std_over_paths",
                "mae_mean_over_paths","mae_std_over_paths",
                "ensemble_mean_rmse","ensemble_mean_mae","ensemble_median_rmse","ensemble_median_mae"]
    for c in num_cols:
        if c in df:
            df[c] = df[c].astype(float).round(6)
    return df
