import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

def simulate_structural_ar1_paths_and_plot(
    model,
    residuals,
    df,
    steps,
    n_sims=1000,
    n_show=30,
    rate_col='rate',
    log_rate_col='log_rate',
    feature_cols=('m_diff', 'y_diff', 'r_diff'),
    exog_future=None,
    use_last_resid=True,
    seed=42,
    test_df=None,
    train_end_date=None,   # <-- NUEVO: te permite forzar el fin de train
):
    # ---------- Determinar la fecha final de entrenamiento ----------
    # Prioridad: argumento explícito > último índice no-NaN de residuales > (fallback) df.index.max()
    if train_end_date is not None:
        last_train_date = pd.to_datetime(train_end_date)
    else:
        res_idx = pd.DatetimeIndex(pd.Series(residuals).dropna().index)
        last_train_date = res_idx.max() if len(res_idx) else pd.to_datetime(df.index.max())

    # ---------- Índice de forecast ----------
    forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1),
                                   periods=steps, freq='D')

    # ---------- Determinístico: exógenas futuras ----------
    exog_names = model.model.exog_names  # p.ej. ['const','m_diff','y_diff','r_diff']

    if exog_future is not None:
        # Alinear robustamente por intersección (evita NaN por reindex)
        cand = exog_future.copy()
        cand = cand.apply(pd.to_numeric, errors='coerce')
        valid_dates = pd.DatetimeIndex(forecast_dates).intersection(cand.index)

        if len(valid_dates) == 0:
            raise ValueError("No hay solapamiento de fechas entre forecast_dates y exog_future.")

        if len(valid_dates) < len(forecast_dates):
            print(f"[WARN] Recortando horizonte: {len(valid_dates)} de {len(forecast_dates)} fechas coinciden.")
            forecast_dates = valid_dates
            steps = len(valid_dates)

        exog_f = cand.loc[forecast_dates, list(feature_cols)].copy()
    else:
        # Mantener constantes los últimos fundamentales observados en *train_end_date*
        last_row = df.loc[[last_train_date], list(feature_cols)]
        exog_f = pd.concat([last_row] * steps, axis=0)
        exog_f.index = forecast_dates

    # Constante y orden de columnas como en el modelo
    if 'const' in exog_names and 'const' not in exog_f.columns:
        exog_f.insert(0, 'const', 1.0)
    exog_f = exog_f[exog_names]

    # Convertir a numérico y sanity check
    exog_f = exog_f.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    if exog_f.isna().any().any():
        bad_cols = exog_f.columns[exog_f.isna().any()].tolist()
        bad_rows = exog_f.index[exog_f.isna().any(axis=1)]
        raise ValueError(f"exog_f aún tiene NaN en {bad_cols}. Ejemplos: {list(bad_rows[:5])}")

    det_path_log = pd.Series(model.predict(exog_f), index=forecast_dates, name='det_log_rate')

    # ---------- AR(1) residuals (igual que antes) ----------
    residuals = pd.to_numeric(pd.Series(residuals), errors='coerce').dropna()
    if residuals.size < 10:
        raise ValueError(f"Not enough numeric residuals for AR(1): got {residuals.size}")

    ar1 = AutoReg(residuals, lags=1, old_names=False).fit()
    phi = ar1.params[1]
    sigma = float(np.std(ar1.resid, ddof=1))
    ar1_params = {'phi': float(phi), 'sigma': float(sigma)}

    rng = np.random.default_rng(seed)
    n_days = steps
    n_paths = n_sims
    u0 = residuals.iloc[-1] if use_last_resid else 0.0

    eps = rng.normal(loc=0.0, scale=sigma, size=(n_days, n_paths))
    u = np.zeros((n_days, n_paths))
    u[0, :] = phi * u0 + eps[0, :]
    for t in range(1, n_days):
        u[t, :] = phi * u[t-1, :] + eps[t, :]

    sim_log = np.tile(det_path_log.values.reshape(-1, 1), (1, n_paths)) + u
    sim_rate = np.exp(sim_log)

    # Última tasa observada en el *fin de train*
    if rate_col in df.columns and last_train_date in df.index:
        last_rate = df.loc[last_train_date, rate_col]
        if pd.isna(last_rate) and log_rate_col in df.columns:
            last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))
    else:
        last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))

    all_paths = pd.DataFrame(sim_rate, index=pd.DatetimeIndex(forecast_dates),
                             columns=[f'path_{i+1}' for i in range(n_paths)])

    # ---------- Plot ----------
    plt.figure(figsize=(14, 6))
    # solo la parte de training para la línea azul
    if rate_col in df.columns:
        train_slice = df.loc[:last_train_date, rate_col].dropna()
        plt.plot(train_slice.index[-30:], train_slice.values[-30:], linewidth=2, label='Train (rate)')
    else:
        train_slice = np.exp(df.loc[:last_train_date, log_rate_col]).dropna()
        plt.plot(train_slice.index[-30:], train_slice.values[-30:], linewidth=2, label='Train (rate)')

    # test (si quieres verlo sobre el forecast)
    if test_df is not None and rate_col in test_df.columns:
        mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
        test_period = test_df.loc[mask]
        if len(test_period) > 0:
            plt.plot(test_period.index, test_period[rate_col], color='orange', linewidth=2, label='Test (forecast period)')

    plt.axvline(x=last_train_date, color='gray', linestyle=':', alpha=0.6, label='Forecast start')

    to_show = min(n_show, n_paths)
    first_label_added = False
    for col in all_paths.columns[:to_show]:
        series_with_anchor = pd.concat([pd.Series([last_rate], index=[last_train_date]), all_paths[col]])
        if not first_label_added:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5, label='Simulated paths')
            first_label_added = True
        else:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5)

    plt.title(f'Structural model + AR(1) residuals — {steps}-day simulations ({n_paths} paths)')
    plt.xlabel('Date'); plt.ylabel('Rate'); plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.show()

    return all_paths, det_path_log, ar1_params


# def simulate_structural_ar1_paths_and_plot(
#     model,
#     residuals,
#     df,
#     steps,
#     n_sims=1000,
#     n_show=30,
#     rate_col='rate',
#     log_rate_col='log_rate',
#     feature_cols=('m_diff', 'y_diff', 'r_diff'),
#     exog_future=None,           # optional DataFrame of fundamentals for forecast window (daily)
#     use_last_resid=True,        # start AR(1) from last residual instead of 0
#     seed=42,
#     test_df=None                # optional test set DataFrame with a 'rate' column
# ):
#     """
#     Simulate multiple paths from a structural model with AR(1) residuals and plot.

#     Parameters
#     ----------
#     model : statsmodels.regression.linear_model.RegressionResults
#         Fitted OLS structural model: log_rate ~ const + feature_cols.
#     residuals : pd.Series
#         OLS residuals aligned with df index.
#     df : pd.DataFrame
#         Training DataFrame containing `log_rate_col` and `feature_cols`, indexed by date (daily).
#     steps : int
#         Forecast horizon in days (e.g., 31, 180, 366, 730).
#     n_sims : int, default 1000
#         Number of simulated paths.
#     n_show : int, default 30
#         Number of paths to display in the plot.
#     rate_col : str, default 'rate'
#         Name of the spot rate column in train_df/test_df (used only for plotting).
#     log_rate_col : str, default 'log_rate'
#         Name of the log-rate column in df (dependent var of the OLS).
#     feature_cols : tuple[str], default ('m_diff','y_diff','r_diff')
#         Names of the structural regressors used to fit the model.
#     exog_future : pd.DataFrame | None
#         If provided, must have daily rows for the forecast window and columns = feature_cols.
#         If None, the last observed row of features is held constant over the horizon.
#     use_last_resid : bool, default True
#         Initialize AR(1) with the last residual u_T. If False, initialize at 0.
#     seed : int, default 42
#         Random seed for reproducibility.
#     test_df : pd.DataFrame | None
#         Optional test DataFrame (indexed by date) with a `rate` column to overlay on the plot.

#     Returns
#     -------
#     all_paths : pd.DataFrame
#         DataFrame of simulated spot-rate paths; columns = path_1..path_n, index = forecast dates.
#     det_path_log : pd.Series
#         Deterministic (model-based) log-rate path over the forecast window.
#     ar1_params : dict
#         {'phi': float, 'sigma': float} for the AR(1) residual process.
#     """

#     # ---------- Build forecast date index ----------
#     last_train_date = df.index[-1]
#     forecast_dates = pd.date_range(start=last_train_date, periods=steps+1, freq='D')[1:]  # t+1..t+steps

#     # ---------- Deterministic component: prepare exog for the forecast window ----------
#     exog_names = model.model.exog_names  # e.g., ['const','m_diff','y_diff','r_diff']
#     k = len(feature_cols)

#     if exog_future is not None:
#         # Validate provided exog_future
#         if not all(col in exog_future.columns for col in feature_cols):
#             missing = [c for c in feature_cols if c not in exog_future.columns]
#             raise ValueError(f"exog_future is missing required columns: {missing}")
#         print(">> DEBUG shapes")
#         print("last_train_date:", last_train_date)
#         print("steps:", steps)
#         print("forecast_dates[0], [-1]:", forecast_dates[0], forecast_dates[-1])

#         print("\nexog_future window:")
#         print("exog_future index[0], [-1]:", exog_future.index[0], exog_future.index[-1])
#         print("exog_future shape:", exog_future.shape)

#         missing_in_future = forecast_dates.difference(exog_future.index)
#         missing_in_fcidx  = exog_future.index.difference(forecast_dates)
#         print("\nFechas en forecast_dates que NO están en exog_future:", len(missing_in_future))
#         print(list(missing_in_future[:5]))
#         print("Fechas en exog_future que NO están en forecast_dates:", len(missing_in_fcidx))
#         print(list(missing_in_fcidx[:5]))

#         # Align to forecast dates and keep only needed cols
#         exog_f = exog_future.reindex(forecast_dates)[list(feature_cols)].copy()
#     else:
#         # Hold last observed fundamentals constant across the horizon
#         last_row = df.loc[[last_train_date], list(feature_cols)]
#         exog_f = pd.concat([last_row] * steps, axis=0)
#         exog_f.index = forecast_dates

#     # Add constant exactly as the model expects (same column order)
#     if 'const' in exog_names and 'const' not in exog_f.columns:
#         exog_f.insert(0, 'const', 1.0)

#     # Reorder columns to match the model's exog order
#     exog_f = exog_f[exog_names]

#     # ensure pure float to avoid object dtypes during model.predict
#     exog_f = exog_f.apply(pd.to_numeric, errors='coerce')
#     if exog_f.isna().any().any():
#         bad_cols = exog_f.columns[exog_f.isna().any()].tolist()
#         raise ValueError(f"exog_future contains non-numeric/NaN after coercion in columns: {bad_cols}")
#     det_path_log = pd.Series(model.predict(exog_f), index=forecast_dates, name='det_log_rate')

#     # Deterministic log-rate path from the structural model
#     det_path_log = pd.Series(model.predict(exog_f), index=forecast_dates, name='det_log_rate')

#     # ---------- Prepare residuals for AR(1) ----------
#     residuals = pd.Series(residuals).copy()
#     # force to float, coerce anything weird to NaN, then drop
#     residuals = pd.to_numeric(residuals, errors='coerce').dropna()

#     # guard: need enough points for AR(1)
#     if residuals.size < 10:
#         raise ValueError(f"Not enough numeric residuals for AR(1): got {residuals.size}")

#     # ---------- AR(1) on residuals ----------
#     ar1 = AutoReg(residuals, lags=1, old_names=False).fit()
#     phi = ar1.params[1]  # AR(1) coefficient
#     sigma = np.std(ar1.resid, ddof=1)  # innovation std dev
#     ar1_params = {'phi': float(phi), 'sigma': float(sigma)}

#     # ---------- Simulate residual paths ----------
#     rng = np.random.default_rng(seed)
#     n_days = steps
#     n_paths = n_sims

#     # Initial residual state
#     u0 = residuals.iloc[-1] if use_last_resid else 0.0

#     # Simulate AR(1) shocks for all paths at once
#     eps = rng.normal(loc=0.0, scale=sigma, size=(n_days, n_paths))
#     u = np.zeros((n_days, n_paths))
#     u[0, :] = phi * u0 + eps[0, :]
#     for t in range(1, n_days):
#         u[t, :] = phi * u[t-1, :] + eps[t, :]

#     # ---------- Combine deterministic + stochastic in LOG space, then back-transform ----------
#     det_matrix = np.tile(det_path_log.values.reshape(-1, 1), (1, n_paths))  # (steps x n_paths)
#     sim_log = det_matrix + u
#     sim_rate = np.exp(sim_log)  # back to rate level

#     # Also prepend the last observed rate (for plotting continuity)
#     last_rate = np.exp(df[log_rate_col].iloc[-1])
#     all_paths = pd.DataFrame(index=pd.DatetimeIndex(forecast_dates), data=sim_rate, columns=[f'path_{i+1}' for i in range(n_paths)])

#     # ---------- Plot ----------
#     plt.figure(figsize=(14, 6))
#     # Plot last 30 days of training rate (if available)
#     if rate_col in df.columns:
#         plt.plot(df.index[-30:], df[rate_col].iloc[-30:], color='blue', linewidth=2, label='Train (rate)')
#     else:
#         # fallback: derive from log_rate
#         plt.plot(df.index[-30:], np.exp(df[log_rate_col]).iloc[-30:], color='blue', linewidth=2, label='Train (rate)')

#     # Plot test set overlapping forecast window (optional)
#     if test_df is not None and rate_col in test_df.columns:
#         mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
#         test_period = test_df.loc[mask]
#         if len(test_period) > 0:
#             plt.plot(test_period.index, test_period[rate_col], color='orange', linewidth=2, label='Test (forecast period)')

#     # Vertical line at forecast start
#     plt.axvline(x=last_train_date, color='gray', linestyle=':', alpha=0.6, label='Forecast start')

#     # Plot simulated paths (show up to n_show)
#     to_show = min(n_show, n_paths)
#     first_label_added = False
#     for col in all_paths.columns[:to_show]:
#         series_with_anchor = pd.concat([pd.Series([last_rate], index=[last_train_date]),all_paths[col]])
#         if not first_label_added:
#             plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5, label='Simulated paths')
#             first_label_added = True
#         else:
#             plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5)

#     plt.title(f'Structural model + AR(1) residuals — {steps}-day simulations ({n_paths} paths)')
#     plt.xlabel('Date')
#     plt.ylabel('Rate')
#     plt.legend()
#     plt.grid(True, alpha=0.2)
#     plt.tight_layout()
#     plt.show()

#     return all_paths, det_path_log, ar1_params

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_structural_sims(
    all_paths: pd.DataFrame,
    test_df: pd.DataFrame,
    rate_col: str = 'rate',
    steps: int | None = None,
    verbose: bool = True
):
    """
    Evaluate simulated paths against ground truth.

    Parameters
    ----------
    all_paths : pd.DataFrame
        Columns = simulated paths (e.g., 'path_1', 'path_2', ...).
        Index = forecast dates (daily).
    test_df : pd.DataFrame
        Must contain a column `rate_col`. Index should be daily dates.
    rate_col : str
        Name of the column with the realized series in `test_df`.
    steps : int or None
        If provided, only evaluate the first `steps` forecast days
        (starting from the first index of `all_paths`).
    verbose : bool
        If True, print a compact summary.

    Returns
    -------
    metrics : dict
        {
            'dates': DatetimeIndex of evaluated dates,
            'n_paths': int,
            'path_names': list[str],
            'rmse_list': list[float],        # per-path RMSE
            'mae_list': list[float],         # per-path MAE
            'expected_rmse': float,          # mean of per-path RMSEs
            'expected_mae': float            # mean of per-path MAEs
        }
    """
    # --- Align dates ---
    test_series = test_df[[rate_col]].dropna().copy()
    common_dates = test_series.index.intersection(all_paths.index)
    if len(common_dates) == 0:
        if verbose:
            print("⚠️ No overlapping dates between predictions and ground truth.")
        return {
            'dates': pd.DatetimeIndex([]),
            'n_paths': 0,
            'path_names': [],
            'rmse_list': [],
            'mae_list': [],
            'expected_rmse': np.nan,
            'expected_mae': np.nan
        }

    # If steps specified, trim to the requested horizon starting at the first forecast date
    if steps is not None:
        start_date = all_paths.index[0]
        horizon = pd.date_range(start=start_date, periods=steps, freq='D')
        eval_dates = common_dates.intersection(horizon)
    else:
        eval_dates = common_dates

    if len(eval_dates) == 0:
        if verbose:
            print("⚠️ After applying 'steps', there are no dates to evaluate.")
        return {
            'dates': pd.DatetimeIndex([]),
            'n_paths': 0,
            'path_names': [],
            'rmse_list': [],
            'mae_list': [],
            'expected_rmse': np.nan,
            'expected_mae': np.nan
        }

    # Ground truth vector
    y_true = test_series.loc[eval_dates, rate_col].values

    # --- Per-path metrics ---
    # Keep only columns (paths) that have no missing values over eval_dates
    sim_cols = [c for c in all_paths.columns if all_paths.loc[eval_dates, c].notna().all()]

    rmse_values, mae_values = [], []
    for c in sim_cols:
        y_pred = all_paths.loc[eval_dates, c].values
        rmse_values.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
        mae_values.append(float(mean_absolute_error(y_true, y_pred)))

    # Convert to arrays for stable means
    rmse_values = np.array(rmse_values, dtype=float) if len(rmse_values) else np.array([])
    mae_values  = np.array(mae_values,  dtype=float) if len(mae_values)  else np.array([])

    expected_rmse = float(rmse_values.mean()) if rmse_values.size else np.nan
    expected_mae  = float(mae_values.mean())  if mae_values.size  else np.nan

    metrics = {
        'dates': eval_dates,
        'n_paths': len(sim_cols),
        'path_names': sim_cols,
        'rmse_list': rmse_values.tolist(),
        'mae_list': mae_values.tolist(),
        'expected_rmse': expected_rmse,
        'expected_mae': expected_mae
    }

    if verbose:
        print(f"Evaluated dates: {len(eval_dates)}")
        print(f"Evaluated paths: {metrics['n_paths']} of {len(all_paths.columns)}")
        if metrics['n_paths'] > 0:
            print(f"✅ Expected RMSE (mean over paths): {expected_rmse:.6f}")
            print(f"✅ Expected MAE  (mean over paths): {expected_mae:.6f}")
        else:
            print("⚠️ No valid paths (all had NaNs on eval dates).")

    return metrics
