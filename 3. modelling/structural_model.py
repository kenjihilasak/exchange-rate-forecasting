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
    base_seed=42,          # <- base seed; path i uses base_seed + i
    seed=None,             # <- deprecated alias; if provided, overrides base_seed
    test_df=None,
    train_end_date=None,   # allows forcing the end of training explicitly
):
    """
    Simulate exchange-rate paths from a structural deterministic path plus AR(1) residuals.
    The deterministic path comes from `model.predict(exog_future)`. Residuals follow:
        u_t = phi * u_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)
    Each path i uses a deterministic seed = base_seed + i for reproducibility.

    Parameters
    ----------
    model : statsmodels result (e.g., OLS/GLS/IV) with in-sample fit and exog names
    residuals : array-like. In-sample residuals from the structural model (index must be datetime or alignable).
    df : pd.DataFrame. Full data with at least `feature_cols` and `log_rate_col` / `rate_col`.
    steps : int. Number of days to forecast.
    n_sims : int. Number of simulated paths.
    n_show : int. Number of paths to plot.
    rate_col : str. Column with rate in levels (if available).
    log_rate_col : str. Column with log rate.
    feature_cols : tuple[str]. Names of exogenous features the structural model expects, in training order.
    exog_future : pd.DataFrame or None. Future exogenous features indexed by date. If None, holds last observed fundamentals constant.
    use_last_resid : bool. Initialize AR(1) recursion with last observed residual (True) or zero (False).
    base_seed : int. Base seed so that path i uses np.random.seed(base_seed + i).
    seed : int or None. Deprecated alias for base_seed. If provided, overrides base_seed.
    test_df : pd.DataFrame or None. Optional test set to overlay on the plot (must contain `rate_col`).
    train_end_date : str|pd.Timestamp|None. Force the training end date (used to anchor the plot and future exog snapshot).

    Returns
    -------
    all_paths : pd.DataFrame. Simulated rate paths (levels) with columns path_1..path_n.
    det_path_log : pd.Series. Deterministic log-rate path implied by the structural model (length == steps).
    ar1_params : dict. {'phi': float, 'sigma': float} estimated from residuals.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.ar_model import AutoReg

    # --- determine last train date ---
    if train_end_date is not None:
        last_train_date = pd.to_datetime(train_end_date)
    else:
        res_idx = pd.DatetimeIndex(pd.Series(residuals).dropna().index)
        last_train_date = res_idx.max() if len(res_idx) else pd.to_datetime(df.index.max())

    # --- forecast index ---
    forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1),
                                   periods=steps, freq='D')

    # --- deterministic path via exogenous fundamentals ---
    exog_names = model.model.exog_names  # e.g., ['const', 'm_diff', 'y_diff', 'r_diff']

    if exog_future is not None:
        cand = exog_future.copy()
        cand = cand.apply(pd.to_numeric, errors='coerce')
        valid_dates = pd.DatetimeIndex(forecast_dates).intersection(cand.index)

        if len(valid_dates) == 0:
            raise ValueError("No overlapping dates between forecast_dates and exog_future index.")

        if len(valid_dates) < len(forecast_dates):
            print(f"[WARN] Truncating horizon: {len(valid_dates)} of {len(forecast_dates)} dates overlap.")
            forecast_dates = valid_dates
            steps = len(valid_dates)

        exog_f = cand.loc[forecast_dates, list(feature_cols)].copy()
    else:
        # hold last observed fundamentals (at train end) fixed during the horizon
        last_row = df.loc[[last_train_date], list(feature_cols)]
        exog_f = pd.concat([last_row] * steps, axis=0)
        exog_f.index = forecast_dates

    # ensure constant and column order consistent with the trained model
    if 'const' in exog_names and 'const' not in exog_f.columns:
        exog_f.insert(0, 'const', 1.0)
    exog_f = exog_f[exog_names]
    exog_f = exog_f.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    if exog_f.isna().any().any():
        bad_cols = exog_f.columns[exog_f.isna().any()].tolist()
        bad_rows = exog_f.index[exog_f.isna().any(axis=1)]
        raise ValueError(f"exog_f still has NaN in {bad_cols}. Examples: {list(bad_rows[:5])}")

    det_path_log = pd.Series(model.predict(exog_f),
                             index=forecast_dates, name='det_log_rate')

    # --- AR(1) on residuals to capture serial correlation ---
    residuals = pd.to_numeric(pd.Series(residuals), errors='coerce').dropna()
    if residuals.size < 10:
        raise ValueError(f"Not enough numeric residuals for AR(1): got {residuals.size}")

    ar1 = AutoReg(residuals, lags=1, old_names=False).fit()
    phi = ar1.params[1]
    sigma = float(np.std(ar1.resid, ddof=1))
    ar1_params = {'phi': float(phi), 'sigma': float(sigma)}

    n_days = steps
    n_paths = n_sims
    u0 = float(residuals.iloc[-1]) if use_last_resid else 0.0

    # --- generate paths with per-path seeding: seed = base_seed + i ---
    u = np.zeros((n_days, n_paths), dtype=float)

    for i in range(n_paths):
        np.random.seed(base_seed + i)  # deterministic seed for path i
        eps_i = np.random.normal(loc=0.0, scale=sigma, size=n_days)

        # AR(1) recursion per path
        u[0, i] = phi * u0 + eps_i[0]
        for t in range(1, n_days):
            u[t, i] = phi * u[t - 1, i] + eps_i[t]

    # combine deterministic log path with residual AR(1) innovations
    det_mat = det_path_log.values.reshape(-1, 1)  # (n_days, 1)
    sim_log = det_mat + u
    sim_rate = np.exp(sim_log)

    # anchor with the last observed rate at train end
    if (rate_col in df.columns) and (last_train_date in df.index):
        last_rate = df.loc[last_train_date, rate_col]
        if pd.isna(last_rate) and (log_rate_col in df.columns):
            last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))
    else:
        last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))

    all_paths = pd.DataFrame(sim_rate, index=pd.DatetimeIndex(forecast_dates),
                             columns=[f'path_{i+1}' for i in range(n_paths)])

    # --- plotting ------
    import matplotlib.dates as mdates
    plt.figure(figsize=(14, 6))

    # training slice (last 30 obs)
    if rate_col in df.columns:
        train_slice = df.loc[:last_train_date, rate_col].dropna()
        y_train = train_slice.values
        x_train = train_slice.index
    else:
        train_slice = np.exp(df.loc[:last_train_date, log_rate_col]).dropna()
        y_train = train_slice.values
        x_train = train_slice.index

    plt.plot(x_train[-30:], y_train[-30:], linewidth=2, label='Train (rate)')

    # optional test overlay
    if (test_df is not None) and (rate_col in test_df.columns):
        mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
        test_period = test_df.loc[mask]
        if len(test_period) > 0:
            plt.plot(test_period.index, test_period[rate_col], color='orange', linewidth=2,
                     label='Test (forecast period)')

    plt.axvline(x=last_train_date, color='gray', linestyle=':', alpha=0.6, label='Forecast start')

    # show subset of paths, anchored at last observed rate
    to_show = min(n_show, n_paths)
    first_label_added = False
    for col in all_paths.columns[:to_show]:
        series_with_anchor = pd.concat(
            [pd.Series([last_rate], index=[last_train_date]), all_paths[col]]
        )
        if not first_label_added:
            plt.plot(series_with_anchor.index, series_with_anchor.values,
                     color='gray', alpha=0.5, label='Simulated paths')
            first_label_added = True
        else:
            plt.plot(series_with_anchor.index, series_with_anchor.values,
                     color='gray', alpha=0.5)

    plt.title(f'Structural model + AR(1) residuals — {steps}-day simulations ({n_paths} paths)')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    return all_paths, det_path_log, ar1_params

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
