import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def simulate_garch_constmean_paths_and_plot_arch(
    df: pd.DataFrame,
    steps: int,
    n_sims: int = 1000,
    n_show: int = 30,
    rate_col: str = 'rate',
    log_rate_col: str = 'log_rate',
    train_end_date: str | pd.Timestamp | None = None,
    dist: str = 't',           # 't' or 'normal'
    base_seed: int = 42,
    test_df: pd.DataFrame | None = None,
    title_suffix: str = '',
    burn: int = 0              # try e.g. 200 if you want a warm-up to stationary variance
):
    """
    Simulate price paths using arch_model.simulate() for a GARCH(1,1) with constant mean.
    Returns:
        all_paths (levels), garch_params (dict)
    """
    # --- Train end ---
    if train_end_date is not None:
        last_train_date = pd.to_datetime(train_end_date)
    else:
        last_train_date = pd.to_datetime(df.index.max())

    work = df.copy()
    if log_rate_col not in work.columns:
        if rate_col not in work.columns:
            raise ValueError(f"Neither '{log_rate_col}' nor '{rate_col}' found in df.")
        work[rate_col] = pd.to_numeric(work[rate_col], errors='coerce').ffill().bfill()
        work[log_rate_col] = np.log(work[rate_col])

    # raw log-returns
    work['ret'] = work[log_rate_col].diff()
    y_train = work.loc[:last_train_date, 'ret'].dropna()
    if y_train.size < 50:
        raise ValueError(f"Need at least ~50 return observations; got {y_train.size}.")

    # ---------- FIX THE DRIFT: mu = mean of returns ----------
    mu_hat = float(y_train.mean())
    y_centered = y_train - mu_hat  # center so we can fit mean='Zero'

    # --- Fit GARCH(1,1) with ZERO mean (since we centered the data) ---
    am = arch_model(
        y_centered,
        mean='Zero',
        vol='GARCH',
        p=1, q=1,
        dist='t' if dist.lower().startswith('t') else 'normal'
    )
    res = am.fit(disp='off')
    params = res.params
    print(f'res.summary: {res.summary()}')
    # extract params
    def _get_param_like(prefix: str, fallback=np.nan):
        for k in params.index:
            if k.lower().startswith(prefix.lower()):
                return float(params[k])
        return float(fallback)

    # report mu as the fixed (external) drift
    mu    = mu_hat
    omega = float(params.get('omega', np.nan))
    alpha = _get_param_like('alpha', np.nan)
    beta  = _get_param_like('beta',  np.nan)
    dof   = float(params.get('nu', np.nan)) if dist.lower().startswith('t') else None

    if np.isfinite(alpha) and np.isfinite(beta):
        print(f"Volatility persistence (alpha + beta): {alpha + beta:.4f}")

    garch_params = {'mu': mu, 'omega': omega, 'alpha': alpha, 'beta': beta}
    if dist.lower().startswith('t') and np.isfinite(dof):
        garch_params['dof'] = dof
    print(f"GARCH(1,1) params: {garch_params}")

    # --- Last log level and forecast dates ---
    last_log = float(work.loc[last_train_date, log_rate_col])
    forecast_dates = pd.date_range(
        start=last_train_date + pd.Timedelta(days=1), periods=steps, freq='D'
    )

    # --- Single global seed (no rng per path, no 'errors' kwarg) ---
    np.random.seed(base_seed)

    # --- Simulate each path ---
    paths_log = np.zeros((n_sims, steps + 1), dtype=float)
    paths_log[:, 0] = last_log

    for i in range(n_sims):
        # arch returns simulated returns in the same units as y_centered (zero-mean)
        sim_i = am.simulate(
            params=res.params,
            nobs=steps,
            burn=burn,
            initial_value=None
        )
        r_path_centered = np.asarray(sim_i['data'], dtype=float)   # zero-mean log-returns
        r_path = mu_hat + r_path_centered                          # add fixed drift mu
        x_path = last_log + np.cumsum(r_path)
        paths_log[i, 1:] = x_path

    sim_rate = np.exp(paths_log)

    all_paths = pd.DataFrame(
        sim_rate[:, 1:].T,  # (steps, n_sims)
        index=forecast_dates,
        columns=[f'path_{i+1}' for i in range(n_sims)]
    )

    # --- Plot ---
    plt.figure(figsize=(14, 6))
    train_slice = np.exp(work.loc[:last_train_date, log_rate_col]).dropna()
    to_plot_train = train_slice.iloc[-30:] if len(train_slice) > 30 else train_slice
    if len(to_plot_train) > 0:
        plt.plot(to_plot_train.index, to_plot_train.values, linewidth=2, label='Train (rate)')

    if test_df is not None and rate_col in test_df.columns:
        mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
        test_period = test_df.loc[mask, rate_col].dropna()
        if len(test_period) > 0:
            plt.plot(test_period.index, test_period.values, color='orange', linewidth=2, label='Test (forecast period)')

    last_rate_anchor = float(np.exp(last_log))
    to_show = min(n_show, n_sims)
    first = True
    for col in all_paths.columns[:to_show]:
        series_with_anchor = pd.concat(
            [pd.Series([last_rate_anchor], index=[last_train_date]), all_paths[col]]
        )
        if first:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5, label='Simulated paths')
            first = False
        else:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5)

    plt.axvline(x=last_train_date, color='gray', linestyle=':', alpha=0.6, label='Forecast start')
    ttl = f'GARCH(1,1) Constant Mean — {steps}-day simulations ({n_sims} paths, simulate())'
    if title_suffix:
        ttl += f' — {title_suffix}'
    plt.title(ttl)
    plt.xlabel('Date'); plt.ylabel('Rate'); plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.show()

    return all_paths, garch_params


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_garch_sims(
    all_paths: pd.DataFrame,
    test_df: pd.DataFrame,
    rate_col: str = 'rate',
    steps: int | None = None,
    verbose: bool = True
):
    """
    Evaluate simulated paths against ground truth using per-path RMSE/MAE,
    and report the expected (mean) RMSE/MAE across paths.
    """
    # Ground truth (levels)
    test_series = test_df[[rate_col]].dropna().copy()

    # Overlapping dates
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

    # Optional horizon truncation
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

    # Targets
    y_true = test_series.loc[eval_dates, rate_col].values

    # Keep only paths with no NaNs over eval_dates
    sim_cols = [c for c in all_paths.columns if all_paths.loc[eval_dates, c].notna().all()]

    # Per-path metrics
    rmse_values, mae_values = [], []
    for c in sim_cols:
        y_pred = all_paths.loc[eval_dates, c].values
        rmse_values.append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
        mae_values.append(float(mean_absolute_error(y_true, y_pred)))

    rmse_values = np.array(rmse_values, dtype=float) if len(rmse_values) else np.array([])
    mae_values  = np.array(mae_values, dtype=float) if len(mae_values)  else np.array([])

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
            print(f"✅ Expected RMSE(%): {metrics['expected_rmse']*100:.2f}")
            print(f"✅ Expected MAE(%): {metrics['expected_mae']*100:.2f}")
        else:
            print("⚠️ No valid paths (all had NaNs on eval dates).")

    return metrics
