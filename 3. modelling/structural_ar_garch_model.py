# --- New: Structural + AR(1) + GARCH(1,1) simulation ---
from math import dist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

def simulate_structural_ar1_garch_paths_and_plot(
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
    train_end_date=None,
    dist='normal',
):
    # ---------- Determine last training date ----------
    if train_end_date is not None:
        last_train_date = pd.to_datetime(train_end_date)
    else:
        res_idx = pd.DatetimeIndex(pd.Series(residuals).dropna().index)
        last_train_date = res_idx.max() if len(res_idx) else pd.to_datetime(df.index.max())

    # ---------- Forecast dates ----------
    forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1),
                                   periods=steps, freq='D')

    # ---------- Build future exog as in your AR(1) version ----------
    exog_names = model.model.exog_names  # e.g. ['const','m_diff','y_diff','r_diff']

    if exog_future is not None:
        cand = exog_future.copy().apply(pd.to_numeric, errors='coerce')
        valid_dates = pd.DatetimeIndex(forecast_dates).intersection(cand.index)
        if len(valid_dates) == 0:
            raise ValueError("No overlap between forecast_dates and exog_future.")
        if len(valid_dates) < len(forecast_dates):
            print(f"[WARN] Trimming horizon to {len(valid_dates)} dates.")
            forecast_dates = valid_dates
            steps = len(valid_dates)
        exog_f = cand.loc[forecast_dates, list(feature_cols)].copy()
    else:
        last_row = df.loc[[last_train_date], list(feature_cols)]
        exog_f = pd.concat([last_row] * steps, axis=0)
        exog_f.index = forecast_dates

    if 'const' in exog_names and 'const' not in exog_f.columns:
        exog_f.insert(0, 'const', 1.0)
    exog_f = exog_f[exog_names].apply(pd.to_numeric, errors='coerce').ffill().bfill()
    if exog_f.isna().any().any():
        raise ValueError("exog_f still has NaNs after ffill/bfill.")

    det_log = pd.Series(model.predict(exog_f), index=forecast_dates, name='det_log_rate')

    # ---------- Residuals prep ----------
    residuals = pd.to_numeric(pd.Series(residuals), errors='coerce').dropna()
    if residuals.size < 50:
        raise ValueError(f"Need at least ~50 residuals; got {residuals.size}.")
    
    # ---------- Fit AR(1)-GARCH(1,1) ----------
    # u_t = Const + phi * u_{t-1} + ε_t    
    # ε_t = sigma_t * z_t  # where z_t ~ N(0,1)
    # sigma_t^2 = omega + alpha * ε_{t-1}^2 + beta * sigma_{t-1}^2
    am = arch_model(residuals, mean='ARX', lags=1, vol='Garch', p=1, q=1, dist=dist)
    res = am.fit(disp='off')  # fit with starting values

    params = res.params
    print(f'params meanARX and vol GARCH:', params)
    # Keys in your environment:
    const = params.get('Const', params.get('const', params.get('mu', 0.0)))
    #   Const, resid[1], omega, alpha[1], beta[1]
    # Add robust fallbacks for portability (AR[1]/ar.L1, etc.)
    phi = (
        params.get('resid[1]',
            params.get('AR[1]',
                params.get('ar.L1', np.nan)))
    )
    omega = params.get('omega', np.nan)
    alpha_key = next((k for k in params.index if k.lower().startswith('alpha')), None)
    beta_key  = next((k for k in params.index if k.lower().startswith('beta')),  None)

    const = float(const)
    phi   = float(phi) if np.isfinite(phi) else np.nan
    omega = float(omega)
    alpha = float(params[alpha_key]) if alpha_key is not None else np.nan
    beta  = float(params[beta_key])  if beta_key  is not None else np.nan
    
    dof   = float(params.get('nu', np.nan)) if dist == 't' else None

    ar1_params   = {'phi': phi, 'const': const}
    garch_params = {'omega': omega, 'alpha': alpha, 'beta': beta}
    if dof is not None:
        garch_params['dof'] = dof

    print("AR(1) params:", ar1_params)
    print("GARCH(1,1) params:", garch_params)
    if np.isfinite(alpha) and np.isfinite(beta):
        print(f"Volatility persistence (alpha+beta): {alpha + beta:.4f}")

    # ---------- Simulate paths ----------
    # rng = np.random.default_rng(seed)
    np.random.seed(seed)  # control all the path sequences

    last_u = float(residuals.iloc[-1]) if use_last_resid else 0.0
    initial_value = np.array([last_u], dtype=float)

    # Simulate n_sims trajectories one-by-one (no 'repetitions' arg in this arch version)
    u_sims = []
    for i in range(n_sims):
        # np.random.seed(seed + i)  # Common Random Numbers CRN
        sim_i = am.simulate(
            params=res.params,
            nobs=steps,
            initial_value=initial_value,
            burn=0
        )
        # sim_i['data'] shape: (steps,)
        u_sims.append(np.asarray(sim_i['data']))

    # Shape to (steps, n_sims)
    u_sim = np.column_stack(u_sims)

    # Add deterministic structural log path and exponentiate to get rates
    sim_log = det_log.values.reshape(-1, 1) + u_sim
    sim_rate = np.exp(sim_log)

    # Anchor last training rate for nice plotting
    if rate_col in df.columns and last_train_date in df.index:
        last_rate = df.loc[last_train_date, rate_col]
        if pd.isna(last_rate) and log_rate_col in df.columns:
            last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))
    else:
        last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))

    all_paths = pd.DataFrame(
        sim_rate,
        index=pd.DatetimeIndex(forecast_dates),
        columns=[f'path_{i+1}' for i in range(n_sims)]
    )

    # ---------- Plot ----------
    plt.figure(figsize=(14, 6))
    if rate_col in df.columns:
        train_slice = df.loc[:last_train_date, rate_col].dropna()
        plt.plot(train_slice.index[-30:], train_slice.values[-30:], linewidth=2, label='Train (rate)')
    else:
        train_slice = np.exp(df.loc[:last_train_date, log_rate_col]).dropna()
        plt.plot(train_slice.index[-30:], train_slice.values[-30:], linewidth=2, label='Train (rate)')

    if test_df is not None and rate_col in test_df.columns:
        mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
        test_period = test_df.loc[mask]
        if len(test_period) > 0:
            plt.plot(test_period.index, test_period[rate_col], color='orange', linewidth=2, label='Test (forecast period)')

    plt.axvline(x=last_train_date, color='gray', linestyle=':', alpha=0.6, label='Forecast start')

    to_show = min(n_show, n_sims)
    first_label_added = False
    for col in all_paths.columns[:to_show]:
        series_with_anchor = pd.concat([pd.Series([last_rate], index=[last_train_date]), all_paths[col]])
        if not first_label_added:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5, label='Simulated paths')
            first_label_added = True
        else:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5)

    plt.title(f'Structural + AR(1)-GARCH(1,1) — {steps}-day simulations ({n_sims} paths)')
    plt.xlabel('Date'); plt.ylabel('Rate'); plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.show()

    return all_paths, det_log, ar1_params, garch_params


def simulate_structural_arma_garch_paths_and_plot(
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
    train_end_date=None,
    dist='normal',          # 'normal' or 't'
    arma_order=None,        # None -> auto BIC; or tuple (p,q)
    max_ar=8, max_ma=8
):
    """
    Structural deterministic mean (from OLS) + ARMA(p,q) on OLS residuals + GARCH(1,1) on ARMA innovations.
    Returns: all_paths (levels), det_log (Series), arma_params (dict), garch_params (dict).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import arma_order_select_ic
    from statsmodels.tsa.arima.model import ARIMA
    from arch import arch_model

    # ---------- Dates ----------
    if train_end_date is not None:
        last_train_date = pd.to_datetime(train_end_date)
    else:
        res_idx = pd.DatetimeIndex(pd.Series(residuals).dropna().index)
        last_train_date = res_idx.max() if len(res_idx) else pd.to_datetime(df.index.max())

    forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1),
                                   periods=steps, freq='D')

    # ---------- Deterministic structural mean (from OLS) ----------
    exog_names = model.model.exog_names  # e.g., ['const','m_diff','y_diff','r_diff',...]
    if exog_future is not None:
        cand = exog_future.copy().apply(pd.to_numeric, errors='coerce')
        valid_dates = pd.DatetimeIndex(forecast_dates).intersection(cand.index)
        forecast_dates = valid_dates
        exog_f = cand.loc[forecast_dates, list(feature_cols)].copy()
    else:
        last_row = df.loc[[last_train_date], list(feature_cols)]
        exog_f = pd.concat([last_row] * len(forecast_dates), axis=0)
        exog_f.index = forecast_dates

    if 'const' in exog_names and 'const' not in exog_f.columns:
        exog_f.insert(0, 'const', 1.0)
    exog_f = exog_f[exog_names].apply(pd.to_numeric, errors='coerce').ffill().bfill()

    det_log = pd.Series(model.predict(exog_f), index=forecast_dates, name='det_log_rate')

    # ---------- ARMA(p,q) on OLS residuals ----------
    u = pd.Series(residuals).dropna().astype(float)

    if arma_order is None:
        sel = arma_order_select_ic(u, max_ar=max_ar, max_ma=max_ma, ic='bic')
        # --- Print all evaluated BIC values ---
        try:
            # Newer statsmodels: ARMAOrderSelectResults with .bic as a DataFrame
            bic_df = sel.bic.copy()
        except AttributeError:
            # Fallback for older versions (rare). Try to reconstruct a DataFrame.
            # If this fails in your env, remove the fallback and rely on sel.bic only.
            bic_df = pd.DataFrame(sel.results_dict['bic'])
            bic_df.index.name = 'p'
            bic_df.columns.name = 'q'

        # Pretty-print the BIC grid (rows = p, columns = q)
        print("\nBIC grid (rows = p, cols = q):")
        print(bic_df.to_string(float_format=lambda x: f"{x:,.3f}"))

        # Long/tidy format: one row per (p,q)
        bic_long = (
            bic_df
            .stack(dropna=True)          # (p,q) -> BIC
            .rename('bic')
            .reset_index()
        )

        # Normalize column names to 'p' and 'q' no matter what the index/column names are
        bic_long.columns = ['p', 'q', 'bic']

        # Sort by BIC ascending (lower is better)
        bic_long = bic_long.sort_values('bic', ignore_index=True)

        print("\nBIC values by (p,q), sorted (lower is better):")
        for _, r in bic_long.iterrows():
            print(f"  p={int(r['p'])}, q={int(r['q'])} -> BIC={r['bic']:.3f}")

        # Selected order and its BIC
        p, q = sel.bic_min_order
        best_bic = bic_df.loc[p, q]
        print(f"\nSelected by BIC: (p,q)=({p},{q}) with BIC={best_bic:.3f}")
    else:
        p, q = arma_order

    arma_res = ARIMA(u, order=(p, 0, q), trend='c').fit()
    eps_hat = arma_res.resid.dropna()  # ε_t innovations
    print(f'params on ARMA:', arma_res.params)
    c = float(arma_res.params.get('const', 0.0))
    ar_coefs = [float(arma_res.params.get(f'ar.L{i}', 0.0)) for i in range(1, p + 1)]
    ma_coefs = [float(arma_res.params.get(f'ma.L{i}', 0.0)) for i in range(1, q + 1)]
    arma_params = {'const': c, 'ar': ar_coefs, 'ma': ma_coefs, 'order': (p, q)}

    # ---------- GARCH(1,1) on ε_t ----------
    dist = (dist or 'normal').lower()
    am = arch_model(eps_hat, mean='Zero', vol='GARCH', p=1, q=1, dist=dist)
    res_g = am.fit(disp='off')
    params_g = res_g.params
    print(f'params GARCH:', params_g)
    omega = float(params_g['omega'])
    alpha = float(params_g['alpha[1]'])
    beta  = float(params_g['beta[1]'])
    dof   = float(params_g['nu']) if 'nu' in params_g.index else None

    garch_params = {'omega': omega, 'alpha': alpha, 'beta': beta}
    if dof is not None:
        garch_params['nu'] = dof

    # Last filtered state (stable start)
    sigma_last = float(res_g.conditional_volatility.iloc[-1])
    eps_last   = float(res_g.resid.iloc[-1])

    # Initial ARMA buffers
    u_hist = list(u.iloc[-p:]) if p > 0 else []
    e_hist = list(eps_hat.iloc[-q:]) if q > 0 else []

    # ---------- Simulate jointly ----------
    rng = np.random.default_rng(seed)
    if dof is not None:
        Z = rng.standard_t(dof, size=(len(forecast_dates), n_sims))
    else:
        Z = rng.standard_normal(size=(len(forecast_dates), n_sims))

    u_paths = np.zeros((len(forecast_dates), n_sims), dtype=float)

    for j in range(n_sims):
        u_buf = u_hist.copy()
        e_buf = e_hist.copy()
        sig_prev = sigma_last
        eps_prev = e_buf[-1] if q > 0 else eps_last

        for t in range(len(forecast_dates)):
            # GARCH recursion
            sig2 = omega + alpha * (eps_prev ** 2) + beta * (sig_prev ** 2)
            sig = np.sqrt(sig2)
            eps_t = sig * Z[t, j]

            # ARMA recursion: u_t = c + Σ phi_i u_{t-i} + Σ theta_j e_{t-j} + eps_t
            ar_part = 0.0 if p == 0 else sum(ar_coefs[i] * u_buf[-(i + 1)] for i in range(p))
            ma_part = 0.0 if q == 0 else sum(ma_coefs[i] * e_buf[-(i + 1)] for i in range(q))
            u_t = c + ar_part + ma_part + eps_t

            u_paths[t, j] = u_t

            if p > 0:
                u_buf.append(u_t)
                if len(u_buf) > p:
                    u_buf.pop(0)
            if q > 0:
                e_buf.append(eps_t)
                if len(e_buf) > q:
                    e_buf.pop(0)

            sig_prev = sig
            eps_prev = eps_t

    # ---------- Add deterministic mean and go back to levels ----------
    sim_log = det_log.values.reshape(-1, 1) + u_paths
    sim_rate = np.exp(sim_log)

    all_paths = pd.DataFrame(
        sim_rate,
        index=pd.DatetimeIndex(forecast_dates),
        columns=[f'path_{i + 1}' for i in range(n_sims)]
    )

    # ---------- Plot (as requested) ----------
    # last_rate to anchor
    if (rate_col in df.columns) and (last_train_date in df.index):
        last_rate = float(df.loc[last_train_date, rate_col])
        if np.isnan(last_rate) and (log_rate_col in df.columns):
            last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))
    else:
        last_rate = float(np.exp(df.loc[last_train_date, log_rate_col]))

    plt.figure(figsize=(14, 6))
    if rate_col in df.columns:
        train_slice = df.loc[:last_train_date, rate_col].dropna()
        plt.plot(train_slice.index[-30:], train_slice.values[-30:], linewidth=2, label='Train (rate)')
    else:
        train_slice = np.exp(df.loc[:last_train_date, log_rate_col]).dropna()
        plt.plot(train_slice.index[-30:], train_slice.values[-30:], linewidth=2, label='Train (rate)')

    if (test_df is not None) and (rate_col in test_df.columns):
        mask = (test_df.index >= forecast_dates[0]) & (test_df.index <= forecast_dates[-1])
        test_period = test_df.loc[mask]
        if len(test_period) > 0:
            plt.plot(test_period.index, test_period[rate_col], color='orange', linewidth=2, label='Test (forecast period)')

    plt.axvline(x=last_train_date, color='gray', linestyle=':', alpha=0.6, label='Forecast start')

    to_show = min(n_show, n_sims)
    first_label_added = False
    for col in all_paths.columns[:to_show]:
        series_with_anchor = pd.concat([pd.Series([last_rate], index=[last_train_date]), all_paths[col]])
        if not first_label_added:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5, label='Simulated paths')
            first_label_added = True
        else:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color='gray', alpha=0.5)

    title = f'Structural + ARMA({p},{q})-GARCH(1,1) — {steps}-day simulations ({n_sims} paths)'
    plt.title(title)
    plt.xlabel('Date'); plt.ylabel('Rate'); plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout(); plt.show()

    return all_paths, det_log, arma_params, garch_params



import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_structural_sims_basic(
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

    y_true = test_series.loc[eval_dates, rate_col].values
    sim_cols = [c for c in all_paths.columns if all_paths.loc[eval_dates, c].notna().all()]

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
