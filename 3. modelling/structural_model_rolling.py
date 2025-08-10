import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

from structural_model import (
    simulate_structural_ar1_paths_and_plot,
    evaluate_structural_sims
)

FEATURE_COLS = ('m_diff','y_diff','r_diff')
Y_COL = 'log_rate'
RATE_COL = 'rate_interpolated'

def fit_structural(df_train, feature_cols=FEATURE_COLS, y_col=Y_COL):
    """
    Fit OLS structural model on df_train (requires y_col and feature_cols).
    Returns model and residuals.
    """
    # Coerce to numeric early - force it to float here and drop problematic rows before fitting
    cols = [y_col] + list(feature_cols)
    tmp = df_train[cols].apply(pd.to_numeric, errors='coerce').dropna()

    X = sm.add_constant(tmp[list(feature_cols)].astype(float))
    y = tmp[y_col].astype(float)

    model = sm.OLS(y, X).fit()
    residuals = pd.Series(model.resid, index=tmp.index)
    return model, residuals

def build_exog_future_from_master(master_df, origin_date, steps, feature_cols=FEATURE_COLS):
    """
    Build exog_future for [origin_date+1 .. origin_date+steps] using *realized* fundamentals
    already present in master_df (Meese–Rogoff style).
    """
    start = origin_date + pd.Timedelta(days=1)
    end   = origin_date + pd.Timedelta(days=steps)
    forecast_idx = pd.date_range(start=start, end=end, freq='D')
    exog_future = master_df.loc[forecast_idx, list(feature_cols)].copy()

    # Safety checks
    if exog_future.shape[0] != steps:
        missing = steps - exog_future.shape[0]
        raise ValueError(f"exog_future is missing {missing} day(s). Check fundamental coverage in master_df.")

    if exog_future.isna().any().any():
        raise ValueError("exog_future contains NaNs. Ensure fundamentals are populated for the horizon.")

    return exog_future

def build_exog_future_frozen(df_train, steps, feature_cols=FEATURE_COLS):
    """
    Alternative: freeze fundamentals at last observed value.
    """
    last_row = df_train.iloc[[-1]][list(feature_cols)]
    forecast_dates = pd.date_range(df_train.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    exog_future = pd.concat([last_row]*steps, axis=0)
    exog_future.index = forecast_dates
    return exog_future



def rolling_forecast(master_df, cutoffs, steps, n_sims=1000,
                     feature_cols=FEATURE_COLS, use_realized_fundamentals=True,
                     seed=42, verbose=True):
    """
    For each cutoff t, re-fit the structural model on data <= t (where Y and features exist),
    then simulate over next 'steps' days using either realized fundamentals (default) or frozen.
    Returns a list of dicts with results.
    """
    results = []
    for t in cutoffs:
        # TRAIN SET: need y and features
        df_train = master_df.loc[:t].dropna(subset=[Y_COL] + list(feature_cols)).copy()

        # Make sure types are numeric (avoids object dtypes downstream)
        for c in [Y_COL] + list(feature_cols):
            df_train[c] = pd.to_numeric(df_train[c], errors='coerce')
        df_train = df_train.dropna(subset=[Y_COL] + list(feature_cols))
        if df_train.empty:
            if verbose:
                print(f"[{t.date()}] Skipping: not enough clean numeric training data.")
            continue

        # If rate missing in train, derive from log_rate for plotting
        if RATE_COL not in df_train.columns or df_train[RATE_COL].isna().any():
            df_train[RATE_COL] = np.exp(df_train[Y_COL])

        if df_train.empty:
            if verbose:
                print(f"[{t.date()}] Skipping: not enough training data.")
            continue

        # Fit structural model at this origin
        model, residuals = fit_structural(df_train, feature_cols=feature_cols, y_col=Y_COL)
        print(residuals)

        # Build exog_future for the horizon
        if use_realized_fundamentals:
            exog_future = build_exog_future_from_master(master_df, origin_date=t, steps=steps, feature_cols=feature_cols)
        else:
            exog_future = build_exog_future_frozen(df_train, steps=steps, feature_cols=feature_cols)

        # make exog_future numeric too (important after reindex)
        exog_future = exog_future.apply(pd.to_numeric, errors='coerce')
        if exog_future.isna().any().any():
            raise ValueError("exog_future contains NaNs or non-numeric values after coercion.")


        # Build test_df containing realized rate for the horizon (for evaluation/overlay)
        start = t + pd.Timedelta(days=1)
        end   = t + pd.Timedelta(days=steps)
        horizon_idx = pd.date_range(start=start, end=end, freq='D')
        test_df = master_df.loc[horizon_idx, [RATE_COL]].copy()

        # If rate missing in test but log_rate exists, derive it
        if test_df[RATE_COL].isna().any():
            if 'log_rate' in master_df.columns:
                fallback = np.exp(master_df.loc[horizon_idx, 'log_rate'])
                test_df[RATE_COL] = test_df[RATE_COL].fillna(fallback)

        # Simulate
        all_paths, det_log, ar1_params = simulate_structural_ar1_paths_and_plot(
            model=model,
            residuals=residuals,
            df=df_train,
            steps=steps,
            n_sims=n_sims,
            n_show=30,
            rate_col=RATE_COL,
            log_rate_col=Y_COL,
            feature_cols=feature_cols,
            exog_future=exog_future,      # key line: use realized fundamentals
            use_last_resid=True,
            seed=seed,
            test_df=test_df if not test_df.empty else None
        )

        # Evaluate
        metrics = None
        if not test_df.empty and test_df[RATE_COL].notna().any():
            metrics = evaluate_structural_sims(
                all_paths=all_paths,
                test_df=test_df,
                rate_col=RATE_COL,
                steps=steps,
                compute_coverage=True,
                verbose=False
            )

        if verbose:
            print(f"[{t.date()}] φ={ar1_params['phi']:.4f}, σ={ar1_params['sigma']:.6f} "
                  f"| horizon={steps} days | paths={all_paths.shape[1]}")

        results.append({
            'origin': t,
            'model': model,
            'ar1_params': ar1_params,
            'det_log': det_log,
            'all_paths': all_paths,
            'metrics': metrics
        })
    return results