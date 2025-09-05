# e-ks_ar_forecasting.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from eks_ar_fitting import make_eks_blocks  # where you defined it
# import the model bundles from eks_ar_fitting.py
# from eks_ar_fitting import EKSModel, AR1ResidualModel

# simulators
def _simulate_iid_eps(steps: int, n_paths: int, sigma: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(steps, n_paths))

def _simulate_bootstrap_eps(steps: int, n_paths: int, resid: pd.Series, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vals = resid.dropna().values.astype(float)
    if vals.size == 0:
        raise ValueError("Residual series is empty for bootstrap.")
    idx = rng.integers(low=0, high=vals.size, size=(steps, n_paths))
    return vals[idx]

# -----------------------------
# Helpers to build the sequential h=1 design
# -----------------------------

def _ensure_log_rate(df: pd.DataFrame, rate_col: str, log_rate_col: str) -> pd.DataFrame:
    if log_rate_col not in df.columns or df[log_rate_col].isna().all():
        if rate_col not in df.columns:
            raise ValueError(f"Neither {log_rate_col} nor {rate_col} available in df_all.")
        if (df[rate_col] <= 0).any():
            raise ValueError(f"{rate_col} has non-positive values; cannot take log.")
        df = df.copy()
        df[log_rate_col] = np.log(df[rate_col].astype(float))
    return df

def _add_daily_diffs(X_raw: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    X = X_raw[base_cols].copy()
    for c in base_cols:
        X[f"d_{c}"] = X[c].diff(1)
    return X

def _build_index_seq(df_all: pd.DataFrame, start_date: str, steps: int) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Returns the index for t (features) and the aligned index for t+1 (predictions).
    """
    t_idx = pd.date_range(pd.to_datetime(start_date), periods=steps, freq="D")
    # Predictions are for t+1
    t1_idx = t_idx + pd.Timedelta(days=1)
    # We won't clip here; downstream evaluation will auto-handle missing y_true if t+1 exceeds df_all.
    return t_idx, t1_idx

def build_X_seq_eks(
    df_all: pd.DataFrame,
    *,
    start_date: str,
    steps: int,
    eks_kwargs: Optional[Dict] = None,   # passed to make_eks_blocks
    rate_col: str = "rate_interpolated",
    log_rate_col: str = "log_rate",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Build sequential X_t = [x1_uip,x2_ppp,x3_mf,x4_taylor] for t in [start_date, start_date+steps-1].
    Aligns y_true = Δlog s_{t+1} using the same (rate/log_rate) used at training.
    """
    eks_kwargs = eks_kwargs or {}
    # 1) Full daily features x1..x4, then restrict to the window
    X_eks_full = make_eks_blocks(df_all, **eks_kwargs)   # daily, ffilled, cleaned
    idx_t  = pd.date_range(pd.to_datetime(start_date), periods=steps, freq="D")
    idx_t1 = idx_t + pd.Timedelta(days=1)

    X_raw = X_eks_full.reindex(idx_t)
    # Drop any rows with NaNs (e.g., at very beginning)
    mask = ~X_raw.isna().any(axis=1)
    X_raw = X_raw.loc[mask]
    idx_t  = X_raw.index
    idx_t1 = idx_t + pd.Timedelta(days=1)

    # 2) Target Δlog s from (rate/log_rate)
    if log_rate_col in df_all.columns and df_all[log_rate_col].notna().any():
        s_log_all = pd.to_numeric(df_all[log_rate_col], errors="coerce")
    else:
        s_all = pd.to_numeric(df_all[rate_col], errors="coerce")
        s_log_all = np.log(s_all)

    s_log_t   = s_log_all.reindex(idx_t)
    s_log_tp1 = s_log_all.reindex(idx_t1)
    y_true = pd.Series(s_log_tp1.values - s_log_t.values, index=idx_t, name="dlog_s_true")

    return X_raw, y_true, s_log_t, idx_t, idx_t1

def build_X_seq(
    df_all: pd.DataFrame,
    *,
    start_date: str,
    steps: int,
    feature_cols: List[str],
    include_daily_diffs: bool,
    rate_col: str = "rate_interpolated",
    log_rate_col: str = "log_rate",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
    df = _ensure_log_rate(df_all, rate_col, log_rate_col)

    idx_t = pd.date_range(pd.to_datetime(start_date), periods=steps, freq="D")
    idx_t1 = idx_t + pd.Timedelta(days=1)

    sub = df.reindex(idx_t)

    # Base features
    missing = [c for c in feature_cols if c not in sub.columns]
    if missing:
        raise ValueError(f"Missing feature columns in df_all: {missing}")

    if include_daily_diffs:
        X_raw = _add_daily_diffs(sub, feature_cols)
    else:
        X_raw = sub[feature_cols].copy()

    # --- clave: y_true indexado por t ---
    s_log_all = df[log_rate_col]
    s_log_t   = s_log_all.reindex(idx_t)
    s_log_tp1 = s_log_all.reindex(idx_t1)
    y_true = pd.Series(s_log_tp1.values - s_log_t.values, index=idx_t, name="dlog_s_true")

    # Si los d_ generan NaN en la primera fila, descarta esa fila en todos
    if include_daily_diffs and X_raw.iloc[0].isna().any():
        X_raw  = X_raw.iloc[1:]
        y_true = y_true.loc[X_raw.index]
        s_log_t = s_log_t.loc[X_raw.index]
        idx_t = X_raw.index
        idx_t1 = idx_t + pd.Timedelta(days=1)

    # Drop donde haya NaNs en X y alinear y_true/s_log_t por índice de X
    mask = ~X_raw.isna().any(axis=1)
    X_raw  = X_raw.loc[mask]
    y_true = y_true.loc[X_raw.index]
    s_log_t = s_log_t.loc[X_raw.index]
    idx_t = X_raw.index
    idx_t1 = idx_t + pd.Timedelta(days=1)

    return X_raw, y_true, s_log_t, idx_t, idx_t1

# -----------------------------
# Deterministic deltas with frozen E-KS
# -----------------------------

def predict_deterministic_deltas(
    eks_model,  # EKSModel
    X_raw: pd.DataFrame
) -> pd.Series:
    """
    Apply training-only scaler and frozen betas to get Δlog s_{t+1} (deterministic).
    """
    # Ensure column order and presence
    X_raw = X_raw[eks_model.feature_names]
    X_std = eks_model.scaler.transform(X_raw)
    deltas = eks_model.predict_deltas_stdX(X_std)
    return pd.Series(deltas, index=X_raw.index, name="dlog_s_det")

# -----------------------------
# AR(1) residual simulation
# -----------------------------

@dataclass
class SimulationResult:
    idx_t1: pd.DatetimeIndex             # timestamps for predictions
    deltas_det: pd.Series               # deterministic deltas (shared across paths)
    deltas_paths: pd.DataFrame          # Δlog s_{t+1} per path (columns=path_k)
    log_s_paths: pd.DataFrame           # log s_t paths (starting from s_log at first t)
    s_paths: pd.DataFrame               # level s_t paths (exp(log_s))
    metrics: Dict[str, float]           # expected_rmse, expected_mae (only when y_true provided)
    details: Dict                       # optional extras

def _simulate_ar1_eps(steps: int, n_paths: int, rho: float, sigma: float, seed: Optional[int], eps0: float) -> np.ndarray:
    """
    Simulate eps_t (t=1..steps) for AR(1): eps_t = rho*eps_{t-1} + eta_t, eta ~ N(0,sigma^2).
    eps_0 is provided (e.g., last training residual).
    Returns shape (steps, n_paths).
    """
    rng = np.random.default_rng(seed)
    eps = np.zeros((steps, n_paths), dtype=float)
    eps_prev = np.full(n_paths, eps0, dtype=float)
    for t in range(steps):
        eta_t = rng.normal(loc=0.0, scale=sigma, size=n_paths)
        eps_t = rho * eps_prev + eta_t
        eps[t, :] = eps_t
        eps_prev = eps_t
    return eps

def _build_paths_from_deltas(
    s0_log: float,
    deltas_det: np.ndarray,           # shape (steps,)
    eps_paths: np.ndarray             # shape (steps, n_paths)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given deterministic deltas and eps paths, build log s paths and levels.
    Paths start at s0_log (log s at the first t).
    """
    steps, n_paths = eps_paths.shape
    # total deltas per path
    deltas_total = deltas_det.reshape(steps, 1) + eps_paths  # (steps, n_paths)
    # cumulative sums (over t) to get log s increments
    incr = deltas_total.cumsum(axis=0)
    log_s = s0_log + incr
    s = np.exp(log_s)
    return log_s, s

def _evaluate_expected_metrics_deltas(
    y_true_dlog: pd.Series,        # Δlog s_{t+1} truth indexed by t
    deltas_paths: pd.DataFrame     # same index (t), cols = paths
) -> dict:
    """Expected RMSE/MAE across paths comparing one-step RETURNS (Δlog)."""
    Y = y_true_dlog.dropna()
    if Y.empty:
        return {"expected_rmse": np.nan, "expected_mae": np.nan}
    P = deltas_paths.loc[Y.index]
    err = P.sub(Y.values, axis=0)
    rmse_each = np.sqrt((err**2).mean(axis=0))
    mae_each  = err.abs().mean(axis=0)
    return {
        "expected_rmse": float(rmse_each.mean()),
        "expected_mae":  float(mae_each.mean()),
        "n_paths": int(P.shape[1]),
        "dates": P.index
    }

import numpy as np
import pandas as pd

def _evaluate_expected_metrics_levels_from_df_cols(
    df_all: pd.DataFrame,
    s_paths: pd.DataFrame,          # simulated LEVEL paths indexed by t+1..t+H
    eval_rate_col: str,
    eval_log_rate_col: str
) -> dict:
    """
    Expected RMSE/MAE across paths comparing LEVELS (rate). Truth taken from df_all
    using eval_rate_col (preferred) or exp(eval_log_rate_col) as fallback.
    """
    if eval_rate_col in df_all.columns and df_all[eval_rate_col].notna().any():
        s_true = df_all[eval_rate_col].reindex(s_paths.index)
    else:
        s_true = np.exp(df_all[eval_log_rate_col]).reindex(s_paths.index)

    Y = s_true.dropna()
    idx = Y.index.intersection(s_paths.index)
    if len(idx) == 0:
        return {"expected_rmse": np.nan, "expected_mae": np.nan, "n_paths": 0, "dates": pd.DatetimeIndex([])}

    P = s_paths.loc[idx]
    valid_cols = [c for c in P.columns if P[c].notna().any()]
    if not valid_cols:
        return {"expected_rmse": np.nan, "expected_mae": np.nan, "n_paths": 0, "dates": idx}

    P = P[valid_cols]
    E = P.sub(Y.loc[idx].values, axis=0)
    rmse_each = np.sqrt((E**2).mean(axis=0, skipna=True)).values.astype(float)
    mae_each  = (E.abs().mean(axis=0, skipna=True)).values.astype(float)

    return {
        "expected_rmse": float(np.mean(rmse_each)) if rmse_each.size else np.nan,
        "expected_mae":  float(np.mean(mae_each))  if mae_each.size  else np.nan,
        "n_paths": len(valid_cols),
        "dates": idx
    }

import matplotlib.pyplot as plt

def plot_paths_with_train_test(
    df_all: pd.DataFrame,
    sim,                               # SimulationResult (de forecast_and_simulate_paths)
    *,
    last_train_date,                   # str o pd.Timestamp
    test_df: pd.DataFrame | None = None,
    rate_col: str = "rate_interpolated",
    log_rate_col: str = "log_rate",
    n_show: int = 100,
    train_window_days: int = 30,
    title: str | None = None,
):
    """
    Plot 100 simulated paths (levels) anclados al último valor de train,
    más el tramo final del train y los valores observados del test en la
    ventana de forecast.

    - sim.s_paths: niveles simulados (index = t+1..t+steps)
    - df_all: contiene rate/log_rate (train+test mergeado)
    - test_df: (opcional) datos observados de test para overlay
    """
    last_train_date = pd.to_datetime(last_train_date)

    # 1) Determinar el último rate disponible en train (para anclar paths)
    def _get_series_rate(df, rcol, lcol):
        if rcol in df.columns and df[rcol].notna().any():
            return df[rcol].astype(float)
        elif lcol in df.columns and df[lcol].notna().any():
            return np.exp(df[lcol].astype(float))
        else:
            raise ValueError(f"Neither '{rcol}' nor '{lcol}' found with data.")

    rate_all = _get_series_rate(df_all, rate_col, log_rate_col)

    # último valor de rate en/antes de last_train_date
    last_rate_slice = rate_all.loc[:last_train_date].dropna()
    if last_rate_slice.empty:
        raise ValueError("No training rate found up to last_train_date.")
    last_rate = float(last_rate_slice.iloc[-1])

    # 2) Preparar ventana de entrenamiento (últimos N días)
    train_slice = last_rate_slice.iloc[-train_window_days:]

    # 3) Ventana de forecast y test observado (si se pasa test_df)
    forecast_dates = sim.s_paths.index  # t+1..t+steps
    test_period = None
    if test_df is not None and not test_df.empty:
        test_rate = _get_series_rate(test_df, rate_col, log_rate_col)
        mask = (test_rate.index >= forecast_dates[0]) & (test_rate.index <= forecast_dates[-1])
        test_period = test_rate.loc[mask].dropna()

    # 4) Plot
    plt.figure(figsize=(14, 6))

    # Train (últimos N días)
    if len(train_slice) > 0:
        plt.plot(train_slice.index, train_slice.values, linewidth=2, label="Train (rate)")

    # Línea vertical inicio forecast
    plt.axvline(x=last_train_date, color="gray", linestyle=":", alpha=0.6, label="Forecast start")

    # Test observado en la ventana de forecast (si hay)
    if test_period is not None and len(test_period) > 0:
        plt.plot(test_period.index, test_period.values, linewidth=2, color="orange", label="Test (observed)")

    # Paths simulados (anclados con el último rate de train)
    to_show = min(n_show, sim.s_paths.shape[1])
    first_label_added = False
    for col in sim.s_paths.columns[:to_show]:
        series_with_anchor = pd.concat([pd.Series([last_rate], index=[last_train_date]), sim.s_paths[col]])
        if not first_label_added:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color="gray", alpha=0.45, label=f"Simulated paths (n={to_show})")
            first_label_added = True
        else:
            plt.plot(series_with_anchor.index, series_with_anchor.values, color="gray", alpha=0.45)

    # Título y labels
    steps_eff = sim.details.get("steps_effective", len(forecast_dates))
    n_paths = sim.details.get("n_paths", sim.s_paths.shape[1])
    proc = sim.details.get("error_process", "ar1").upper()
    default_title = f"E-KS + {proc} — {steps_eff}-day simulations ({n_paths} paths)"

    plt.title(title or default_title)
    plt.xlabel("Date"); plt.ylabel("Rate")

    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Top-level: forecast + simulate +  evaluate
# -----------------------------

def forecast_and_simulate_paths(
    df_all: pd.DataFrame,
    *,
    eks_model,                 # EKSModel (frozen)
    ar1_model=None,            # AR1ResidualModel (opcional si iid/none)
    start_date: str,
    steps: int,
    n_paths: int = 1000,
    seed: Optional[int] = None,

    # NEW: choose how to build features for the forecast window
    features_mode: str = "eks",            # "eks" | "custom"
    eks_kwargs: Optional[Dict] = None,     # forwarded to make_eks_blocks

    feature_cols: List[str] = ("m_diff","y_diff","r_diff"),
    include_daily_diffs: bool = True,

    # SERIES for FEATURES (X_t / returns) and eval
    feature_rate_col: str = "rate_interpolated",
    feature_log_rate_col: str = "log_rate",
    eval_rate_col: str = "rate",
    eval_log_rate_col: str = "log_rate",

    # SERIES for anchor from PATH (by default eval_rate_col; fallback a feature)
    anchor_rate_col: Optional[str] = None,

    # ERROR PROCESS
    error_process: str = "ar1",          # "ar1" | "iid" | "bootstrap" | "none"
    sigma_resid: Optional[float] = None, # for "iid" if not ar1_model
    resid_series: Optional[pd.Series] = None,  # for "bootstrap" if not ar1_model
    init_eps: str = "last",              # "last" | "stationary" (only for ar1)

    # METRICS
    metrics_target: str = "levels"       # "levels" | "returns"
) -> SimulationResult:

    # 1) Build sequential X_t and y_true
    if features_mode.lower() == "eks":
        X_raw, y_true_feat, s_log_t_feat, idx_t, idx_t1 = build_X_seq_eks(
            df_all,
            start_date=start_date,
            steps=steps,
            eks_kwargs=eks_kwargs,
            rate_col=feature_rate_col,
            log_rate_col=feature_log_rate_col
        )
        # Optional safety: ensure columns match training order
        missing = [c for c in eks_model.feature_names if c not in X_raw.columns]
        if missing:
            raise ValueError(f"E-KS features missing in forecast window: {missing}")
        # reorder to training order
        X_raw = X_raw[eks_model.feature_names]

    elif features_mode.lower() == "custom":
        X_raw, y_true_feat, s_log_t_feat, idx_t, idx_t1 = build_X_seq(
            df_all,
            start_date=start_date,
            steps=steps,
            feature_cols=list(feature_cols),
            include_daily_diffs=include_daily_diffs,
            rate_col=feature_rate_col,
            log_rate_col=feature_log_rate_col
        )
    else:
        raise ValueError("features_mode must be 'eks' or 'custom'.")

    if X_raw.empty:
        raise ValueError("No valid feature rows in the requested window. Check start_date/steps/data coverage.")

    # 2) Deterministic Δlog from frozen E-KS (features standardized with training scaler)
    deltas_det = predict_deterministic_deltas(eks_model, X_raw)  # index = t
    steps_eff = len(deltas_det)

    # 3) SIMULATON OF ERROR according to 'error_process'
    if error_process == "none":
        eps_paths = np.zeros((steps_eff, n_paths), dtype=float)

    elif error_process == "iid":
        # Si no especificas sigma_resid, reutilizamos la del AR(1) (si la tienes)
        if sigma_resid is None:
            if ar1_model is None:
                raise ValueError("Provide sigma_resid or ar1_model when error_process='iid'.")
            sigma_resid = float(ar1_model.sigma_eta)
        eps_paths = _simulate_iid_eps(steps_eff, n_paths, sigma_resid, seed)

    elif error_process == "bootstrap":
        if resid_series is None:
            if ar1_model is None:
                raise ValueError("Provide resid_series or ar1_model when error_process='bootstrap'.")
            resid_series = ar1_model.resid_series
        eps_paths = _simulate_bootstrap_eps(steps_eff, n_paths, resid_series, seed)

    elif error_process == "ar1":
        if ar1_model is None:
            raise ValueError("ar1_model is required when error_process='ar1'.")
        if init_eps == "last":
            eps0 = float(ar1_model.last_resid)
        elif init_eps == "stationary":
            var = ar1_model.sigma_eta**2 / max(1.0 - ar1_model.rho**2, 1e-12)
            rng0 = np.random.default_rng(seed)
            eps0 = float(rng0.normal(0.0, np.sqrt(var)))
        else:
            raise ValueError("init_eps must be 'last' or 'stationary'.")
        eps_paths = _simulate_ar1_eps(
            steps=steps_eff, n_paths=n_paths, rho=ar1_model.rho,
            sigma=ar1_model.sigma_eta, seed=seed, eps0=eps0
        )

    else:
        raise ValueError("error_process must be 'ar1', 'iid', 'bootstrap', or 'none'.")

    # 4) Choose anchor series (prefer evaluation rate; fallback to feature rate)
    anchor_col = anchor_rate_col or eval_rate_col
    if anchor_col in df_all.columns and df_all[anchor_col].notna().any():
        s0 = float(df_all[anchor_col].reindex([idx_t[0]]).dropna().iloc[0])
        s0_log = float(np.log(s0))
    else:
        s0_log = float(s_log_t_feat.iloc[0])

    log_s_paths_arr, s_paths_arr = _build_paths_from_deltas(
        s0_log=s0_log,
        deltas_det=deltas_det.values,
        eps_paths=eps_paths
    )
    idx_t1_eff = idx_t + pd.Timedelta(days=1)

    # 5) Pack outputs
    cols = [f"path_{i+1}" for i in range(n_paths)]
    deltas_paths = pd.DataFrame(
        data=deltas_det.values.reshape(-1,1) + eps_paths,  # Δlog s_{t+1} per path
        index=idx_t, columns=cols
    )
    log_s_paths = pd.DataFrame(log_s_paths_arr, index=idx_t1_eff, columns=cols)
    s_paths     = pd.DataFrame(s_paths_arr,     index=idx_t1_eff, columns=cols)

    # 6) Metrics
    if metrics_target == "levels":
        metrics = _evaluate_expected_metrics_levels_from_df_cols(
            df_all=df_all,
            s_paths=s_paths,
            eval_rate_col=eval_rate_col,            # <— verdad en NIVELES: 'rate'
            eval_log_rate_col=eval_log_rate_col
        )
    elif metrics_target == "returns":
        # Si alguna vez quieres evaluar retornos frente a la VERDAD real,
        # recomputamos y_true desde eval series (no desde feature series)
        s_eval = (df_all[eval_rate_col] if eval_rate_col in df_all.columns and df_all[eval_rate_col].notna().any()
                  else np.exp(df_all[eval_log_rate_col]))
        s_eval = s_eval.reindex(idx_t.union(idx_t1_eff))
        y_true_eval = s_eval.reindex(idx_t1_eff).values - s_eval.reindex(idx_t).values
        y_true_eval = pd.Series(y_true_eval, index=idx_t, name="dlog_s_true_eval")
        metrics = _evaluate_expected_metrics_deltas(y_true_eval, deltas_paths)
    else:
        raise ValueError("metrics_target must be 'levels' or 'returns'.")
    
    details = {
        "start_date": str(pd.to_datetime(start_date).date()),
        "steps_requested": steps,
        "steps_effective": steps_eff,
        "idx_t_start": str(idx_t.min().date()) if len(idx_t) else None,
        "idx_t_end": str(idx_t.max().date()) if len(idx_t) else None,
        "n_paths": n_paths,
        "seed": seed,
        "error_process": error_process,
        "sigma_resid": sigma_resid,
        "init_eps": init_eps if error_process=='ar1' else None,
        "metrics_target": metrics_target,
        "feature_rate_col": feature_rate_col,
        "eval_rate_col": eval_rate_col,
        "anchor_rate_col": anchor_col
    }

    return SimulationResult(
        idx_t1=idx_t1_eff,
        deltas_det=deltas_det,
        deltas_paths=deltas_paths,
        log_s_paths=log_s_paths,
        s_paths=s_paths,
        metrics=metrics,
        details=details
    )
