# e-ks_ar_fitting.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Optional: HP filter (for output gap)
try:
    from statsmodels.tsa.filters.hp_filter import hpfilter
except Exception:
    hpfilter = None  # we'll fall back if unavailable

# ---------------------------------------------------------------------
# 1) Compute EKS feature blocks: x1 UIP, x2 PPP, x3 MF, x4 Taylor
# ---------------------------------------------------------------------
def make_eks_blocks(
    df_all: pd.DataFrame,
    *,
    rate_col: str = "rate_interpolated",
    log_rate_col: str = "log_rate",              # computed if missing
    cpi_cols: Tuple[str, str] = ("CPI_US", "CPI_EU"),
    m1_cols:  Tuple[str, str] = ("M1_US", "M1_EU"),
    ip_cols:  Tuple[str, str] = ("IP_US", "IP_EU"),
    r3m_cols: Tuple[str, str] = ("R3M_US", "R3M_EU"),
    pi_cols:  Optional[Tuple[str, str]] = ("PI_US", "PI_EU"),  # YoY inflation in p.p., if available
    ip_is_log: bool = True,        # your IP_* already look logged (≈4.63)
    m1_is_log: bool = False,       # your M1_* look in levels (≈1.14e7, 1.9e4)
    cpi_is_log: bool = False,      # your CPI_* look in levels (≈120, 299)
    scale_100: bool = True,        # Li et al. work with logs×100
    horizon_mode: str = "daily",   # "daily" -> UIP ≈ diff/360, "monthly" -> diff/12
    monthly_freq: str = "M",       # for HP filter on IP to compute output gap
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['x1_uip','x2_ppp','x3_mf','x4_taylor'] aligned to df_all.index (daily).
    """
    us, eu = "US", "EU"
    cpi_us, cpi_eu = cpi_cols
    m1_us,  m1_eu  = m1_cols
    ip_us,  ip_eu  = ip_cols
    r_us,   r_eu   = r3m_cols

    df = df_all.copy()

    # --- log s_t ---
    if log_rate_col not in df.columns:
        if rate_col not in df.columns:
            raise ValueError("Need either log_rate_col or rate_col in df_all.")
        if (df[rate_col] <= 0).any():
            raise ValueError(f"{rate_col} has non-positive values; cannot log.")
        df[log_rate_col] = np.log(df[rate_col])
    s_log = df[log_rate_col].astype(float)

    # --- logs for CPI, M1, IP ---
    def _as_log(x: pd.Series, already_log: bool, name: str) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if not already_log:
            if (x <= 0).any():
                raise ValueError(f"{name} has non-positive values; cannot log.")
            x = np.log(x)
        return x.astype(float)

    p_us = _as_log(df[cpi_us], cpi_is_log, cpi_us)
    p_eu = _as_log(df[cpi_eu], cpi_is_log, cpi_eu)
    m_us = _as_log(df[m1_us],  m1_is_log,  m1_us)
    m_eu = _as_log(df[m1_eu],  m1_is_log,  m1_eu)
    y_us = _as_log(df[ip_us],  ip_is_log,  ip_us)
    y_eu = _as_log(df[ip_eu],  ip_is_log,  ip_eu)

    # --- Inflation differential (YoY, p.p.) ---
    # Prefer provided PI_* (YoY %). If missing, derive from CPI logs: 100 * (p_t - p_{t-12M}).
    if pi_cols is not None and all(c in df.columns for c in pi_cols):
        pi_us, pi_eu = pi_cols
        pi_diff = pd.to_numeric(df[pi_us], errors="coerce") - pd.to_numeric(df[pi_eu], errors="coerce")
    else:
        # Derive monthly YoY from CPI logs, then ffill to daily
        p_us_m = p_us.resample(monthly_freq).last()
        p_eu_m = p_eu.resample(monthly_freq).last()
        pi_us_m = 100.0 * (p_us_m - p_us_m.shift(12))
        pi_eu_m = 100.0 * (p_eu_m - p_eu_m.shift(12))
        pi_diff = (pi_us_m - pi_eu_m).reindex(df.index, method="ffill")

    # --- Output gap via HP filter on monthly log IP, then daily align ---
    if hpfilter is None:
        # Fallback: centered 12m MA as a crude trend if hpfilter unavailable
        y_us_m = y_us.resample(monthly_freq).last()
        y_eu_m = y_eu.resample(monthly_freq).last()
        trend_us = y_us_m.rolling(24, center=True, min_periods=6).mean()
        trend_eu = y_eu_m.rolling(24, center=True, min_periods=6).mean()
    else:
        y_us_m = y_us.resample(monthly_freq).last()
        y_eu_m = y_eu.resample(monthly_freq).last()
        # λ for monthly data per Ravn & Uhlig ≈ 129600
        trend_us, _ = hpfilter(y_us_m.dropna(), lamb=129600)
        trend_us = trend_us.reindex(y_us_m.index)
        trend_eu, _ = hpfilter(y_eu_m.dropna(), lamb=129600)
        trend_eu = trend_eu.reindex(y_eu_m.index)

    gap_us_m = (y_us_m - trend_us)
    gap_eu_m = (y_eu_m - trend_eu)
    gap_diff = (gap_us_m - gap_eu_m).reindex(df.index, method="ffill")

    # --- x1 UIP: forward premium ~ interest differential scaled to horizon ---
    if horizon_mode == "monthly":
        scale = 1.0 / 12.0
    elif horizon_mode == "daily":
        scale = 1.0 / 360.0
    else:
        raise ValueError("horizon_mode must be 'daily' or 'monthly'.")

    i_us = pd.to_numeric(df[r_us], errors="coerce")
    i_eu = pd.to_numeric(df[r_eu], errors="coerce")
    x1_uip = (i_us - i_eu) * scale  # no logs on interest rates

    # --- x2 PPP: p_us - p_eu - s_log ---
    x2_ppp = (p_us - p_eu - s_log)

    # --- x3 MF: (m_us - m_eu) - (y_us - y_eu) - s_log ---
    x3_mf = (m_us - m_eu) - (y_us - y_eu) - s_log

    # --- x4 Taylor: 1.5*pi_diff + 0.1*gap_diff + 0.1*(s_log + p_eu - p_us) ---
    x4_taylor = (1.5 * pi_diff) + (0.1 * gap_diff) + (0.1 * (s_log + p_eu - p_us))

    # Optional: scale logs by 100 to be in "percent" units
    if scale_100:
        x2_ppp = 100.0 * x2_ppp
        x3_mf  = 100.0 * x3_mf
        # The 0.1*(s + p* - p) term should be consistent; if you scale logs by 100,
        # scale that subterm accordingly:
        x4_taylor = (1.5 * pi_diff) + (0.1 * gap_diff) + (0.1 * (100.0 * (s_log + p_eu - p_us)))

    # Assemble
    X_eks = pd.DataFrame({
        "x1_uip": x1_uip,
        "x2_ppp": x2_ppp,
        "x3_mf":  x3_mf,
        "x4_taylor": x4_taylor
    }, index=df.index)

    # Forward-fill monthly-to-daily and clean NaNs/Infs
    X_eks = X_eks.ffill().replace([np.inf, -np.inf], np.nan)
    X_eks = X_eks.dropna(how="any")

    return X_eks

# -----------------------------
# 1) Data containers
# -----------------------------

@dataclass
class PredictorBuildResult:
    X: pd.DataFrame             # Features X_t aligned to y_{t+1}
    y: pd.Series                # Target Δlog s_{t+1}
    s_log: pd.Series            # log spot aligned to X index (useful for later paths)
    feature_names: List[str]
    meta: Dict

@dataclass
class SplitResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex

# -----------------------------
# 2) Standardizer (training-only)
# -----------------------------

class Standardizer:
    """
    Simple training-only standardizer (mean/std).
    Use .fit() on X_train, then .transform() both train and test.
    """
    def __init__(self):
        self.means_: Optional[pd.Series] = None
        self.stds_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame) -> "Standardizer":
        self.means_ = X.mean(axis=0)
        self.stds_  = X.std(axis=0, ddof=1).replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("Standardizer must be fit before transform.")
        # align columns just in case
        Xc = X.copy()
        Xc = Xc[self.means_.index]
        return (Xc - self.means_) / self.stds_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

# -----------------------------
# 3) Feature builder
# -----------------------------

def build_predictors_from_merged_df_eks(
    df_all: pd.DataFrame,
    *,
    horizon_days: int = 1,              # set 22 to approximate 1M horizon
    **eks_kwargs
) -> PredictorBuildResult:
    """
    Build X_t=[x1..x4], y_{t+h}=(log s_{t+h}-log s_t) from df_all using EKS blocks.
    eks_kwargs are passed to make_eks_blocks (rate_col, log_rate_col, etc.).
    """
    # 1) features
    X = make_eks_blocks(df_all, **eks_kwargs)

    # 2) target Δlog s_{t+h}
    log_rate_col = eks_kwargs.get("log_rate_col", "log_rate")
    rate_col     = eks_kwargs.get("rate_col", "rate_interpolated")

    if log_rate_col in df_all.columns:
        s_log = pd.to_numeric(df_all[log_rate_col], errors="coerce")
    else:
        s = pd.to_numeric(df_all[rate_col], errors="coerce")
        if (s <= 0).any():
            raise ValueError(f"{rate_col} has non-positive values; cannot log.")
        s_log = np.log(s)

    y = s_log.shift(-horizon_days) - s_log

    # 3) align to X (drop any remaining NaNs at edges)
    data = pd.concat([X, y.rename("dlog_s_lead")], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    X_aligned = data[X.columns]
    y_aligned = data["dlog_s_lead"]
    s_log_aligned = s_log.reindex(X_aligned.index)

    meta = {
        "horizon_days": horizon_days,
        "feature_names": list(X_aligned.columns),
        "n_obs": int(len(X_aligned)),
        "index_start": str(X_aligned.index.min().date()),
        "index_end": str(X_aligned.index.max().date()),
        "horizon_mode": eks_kwargs.get("horizon_mode", "daily"),
        "scale_100": eks_kwargs.get("scale_100", True),
    }

    return PredictorBuildResult(
        X=X_aligned, y=y_aligned, s_log=s_log_aligned,
        feature_names=list(X_aligned.columns), meta=meta
    )

# -----------------------------
# 4) Train/Test split helpers
# -----------------------------

def split_train_test_by_dates(
    X: pd.DataFrame, y: pd.Series,
    *, train_start: str, train_end: str,
    test_start: str | None = None, test_end: str | None = None
):
    """
    Split X,y by calendar dates on the *feature index* t (not on y's t+1).
    If test_start == train_end, we avoid overlap by excluding that boundary
    date from the training set (train uses [train_start, test_start), test uses [test_start, ...]).

    Returns a SplitResult with disjoint train/test indices.
    """
    X = X.sort_index(); y = y.sort_index()
    if test_start is None:
        test_start = train_end  # your rule

    ts = pd.to_datetime(test_start)
    te = pd.to_datetime(train_end)

    if ts <= te:
        # Disjoint: training uses dates < test_start
        train_mask = (X.index >= pd.to_datetime(train_start)) & (X.index < ts)
    else:
        # Standard case: training uses <= train_end
        train_mask = (X.index >= pd.to_datetime(train_start)) & (X.index <= te)

    test_mask = (X.index >= ts)
    if test_end is not None:
        test_mask &= (X.index <= pd.to_datetime(test_end))

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test,  y_test  = X.loc[test_mask],  y.loc[test_mask]

    return SplitResult(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        train_index=X_train.index, test_index=X_test.index
    )

def standardize_train_and_apply(
    split: SplitResult,
) -> Tuple[SplitResult, Standardizer]:
    """
    Fit a Standardizer on training features, then transform both train and test.
    Returns updated SplitResult and the fitted Standardizer for reuse in forecasting.
    """
    scaler = Standardizer().fit(split.X_train) # (x - mu) / sd
    X_train_std = scaler.transform(split.X_train)
    X_test_std  = scaler.transform(split.X_test)

    return SplitResult(
        X_train=X_train_std, y_train=split.y_train,
        X_test=X_test_std,  y_test=split.y_test,
        train_index=split.train_index, test_index=split.test_index
    ), scaler


# -----------------------------


from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.api as sm

# -----------------------------
# Model bundle to pass to forecasting
# -----------------------------

@dataclass
class EKSModel:
    intercept_: float
    coef_: np.ndarray
    feature_names: List[str]
    l1_ratio: float       # elastic-net mixing (0=ridge,1=lasso)
    alpha_penalty: float  # lambda
    scaler: "Standardizer"

    def predict_deltas_stdX(self, X_std: pd.DataFrame) -> np.ndarray:
        """Predict Δlog s_{t+1} given standardized features."""
        # Align columns defensively
        X_use = X_std[self.feature_names]
        return self.intercept_ + X_use.values @ self.coef_

    def predict_deltas_rawX(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Predict Δlog s_{t+1} given raw features by applying stored scaler."""
        X_std = self.scaler.transform(X_raw[self.feature_names])
        return self.predict_deltas_stdX(X_std)

@dataclass
class AR1ResidualModel:
    rho: float
    sigma_eta: float         # std of AR(1) innovations η_t
    last_resid: float        # ε_T (last in-sample residual)
    resid_series: pd.Series  # full in-sample residuals (for diagnostics)

# -----------------------------
# Elastic-Net (E-KS) fitting with 5-fold TimeSeries CV
# -----------------------------

def fit_eks_once_with_tscv(
    X_train_std: pd.DataFrame,
    y_train: pd.Series,
    *,
    l1_ratio_grid=(0.0,0.01,0.05,0.1),
    alpha_grid= np.logspace(-8,-3, 60),
    n_splits=5,
    max_iter=20000,
    tol=1e-6,
) -> Tuple[EKSModel, pd.DataFrame]:
    """
    Fit ElasticNet once using 5-fold time-series CV over (l1_ratio, alpha).
    Returns the frozen EKSModel and a CV summary DataFrame.
    """
    if alpha_grid is None:
        # Lambda grid roughly log-spaced for daily data
        alpha_grid = np.logspace(-5, 0, 12)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    best = {"rmse": np.inf, "l1_ratio": None, "alpha": None, "coef": None, "intercept": None}
    rows = []

    for l1 in l1_ratio_grid:
        for a in alpha_grid:
            rmses = []
            for tr_idx, va_idx in tscv.split(X_train_std):
                X_tr, X_va = X_train_std.iloc[tr_idx], X_train_std.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                enet = ElasticNet(
                    alpha=a, l1_ratio=l1, fit_intercept=True,
                    max_iter=max_iter, tol=tol, selection="cyclic", random_state=None
                )
                enet.fit(X_tr, y_tr)
                y_hat = enet.predict(X_va)
                rmses.append(sqrt(mean_squared_error(y_va, y_hat)))

            mean_rmse = float(np.mean(rmses))
            rows.append({"l1_ratio": l1, "alpha": a, "cv_rmse": mean_rmse})

            if mean_rmse < best["rmse"]:
                best.update({
                    "rmse": mean_rmse, "l1_ratio": l1, "alpha": a,
                    "coef": enet.coef_.copy(), "intercept": float(enet.intercept_)
                })

    cv_table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)

    # Refit on full training set with best hyper-params
    final = ElasticNet(
        alpha=best["alpha"], l1_ratio=best["l1_ratio"],
        fit_intercept=True, max_iter=max_iter, tol=tol, selection="cyclic"
    ).fit(X_train_std, y_train)

    model = EKSModel(
        intercept_=float(final.intercept_),
        coef_=final.coef_.copy(),
        feature_names=list(X_train_std.columns),
        l1_ratio=float(best["l1_ratio"]),
        alpha_penalty=float(best["alpha"]),
        scaler=None  # attach after we call this function
    )
    return model, cv_table

# -----------------------------
# Residual AR(1) fit on training residuals
# -----------------------------

def fit_ar1_on_training_residuals(
    X_train_std: pd.DataFrame,
    y_train: pd.Series,
    eks_model: EKSModel
) -> AR1ResidualModel:
    """
    Fit ε_{t+1} = ρ ε_t + η_{t+1}, with no constant.
    Uses statsmodels ARIMA(1,0,0) with trend='n' for robustness.
    """
    # 1) In-sample fitted deltas
    y_hat = eks_model.predict_deltas_stdX(X_train_std)
    resid = pd.Series(y_train.values - y_hat, index=y_train.index, name="resid")

    # 2) Fit AR(1) with no constant
    ar1_res = sm.tsa.ARIMA(resid, order=(1,0,0), trend='n').fit()
    rho = float(ar1_res.params.get("ar.L1", ar1_res.params[0]))
    sigma_eta = float(np.sqrt(ar1_res.params.get("sigma2", ar1_res.params[-1])))

    # # 3) Extract rho robustly (no positional indexing)
    # if "ar.L1" in ar1_res.params.index:
    #     rho = float(ar1_res.params["ar.L1"])
    # else:
    #     # Fallback: first parameter by position
    #     rho = float(ar1_res.params.iloc[0])

    # # 4) Innovation variance: prefer sigma2/scale on the results
    # if hasattr(ar1_res, "sigma2"):
    #     sigma2 = float(ar1_res.sigma2)
    # elif hasattr(ar1_res, "scale"):
    #     sigma2 = float(ar1_res.scale)
    # else:
    #     # Fallback: variance of model innovations (resid) as a last resort
    #     sigma2 = float(np.var(ar1_res.resid, ddof=0))

    # sigma_eta = float(np.sqrt(sigma2))

    return AR1ResidualModel(
        rho=rho,
        sigma_eta=sigma_eta,
        last_resid=float(resid.iloc[-1]),
        resid_series=resid
    )
