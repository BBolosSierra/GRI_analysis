import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# own functions
from metrics.score import KM_CensoringDistribution
from metrics.score import CompRiskMetricsCompute
# sksurv
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv
from sksurv.datasets import load_bmt
from sksurv.nonparametric import cumulative_incidence_competing_risks
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
# sklearn
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold 
# to look at r data
import pyreadr
from lifelines import KaplanMeierFitter

#### For preprocessing ####
def smart_fix_dtypes(df, cat_threshold=20):
    df = df.copy()
    # A) floats that are 0/1 → boolean
    mask_bool = {
        c for c in df.columns
        if pd.api.types.is_float_dtype(df[c]) and set(df[c].dropna().unique()).issubset({0,1})
    }
    for c in mask_bool:
        df[c] = df[c].astype("Int64").map({0: False, 1: True}).astype("boolean")

    # B) object numeric strings → numeric
    for c in df.select_dtypes(include="object").columns:
        frac_num = df[c].astype("string").str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False).mean()
        if frac_num > 0.9:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # C) low-cardinality numerics → category
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        if nun <= cat_threshold and (pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)):
            if pd.api.types.is_float_dtype(s) and (s.dropna() == s.dropna().round()).all():
                df[c] = s.round().astype("Int64").astype("category")
            else:
                df[c] = s.astype("string").astype("category")
        elif pd.api.types.is_object_dtype(s) and nun <= 50:
            df[c] = s.astype("string").str.strip().astype("category")
    return df

def split_num_cat(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    return num_cols, cat_cols

def fit_preprocessor(X_train):
    """Fit all preprocessors on TRAIN ONLY. Return a dict you can reuse."""
    num_cols, cat_cols = split_num_cat(X_train)

    # --- numeric: MICE -> scaler
    mice = IterativeImputer(random_state=42, sample_posterior=True,
                            max_iter=10, initial_strategy="median")
    Xn_imp = mice.fit_transform(X_train[num_cols]) if num_cols else np.empty((len(X_train), 0))

    scaler = StandardScaler()
    Xn = scaler.fit_transform(Xn_imp) if num_cols else np.empty((len(X_train), 0))

    # --- categorical: mode impute -> OHE (drop='first' to avoid singularity)
    cat_imp = SimpleImputer(strategy="most_frequent")
    Xc_imp = cat_imp.fit_transform(X_train[cat_cols]) if cat_cols else np.empty((len(X_train), 0))

    ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    Xc = ohe.fit_transform(Xc_imp) if cat_cols else np.empty((len(X_train), 0))

    # --- combine & remove constant columns
    X_comb = np.hstack([Xn, Xc]) if (Xn.size or Xc.size) else np.empty((len(X_train), 0))
    nzv = VarianceThreshold(threshold=0.0)
    Xnz = nzv.fit_transform(X_comb) if X_comb.size else X_comb

    # feature names (optional but handy)
    num_names = [f"num_{c}" for c in num_cols]
    cat_names = []
    if cat_cols:
        # names with drop='first'
        ohe_out = ohe.get_feature_names_out(cat_cols)
        cat_names = ohe_out.tolist()
    names_all = num_names + cat_names
    if X_comb.size:
        names_all = [n for keep, n in zip(nzv.get_support().tolist(), names_all) if keep]
    else:
        names_all = []

    pre = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "mice": mice,
        "scaler": scaler,
        "cat_imp": cat_imp,
        "ohe": ohe,
        "nzv": nzv,
        "feature_names_": names_all,
    }
    return pre, Xnz

def transform_with_preprocessor(X: pd.DataFrame, pre: dict):
    """Apply TRAIN-fitted preprocessors to new data (validation/test)."""
    num_cols, cat_cols = pre["num_cols"], pre["cat_cols"]

    # numeric
    if num_cols:
        Xn_imp = pre["mice"].transform(X[num_cols])
        Xn = pre["scaler"].transform(Xn_imp)
    else:
        Xn = np.empty((len(X), 0))

    # categorical
    if cat_cols:
        Xc_imp = pre["cat_imp"].transform(X[cat_cols])
        Xc = pre["ohe"].transform(Xc_imp)
    else:
        Xc = np.empty((len(X), 0))

    X_comb = np.hstack([Xn, Xc]) if (Xn.size or Xc.size) else np.empty((len(X), 0))
    Xnz = pre["nzv"].transform(X_comb) if X_comb.size else X_comb
    return Xnz

### Dataset generation ###
def sim_cmprks(n=30000,
               p_block=4,
               gammas=10,         # scalar or length-p_block array-like
               cens_frac=0.5,
               seed=123):
    """
    Competing risks simulator akin to the R version (DeepHit-like).
    
    Returns a DataFrame with columns:
      Tobs, status (1=event, 0=censored), cause (0=censored, 1 or 2=event type),
      x1_1..x1_p, x2_1..x2_p, x3_1..x3_p
    """

    rng = np.random.default_rng(seed)

    # gamma vectors
    if np.isscalar(gammas):
        gamma1 = np.repeat(gammas, p_block).astype(float)
        gamma2 = np.repeat(gammas, p_block).astype(float)
        gamma3 = np.repeat(gammas, p_block).astype(float)
    else:
        g = np.asarray(gammas, dtype=float)
        if g.size != p_block:
            raise ValueError("gammas must be scalar or length p_block.")
        gamma1 = g.copy()
        gamma2 = g.copy()
        gamma3 = g.copy()

    # covariates (n x p_block)
    x1 = rng.normal(size=(n, p_block))
    x2 = rng.normal(size=(n, p_block))
    x3 = rng.normal(size=(n, p_block))

    # linear effects
    g1_x1 = x1 @ gamma1
    g2_x2 = x2 @ gamma2
    g3_x3 = x3 @ gamma3

    # quadratic effects & rates
    lambda1 = np.exp(g3_x3**2 + g1_x1)
    lambda2 = np.exp(g3_x3**2 + g2_x2)

    # hitting times: rexp(rate=lambda) -> exponential with scale=1/lambda
    T1 = rng.exponential(scale=1.0 / lambda1)
    T2 = rng.exponential(scale=1.0 / lambda2)
    si = np.minimum(T1, T2)

    # censoring times
    sc = np.full(n, np.inf, dtype=float)
    if cens_frac > 0.0:
        m = np.floor(cens_frac * n)
        if m > 0:
            cens_idx = rng.choice(n, size=m, replace=False)
            # Uniform(0, si[i]) for selected i
            sc[cens_idx] = rng.random(size=m) * si[cens_idx]

    # observed time and cause
    Tobs = np.minimum(np.minimum(T1, T2), sc)

    is_cens = sc <= si          
    uncens = ~is_cens

    c1 = np.zeros(n, dtype=bool)
    c2 = np.zeros(n, dtype=bool)

    # compare only among uncensored
    c1[uncens] = T1[uncens] < T2[uncens]
    c2[uncens] = T2[uncens] < T1[uncens]
    tie_mask = uncens & ~(c1 | c2)

    cause = np.zeros(n, dtype=int)
    cause[c1] = 1
    cause[c2] = 2
    if np.any(tie_mask):
        # randomize ties between {1,2}
        cause[tie_mask] = rng.choice([1, 2], size=tie_mask.sum(), replace=True)

    status = (cause != 0).astype(int)

    # build DataFrame with matching column names
    df = pd.DataFrame({
        "Tobs": Tobs.astype(float),
        "status": status.astype(int),
        "cause": cause.astype(int),
    })

    for j in range(p_block):
        df[f"x1_{j+1}"] = x1[:, j]
        df[f"x2_{j+1}"] = x2[:, j]
        df[f"x3_{j+1}"] = x3[:, j]

    return df

## Estimators and model fitters:
def km_test():
    cens = KM_CensoringDistribution(eps = 1e-8)
    # create synthetic dataset
    df = sim_cmprks(n=30000, p_block=4, gammas=10, cens_frac=0.5, seed=123) 

    Tobs = df["Tobs"]
    delta = df["cause"]

    cens.fit(Tobs, delta) # fitting the Censoring Distribution

    # create a grid of times
    t = np.linspace(0.0,  np.quantile(Tobs, 0.99), 50)
    G_vals = cens.G(t)
    W = 1/G_vals
    
    # the G(t) is a decreasing function, the probability of remaining uncensored decreeases over time
    # the IPCW is a increasing function, as the inverse of the G(t).
    plt.figure()
    plt.plot(t, G_vals, label="G(t)")
    plt.plot(t, W, label="IPCW")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Uncensored distribution and IPCW")
    plt.legend()
    plt.show()

def fit_cs_cox_manual(Xtr_df, t_tr, s_tr, cause, alpha=1.0):
    """Fit CoxPH on preprocessed TRAIN for a specific cause (others censored)."""
    pre, Xtr = fit_preprocessor(Xtr_df)  # fit preprocessors on TRAIN
    y_tr = Surv.from_arrays(event=(s_tr == cause), time=t_tr)

    # ridge penalty (alpha>0) helps avoid singularities
    cox = CoxPHSurvivalAnalysis(alpha=alpha).fit(Xtr, y_tr)
    return {"pre": pre, "model": cox}

def cumhaz_on_grid_manual(fitted, X_df, tgrid):
    """Transform X_df with TRAIN-fitted preprocessors and get cum hazards."""
    Xv = transform_with_preprocessor(X_df, fitted["pre"])
    chf_list = fitted["model"].predict_cumulative_hazard_function(Xv)
    out = np.empty((len(chf_list), len(tgrid)), float)
    for i, f in enumerate(chf_list):
        out[i] = f(tgrid)
    return out

def fit_cs_cox(X, time, status, cause):
    y_cs = Surv.from_arrays(event=(status == cause), time=time)
    # simple CoxPH; add penalizer if needed
    model = CoxPHSurvivalAnalysis().fit(X, y_cs)
    return model

def cumhaz_on_grid(model, X_matrix, tgrid):
    # returns array (n, m) with cumulative hazard Λ(t|x)
    ch_funcs = model.predict_cumulative_hazard_function(X_matrix)
    m = len(tgrid)
    out = np.empty((len(ch_funcs), m), float)
    for i, f in enumerate(ch_funcs):
        out[i] = f(tgrid)  # StepFunction is callable
    return out

def cifs_from_cs_hazards(L1, L2):
    """
    Aalen–Johansen discrete step:
      ΔΛ_kj = Λ_k(t_j)-Λ_k(t_{j-1}), ΔΛ_tot_j = ΔΛ_1j+ΔΛ_2j
      dF_kj = S_{j-1} * (1 - exp(-ΔΛ_tot_j)) * (ΔΛ_kj / ΔΛ_tot_j)  (0 if ΔΛ_tot_j=0)
      S_j   = S_{j-1} * exp(-ΔΛ_tot_j),  with S_0 = 1
    """
    n, m = L1.shape
    F1 = np.zeros((n, m), float)
    F2 = np.zeros((n, m), float)
    S  = np.ones(n, float)

    dL1 = np.diff(np.column_stack([np.zeros(n), L1]), axis=1)
    dL2 = np.diff(np.column_stack([np.zeros(n), L2]), axis=1)
    dLtot = dL1 + dL2

    for j in range(m):
        dl1 = dL1[:, j]
        dl2 = dL2[:, j]
        dlt = dLtot[:, j]
        # prob of any event in (t_{j-1}, t_j]
        p_any = 1.0 - np.exp(-dlt)
        share1 = np.divide(dl1, dlt, out=np.zeros_like(dl1), where=(dlt > 0))
        share2 = np.divide(dl2, dlt, out=np.zeros_like(dl2), where=(dlt > 0))
        dF1 = S * p_any * share1
        dF2 = S * p_any * share2
        # update
        F1[:, j] = (F1[:, j-1] if j > 0 else 0) + dF1
        F2[:, j] = (F2[:, j-1] if j > 0 else 0) + dF2
        S = S * np.exp(-dlt)
    # clip numerically to [0,1]
    return np.clip(F1, 0, 1), np.clip(F2, 0, 1)


''' This block corresponds with translations from my R package
To check if the function was working I use the LumcData, based on the
splits, risk was predicted with the riskRegression cause-specific 
cox PH in R ato be loaded directly for straightforward comparison.
The prediction is simply the risk at a specific time point or horizon 
of interest and we evaluate at that same time (tau). 

Therefore this implmentation is a simplification of the problem, where the
goal is to have a weighted brier score that adjust for censoring, but also 
can allow competing risks. 
We ultise lifelines for estimating the Kaplan Meier of the censoring 
distribution.

With this test we get the same results as in riskregression in R and my 
own functions in R '''

def read_lumc():
    # read the two RDS files from R
    rdata = pyreadr.read_r("metrics/LumcData/rdata.rds")[None]  # a pandas.DataFrame
    vdata = pyreadr.read_r("metrics/LumcData/vdata.rds")[None]

    # predictions done in R with riskRegression Cause-Specific hazards for cause 1
    vdata_pred = pyreadr.read_r("metrics/LumcData/vdata_CSC_pred.rds")[None]

    return rdata, vdata, vdata_pred

def censor_prob_KM(time, status, cens_code=0, step=0.1, eps=0.01, left_limit=True):

    time   = np.asarray(time, float)
    status = np.asarray(status, int)

    # Fit KM for censoring: event_observed=1 indicates censoring occurred
    km = KaplanMeierFitter().fit(durations=time, event_observed=(status == cens_code).astype(int))

    # Raw step function from lifelines
    tt = km.survival_function_.index.to_numpy(float)                    # jump times including 0
    gg = km.survival_function_["KM_estimate"].to_numpy(float)           # step values

    # Build regular grid
    tmax = math.floor(float(np.max(time)))
    time_grid = np.arange(0.0, tmax + step/2, step)                     # inclusive of tmax

    # Left-limit lookup: last value strictly before t
    if left_limit:
        idx = np.searchsorted(tt, time_grid, side="right") - 1
        idx = np.clip(idx, 0, len(gg)-1)
        G = gg[idx]
    else:
        # right-continuous value at t
        idx = np.searchsorted(tt, time_grid, side="left")
        idx = np.clip(idx, 0, len(gg)-1)
        G = gg[idx]

    # Ensure monotone and clip away from zero
    G = np.minimum.accumulate(G)                    # safety; KM should be nonincreasing
    G = np.clip(G, eps, 1.0)

    return np.column_stack([time_grid, G])

def weighted_brier_score(pred, tau, time, status, cause, cens_code=0, cmprsk=True, eps=0.01):
    # as in Albergue 2025 (who set it from Kretowska 2018)
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    pred = np.asarray(pred, float)  # prediction at tau

    Gtbl = censor_prob_KM(time, status, cens_code=cens_code)
    tG, Gv = Gtbl[:,0], Gtbl[:,1]

    def G_left(x):   # G(T-) : last grid point strictly < Ti
        i = np.searchsorted(tG, float(x), side="right") - 1
        i = 0 if i < 0 else (len(Gv) - 1 if i >= len(Gv) else i)
        return float(Gv[i])

    def G_right(x):  # G(t+) ≈ right-continuous value at/after t
        i = np.searchsorted(tG, float(x), side="left")
        i = len(Gv) - 1 if i >= len(Gv) else i
        return float(Gv[i])

    # G1 = G(T_i-), G2 = G(t+)
    G1 = np.clip(np.array([G_left(ti) for ti in time]), eps, 1.0)
    G2 = np.clip(G_right(float(tau)), eps, 1.0)

    alive_after_tau = (time > tau)
    evt_interest    = (time <= tau) & (status == cause)
    evt_other       = (time <= tau) & (status != cause) & (status != cens_code)
    cens_before_tau = (time <= tau) & (status == cens_code)

    if cmprsk:
        resid = np.zeros_like(pred, float)
        resid[evt_interest]  = ((1.0 - pred[evt_interest])**2) / G1[evt_interest]
        resid[evt_other]     = (pred[evt_other]**2)            / G1[evt_other]
        resid[cens_before_tau]= 0.0
        resid[alive_after_tau]= (pred[alive_after_tau]**2)     / G2
    else:
        resid = np.zeros_like(pred, float)
        resid[(time <= tau) & (status == cause)] = ((1.0 - pred[(time <= tau) & (status == cause)])**2) / G1[(time <= tau) & (status == cause)]
        resid[(time <= tau) & (status == cens_code)] = 0.0
        resid[alive_after_tau] = (pred[alive_after_tau]**2) / G2

    return dict(weighted_brier_score=float(np.mean(resid)),
                tau=float(tau),
                n=int(len(pred)),
                n_risk=int(np.sum(alive_after_tau)))


