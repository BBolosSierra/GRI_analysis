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
def synthetic_censoring_times(n=1000, lam_event=0.3, lam_cens=0.2, rng=42):
    """ T ~ Exp(lam_event), C ~ Exp(lamb_cens)
    The observed time = min(T, C). 
    Event indicator depends on the observed time. """
    
    rng= np.random.default_rng(rng)

    T = rng.exponential(scale=1/lam_event, size=n)
    C = rng.exponential(scale=1/lam_cens, size=n)
    
    T = np.asarray(T, dtype=float)
    C = np.asarray(C, dtype=float)

    Tobs = np.minimum(T, C)
    delta = np.where(T <= C, 1, 0).astype(int)
    return Tobs, delta


def sim_cmprsk_dataset(n, lambda1, lambda2, lambda_c, rng):
    # only times and no covariates based on exponen
    #  T1 ~ Exp(l1), T2 ~ Exp(l2), C ~ Exp(lc)
    rng= np.random.default_rng(rng)

    T1 = rng.exponential(1/lambda1, size=n)
    T2 = rng.exponential(1/lambda2, size=n)
    C  = rng.exponential(1/lambda_c, size=n)
    # observed time and type
    T  = np.minimum(np.minimum(T1, T2), C)
    D  = np.where((T1 < T2) & (T1 < C), 1,
         np.where((T2 < T1) & (T2 < C), 2, 0)).astype(int)
    
    return T, D

def cif_cmprsk_data(n_train, n_val, lambda1, lambda2, lambda_c):
    # generate simple dataset without covariates
    T_train, D_train = sim_cmprsk_dataset(n_train, lambda1, lambda2, lambda_c)
    T_val,   D_val   = sim_cmprsk_dataset(n_val, lambda1, lambda2, lambda_c)
    
    # but double check this:
    # S(t) = exp(-(l1+l2)t), F1(t) = (l1/(l1+l2))*(1 - S), 
    # F2(t) = (l2/(l1+l2))*(1 - S).
    time_grid = np.linspace(0.1, 10.0, 50)
    S = np.exp(-(lambda1+lambda2)*time_grid) # survival
    F1_true = (lambda1/(lambda1+lambda2))*(1 - S) # cif cause 1
    F2_true = (lambda2/(lambda1+lambda2))*(1 - S) # cif cause 2

    # cifs as a matrix (here all patients)
    pred_cif_val_c1 = np.tile(F1_true, (T_val.size, 1))
    pred_cif_val_c2 = np.tile(F2_true, (T_val.size, 1))

    return T_train, D_train, T_val, D_val, pred_cif_val_c1, pred_cif_val_c2, time_grid
    

## Estimators and model fitters:
def km_test():
    cens = KM_CensoringDistribution(eps = 1e-8)
    # create synthetic dataset
    Tobs, delta = synthetic_censoring_times(n=2000, lam_event=0.3, lam_cens=0.2)
    print(Tobs)

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

