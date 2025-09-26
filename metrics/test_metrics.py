import numpy as np
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
# to look at r data
import pyreadr
from lifelines import KaplanMeierFitter

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

'''Block to test Cifs per patient'''

def fit_cs_cox(X, time, status, cause):
    y_cs = Surv.from_arrays(event=(status == cause), time=time)
    # simple CoxPH; add penalizer if needed
    model = CoxPHSurvivalAnalysis().fit(X, y_cs)
    return model

def cumhaz_on_grid(model, X_query, tgrid):
    # returns array (n, m) with cumulative hazard Λ(t|x)
    ch_funcs = model.predict_cumulative_hazard_function(X_query)
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


if __name__ == "__main__":

    ### testing R code translation on 1 vector of risks
    rdata, vdata, vdata_pred = read_lumc()

    time   = vdata["time"].to_numpy(float)
    status = vdata["status_num"].to_numpy(int)

    # here the prediction is a risk at a specific time point
    pred = vdata_pred.to_numpy(float).reshape(-1)

    cause = 1
    tau = 5 

    out = weighted_brier_score(pred, tau, time, status, cause, cens_code=0, cmprsk=True)
    print(out) # get the same result as in R 

    ### testing for a matrix with a Cif per patient:
    dis, bmt_df = load_bmt()

    status = np.asarray(bmt_df["status"], dtype=int)
    ftime  = np.asarray(bmt_df["ftime"], dtype=float)
    X      = np.asarray(dis, dtype=int)

    X_train, X_val, ftime_train, ftime_val, status_train, status_val = train_test_split(
    X, ftime, status, test_size=0.4, random_state=1, stratify=status)

    time_grid = np.unique(ftime_train[ftime_train > 0])

    cox_trm     = fit_cs_cox(X_train, ftime_train, status_train, cause=1)
    cox_relapse = fit_cs_cox(X_train, ftime_train, status_train, cause=2)
    
    # from hazards to cifs
    A1 = cumhaz_on_grid(cox_trm,     X_val, time_grid)  # (n_val, m)
    A2 = cumhaz_on_grid(cox_relapse, X_val, time_grid)

    F1_pred, F2_pred = cifs_from_cs_hazards(A1, A2)     # (n_val, m)

    metrics = CompRiskMetricsCompute(
        time_grid=time_grid,
        durations_train=ftime_train,
        delta_train=status_train,
        eps=1e-6
    )

    bs_trm = metrics.weighted_brier_cif(F1_pred, ftime_val, status_val, cause=1)
    ibs_trm = metrics.integrated_brier_cif(F1_pred, ftime_val, status_val, cause=1)

    bs_rel = metrics.weighted_brier_cif(F2_pred, ftime_val, status_val, cause=2)
    ibs_rel = metrics.integrated_brier_cif(F2_pred, ftime_val, status_val, cause=2)

    print(f"IBS(TRM)    = {ibs_trm:.4f}")
    print(f"IBS(Relapse)= {ibs_rel:.4f}")
    # Optional sanity checks
    assert np.all(np.isfinite(bs_trm)) and np.all(np.isfinite(bs_rel))
    assert 0 <= ibs_trm <= 1 and 0 <= ibs_rel <= 1


    






