import numpy as np
import matplotlib.pyplot as plt
import math
# own functions
from metrics.brier_score import KM_CensoringDistribution
from metrics.brier_score import CompRiskMetricsCompute
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

rng= np.random.default_rng(42)

def synthetic_censoring_times(n=1000, lam_event=0.3, lam_cens=0.2):
    """ T ~ Exp(lam_event), C ~ Exp(lamb_cens)
    The observed time = min(T, C). 
    Event indicator depends on the observed time. """

    T = rng.exponential(scale=1/lam_event, size=n)
    C = rng.exponential(scale=1/lam_cens, size=n)
    
    T = np.asarray(T, dtype=float)
    C = np.asarray(C, dtype=float)

    Tobs = np.minimum(T, C)
    delta = np.where(T <= C, 1, 0).astype(int)
    return Tobs, delta

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


def sim_cmprsk_dataset(n, lambda1, lambda2, lambda_c):
    # only times and no covariates based on exponen
    #  T1 ~ Exp(l1), T2 ~ Exp(l2), C ~ Exp(lc)
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
    

