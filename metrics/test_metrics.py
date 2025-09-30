import numpy as np
import matplotlib.pyplot as plt
import math
# own functions
from score import KM_CensoringDistribution
from score import CompRiskMetricsCompute
# sksurv
from sklearn.model_selection import train_test_split

import helper


### testing R code translation on 1 vector of risks
rdata, vdata, vdata_pred = helper.read_lumc()

time   = vdata["time"].to_numpy(float)
status = vdata["status_num"].to_numpy(int)

# here the prediction is a risk at a specific time point
pred = vdata_pred.to_numpy(float).reshape(-1)

cause = 1
tau = 5 

out = helper.weighted_brier_score(pred, tau, time, status, cause, cens_code=0, cmprsk=True)
print(out) # get the same result as in R 





### testing for a matrix with a Cif per patient:
dis, bmt_df = helper.load_bmt()

status = np.asarray(bmt_df["status"], dtype=int)
ftime  = np.asarray(bmt_df["ftime"], dtype=float)
X      = np.asarray(dis, dtype=int)

X_train, X_val, ftime_train, ftime_val, status_train, status_val = train_test_split(
X, ftime, status, test_size=0.4, random_state=1, stratify=status)

time_grid = np.unique(ftime_train[ftime_train > 0])

cox_trm     = helper.fit_cs_cox(X_train, ftime_train, status_train, cause=1)
cox_relapse = helper.fit_cs_cox(X_train, ftime_train, status_train, cause=2)

# from hazards to cifs
A1 = helper.cumhaz_on_grid(cox_trm,     X_val, time_grid)  # (n_val, m)
A2 = helper.cumhaz_on_grid(cox_relapse, X_val, time_grid)

F1_pred, F2_pred = helper.cifs_from_cs_hazards(A1, A2)     # (n_val, m)

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









