import numpy as np 
from lifelines import KaplanMeierFitter

class KM_CensoringDistribution:
    """Estimation of the Inverse probability censoring weights (IPCW) from the Kaplan Meier.
    
    The IPCW are calculated as the inverse of the probability of being uncensored: 1/G(T), where G(t) 
    is estimated with the Kaplan Meier (it is a population level estimate, and does not depend on 
    covariates). For competing risks we will consider event of interest and competing risk as death, to 
    calculate the uncensoring discribution. 
    
    It is calculated to correct the bias on the performance metrics caused by the right-censoring. """

    def __init__(self, eps = 1e-8):
        self.km_fitter = KaplanMeierFitter()
        self.eps = float(eps)
        self._fitted = False

    def fit(self, durations, delta):
        # fitting KM to get the G(t)
        censoring_indicator = (np.asarray(delta) == 0).astype(int) # censor 1, else 0
        # no need to reverse now the status, since 1 is censoring
        self.km_fitter.fit(durations = np.asarray(durations, float), event_observed = censoring_indicator)
        self._fitted = True
        return self
    
    def _check_fitted(self):
        # need to be fitted before in can take the prediction over a range of t
        if not self._fitted:
            raise RuntimeError("KM_IPCW not fitted. Use fit.")

    def G(self, t):
        self._check_fitted()
        val = np.array(self.km_fitter.predict(t), float)
        return np.clip(val, self.eps, 1.0)
    

# to add here any other class 
class CompRiskMetricsCompute:
    """Compute IPCW weighted metrics for competing risks """
    # we need the training for this part
    def __init__(self, time_grid: np.ndarray, 
                 durations_train: np.ndarray,
                 delta_train: np.ndarray,
                 stabilize: bool = True, 
                 eps: float = 1e-8):
        self.time_grid = np.asarray(time_grid, float)
        if np.any(np.diff(self.time_grid) <= 0):
            raise ValueError("time_grid must be strictly increasing.")
        self.stabilize = bool(stabilize)
        # if we calculate the G here, then we use the same one for all the metrics, and we do not need to recalculate
        self.censoring = KM_CensoringDistribution(eps = eps).fit(durations = durations_train, delta = delta_train)
        self.G = self.censoring.G(self.time_grid)
        self.ipcw = 1/self.G


    def brier_cif(self, pred_cif: np.ndarray, durations_val: np.ndarray, delta_val: np.ndarray, cause = int):
        T = np.asarray(durations_val, float)
        D = np.asarray(delta_val, int)
        F = np.asarray(pred_cif, float)

        n, m = F.shape
        if (n, m) != (T.size, self.time_grid.size):
            raise ValueError("The predicted CIF should be of (n_patients, len(time_grid))")
        # Here we can now calculate the subject specific weight 1/G(T_i)

        G_T = self.censoring.G(T)
        # avoiding 0s
        G_T = np.clip(G_T, self.censoring.eps, 1.0) 
        ipcw_T = 1/G_T

        any_event = (D != 0) 
        is_k      = (D == cause) # this is death by the event of interest
        is_other  = any_event & (~is_k) # this is death by the competing risk

        bs = np.empty(m, float)
        for j, tj in enumerate(self.time_grid):
            Fk = F[:,j]
            ipcw_t = self.ipcw[j] # 1/G(tj) 
            alive = (T > tj)

            numer = (
                np.sum(alive * (Fk**2) * ipcw_t) +
                np.sum(((T <= tj) & is_k) * ((1.0 - Fk)**2) * ipcw_t) +
                np.sum(((T <= tj) & is_other) * (Fk**2) * ipcw_t)
            )
        # sometimes dividing by n can be unstable 
        if self.stabilize:
            denom = np.sum(alive * ipcw_t + ((T <= tj) & any_event) * ipcw_t)
        else:
            denom = n

        bs[j] = np.nan if denom <= 0 else numer / denom

        return bs
        

    def integrated_brier_cif(self, pred_cif, durations_val, delta_val, cause_k: int, return_bs: bool = False):
        bs = self.brier_cif(pred_cif, durations_val, delta_val, cause_k)
        area = np.trapz(bs, self.time_grid)
        ibs = area / (self.time_grid[-1] - self.time_grid[0])
        return (ibs, bs) if return_bs else ibs
