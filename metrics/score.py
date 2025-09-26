import numpy as np 
from lifelines import KaplanMeierFitter

class KM_CensoringDistribution:
    """Estimation of the Inverse probability censoring weights (IPCW) from the Kaplan Meier.
    
    The IPCW are calculated as the inverse of the probability of being uncensored: 1/G(T), where G(t) 
    is estimated with the Kaplan Meier (it is a population level estimate, and does not depend on 
    covariates). For competing risks we will consider event of interest and competing risk as death, to 
    calculate the uncensoring discribution. 
    
    It is calculated to correct the bias on the performance metrics caused by the right-censoring. """

    def __init__(self, eps = 0.01):
        self.km_fitter = KaplanMeierFitter()
        self.eps = float(eps)
        self._fitted = False

    def fit(self, durations, delta):
        # 0=censor, >0=event(any)
        # fitting KM to get the G(t)
        censoring_indicator = (np.asarray(delta) == 0).astype(int) # censor 1, else 0
        # no need to reverse now the status, since 1 is censoring
        self.km_fitter.fit(durations = np.asarray(durations, float), event_observed = censoring_indicator)
        # keep the 
        sf = self.km_fitter.survival_function_["KM_estimate"].to_numpy(float)
        tt = self.km_fitter.survival_function_.index.to_numpy(float)
        sf = np.minimum.accumulate(sf)
        sf = np.clip(sf, self.eps, 1.0)
        self._times = tt
        self._Gvals = sf
        self._fitted = True
        return self
    
    def _check_fitted(self):
        # need to be fitted before in can take the prediction over a range of t
        if not self._fitted:
            raise RuntimeError("KM_IPCW not fitted. Use fit.")

    def G(self, t): # where t is the time grid
        # for G(t)
        self._check_fitted()
        t = np.asarray(t, float)
        idx = np.searchsorted(self._times, t, side="left")
        idx = np.clip(idx, 0, len(self._Gvals)-1)
        return self._Gvals[idx]
    
    def G_left(self, t):
        # For Ti-
        self._check_fitted()
        t = np.asarray(t, float)
        idx = np.searchsorted(self._times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self._Gvals)-1)
        return self._Gvals[idx]
    

# to add here any other class 
class CompRiskMetricsCompute:
    """Compute IPCW weighted metrics for competing risks """
    # we need the training for this part
    def __init__(self, time_grid: np.ndarray, 
                 durations_train: np.ndarray,
                 delta_train: np.ndarray,
                 eps: float = 0.01):
        self.time_grid = np.asarray(time_grid, float)
        if np.any(np.diff(self.time_grid) <= 0):
            raise ValueError("time_grid must be strictly increasing.")
        # if we calculate the G here, then we use the same one for all the metrics, and we do not need to recalculate
        self.censoring = KM_CensoringDistribution(eps = eps).fit(durations = durations_train, delta = delta_train)
        self.G_t = self.censoring.G(self.time_grid) # G(t)
        self.ipcw_t = 1/self.G_t


    def weighted_brier_cif(self, pred_cif: np.ndarray, 
                           durations_val: np.ndarray, 
                           delta_val: np.ndarray, 
                           cause = int):
        "Computing Brier score with IPCW adjustment which does not exist in sksurv."
        T = np.asarray(durations_val, float)
        D = np.asarray(delta_val, int)
        F = np.asarray(pred_cif, float)

        # if a vector instead
        if F.ndim == 1:
            F = np.tile(F, (T.size, 1))

        n, m = F.shape
        if (n, m) != (T.size, self.time_grid.size):
            raise ValueError("The predicted CIF should be of (n_patients, len(time_grid))")
        
        # Here we can now calculate the subject specific weight 1/G(T_i)
        G_T = self.censoring.G_left(T)
        # avoiding 0s
        G_T = np.clip(G_T, self.censoring.eps, 1.0) 
        ipcw_T = 1.0 / G_T

        any_event = (D != 0)  
        is_k      = (D == cause) # this is death by the event of interest
        is_other  = any_event & (~is_k) # this is death by the competing risk

        bs = np.empty(m, float)
        # accross timepoints
        for j, tj in enumerate(self.time_grid):
            Fk = F[:,j]
            w_t = self.ipcw_t[j]          # 1 / G(t_j+)
            alive = (T > tj)
            evt_le_t = (T <= tj)

            numer = (
                np.sum(alive * (Fk**2) * w_t) +
                np.sum((evt_le_t & is_k)     * ((1.0 - Fk)**2) * ipcw_T) +
                np.sum((evt_le_t & is_other) * (Fk**2)         * ipcw_T)
            )

            bs[j] = numer / n

        return bs

    def integrated_brier_cif(self, 
                             pred_cif, 
                             durations_val, 
                             delta_val, 
                             cause: int, 
                             return_bs: bool = False):
        bs = self.weighted_brier_cif(pred_cif, durations_val, delta_val, cause)
        area = np.trapezoid(bs, self.time_grid)
        ibs = area / (self.time_grid[-1] - self.time_grid[0])
        return (ibs, bs) if return_bs else ibs
