import numpy as np 
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.nonparametric import kaplan_meier_estimator

from scipy.interpolate import interp1d

import warnings


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
        
        # Lifelines:
        #censoring_indicator = (np.asarray(delta) == 0).astype(int) # censor 1, else 0
        # no need to reverse now the status, since 1 is censoring
        #self.km_fitter.fit(durations = np.asarray(durations, float), event_observed = censoring_indicator)
        #sf = self.km_fitter.survival_function_["KM_estimate"].to_numpy(float)
        #tt = self.km_fitter.survival_function_.index.to_numpy(float)

        # sksurv
        durations = np.asarray(durations, dtype=float)
        cens_event = (np.asarray(delta) == 0).astype(bool)  # True if censored
        # sksurv returns times and survival as numpy arrays
        tt, sf = kaplan_meier_estimator(event=cens_event, time_exit=durations)
        
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
                           cause : int):
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


"""From hazardous"""

from sklearn.utils.validation import check_scalar
from numbers import Integral


def check_y_survival(y):
    """Convert DataFrame and dictionnary to record array."""
    y_keys = ["event", "duration"]

    if (
        isinstance(y, np.ndarray)
        and sorted(y.dtype.names, reverse=True) == y_keys
        or isinstance(y, dict)
        and sorted(y, reverse=True) == y_keys
    ):
        return np.ravel(y["event"]), np.ravel(y["duration"])

    elif isinstance(y, pd.DataFrame) and sorted(y.columns, reverse=True) == y_keys:
        return y["event"].values, y["duration"].values

    else:
        raise ValueError(
            "y must be a record array, a pandas DataFrame, or a dict "
            "whose dtypes, keys or columns are 'event' and 'duration'. "
            f"Got:\n{repr(y)}"
        )
def check_event_of_interest(k):
    """`event_of_interest` must be the string 'any' or a positive integer."""
    check_scalar(k, "event_of_interest", target_type=(str, Integral))
    not_str_any = isinstance(k, str) and k != "any"
    not_positive = isinstance(k, int) and k < 1
    if not_str_any or not_positive:
        raise ValueError(
            "event_of_interest must be a strictly positive integer or 'any', "
            f"got: event_of_interest={k}"
        )
    return
class KaplanMeierIPCW:
    """Estimate the Inverse Probability of Censoring Weight (IPCW).

    This class estimates the inverse probability of 'survival' to censoring using the
    Kaplan-Meier estimator applied to a binary indicator for censoring, defined as the
    negation of the binary indicator for any event occurrence. This estimator assumes
    that the censoring distribution is independent of the covariates X. If this
    assumption is violated, the estimator may be biased, and a conditional estimator
    might be more appropriate.

    This approach is useful for correcting the bias introduced by right censoring in
    survival analysis, particularly when computing model evaluation metrics such as
    the Brier score or the concordance index.

    Note that the term 'IPCW' can be somewhat misleading: IPCW values represent the
    inverse of the probability of remaining censor-free (or uncensored) at a given time.
    For instance, at t=0, the probability of being censored is 0, so the probability of
    being uncensored is 1.0, and its inverse is also 1.0.

    By construction, IPCW values are always greater than or equal to 1.0 and can only
    increase over time. If no observations are censored, the IPCW values remain
    uniformly at 1.0.

    Note: This estimator extrapolates by maintaining a constant value equal to the last
    observed IPCW value beyond the last recorded time point.

    Parameters
    ----------
    epsilon_censoring_prob : float, default=0.05
        Lower limit of the predicted censoring probabilities. It helps avoiding
        instabilities during the division to obtain IPCW.

    Attributes
    ----------
    min_censoring_prob_ : float
        The effective minimal probability used, defined as the max between
        min_censoring_prob and the minimum predicted probability.

    unique_times_ : ndarray of shape (n_unique_times,)
        The observed censoring durations from the training target.

    censoring_survival_probs_ : ndarray of shape (n_unique_times,)
        The estimated censoring survival probabilities.

    censoring_survival_func_ : callable
        The linear interpolation function defined with unique_times_ (x) and
        censoring_survival_probs_ (y).
    """

    def __init__(self, epsilon_censoring_prob=0.05):
        self.epsilon_censoring_prob = epsilon_censoring_prob

    def fit(self, y, X=None):
        """Marginal estimation of the censoring survival function

        In addition to running the Kaplan-Meier estimator on the negated event
        labels (1 for censoring, 0 for any event), this methods also fits
        interpolation function to be able to make prediction at any time.

        Parameters
        ----------
        y : array-like of shape (n_samples, 2)
            The target data.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        event, duration = check_y_survival(y)
        censoring = event == 0

        km = KaplanMeierFitter()
        km.fit(
            durations=duration,
            event_observed=censoring,
        )

        df = km.survival_function_
        self.unique_times_ = df.index
        self.censoring_survival_probs_ = df.values[:, 0]

        min_censoring_prob = self.censoring_survival_probs_[
            self.censoring_survival_probs_ > 0
        ].min()

        self.min_censoring_prob_ = max(
            min_censoring_prob,
            self.epsilon_censoring_prob,
        )
        self.censoring_survival_func_ = interp1d(
            self.unique_times_,
            self.censoring_survival_probs_,
            kind="previous",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return self

    def compute_ipcw_at(self, times, X=None, ipcw_training=False):
        """Estimate the inverse probability of censoring weights at given time horizons.

        Compute the inverse of the linearly interpolated censoring survival
        function.

        Parameters
        ----------
        times : np.ndarray of shape (n_samples,)
            The input times for which to predict the IPCW for each sample.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        Returns
        -------
        ipcw : np.ndarray of shape (n_samples,)
            The IPCW for each sample at each time horizon.
        """

        cs_prob = self.compute_censoring_survival_proba(
            times,
            X=X,
            ipcw_training=ipcw_training,
        )
        cs_prob = np.clip(cs_prob, self.min_censoring_prob_, 1)
        return 1 / cs_prob

    def compute_censoring_survival_proba(self, times, X=None, ipcw_training=False):
        """Estimate probability of not experiencing censoring at times.

        Linearly interpolate the censoring survival function.

        Parameters
        ----------
        times : np.ndarray of shape (n_times,)
            The input times for which to predict the IPCW.

        X : None
            The input samples. Unused since this estimator is non-conditional.

        ipcw_training : bool, default=False
            Unused.

        Returns
        -------
        ipcw : np.ndarray of shape (n_times,)
            The IPCW for times
        """
        return self.censoring_survival_func_(times)


class IncidenceScoreComputer:
    """Censoring adjusted, time-dependent scoring rules.

    This class factorizes the computation of scoring rules such as the
    time-dependent Brier score for single-event or any event survival functions
    and cause-specific cumulative incidence functions.

    It leverages the Inverse Probability of Censoring Weighting (IPCW) scheme
    using a Kaplan-Meier of the censoring distribution to weight the terms.

    Parameters
    ----------
    y_train : array-like of shape (n_samples, 2)
        The target data, used to fit the IPCW estimator.

    event_of_interest : int or "any", default="any"
        The event to consider in a competing events setting.

        "any" indicates that all events except the censoring marker 0 are considered
        collapsed together as a single event. In a single event (survival) setting,
        "any" and 1 are equivalent.

    ipcw_estimator : object, default=None
        The estimator used to compute the IPCW. If set to ``None``,
        the ``KaplanMeierIPCW`` is used.
    """

    def __init__(
        self,
        y_train,
        event_of_interest="any",
        ipcw_estimator=None,
    ):
        self.y_train = y_train
        self.event_train, self.duration_train = check_y_survival(y_train)
        self.event_ids_ = np.unique(self.event_train)
        self.any_event_train = self.event_train > 0
        self.event_of_interest = event_of_interest

        y = dict(
            event=self.any_event_train,
            duration=self.duration_train,
        )
        # Estimate the censoring distribution from the training set.
        if ipcw_estimator is None:
            ipcw_estimator = KaplanMeierIPCW()
        self.ipcw_estimator = ipcw_estimator.fit(y)

    def brier_score_survival(self, y_true, y_pred, times):
        """Time-dependent Brier score of a survival function estimate.

        Compute the time-dependent Brier score value for each individual and
        each time point in times and then average over individuals.

        This estimate is adjusted for censoring by leveraging the Inverse
        Probability of Censoring Weighting (IPCW) scheme.

        Parameters
        ----------
        y_true : record-array, dict or dataframe of shape (n_samples, 2)
            The ground truth, consisting in the 'event' and 'duration' columns.
            In a survival setting, we expect the event to be a binary
            indicator: 1 for the event of interest and 0 for censoring.
            Alternatively, all competing event types should be collapsed by
            setting event_of_interest="any".

        y_pred : array-like of shape (n_samples, n_times)
            Survival probability estimates predicted at times. In the
            binary event settings, this is 1 - incidence_probability.

        times : array-like of shape (n_times)
            Times to estimate the survival probability and to compute the Brier
            Score.

        Returns
        -------
        brier_score : np.ndarray of shape (n_times)
            Time-dependent Brier scores averaged over the individuals.

        """
        if (self.event_ids_ > 0).sum() > 1 and self.event_of_interest != "any":
            warnings.warn(
                "Computing the survival Brier score only makes "
                "sense with a binary event indicator or when setting "
                "event_of_interest='any'. "
                "Instead this model is evaluated on data with event ids "
                f"{self.event_ids_.tolist()} and with "
                f"event_of_interest={self.event_of_interest}."
            )
        return self.brier_score_incidence(y_true, 1 - y_pred, times)

    def brier_score_incidence(self, y_true, y_pred, times):
        """Brier score for the cause-specific cumulative incidence function.

        Compute the Brier score values with IPCW adjustment for censoring for
        each cumulative incidence estimate for the event of interest and each
        requested time point and return the time-dependent Brier score averaged
        over individuals.

        Parameters
        ----------
        y_true : record-array, dictionary or dataframe of shape (n_samples, 2)
            The ground truth, consisting in the 'event' and 'duration' columns.

        y_pred : array-like of shape (n_samples, n_times)
            Cause-specific cumulative incidence estimates predicted at
            times for the event of interest. In the single event type
            settings, or when event_of_interest == "any", this is 1 -
            survival_probability.

        times : array-like of shape (n_times)
            Times to estimate the survival probability and to compute the Brier
            score.

        Returns
        -------
        brier_score_incidence : np.ndarray
            Average value of the time-dependent Brier scores computed at time
            locations specified in the times argument.
        """
        event_true, duration_true = check_y_survival(y_true)
        check_event_of_interest(self.event_of_interest)

        if self.event_of_interest == "any":
            if y_true is self.y_train:
                event_true = self.any_event_train
            else:
                event_true = event_true > 0

        if y_pred.ndim != 2:
            raise ValueError(
                "'y_pred' must be a 2D array with shape (n_samples, n_times), got"
                f" shape {y_pred.shape}."
            )
        if y_pred.shape[0] != event_true.shape[0]:
            raise ValueError(
                "'y_true' and 'y_pred' must have the same number of samples, "
                f"got {event_true.shape[0]} and {y_pred.shape[0]} respectively."
            )
        times = np.atleast_1d(times)
        if y_pred.shape[1] != times.shape[0]:
            raise ValueError(
                f"'times' length ({times.shape[0]}) "
                f"must be equal to y_pred.shape[1] ({y_pred.shape[1]})."
            )

        n_samples = event_true.shape[0]
        n_time_steps = times.shape[0]
        brier_scores = np.empty(
            shape=(n_samples, n_time_steps),
            dtype=np.float64,
        )
        ipcw_y = self.ipcw_estimator.compute_ipcw_at(duration_true)
        for t_idx, t in enumerate(times):
            y_true_binary, weights = self._weighted_binary_targets(
                y_event=event_true,
                y_duration=duration_true,
                times=np.full(shape=n_samples, fill_value=t),
                ipcw_y_duration=ipcw_y,
            )
            # XXX: refactor and rename this function to make it possible to
            # also compute the time-dependent binary cross-entropy loss.
            squared_error = (y_true_binary - y_pred[:, t_idx]) ** 2
            brier_scores[:, t_idx] = weights * squared_error

        return brier_scores.mean(axis=0)

    def integrated_brier_score_incidence(self, y_true, y_pred, times):
        brier_scores = self.brier_score_incidence(
            y_true,
            y_pred,
            times,
        )
        return self._time_integrated(brier_scores, times)

    def _time_integrated(self, scores, times):
        ordering = np.argsort(times)
        sorted_times = times[ordering]
        sorted_scores = scores[ordering]
        time_span = sorted_times[-1] - sorted_times[0]
        return np.trapz(sorted_scores, sorted_times) / time_span

    def _weighted_binary_targets(
        self,
        y_event,
        y_duration,
        times,
        ipcw_y_duration,
        ipcw_training=False,
        X=None,
    ):
        if self.event_of_interest == "any":
            # y should already be provided as binary indicator
            k = 1
        else:
            k = self.event_of_interest

        # Specify the binary classification target for each record in y and a
        # reference time horizon:
        #
        # - 1 when event of interest was observed before the reference time
        #   horizon,
        #
        # - 0 otherwise: any competing event happening at any time, censored
        #   record or event of interest happening after the reference time
        #   horizon.
        #
        #   Note: censored events only contribute (as negative target) when
        #   their duration is larger than the reference target horizon.
        #   Otherwise, they are discarded by setting their weight to 0 in the
        #   following.
        #
        #   Contrary to censored records, competing events always contribute as
        #   negative targets. There weight is always non-zero but differ if
        #   they happen either before or after the reference time horizon.
        #
        # This IPCW scheme for survival analysis (binary events) is described
        # in [Graf1999] and is extended to multiple competing events in
        # [Kretowska2018].
        event_k_before_horizon = (y_event == k) & (y_duration <= times)
        y_binary = event_k_before_horizon.astype(np.int32)

        ipcw_times = self.ipcw_estimator.compute_ipcw_at(
            times,
            X=X,
            ipcw_training=ipcw_training,
        )
        any_event_or_censoring_after_horizon = y_duration > times
        weights = np.where(any_event_or_censoring_after_horizon, ipcw_times, 0)

        any_observed_event_before_horizon = (y_event > 0) & (y_duration <= times)
        weights = np.where(any_observed_event_before_horizon, ipcw_y_duration, weights)

        return y_binary, weights


def brier_score_incidence(
    y_train,
    y_test,
    y_pred,
    times,
    event_of_interest="any",
):
    r"""Time-dependent Brier score for the kth cause of event.

    .. math::

        \mathrm{BS}_k(t) = \frac{1}{n} \sum_{i=1}^n \hat{\omega}_i(t)
        (\mathbb{I}(t_i \leq t, \delta_i = k) - \hat{F}_k(t|\mathbf{x}_i))^2

    where :math:`\hat{F}_k(t | \mathbf{x}_i)` is an estimate of the
    (uncensored) cumulative incidence for the kth event up to time point
    :math:`t` for a feature vector :math:`\mathbf{x}_i` [Edwards2016]_:

    .. math::

            \hat{F}_k(t | \mathbf{x}_i) \approx P(T_i \leq t, \Delta_i = k |
            \mathbf{x}_i)

    and :math:`\hat{\omega}_i(t)` are IPCW weigths based on the Kaplan-Meier
    estimate of the censoring distribution :math:`\hat{G}(t)`:

    .. math::

        \hat{\omega}_i(t)=\frac{\mathbb{I}(t_i \leq t, \delta_i \neq
        0)}{\hat{G}(t_i)} + \frac{\mathbb{I}(t_i > t)}{\hat{G}(t)}

    This scheme was introduced in [Graf1999]_ in the context of survival
    analysis and extended to competing events in [Kretowska2018]_.

    Note that this assumes independence between censoring and the covariates.
    When this assumption is violated, the IPCW weights are biased and the Brier
    score is not a proper scoring rule anymore. See [Gerds2006]_ for a study of
    this bias.

    Parameters
    ----------
    y_train : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The target, consisting in the 'event' and 'duration' columns. This is
        used to fit the IPCW estimator.

    y_test : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The ground truth, consisting in the 'event' and 'duration' columns. In
        the "event" column, `0` indicates censoring, and any other values
        indicate competing event types.

    y_pred : array-like of shape (n_samples, n_times)
        Incidence probability estimates predicted at ``times``. In the binary
        event settings, this is 1 - survival_probability.

    times : array-like of shape (n_times)
        Times at which the survival probability ``y_pred`` has been estimated
        and for which we compute the Brier score.

    event_of_interest : int or "any", default="any"
        The event to consider in a competing events setting. When an integer,
        this should be one of the non-zero values in the "event" column of
        ``y_train`` and ``y_test``.

        ``"any"`` indicates that all events except the censoring marker ``0``
        are considered collapsed together as a single event. In a single event
        setting, ``"any"`` and ``1`` are equivalent.

    Returns
    -------
    brier_score : np.ndarray of shape (n_times)

    See Also
    --------
    integrated_brier_score_incidence : Time-integrated Brier score for the kth
        cause of event.

    References
    ----------
    .. [Graf1999] E. Graf, C. Schmoor, W. Sauerbrei, M. Schumacher, "Assessment
       and comparison of prognostic classification schemes for survival data",
       1999

    .. [Kretowska2018] M. Kretowska, "Tree-based models for survival data with
       competing risks", 2018

    .. [Gerds2006] T. Gerds and M. Schumacher, "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times", 2006

    .. [Edwards2016] J. Edwards, L. Hester, M. Gokhale, C. Lesko,
       "Methodologic Issues When Estimating Risks in Pharmacoepidemiology.",
       2016, doi:10.1007/s40471-016-0089-1
    """
    # XXX: make times an optional kwarg to be compatible with
    # sksurv.metrics.brier_score?
    # XXX: 'times' must match the times of y_pred,
    # but we have no way to check that.
    # In this sense, 'y_pred[:, t_idx]' is incorrect when 'times'
    # is not the time used during the prediction.
    computer = IncidenceScoreComputer(
        y_train,
        event_of_interest=event_of_interest,
    )
    return computer.brier_score_incidence(y_test, y_pred, times)


def integrated_brier_score_incidence(
    y_train,
    y_test,
    y_pred,
    times,
    event_of_interest="any",
):
    r"""Time-integrated Brier score of a cause-specific cumulative incidence estimate.

    .. math::

        \mathrm{IBS}_k = \frac{1}{t_{max} - t_{min}} \int^{t_{max}}_{t_{min}}
        \mathrm{BS}_k(u) du

    This scheme was introduced in [Graf1999]_ for survival analysis and
    extended to competing events in [Kretowska2018]_.

    Note that this assumes independence between censoring and the covariates.
    When this assumption is violated, the IPCW weights are biased and the Brier
    score is not a proper scoring rule anymore. See [Gerds2006]_ for a study of
    this bias.

    Parameters
    ----------
    y_train : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The target, consisting in the 'event' and 'duration' columns.
        This is used to fit the IPCW estimator.

    y_test : record-array, dictionnary or dataframe of shape (n_samples, 2)
        The ground truth, consisting in the 'event' and 'duration' columns.
        In the "event" column, `0` indicates censoring, and any other values
        indicate competing event types.

    y_pred : array-like of shape (n_samples, n_times)
        Incidence probability estimates predicted at ``times``.
        In the binary event settings, this is 1 - survival_probability.

    times : array-like of shape (n_times)
        Times at which the survival probabilities ``y_pred`` has been estimated
        and for which we compute the Brier score.

    event_of_interest : int or "any", default="any"
        The event to consider in a competing events setting. When an integer,
        this should be one of the non-zero values in the "event" column of
        ``y_train`` and ``y_test``.

        ``"any"`` indicates that all events except the censoring marker ``0``
        are considered collapsed together as a single event. In a single event
        setting, ``"any"`` and ``1`` are equivalent.

    Returns
    -------
    ibs : float

    See Also
    --------
    brier_score_incidence : Time-dependent Brier score for the kth cause of event.

    References
    ----------
    .. [Graf1999] E. Graf, C. Schmoor, W. Sauerbrei, M. Schumacher, "Assessment
       and comparison of prognostic classification schemes for survival data",
       1999

    .. [Kretowska2018] M. Kretowska, "Tree-based models for survival data with
       competing risks", 2018

    .. [Gerds2006] T. Gerds and M. Schumacher, "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times", 2006
    """
    computer = IncidenceScoreComputer(
        y_train,
        event_of_interest=event_of_interest,
    )
    return computer.integrated_brier_score_incidence(y_test, y_pred, times)