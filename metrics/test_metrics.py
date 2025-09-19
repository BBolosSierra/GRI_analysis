import numpy as np
import matplotlib.pyplot as plt
from brier_score import KM_CensoringDistribution

rng= np.random.default_rng(42)

def synthetic_censoring_times(n=1000, lam_event=0.3, lam_cens=0.2):
    """ T ~ Exp(lam_event), C ~ Exp(lamb_cens)
    The observed time = min(T, C). 
    Event indicator depends on the observed time. """

    T = rng.exponential(lam_event, size=n)
    C = rng.exponential(lam_cens, size=n)
    
    T = np.asarray(T, dtype=float)
    C = np.asarray(C, dtype=float)

    Tobs = np.minimum(T, C)
    delta = np.where(T <= C, 1, 0).astype(int)
    return Tobs, delta

def main():
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

if __name__ == "__main__":
    main()



