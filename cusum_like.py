import numpy as np
from numpy.linalg import inv


def compute_llr(x_t, mu0, mu1, Sigma):
    """log-likelihood ratio assuming known shared covariance."""
    delta = (x_t - mu0).T @ inv(Sigma) @ (x_t - mu0) - (x_t - mu1).T @ inv(Sigma) @ (x_t - mu1)
    return 0.5 * delta



def online_cusum_like(data, window_size=65, alpha=0.01, h=5.0, drift=0.0):
    """Online change point detection using CUSUM with likelihood function."""
    alarms = []
    t = window_size
    g = 0
    T = len(data)
    scores = []
    windows = []
    llrs = []

    while t < T:
        ref_window = data[t - window_size:t]
        window_list = list(range(t - window_size, t))
        windows.append(window_list)
        Sigma = np.cov(ref_window.T)
        mu0 = ref_window.mean(axis=0)
        mu1 = mu0.copy() 
        g = 0  #update the assumed params and reset score after each shift

        while t < T:
            x_t = data[t]

            #update the new mean with EWMA
            mu1 = (1 - alpha) * mu1 + alpha * x_t

            score = compute_llr(x_t, mu0, mu1, Sigma)
            llrs.append(score - drift)
            g = max(0, g + score - drift)
            scores.append(g)
            if g > h:
                alarms.append(t-window_size)
                # Reset reference window starting from alarm point
                if t < T:
                    t += 1  # move to next time to avoid loop freeze
                    break
                else:
                    return alarms, scores  # can't build new reference window
            t += 1
    return alarms, scores, windows
