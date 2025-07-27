from sklearn.metrics.pairwise import rbf_kernel



def kernel_mmd2(X, Y, gamma):
    """Compute MMD^2 between X and Y using RBF kernel."""
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd2

def kernel_cusum_score(ref_window, new_point, gamma, nu):
    """One-step kernel CUSUM score update."""
    mmd2 = kernel_mmd2(ref_window, new_point[None, :], gamma)
    return mmd2 - nu

def online_kernel_cusum(data, window_size=65, gamma=0.5, nu=0.5, h=3.0):
    """Online change point detection using kernel CUSUM."""
    alarms = []
    t = window_size
    g = 0
    T = len(data)
    scores = []
    windows = []
    mmd2s = []

    while t < T:
        ref_window = data[t - window_size:t]
        window_list = list(range(t - window_size, t))
        windows.append(window_list)
        g = 0  # Reset score after every reference window shift

        while t < T:
            x_t = data[t]
            score = kernel_cusum_score(ref_window, x_t, gamma, nu)
            mmd2s.append(score + nu)
            g = max(0, g + score)
            scores.append(g)
            if g > h:
                alarms.append(t-window_size)
                # Reset reference window starting from alarm point
                if t < T:
                    t += 1  # move to next time to avoid loop freeze
                    break
                else:
                    return alarms, scores  # can't build new reference windo
            t += 1
    return alarms, scores, windows

