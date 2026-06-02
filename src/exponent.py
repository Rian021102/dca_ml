import numpy as np
from scipy.optimize import curve_fit


def exponential(df, qi=None):
    if qi is None:
        qi = float(df["OIL"].iloc[0])

    t = np.asarray(df["t"], dtype=float)
    t = t - t[0]
    oil = np.asarray(df["OIL"], dtype=float)

    def exp_func(t_arr, di):
        return qi * np.exp(-di * t_arr)

    popt, _ = curve_fit(exp_func, t, oil, p0=[0.01], maxfev=10000)
    di = float(popt[0])
    return di


def exponential_with_uncertainty(df, qi=None, z_score=1.96):
    if qi is None:
        qi = float(df["OIL"].iloc[0])

    t = np.asarray(df["t"], dtype=float)
    t = t - t[0]
    oil = np.asarray(df["OIL"], dtype=float)

    def exp_func(t_arr, di):
        return qi * np.exp(-di * t_arr)

    popt, pcov = curve_fit(exp_func, t, oil, p0=[0.01], maxfev=10000)
    di = float(popt[0])

    di_std = 0.0
    if pcov is not None and np.ndim(pcov) == 2 and pcov.shape[0] > 0 and np.isfinite(pcov[0, 0]):
        di_std = float(np.sqrt(max(pcov[0, 0], 0.0)))

    di_lower = max(di - z_score * di_std, 0.0)
    di_upper = max(di + z_score * di_std, di_lower)
    return di, di_lower, di_upper