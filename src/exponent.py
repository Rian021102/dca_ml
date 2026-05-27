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