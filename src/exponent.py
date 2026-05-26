import numpy as np
from scipy.optimize import curve_fit


def exponential(df):
    qi=df["OIL"].iloc[0]
    def exp_func(t, Di):
        return qi * np.exp(-Di * t)
    popt,pcov=curve_fit(exp_func,df['t'],df['OIL'],p0=[0.01])
    di=popt[0]
    return di