from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter

def scale_invariant_transform(signal):
    """Skálafüggetlen állapot: zéró középérték, egységnyi variancia"""
    signal = signal - np.mean(signal)
    return signal / (np.std(signal) + 1e-8)

def gpr_fill(y):
    """
    Hiányzó értékek globális kitöltése Gaussian Process Regression segítségével.
    """
    y = pd.Series(y)
    X = y.index.values.reshape(-1, 1)

    known_mask = y.notna()
    X_known = X[known_mask]
    y_known = y[known_mask]

    X_missing = X[~known_mask]

    kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-2)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    gpr.fit(X_known, y_known)
    y_pred = gpr.predict(X_missing)

    y_filled = y.copy()
    y_filled[~known_mask] = y_pred
    return y_filled


def gpr_fill_local(y, context=20):
    """
    Hiányzó szakaszok lokális GPR interpolációja szomszédos ismert értékek alapján.
    """
    y = pd.Series(y)
    y_filled = y.copy()
    is_nan = y.isna()
    group = (is_nan != is_nan.shift()).cumsum()

    for gid, segment in y.groupby(group):
        if segment.isna().all():
            start = segment.index[0]
            end = segment.index[-1]

            left = y[max(0, start - context):start].dropna()
            right = y[end + 1:end + 1 + context].dropna()

            if not left.empty and not right.empty:
                X_known = np.concatenate([left.index, right.index]).reshape(-1, 1)
                y_known = np.concatenate([left.values, right.values])
                X_missing = np.arange(start, end + 1).reshape(-1, 1)

                kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=1e-2)
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

                gpr.fit(X_known, y_known)
                y_pred = gpr.predict(X_missing)

                y_filled[start:end + 1] = y_pred

    return y_filled


def butter_lowpass_filter(signal, cutoff=30.0, fs=250.0, order=5):
    """
    Butterworth aluláteresztő szűrő.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def normalize_signal(signal):
    """
    Z-score standardizálás.
    """
    signal = np.asarray(signal)
    return (signal - np.mean(signal)) / np.std(signal)


def moving_average(signal, window_size=5):
    """
    Egyszerű csúszóátlag szűrés.
    """
    signal = np.asarray(signal)
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


def savitzky_golay_filter(signal, window_length=11, polyorder=2):
    """
    Savitzky-Golay simító szűrő.
    """
    signal = np.asarray(signal)
    if window_length % 2 == 0:
        window_length += 1  # biztosan páratlan legyen
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def log_scale_transform(signal):
    """
    Logaritmikus skálázás (log(1 + |x|) jellel).
    """
    signal = np.asarray(signal)
    return np.sign(signal) * np.log1p(np.abs(signal))
