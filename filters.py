from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import pandas as pd

def gpr_fill(y):
    y = pd.Series(y)
    X = y.index.values.reshape(-1, 1)

    known_mask = y.notna()
    X_known = X[known_mask]
    y_known = y[known_mask]

    X_missing = X[~known_mask]

    # Kernel: simaság (RBF) + zaj
    kernel = RBF(length_scale=10.0) + WhiteKernel(noise_level=1e-2)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

    gpr.fit(X_known, y_known)
    y_pred = gpr.predict(X_missing)

    y_filled = y.copy()
    y_filled[~known_mask] = y_pred
    return y_filled

def gpr_fill_local(y, context=20):
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
                # Lokális ismert minta összeállítása
                X_known = np.concatenate([left.index, right.index]).reshape(-1, 1)
                y_known = np.concatenate([left.values, right.values])

                X_missing = np.arange(start, end + 1).reshape(-1, 1)

                # GPR betanítás és predikció
                kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=1e-2)
                gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

                gpr.fit(X_known, y_known)
                y_pred = gpr.predict(X_missing)

                y_filled[start:end + 1] = y_pred

    return y_filled