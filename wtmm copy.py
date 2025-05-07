import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import streamlit as st

# --- Numba-kompatibilis alapfüggvények ---

@njit
def pad_or_truncate(arr, target_len):
    result = np.zeros(target_len)
    n = min(len(arr), target_len)
    for i in range(n):
        result[i] = arr[i]
    return result

@njit
def convolve_same(signal, kernel):
    N = len(signal)
    M = len(kernel)
    output = np.zeros(N)
    half = M // 2
    for i in range(N):
        s = 0.0
        for j in range(M):
            k = i - half + j
            if 0 <= k < N:
                s += signal[k] * kernel[j]
        output[i] = s
    return output

@njit
def generate_ricker_wavelet(length, a):
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    x = np.linspace(-length // 2, length // 2, length)
    mod = 1.0 - (x ** 2) / wsq
    gauss = np.exp(-x ** 2 / (2.0 * wsq))
    return A * mod * gauss

@njit
def fast_ricker_cwt_numba(signal, scales):
    N = len(signal)
    num_scales = len(scales)
    cwt_matrix = np.zeros((num_scales, N))
    for idx in range(num_scales):
        scale = scales[idx]
        wavelet_length = min(10 * scale, N)
        if wavelet_length % 2 == 0:
            wavelet_length += 1
        wavelet = generate_ricker_wavelet(wavelet_length, scale)
        conv_result = convolve_same(signal, wavelet)
        cwt_matrix[idx, :] = pad_or_truncate(conv_result, N)
    return cwt_matrix

# --- Skála és q optimalizáló ---
def estimate_optimal_scales(signal_len):
    return np.arange(2, max(16, signal_len // 10))

def estimate_optimal_q_values():
    return np.linspace(-3, 3, 21)

# --- WTMM multifraktál spektrum ---

def wtmm_multifractal_spectrum(signal, scales=None, q_values=None, plot=True, streamlit_mode=True):
    signal = np.asarray(signal)
    N = len(signal)
    time = np.linspace(0, 1, N)

    if scales is None:
        scales = estimate_optimal_scales(N)
    if q_values is None:
        q_values = estimate_optimal_q_values()

    coef = fast_ricker_cwt_numba(signal, scales)

    def find_modulus_maxima(coef_matrix):
        modulus = np.abs(coef_matrix)
        maxima = np.zeros_like(modulus)
        for scale_idx in range(modulus.shape[0]):
            local_max_idx = argrelextrema(modulus[scale_idx], np.greater)[0]
            maxima[scale_idx, local_max_idx] = modulus[scale_idx, local_max_idx]
        return maxima

    maxima = find_modulus_maxima(coef)

    Z_q_s = np.zeros((len(q_values), len(scales)))
    for i, q in enumerate(q_values):
        for j in range(len(scales)):
            maxima_scale = maxima[j, :]
            maxima_scale = maxima_scale[maxima_scale > 0]
            if len(maxima_scale) > 0:
                Z_q_s[i, j] = np.sum(maxima_scale ** q)
            else:
                Z_q_s[i, j] = np.nan

    tau_q = np.zeros(len(q_values))
    log_scales = np.log2(scales)
    for i in range(len(q_values)):
        valid = ~np.isnan(Z_q_s[i, :]) & (Z_q_s[i, :] > 0)
        if np.sum(valid) > 2:
            slope, _ = np.polyfit(log_scales[valid], np.log2(Z_q_s[i, valid]), 1)
            tau_q[i] = slope
        else:
            tau_q[i] = np.nan

    alpha = np.diff(tau_q) / np.diff(q_values)
    q_mid = (q_values[1:] + q_values[:-1]) / 2
    f_alpha = q_mid * alpha - tau_q[1:]

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        ax1.plot(time, signal, color='black')
        ax1.set_title('Eredeti jel')
        ax1.set_ylabel('Érték')
        extent = [time[0], time[-1], scales[-1], scales[0]]
        ax2.imshow(np.abs(coef), extent=extent, cmap='jet', aspect='auto')
        ax2.set_title('Continuous Wavelet Transform (Ricker)')
        ax2.set_ylabel('Skálák')
        ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto')
        ax3.set_title('WTMM modulus maxima')
        ax3.set_ylabel('Skálák')
        ax3.set_xlabel('Idő')
        plt.tight_layout()
        if streamlit_mode:
            st.pyplot(fig)
        else:
            plt.show()

        plt.figure(figsize=(8, 6))
        #valid = np.isfinite(alpha) & np.isfinite(f_alpha)
        #plt.plot(alpha[valid], f_alpha[valid], '-o', color='navy')
        plt.plot(alpha, f_alpha, '-o', color='navy')
        plt.xlabel('α (Szingularitás erőssége)')
        plt.ylabel('f(α) (Spektrum)')
        plt.title('Multifraktál spektrum (WTMM +CWT)')
        plt.grid(True)
        if streamlit_mode:
            st.pyplot(plt)
        else:
            plt.show()

    return alpha, f_alpha, tau_q, Z_q_s, q_values, scales, coef, maxima
