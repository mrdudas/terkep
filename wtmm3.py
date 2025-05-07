import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from wtmm import cwt


#@njit
def morlet_wavelet(length, scale, w0=5.0):
    x = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    x = x / scale
    cos_part = np.cos(w0 * x)
    sin_part = np.sin(w0 * x)
    gaussian = np.exp(-x ** 2 / 2)
    wavelet = gaussian * cos_part + 1j * gaussian * sin_part
    norm = np.sqrt(scale * np.sqrt(np.pi))
    return wavelet / norm

@njit
def ricker_wavelet(length, scale):
    x = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    a2 = scale ** 2
    A = 2 / (np.sqrt(3 * scale) * (np.pi ** 0.25))
    wavelet = A * (1 - (x**2) / a2) * np.exp(-x**2 / (2 * a2))
    wavelet /= np.sqrt(scale)
    return wavelet

@njit
def convolve_same_complex(signal, kernel):
    N = len(signal)
    M = len(kernel)
    half = M // 2
    output = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        acc = 0.0 + 0.0j
        for j in range(M):
            k = i - half + j
            if 0 <= k < N:
                acc += signal[k] * kernel[j]
        output[i] = acc
    return output

@njit
def convolve_same_real(signal, kernel):
    N = len(signal)
    M = len(kernel)
    half = M // 2
    output = np.zeros(N)
    for i in range(N):
        acc = 0.0
        for j in range(M):
            k = i - half + j
            if 0 <= k < N:
                acc += signal[k] * kernel[j]
        output[i] = acc
    return output

def cwt_morlet(signal, scales):
    N, S = len(signal), len(scales)
    coef = np.zeros((S, N), dtype=np.complex128)
    for i in range(S):
        scale = scales[i]
        width = int(min(10 * scale, N)) | 1
        wavelet = morlet_wavelet(width, scale)
        coef[i] = convolve_same_complex(signal, wavelet)
    return coef

@njit(parallel=True)
def cwt_ricker(signal, scales):
    N, S = len(signal), len(scales)
    coef = np.zeros((S, N))
    for i in prange(S):
        scale = scales[i]
        width = int(min(10 * scale, N)) | 1
        wavelet = ricker_wavelet(width, scale)
        coef[i] = convolve_same_real(signal, wavelet)
    return coef

@njit(parallel=True)
def modulus_maxima(coef_matrix, threshold_ratio=0.01):
    S, N = coef_matrix.shape
    maxima = np.zeros((S, N))
    for s in prange(S):
        row_abs = np.abs(coef_matrix[s])
        threshold = threshold_ratio * np.max(row_abs)
        for i in range(1, N - 1):
            val = row_abs[i]
            if val > row_abs[i - 1] and val > row_abs[i + 1] and val > threshold:
                maxima[s, i] = val
    return maxima

@njit(parallel=True)
def compute_Z_q_s(maxima, q_values):
    nq, ns = len(q_values), maxima.shape[0]
    Z_q_s = np.zeros((nq, ns))
    for i in prange(nq):
        q = q_values[i]
        for j in range(ns):
            m = maxima[j]
            nonzero = m[m > 0]
            Z_q_s[i, j] = np.sum(nonzero ** q) if len(nonzero) > 0 else np.nan
    return Z_q_s

@njit
def linear_fit(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sum((x - mean_x) ** 2)
    slope = num / den if den else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept

@njit
def compute_tau_q(Z_q_s, log_scales):
    nq = Z_q_s.shape[0]
    tau_q = np.zeros(nq)
    for i in range(nq):
        valid = (~np.isnan(Z_q_s[i])) & (Z_q_s[i] > 0)
        if valid.sum() > 2:
            y, x = np.log2(Z_q_s[i][valid]), log_scales[valid]
            slope, _ = linear_fit(x, y)
            tau_q[i] = slope
        else:
            tau_q[i] = np.nan
    return tau_q

def wtmm_multifractal(signal, scales, q_values, wavelet_type="morlet"):
    if wavelet_type == "morlet":
        coef = cwt_morlet(signal, scales)
    else:
        coef = cwt_ricker(signal, scales)

    maxima = modulus_maxima(coef, threshold_ratio=0.01)
    Z_q_s = compute_Z_q_s(maxima, q_values)
    log_scales = np.log2(scales)
    tau_q = compute_tau_q(Z_q_s, log_scales)

    alpha = np.diff(tau_q) / np.diff(q_values)
    f_alpha = (q_values[:-1] + q_values[1:]) / 2 * alpha - (tau_q[:-1] + tau_q[1:]) / 2

    return alpha, f_alpha, tau_q, Z_q_s, coef, maxima, log_scales

def wtmm_compute_r2_per_q(Z_q_s, log_scales):
    nq = Z_q_s.shape[0]
    r2 = np.zeros(nq)
    for i in range(nq):
        valid = (~np.isnan(Z_q_s[i])) & (Z_q_s[i] > 0)
        if valid.sum() > 2:
            y, x = np.log2(Z_q_s[i][valid]), log_scales[valid]
            slope, intercept = linear_fit(x, y)
            y_fit = slope * x + intercept
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2[i] = 1 - ss_res / ss_tot if ss_tot else np.nan
        else:
            r2[i] = np.nan
    return r2

def run_wtmm_analysis(signal, scales, q_values, wavelet_type="morlet"):
    
      # normalize the signal to fit in the wavelet
    sig_max = signal.max()
    sig_min = signal.min()
    signal = (signal - (sig_min - 0.01)) / (sig_max - sig_min + 0.02)


    alpha, f_alpha, tau_q, Z_q_s, coef, maxima, log_scales = wtmm_multifractal(signal, scales, q_values, wavelet_type)
    r2 = wtmm_compute_r2_per_q(Z_q_s, log_scales)
    return {
        'alpha': alpha,
        'f_alpha': f_alpha,
        'tau_q': tau_q,
        'Z_q_s': Z_q_s,
        'coef': coef,
        'maxima': maxima,
        'r2': r2,
        'q_values': q_values,
        'log_scales': log_scales
    }

def multiplicative_binomial(n, p=0.7):
    signal = np.ones(1)
    for _ in range(n):
        signal = np.concatenate([signal * p, signal * (1 - p)])
    return signal

if __name__ == "__main__":
    signal = multiplicative_binomial(12)
    N = len(signal)
    min_scale = 4
    max_scale = N // 10
    scales = 2 ** np.linspace(np.log2(min_scale), np.log2(max_scale), 40)
    #scales = np.logspace(np.log2(min_scale), np.log2(max_scale), num=40, base=2)
    q_values = np.linspace(-3, 3, 25)

    #wavelet_type = "morlet"  # vagy "ricker"
    wavelet_type = "ricker"
    
    result = run_wtmm_analysis(signal, scales, q_values, wavelet_type=wavelet_type)

    # Multifraktál spektrum ábra
    plt.figure(figsize=(6, 4))
    plt.plot(result['alpha'], result['f_alpha'], 'o-', label="f(α)")
    plt.xlabel("α")
    plt.ylabel("f(α)")
    plt.title(f"Multifraktál spektrum ({wavelet_type} wavelet)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # R²(q) ábra
    plt.figure(figsize=(6, 4))
    plt.plot(q_values, result['r2'], 'o-', label="R²(q)", color='darkred')
    plt.axhline(1.0, linestyle="--", color="gray", linewidth=0.5)
    plt.axhline(0.98, linestyle="--", color="blue", linewidth=0.5, label="0.98 küszöb")
    plt.xlabel("q")
    plt.ylabel("R²")
    plt.title("Skálázási illeszkedés minősége")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # WTMM modulus maxima ábra
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(result['maxima']), aspect='auto', cmap='hot', extent=[0, N, max_scale, min_scale])
    plt.colorbar(label="Modulus maxima")
    plt.title("WTMM modulus maxima")
    plt.xlabel("Idő")
    plt.ylabel("Skálák")
    plt.tight_layout()
    plt.show()
    # μ(q, i) hőtérkép ábra
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log10(result['Z_q_s'] + 1e-12), aspect='auto', cmap='inferno', extent=[0, result['Z_q_s'].shape[1], q_values[0], q_values[-1]])
    plt.colorbar(label="log₁₀(μ(q, i))")
    plt.title("μ(q, i) hőtérkép")
    plt.xlabel("Ablak index (i)")
    plt.ylabel("q")
    plt.tight_layout()
    plt.show()
    # WTMM CWT ábra
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(result['coef']), aspect='auto', cmap='jet', extent=[0, N, max_scale, min_scale])
    plt.colorbar(label="CWT modulus")
    plt.title(f"WTMM CWT ({wavelet_type} wavelet)")
    plt.xlabel("Idő")
    plt.ylabel("Skálák")
    plt.tight_layout()
    plt.show()
    
    spectrum = cwt.wtmm (signal, width_step=0.25, max_scale=150, plot=True)
    plt.plot(spectrum['alpha'], spectrum['f_alpha'])
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$f(\alpha)$')
    plt.title('Multifractal Spectrum (WTMM)')
    plt.grid(True)
    plt.show()

    



