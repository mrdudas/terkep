import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# === Multiplicative Cascade Generator ===
def multiplicative_cascade(target_length=1024, weights=(0.7, 0.3), seed=None):
    if seed is not None:
        np.random.seed(seed)
    levels = int(np.ceil(np.log2(target_length)))
    signal = np.array([1.0])
    for _ in range(levels):
        signal = np.concatenate([signal * weights[0], signal * weights[1]])
    return signal[:target_length]

# === Numba-accelerated Linear Regression ===
@njit(parallel=True)
def simple_linregress(x, y):
    n = x.shape[0]
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x * x)
    sum_xy = np.sum(x * y)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0, 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    r_value = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_xx - sum_x**2) * (n * np.sum(y**2) - sum_y**2))
    return slope, intercept, r_value

# === Multifractal Spectrum (Chhabra–Jensen) ===
@njit(parallel=True)
def get_multifractal_spectrum(signal, q_values, window_sizes):
    nq = q_values.shape[0]
    ns = window_sizes.shape[0]
    Ma = np.zeros((nq, ns))
    Mf = np.zeros((nq, ns))

    for i in prange(nq):
        q = q_values[i]
        for j in range(ns):
            win = window_sizes[j]
            valid_len = (len(signal) // win) * win
            if valid_len == 0:
                continue
            reshaped = signal[:valid_len].reshape(-1, win).T
            p = np.sum(reshaped, axis=0)
            total = np.sum(p)
            if total == 0:
                continue
            p = p / total
            safe_p = np.where(p > 0, p, 1e-12)
            p_q = safe_p ** q
            Zq = np.sum(p_q)
            mu_q = p_q / Zq if Zq > 0 else np.zeros_like(p_q)
            safe_mu = np.where(mu_q > 0, mu_q, 1e-12)
            Ma[i, j] = np.sum(mu_q * np.log2(safe_p))
            Mf[i, j] = np.sum(mu_q * np.log2(safe_mu))

    alpha = np.zeros(nq)
    falpha = np.zeros(nq)
    log_scales = np.log2(window_sizes.astype(np.float64))
    for i in range(nq):
        slope_a, _, _ = simple_linregress(log_scales, Ma[i])
        slope_f, _, _ = simple_linregress(log_scales, Mf[i])
        alpha[i] = slope_a
        falpha[i] = slope_f

    return alpha, falpha

# === μ(q, i) Matrix Calculation ===
@njit(parallel=True)
def compute_mu_matrix(signal, q_values, window_size):
    nq = len(q_values)
    valid_length = (len(signal) // window_size) * window_size
    reshaped = signal[:valid_length].reshape(-1, window_size).T
    window_sums = np.sum(reshaped, axis=0)
    total = np.sum(window_sums)

    if total <= 0:
        return np.zeros((nq, reshaped.shape[1]))

    p = window_sums / total
    safe_p = np.where(p > 0, p, 1e-12)
    num_windows = len(p)

    mu_matrix = np.zeros((nq, num_windows))
    for i in prange(nq):
        q = q_values[i]
        p_q = safe_p ** q
        Zq = np.sum(p_q)
        if Zq > 0:
            mu_matrix[i, :] = p_q / Zq
        else:
            mu_matrix[i, :] = 0.0
    return mu_matrix

# === Main Execution ===
signal = multiplicative_cascade(target_length=1024, weights=(0.7, 0.3), seed=42)
q_vals = np.linspace(-5, 5, 21)
scales = 2 ** np.arange(2, 7)  # [4, 8, 16, 32, 64]

# f(α) spectrum
alpha, falpha = get_multifractal_spectrum(signal, q_vals, scales)

plt.figure(figsize=(7, 5))
plt.plot(alpha, falpha, 'o-', label='f(α)')
plt.xlabel('α')
plt.ylabel('f(α)')
plt.title('Multifraktál spektrum (Multiplikatív kaszkád)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# μ(q, i) hőtérkép
mu_matrix = compute_mu_matrix(signal, q_vals, window_size=32)
plt.figure(figsize=(10, 5))
plt.imshow(np.log10(mu_matrix + 1e-12), aspect='auto', origin='lower',
           extent=[0, mu_matrix.shape[1], q_vals[0], q_vals[-1]],
           cmap='inferno')
plt.colorbar(label="log₁₀(μ(q, i))")
plt.xlabel("Ablak index (i)")
plt.ylabel("q")
plt.title("μ(q, i) hőtérkép (scale = 32)")
plt.tight_layout()
plt.show()
