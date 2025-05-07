
import numpy as np
from numba import njit , prange
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import streamlit as st
import os
from scipy.signal import detrend
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress 

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
def ricker_wavelet_old(length, scale):
    x = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    a2 = scale ** 2
    A = 2 / (np.sqrt(3 * scale) * (np.pi ** 0.25))
    wavelet = A * (1 - (x**2) / a2) * np.exp(-x**2 / (2 * a2))
    wavelet /= np.sqrt(scale)
    return wavelet

def ricker_wavelet(length, scale):
    """Ricker wavelet = second derivative of Gaussian"""
    points = np.arange(-length // 2, length // 2 + 1)
    a = scale
    return (2 / (np.sqrt(3 * a) * (np.pi**0.25))) * (1 - (points**2) / a**2) * np.exp(-(points**2) / (2 * a**2))


@njit
def convolve_same_real(signal, kernel):
    N = len(signal)
    M = len(kernel)
    half = M // 2

    output = np.zeros(N)
    for i in range(N):
        acc = 0.0
        for j in range(M):
            idx = i - half + j
            if 0 <= idx < N:
                acc += signal[idx] * kernel[j]
        output[i] = acc
    return output

@njit
def convolve_same_complex(signal, kernel):
    N = len(signal)
    M = len(kernel)
    half = M // 2

    output = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        acc = 0.0 + 0.0j  # kezdeti érték: komplex nulla
        for j in range(M):
            idx = i - half + j
            if 0 <= idx < N:
                acc += signal[idx] * kernel[j]
        output[i] = acc.real
    return output


@njit
def convolve_same_valid(signal, kernel):
    N = len(signal)
    M = len(kernel)
    output_length = N - M + 1
    output = np.zeros(output_length)
    for i in range(output_length):
        s = 0.0
        for j in range(M):
            s += signal[i + j] * kernel[j]
        output[i] = s
    return output


def cwt_morlet(
    signal,
    scales=None,
    kernel_size=1.0,
    auto_scale=False,
    min_exp=2,
    max_exp=None,
    num_scales=None,
    return_confidence=True ):

    N = len(signal)

    if auto_scale or scales is None:
        scales = generate_multifractal_scales(N, min_exp=min_exp, max_exp=max_exp, num_scales=num_scales)

   
    N, S = len(signal), len(scales)
    coef = np.zeros((S, N), dtype=np.complex128)
    confidence = np.zeros((S, N), dtype=float)   
    width = int(min(10 * 2, N)) | 1
    wavelet = morlet_wavelet(width, 2) * kernel_size
    #wavlet_mean = np.mean(wavelet)
    #signal= signal - signal.mean() + wavlet_mean


    for i in range(S):
        scale = scales[i]
        width = int(min(8 * scale, N)) | 1
        half_width = width // 2

        wavelet = morlet_wavelet(width, scale)

        # Padding tükrözéssel
        padded = np.pad(signal, pad_width=half_width, mode='reflect')

        # Konvolúció
        #conv_full = convolve_same_real(padded, wavelet)
        conv_full = convolve_same_complex (padded, wavelet)
        coef[i] = conv_full[half_width:half_width + N]

        # Élszéli zónák megjelölése
        confidence[i, :half_width] = 0
        confidence[i, -half_width:] = 0

    if return_confidence:
        return coef, scales, confidence

    return coef, scales
    
    N, S = len(signal), len(scales)
    coef = np.zeros((S, N), dtype=np.complex128)
    confidence = np.zeros((S, N), dtype=np.complex128)
    
    for i in range(S):
        scale = scales[i]
        width = int(min(10 * scale, N)) | 1
        wavelet = morlet_wavelet(width, scale)
        coef[i] = convolve_same_complex(signal, wavelet)
    return coef, scales, confidence


def trim_edges_from_spectrum(Z_q_s, log_scales, trim=None):
    """
    Kiszűri a spektrumból a torz széli értékeket (pl. valid konvolúcióból eredően).
    trim=None esetén automatikusan 10%-ot vág le.
    """
    if trim is None:
        trim = max(1, int(Z_q_s.shape[1] * 0.1))
    if trim == 0:
        return Z_q_s, log_scales
    return Z_q_s[:, trim:-trim], log_scales[trim:-trim]

def generate_multifractal_scales(signal_length, min_exp=2, max_exp=None, num_scales=None, base=2, min_repeats=8):
    if max_exp is None:
        max_exp = int(np.floor(np.log2(signal_length // min_repeats)))
    if num_scales is None:
        exponents = np.arange(min_exp, max_exp + 1)
        scale_vals = base ** exponents
    else:
        exponents = np.linspace(min_exp, max_exp, num=num_scales)
        scale_vals = np.round(base ** exponents).astype(int)
        scale_vals = np.unique(scale_vals)
    scale_vals = scale_vals[scale_vals * min_repeats <= signal_length]
    return scale_vals.astype(float)

def cwt_ricker(
    signal,
    scales=None,
    kernel_size=1.0,
    auto_scale=False,
    min_exp=2,
    max_exp=None,
    num_scales=None,
    return_confidence=True
    ):
    N = len(signal)

    if auto_scale or scales is None:
        scales = generate_multifractal_scales(N, min_exp=min_exp, max_exp=max_exp, num_scales=num_scales)

    S = len(scales)
    coef = np.zeros((S, N))
    confidence = np.ones((S, N))  # 1=megbízható, 0=padding zóna
    width = int(min(10 , N)) | 1
    wavelet = ricker_wavelet(width, 2) * kernel_size
    wavlet_mean = np.mean(wavelet)
    signal= signal - signal.mean() + wavlet_mean


    for i in range(S):
        scale = scales[i]
        width = int(min(8 * scale, N)) | 1
        half_width = width // 2

        wavelet = ricker_wavelet(width, scale) * kernel_size

        # Padding tükrözéssel
        padded = np.pad(signal, pad_width=half_width, mode='reflect')

        # Konvolúció
        conv_full = convolve_same_real(padded, wavelet)
        coef[i] = conv_full[half_width:half_width + N]

        # Élszéli zónák megjelölése
        confidence[i, :half_width] = 0
        confidence[i, -half_width:] = 0

    if return_confidence:
        return coef, scales, confidence

    return coef, scales

@njit(parallel=True)
def modulus_maxima(coef_matrix, threshold_ratio=0.05):
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


@njit
def linear_fit_r2(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    x_diff = x - mean_x
    y_diff = y - mean_y

    num = np.sum(x_diff * y_diff)
    den = np.sum(x_diff ** 2)
    slope = num / den if den != 0.0 else 0.0
    intercept = mean_y - slope * mean_x

    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - mean_y) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    return slope, intercept, r_squared

@njit
def compute_tau_q(Z_q_s, log_scales, r2_threshold=0.90, eps=1e-12):
    nq = Z_q_s.shape[0]
    tau_q = np.empty(nq)
    r2 = np.empty(nq)
    tau_q[:] = np.nan
    r2[:] = np.nan

    for i in range(nq):
        row = Z_q_s[i]
        valid = (~np.isnan(row)) & (row > 0)

        if valid.sum() > 2:
            z_vals = row[valid]
            z_vals = np.where(z_vals < eps, eps, z_vals)
            y = np.log2(z_vals)
            x = log_scales[valid]
            slope, _, r_squared = linear_fit_r2(x, y)
            if r_squared < 0.90:
                slope, intercept, r_squared = robust_fit (x, y, drop=5)
                continue
            #if r_squared < 0.90:
            #    slope, intercept, r_squared, nx, ny = robust_fit (nx, ny, drop=2)
            #    continue
            

            r2[i] = r_squared
            tau_q[i] = slope if r_squared >= r2_threshold else np.nan
        else:
            r2[i] = np.nan
            tau_q[i] = np.nan

    return tau_q, r2

@njit   
def robust_fit(x, y, drop=1):
    n = len(x)
    if n - drop < 3:
        slope, intercept, r2 = linear_fit_r2(x, y)
        return slope, intercept, r2

    slope, intercept, _ = linear_fit_r2(x, y)
    residuals = np.abs(y - (slope * x + intercept))

    # Find indices of smallest residuals
    keep_mask = np.ones(n, dtype=np.bool_)
    for _ in range(drop):
        max_i = np.argmax(residuals)
        keep_mask[max_i] = False
        residuals[max_i] = -1.0  # már eldobtuk

    # Filtered arrays
    kept_x = x[keep_mask]
    kept_y = y[keep_mask]

    slope_new, intercept_new, r2_new = linear_fit_r2(kept_x, kept_y)
    return slope_new, intercept_new, r2_new 

def debug_plot_all_regressions(Z_q_s, log_scales, q_values):
    """
    Debug célú regressziós ábrák minden q-hoz:
    log2(Z(q,s)) vs log2(s), lineáris illesztéssel.
    """
    nq = Z_q_s.shape[0]
    
    for i in range(nq):
        Z = Z_q_s[i]
        q = q_values[i]
        
        # Csak érvényes (pozitív, nem NaN) értékek
        valid = (~np.isnan(Z)) & (Z > 0)
        if np.sum(valid) < 3:
            continue  # túl kevés adat az illesztéshez
        
        log_s = log_scales[valid]
        log_Z = np.log2(Z[valid])
        
        # Lineáris illesztés
        #slope, intercept = np.polyfit(log_s, log_Z, 1)
        slope, intercept, r_squared = linear_fit_r2(log_s, log_Z)
        if r_squared < 0.90:
            slope, intercept, r_squared, = robust_fit(log_s, log_Z, drop=1)
            continue

        fit_line = slope * log_s + intercept

        # Ábra
        plt.figure(figsize=(6, 4))
        plt.plot(log_s, log_Z, 'o', label='log₂ Z(q, s)', color='blue')
        #plt.plot(log_s, r_squared, 'o', label='r_squared', color='red')
        
        plt.plot(log_s, fit_line, '-', label=f'Illesztés: slope = {slope:.3f} rsq = {r_squared:.3f}', color='red')
        #plt.plot(log_s, fit_line, '-', label=f'Illesztés: slope = {slope:.3f}', color='red')
        
        plt.xlabel("log₂(s)")
        plt.ylabel("log₂ Z(q, s)")
        plt.title(f"Lineáris regresszió (q = {q:.2f})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt) 
        plt.close()

def compute_Z_q_s(maxima, q_values, threshold=0.001):
    """
    Stabil Z(q, s) számítás: kis értékeket kizár, hatványozás maszkkal történik.
    """
    maxima = np.nan_to_num(maxima, nan=0.0)
    mask = maxima > threshold
    num_q = len(q_values)
    num_s = maxima.shape[0]

    Z_q_s = np.full((num_q, num_s), np.nan)

    for qi, q in enumerate(q_values):
        for si in range(num_s):
            row_mask = mask[si]
            row_vals = maxima[si][row_mask]

            if len(row_vals) == 0:
                continue

            Z_q_s[qi, si] = np.sum(row_vals ** q)

    return Z_q_s
#
# Numpy-alapú mozgóátlag + lineáris interpoláció
def moving_average_with_interpolation(x, window=3):
    kernel = np.ones(window) / window
    smoothed = np.convolve(x, kernel, mode='same')

    # Kezdeti és végpontok interpolálása
    mask = np.full_like(smoothed, True, dtype=bool)
    mask[window//2:-window//2] = False  # középső, biztos értékek

    x_idx = np.arange(len(x))
    smoothed[mask] = np.interp(x_idx[mask], x_idx[~mask], smoothed[~mask])
    
    return smoothed


def prepare_signal (signal, padding_len=0, smoothing=0.1):
    """
    A bemeneti jelet max -1 és + 1 között tartja.
    """
    #A bemeneti jelet max -1 és + 1 között tartja.
    signal_max = np.max(np.abs(signal))
    signal = moving_average_with_interpolation(signal, window=3)
    signal = signal / signal_max if signal_max > 0 else signal
    signal = signal* 1
    # Ha kell, padding
    

    #signal.astype(np.float32)

    # spline interpoláció
    #signal = UnivariateSpline(np.arange(len(signal)), signal, s=smoothing)(np.arange(len(signal)))
    
    #st.line_chart(signal)
    #st.write (f"Jel előkészítve: {len(signal)}")

    return signal
from io import BytesIO
def debug_image_show (array, dpi=1):
    #st.write (array)
    # Paraméterek
    height, width = array.shape
    st.write (f"Debug image: {height}x{width}")
    
    figsize = (width / dpi, height / dpi)  # inch-ben

    # Ábra készítése
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(array, cmap='inferno', origin='lower', aspect='auto')
    ax.axis('off')  # nincs tengely

    # Kép mentése bufferbe
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # ne mutassa újra

    # Kép megjelenítése streamlitben pixelpontos méretben
    st.image(buf, caption="Eredeti pixelméretű numpy kép", use_column_width=False)

def normalize_nan_safe(mat, axis=1):
    """Minden sort [0–1] közé skáláz, NaN-ok figyelembevételével"""
   
    row_max = np.nanmax(mat, axis=axis, keepdims=True)
    row_min = np.nanmin(mat, axis=axis, keepdims=True)
    denom = row_max - row_min

    # Elkerüljük az osztást nullával (pl. konstans vagy csak NaN sor)
    denom[denom == 0] = np.nan

    normed = (mat - row_min) / denom
    return normed

def extract_top_local_maxima(modulus,  keep_ratio=1, limit=0.005):
    """
    Lokális maximumok kiszűrése CWT modulus mátrixból (globális szűrés).
    
    Paraméterek:
        modulus (2D np.ndarray): CWT abszolútérték mátrix [skála × idő]
        keep_ratio (float): megtartandó maximumok aránya (0–1)
    
    Visszaadja:
        maxima (2D np.ndarray): csak a kiválasztott maximumokat tartalmazó mátrix
    """
    # 0–1 normalizálás soronként
    #modulus = modulus.copy()
    #modulus = normalize_rows_nan_safe(modulus)
    
    # NaN ami kisebb mint limit
    #limit = 0.0001
    modulus[modulus < limit] = 0


    S, N = modulus.shape
    maxima_mask = np.zeros_like(modulus, dtype=bool)

    # 1. Lokális maximumok keresése minden skálasorban (1D)
    for s in range(S):
        row = modulus[s]
        # 1D maximumok: bal és jobb szomszédnál nagyobb
        local_max = (row[1:-1] > row[:-2]) & (row[1:-1] > row[2:])
        maxima_mask[s, 1:-1] = local_max
    #st.write ("Maxima:", modulus * maxima_mask)
    return modulus * maxima_mask

def detect_strict_local_maxima(mat, keep_naigbhbor=4, limit=0.0001):
    """
    Módosított lokális maximumdetekció:
    csak akkor maximum, ha a szomszédok közül max. 1 lehet nála nem kisebb.
    """
    mat[mat < limit] = 0
    padded = np.pad(mat, pad_width=1, mode='reflect')
    output = np.zeros_like(mat, dtype=bool)

    # A 8 szomszédból minden pixelre kigyűjtjük a szomszédértékeket
    neighbors = np.stack([
        padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:],
        padded[1:-1, 0:-2],                   padded[1:-1, 2:],
        padded[2:  , 0:-2], padded[2:  , 1:-1], padded[2:  , 2:]
    ], axis=0)  # shape: (8, H, W)

    center = mat
    # Minden pixel szomszédaihoz: megszámoljuk hány szomszéd NEM kisebb nála
    not_strictly_greater = (center <= neighbors)
    worse_count = np.sum(not_strictly_greater, axis=0)

    # Akkor tekintjük maximumként, ha legfeljebb 1 ilyen szomszéd van
    # Tehát az érték nem dominál mindenkit, de majdnem mindenkit
    output = (mat > 0) & (worse_count <= keep_naigbhbor)

    # Ha értékekre van szükség, nem maszkra:
    return output * mat

def wtmm_multifractal_spectrum(signal, scales, q_values, wavelet_type="ricker", keep_ratio=0.2, kernel_size=1):

    scales= scales.astype(np.int32)

    log2_scales = np.log2(scales)
    if wavelet_type == "morlet":
        coef, tmp, confidence = cwt_morlet(signal, scales)
    else:
        coef, tmp, confidence = cwt_ricker(signal, scales, kernel_size=kernel_size)

    modulus = np.abs(coef)
    modulus = modulus*confidence


    maxima = extract_top_local_maxima (modulus, keep_ratio=1, limit=0.001)
    #st.write ("Maxima:", maxima.shape)
    #st.write ("q_values", q_values)
    
    Z_q_s = compute_Z_q_s(maxima, q_values)
    #st.write ("Z_q_s:", Z_q_s)
    
    
    
    tau_q, r2 = compute_tau_q(Z_q_s, log2_scales)
    #st.write (tau_q)
    #st.write (r2)

    #debug_plot_all_regressions (Z_q_s, log2_scales, q_values)
    
    alpha = np.diff(tau_q) / np.diff(q_values)
    
    f_alpha = (q_values[:-1] + q_values[1:]) / 2 * alpha - (tau_q[:-1] + tau_q[1:]) / 2
    return alpha, f_alpha, tau_q, Z_q_s, coef, maxima, 0, r2

def multiplicative_binomial(n, p=0.7):
    signal = np.ones(1)
    for _ in range(n):
        signal = np.concatenate([signal * p, signal * (1 - p)])
    return signal

# --- Skála és q optimalizáló ---
def estimate_optimal_scales(signal_len, num_scales=20):
    return np.unique(np.logspace(np.log2(2), np.log2(signal_len // 4), num=num_scales, base=2).astype(int))


def estimate_optimal_q_values():
    return np.linspace(-3, 3, 21)

def wtmm_compute_r2(Z_q_s, log_scales):
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


def plot_wtmm_results(
    signal,
    scales,
    q_values,
    alpha,
    f_alpha,
    tau_q,
    Z_q_s,
    coef,
    maxima,
    save_path=None,
    dpi=300
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    N = len(signal)
    t = np.linspace(0, 1, N)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

    target_width = 1000
    coef_ds = preprocess_for_plotting(np.abs(coef), target_width)
    maxima_ds = preprocess_for_plotting(maxima, target_width)

    # 1. Eredeti jel
    axs[0].plot(t, signal, color='black')
    axs[0].set_title("Eredeti jel")
    axs[0].set_ylabel("Amplitúdó")

    # 2. CWT spektrum
    extent = [0, 1, np.max(scales), np.min(scales)]
    axs[1].imshow(np.abs(coef_ds), extent=extent, aspect='auto', cmap='viridis')
    axs[1].set_title("CWT amplitúdó (Ricker hullám)")
    axs[1].set_ylabel("Skála")

    # 3. Modulus maxima
    axs[2].imshow(maxima_ds, extent=extent, aspect='auto', cmap='hot')
    axs[2].set_title("Modulus maximumok")
    axs[2].set_ylabel("Skála")

    #

    plt.tight_layout()

    # Mentés fájlba, ha kérted
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"📁 Ábra mentve: {save_path}")

    return fig, axs

def preprocess_for_plotting(matrix, target_width):
    """
    Időtengely menti összevonás és normalizálás 0-1 közé.
    """
    n_scales, n_times = matrix.shape
    if target_width >= n_times:
        data = matrix.copy()
    else:
        factor = n_times // target_width
        trimmed_width = factor * target_width
        matrix = matrix[:, :trimmed_width]
        reshaped = matrix.reshape(n_scales, target_width, factor)
        data = reshaped.sum(axis=2)  # vagy .mean(axis=2) ha inkább átlagolnád

    # Normalizálás 0–1 közé
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        data = (data - min_val) / (max_val - min_val)
    else:
        data[:] = 0  # konstans mátrix

    return data


def plot_wtmm_all_outputs(
    signal,
    scales,
    q_values,
    alpha,
    f_alpha,
    tau_q,
    Z_q_s,
    coef,
    maxima,
    output_dir="output",
    filename_prefix="wtmm_result",
    dpi=300
    ):
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    target_width = 1000
    coef_ds = preprocess_for_plotting(np.abs(coef), target_width)
    maxima_ds = preprocess_for_plotting(maxima, target_width)

    N = len(signal)
    t = np.linspace(0, 1, N)
    extent = [0, 1, np.max(scales), np.min(scales)]

    # 1. Eredeti jel
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, signal, color='black')
    ax.set_title("Eredeti jel")
    ax.set_ylabel("Amplitúdó")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename_prefix}_signal.png", dpi=dpi, bbox_inches='tight')
    plt.close()

    # 2. CWT
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(np.abs(coef_ds), extent=extent, aspect='auto', cmap='viridis')
    ax.set_title("CWT amplitúdó (Ricker hullám)")
    ax.set_ylabel("Skála")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename_prefix}_cwt.png", dpi=dpi, bbox_inches='tight')
    plt.close()

    # 3. Modulus maxima
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(maxima_ds, extent=extent, aspect='auto', cmap='hot')
    ax.set_title("Modulus maximumok")
    ax.set_ylabel("Skála")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename_prefix}_modulus_maxima.png", dpi=dpi, bbox_inches='tight')
    plt.close()

    # 4. Multifraktál spektrum
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(alpha, f_alpha, '-o', color='navy')
    ax.set_title("Multifraktál spektrum (f(α))")
    ax.set_xlabel("α (szingularitás erőssége)")
    ax.set_ylabel("f(α)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename_prefix}_spectrum.png", dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"✅ WTMM eredmények elmentve mappába: {output_dir}")
    return fig, ax

def plot_wtmm_dq_spectrum(q_values, tau_q, title="WTMM D(q) spektrum"):
    D_q = np.gradient(tau_q, q_values)
    plt.figure(figsize=(7, 4))
    plt.plot(q_values, D_q, 'o-', label='D(q)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('q')
    plt.ylabel('D(q) = dτ(q)/dq')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return D_q, plt


    """
    Minden q-hoz kiszámítja a log2(Z_q) vs. log2(scale) lineáris illesztésének R² értékét.
    """
    r2_array = np.full(len(q_values), np.nan)
    for i, q in enumerate(q_values):
        zq = Z_q_s[i]
        valid = (~np.isnan(zq)) & (zq > 0)
        if np.sum(valid) > 2:
            y = np.log2(zq[valid])
            x = log_scales[valid]
            slope, intercept = linear_fit(x, y)
            y_fit = slope * x + intercept
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            r2_array[i] = r2
    return r2_array

# --- WTMM: R² görbe ábrázolása ---
def plot_wtmm_r2_vs_q(q_values, r2_array, threshold=0.98, title="WTMM Z_q skálázási R²"):
    plt.figure(figsize=(7, 4))
    plt.plot(q_values, r2_array, 'o-', label='R²(q)')
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(threshold, color='red', linestyle='--', linewidth=0.5, label=f'{threshold} küszöb')
    plt.xlabel('q')
    plt.ylabel('R²')
    plt.title(title)
    plt.grid(True)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    return plt


def analyze_multifractal_over_time(signal, sampling_rate, window_sec, scales, q_values, keep_ratio=0.3, wavelet_type="ricker"):
    """
    Felbontja a jelet időablakokra, majd minden ablakra kiszámolja a multifraktál spektrumot,
    és visszaadja a szegmensenkénti spektrumszélességet (alpha_max - alpha_min).

    Paraméterek:
        signal        : 1D numpy array, a teljes időjel
        sampling_rate : int, pl. 1000 (minták/s)
        window_sec    : float, pl. 10.0 (ablak hossza másodpercben)
        scales        : numpy array, a skálák listája (pl. log-skála)
        q_values      : numpy array, q értékek (pl. np.linspace(-2, 3, 21))
        keep_ratio    : float (0..1), a megtartandó maximumok aránya globálisan

    Visszatér:
        times_sec : időpontok (az ablakok közepei másodpercben)
        widths    : alpha_max - alpha_min értékek idő szerint
    """
    window_size = int(window_sec * sampling_rate)
    num_segments = (len(signal) - window_size) // window_size + 1
    #st.write (f"Jel hossza: {len(signal)} minták ({len(signal)/sampling_rate:.2f}s)")
    #st.write (f"Numser of segments: {num_segments} ({window_size} samples per segment)")

    times_sec = []
    widths = []

    for seg_idx in range(num_segments):
        start = seg_idx * window_size
        end = start + window_size
        segment = signal[start:end]
        #st.write (f"Szegment {seg_idx + 1}/{num_segments} ({start/sampling_rate:.2f}s - {end/sampling_rate:.2f}s)")
        #if True : 
        try:
            alpha, _, _, _, _, _, _, _ = wtmm_multifractal_spectrum(segment, scales, q_values, keep_ratio=keep_ratio, wavelet_type=wavelet_type)
            
            valid_alpha = alpha[np.isfinite(alpha)]

            if valid_alpha.size >= 2:
                width = np.max(valid_alpha) - np.min(valid_alpha)
                #st.write(f"Alpha width: {width:.4f}")
            else:
                width = np.nan
                #st.warning("Nem volt elég érvényes α érték a szélesség számításhoz.")
        except Exception as e:
            print(f"❌ Szegmens {seg_idx}: hiba a WTMM-ben: {e}")
            width = np.nan

        center_time = (start + end) / 2 / sampling_rate
        times_sec.append(center_time)
        widths.append(width)

    return np.array(times_sec), np.array(widths)
