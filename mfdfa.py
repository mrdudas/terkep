import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress 
from numpy.polynomial.polynomial import Polynomial
import streamlit as st

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




def mfdfa_multifractal_spectrum(signal, scales=None, q_values=np.linspace(-5, 5, 41),  m=3):
    """
    Részletes MFDFA implementáció.
    Visszatér: alpha, f_alpha, tau_q, Z_q_s (scaling), illesztési együtthatók, R^2, stb.
    """
    N = len(signal)
    Y = np.cumsum(signal - np.mean(signal))  # profil
    #Y = signal.copy()
    if scales is None:
        scales = np.unique(np.logspace(2, np.log2(N//4), num=20, base=2, dtype=int))

    Z_q_s = []  # Z(q, s) minden skálára
    log_scales = np.log2(scales)
    q_values = np.array(q_values)

    for s in scales:
        Ns = N // s
        F_s = []
        #cfs =np.range(0, Ns * s, s)
        #st.write (range(0, Ns * s, s)) 
        cfs = []
        for start in range(0, Ns * s, s):
            segment = Y[start:start + s]
            x = np.arange(s)
            # polinom detrend
            coefs = Polynomial.fit(x, segment, m).convert().coef
            cfs[int(start//s)] = coefs
            fit = np.polyval(coefs[::-1], x)
            F_s.append(np.mean((segment - fit) ** 2))
        F_s = np.array(F_s)
        #cfs = np.array(cfs)
        st.write (f"F_s: ",F_s)
        st.write (f"cfs: ",cfs.shape)
        
        Z_q = []

        for q in q_values:
            if q == 0:
                Z = np.exp(0.5 * np.mean(np.log(F_s + 1e-12)))
            else:
                Z = (np.mean(F_s ** (q / 2))) ** (1 / q)
            Z_q.append(Z)

        Z_q_s.append(Z_q)

    Z_q_s = np.array(Z_q_s)  # shape: (n_scales, n_q)
    #st.write (f"Z_q_s shape: {Z_q_s.shape}")
    #st.write (Z_q_s)
    log_Z_q_s = np.log2(Z_q_s + 1e-12)

    h_q = []
    tau_q = []
    coef = []
    r2 = []
    for i, q in enumerate(q_values):
        y = log_Z_q_s[:, i]
        x = log_scales
        slope, intercept, r_val, _, _ = linregress(x, y)
        h = slope
        h_q.append(h)
        tau_q.append(q * h - 1)
        coef.append((slope, intercept))
        r2.append(r_val**2)

    h_q = np.array(h_q)
    tau_q = np.array(tau_q)
    coef = np.array(coef)
    r2 = np.array(r2)

    # Deriválással: α(q), f(α)
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q
    
    return alpha, f_alpha, tau_q, Z_q_s, coef, h_q, 0, r2
    
    #return {
    #    "q": q_vals,
    #    "h_q": h_q,
    #    "tau_q": tau_q,
    #    "alpha": alpha,
    #    "f_alpha": f_alpha,
    #    "Z_q_s": Z_q_s,
    #    "log_scales": log_scales,
    #    "coef": coef,
    #    "r2": r2
    #}
    #return q_vals, h_q, alpha, f_alpha

def prepare_signal (signal, padding_len=0, smoothing=0.1):
    """
    A bemeneti jelet max -1 és + 1 között tartja.
    """
    #A bemeneti jelet max -1 és + 1 között tartja.
    signal_max = np.max(np.abs(signal))
    signal = moving_average_with_interpolation(signal, window=3)
    signal = signal / signal_max if signal_max > 0 else signal
    #signal = signal* 1
    # Ha kell, padding
    

    #signal.astype(np.float32)

    # spline interpoláció
    #signal = UnivariateSpline(np.arange(len(signal)), signal, s=smoothing)(np.arange(len(signal)))
    
    #st.line_chart(signal)
    #st.write (f"Jel előkészítve: {len(signal)}")

    return signal

def plot_mfdfa_r2_vs_q (q_values, r2_array, threshold=0.98, title="mfdfa Z_q skálázási R²"):
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

def plot_mfdfa_dq_spectrum(q_values, tau_q, title="MFDFA D(q) spektrum"):
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

def plot_mfdfa_results(
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
    axs[2].imshow(Z_q_s, extent=extent, aspect='auto', cmap='hot')
    axs[2].set_title("Z_q_s")
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

def analyze_multifractal_over_time(signal, sampling_rate, window_sec, scales, q_values, keep_ratio=0.3):
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
            alpha, _, _, _, _, _, _, _ = mfdfa_multifractal_spectrum(segment, scales, q_values)
            
            valid_alpha = alpha[np.isfinite(alpha)]

            if valid_alpha.size >= 2:
                width = np.max(valid_alpha) - np.min(valid_alpha)
                #st.write(f"Alpha width: {width:.4f}")
            else:
                width = np.nan
                #st.warning("Nem volt elég érvényes α érték a szélesség számításhoz.")
        except Exception as e:
            print(f"❌ Szegmens {seg_idx}: hiba a mfdfa-ben: {e}")
            width = np.nan

        center_time = (start + end) / 2 / sampling_rate
        times_sec.append(center_time)
        widths.append(width)

    return np.array(times_sec), np.array(widths)


def preprocess_for_plotting(matrix, target_width):
    """
    Időtengely menti összevonás és normalizálás 0-1 közé.
    """
    #st.write (matrix)
    if len (matrix.shape) != 2:
        #raise ValueError("A bemeneti mátrixnak 2D-nek kell lennie.")
        return []
    #st.write (f"Preprocess for plotting: {matrix.shape}")

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


def plot_mfdfa_all_outputs(
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
    filename_prefix="mfdfa_result",
    dpi=300
    ):
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    target_width = 1000
    if len(coef) > 1:
        coef_ds = preprocess_for_plotting(np.abs(coef), target_width)
    else:
        coef_ds = [[]]
    #if len(maxima) > 1:
    #    maxima_ds = preprocess_for_plotting(maxima, target_width)
    #else:
    maxima_ds = preprocess_for_plotting(np.abs(Z_q_s), target_width)

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
    if len(coef_ds.shape) == 2:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(np.abs(coef_ds), extent=extent, aspect='auto', cmap='viridis')
        ax.set_title("CWT amplitúdó (Ricker hullám)")
        ax.set_ylabel("Skála")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{filename_prefix}_cwt.png", dpi=dpi, bbox_inches='tight')
        plt.close()

    # 3. Modulus maxima
    if False:
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

    print(f"✅ mfdfa eredmények elmentve mappába: {output_dir}")
    return fig, ax