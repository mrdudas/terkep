import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import streamlit as st
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline


# --- 1. Log-log lineáris regresszió ---
@njit(parallel=True)
def linear_regression_loglog(x, y):
    n = x.shape[0]
    sum_x, sum_y = np.sum(x), np.sum(y)
    sum_xx, sum_xy = np.sum(x * x), np.sum(x * y)
    sum_yy = np.sum(y * y)
    denom = n * sum_xx - sum_x ** 2
    if denom == 0:
        return 0.0, 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R² determinációs együttható (mindig >= 0)
    numerator = (n * sum_xy - sum_x * sum_y) ** 2
    denominator = (n * sum_xx - sum_x**2) * (n * sum_yy - sum_y**2)
    r_squared = numerator / denominator if denominator != 0 else 0.0

    return slope, intercept, r_squared

@njit
def compute_zq_r2_single(signal, q, window_sizes):
    """
    Egyetlen q értékre kiszámítja a Z_q skálázás log-log illesztési R² értékét.
    """
    log_scales = np.log2(window_sizes.astype(np.float64))
    log_zq = []

    for scale in window_sizes:
        valid_len = (len(signal) // scale) * scale
        if valid_len == 0:
            log_zq.append(np.nan)
            continue
        reshaped = signal[:valid_len].reshape(-1, scale).T
        p = np.sum(reshaped, axis=0)
        total = np.sum(p)
        if total == 0:
            log_zq.append(np.nan)
            continue
        p = p / total
        safe_p = np.where(p > 0, p, 1e-12)
        Zq = np.sum(safe_p ** q)
        log_zq.append(np.log2(Zq) if Zq > 0 else np.nan)

    log_zq = np.array(log_zq)
    valid = ~np.isnan(log_zq)
    if np.sum(valid) >= 2:
        _, _, r2 = linear_regression_loglog(log_scales[valid], log_zq[valid])
        return r2
    else:
        return np.nan

@njit
def compute_zq_r2_vs_q(signal, q_values, window_sizes):
    ret = np.zeros(len(q_values))
    for i in prange(len(q_values)):
        q=q_values[i]
        R2 = compute_zq_r2_single(signal, q, window_sizes)
        if R2 is not None:
            ret[i] = np.abs(R2)
        else:
            ret[i] = np.nan
    return ret
 


def chhabra_jensen(signal, q_values, window_sizes, use_abs=True):
    """
    Chhabra-Jensen multifraktális spektrum számítása
    - signal: 1D tömb (pozitív mérték vagy időjel)
    - q_values: np.array, q skálák
    - window_sizes: np.array, vizsgált dobozméretek
    - use_abs: ha True, a mérték számításakor abszolút értéket használ
    Visszatér: alpha, f_alpha, tau_q, Z_q_s, coef, R^2
    """
    nq, ns = q_values.shape[0], window_sizes.shape[0]
    Ma = np.zeros((nq, ns))  # alpha-related moments
    Mf = np.zeros((nq, ns))  # f(alpha)-related moments
    Md = np.zeros((nq, ns))  # partition function Z(q,s)

    for i, q in enumerate(q_values):
        for j, scale in enumerate(window_sizes):
            valid_len = (len(signal) // int(scale)) * int(scale)
            reshaped = signal[:valid_len].reshape(-1, int(scale)).T
            if use_abs:
                p = np.sum(np.abs(reshaped), axis=0)
            else:
                p = np.sum(reshaped, axis=0)

            total = np.sum(p)
            if total == 0:
                continue
            p = p / total
            safe_p = np.where(p > 0, p, 1e-12)

            p_q = safe_p ** q
            Z_q = np.sum(p_q)
            mu_q = p_q / Z_q if Z_q > 0 else np.zeros_like(p_q)
            safe_mu = np.where(mu_q > 0, mu_q, 1e-12)

            Ma[i, j] = np.sum(mu_q * np.log2(safe_p))
            Mf[i, j] = np.sum(mu_q * np.log2(safe_mu))
            Md[i, j] = np.log2(Z_q) if q != 1 else -np.sum(safe_p * np.log2(safe_p))

    alpha, f_alpha, tau_q, R2, h_q = np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nq)
    coef = np.zeros((nq, 2))  # slope, intercept for f(α)
    log_scales = np.log2(window_sizes.astype(np.float64))

    for i in range(nq):
        alpha[i], _, _ = linear_regression_loglog(log_scales, Ma[i])
        f_alpha[i], intercept_f, R2[i] = linear_regression_loglog(log_scales, Mf[i])
        slope_dq, _, _ = linear_regression_loglog(log_scales, Md[i])
        tau_q[i] = slope_dq if q_values[i] == 1 else (q_values[i] - 1) * slope_dq
        coef[i] = (f_alpha[i], intercept_f)
    return alpha, f_alpha, tau_q, Md, coef, h_q, 0, R2
    return {
        "alpha": alpha,
        "f_alpha": f_alpha,
        "tau_q": tau_q,
        "Z_q_s": Md,
        "coef": coef,
        "R2": R2,
        "q": q_values,
        "log_scales": log_scales
    }


# --- 2. Chhabra–Jensen multifraktál spektrum kiszámítása ---
@njit(parallel=True)
def chhabra_jensen_multifractal(signal, q_values, window_sizes, keep_ratio=0.95):
    nq, ns = q_values.shape[0], window_sizes.shape[0]
    Ma, Mf, Md = np.zeros((nq, ns)), np.zeros((nq, ns)), np.zeros((nq, ns))

    for i in prange(nq):
        q = q_values[i]
        for j in range(ns):
            scale = window_sizes[j]

            # számítsuk ki a leghosszabb, scale-méretre osztható prefixet
            valid_len = (len(signal) // int(scale)) * int(scale)
            reshaped = signal[:valid_len].reshape(-1, int(scale)).T
            p = np.sum(reshaped, axis=0)
            total = np.sum(p)
            if total == 0:
                continue
            p = p / total
            safe_p = np.where(p > 0, p, 1e-12)

            p_q = safe_p ** q
            Z_q = np.sum(p_q)
            mu_q = p_q / Z_q if Z_q > 0 else np.zeros_like(p_q)
            safe_mu = np.where(mu_q > 0, mu_q, 1e-12)

            Ma[i, j] = np.sum(mu_q * np.log2(safe_p))
            Mf[i, j] = np.sum(mu_q * np.log2(safe_mu))
            Md[i, j] = np.log2(Z_q) if q != 1 else -np.sum(safe_p * np.log2(safe_p))

    alpha, falpha, Dq, Rsqr_falpha = np.zeros(nq), np.zeros(nq), np.zeros(nq), np.zeros(nq)
    log_scales = np.log2(window_sizes.astype(np.float64))

    for i in range(nq):
        alpha[i], _, _ = linear_regression_loglog(log_scales, Ma[i])
        falpha[i], _, Rsqr_falpha[i] = linear_regression_loglog(log_scales, Mf[i])
        slope_dq, _, _ = linear_regression_loglog(log_scales, Md[i])
        Dq[i] = slope_dq / (q_values[i] - 1) if q_values[i] != 1 else slope_dq
        #print (f" α: {alpha[i]:.3f}, f(α): {falpha[i]:.3f}, D(q): {Dq[i]:.3f}, R²: {Rsqr_falpha[i]:.3f}")
    
    #print (f"Z_q skálázás R²: {Rsqr_falpha}")
    #print (f" α: {alpha}")
    return alpha, falpha, Dq, Rsqr_falpha


# --- Z_q skálázási R² ábrázolása ---
def plot_r2_vs_q(q_values, r2_values, title="Z_q skálázás illesztési R²"):
    plt.figure(figsize=(7, 4))
    plt.plot(q_values, r2_values, 'o-', label='R²(q)')
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(0.95, color='red', linestyle='--', linewidth=0.5, label='0.98 küszöb')
    plt.xlabel('q')
    plt.ylabel('R²')
    plt.title(title)
    plt.grid(True)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    return plt


def filter_multifractal_spectrum(alpha, falpha, Rsqr_falpha=None, q_vals=None,
                                 r2_threshold=0.95, exclude_edges=2, keep_peak=True,
                                 compute_width=True, preserve_symmetry=False):
    n = len(alpha)
    mask = np.ones(n, dtype=bool)
    #st.write (f"Szűrés: {n}")
    #st.write (f"Szűrés: {alpha}")


    # Alap szélső érték szűrés
    if exclude_edges > 0:
        mask[:exclude_edges] = False
        mask[-exclude_edges:] = False

    # R² küszöb szűrés
    if Rsqr_falpha is not None and len(Rsqr_falpha) == n:
        mask &= (Rsqr_falpha >= r2_threshold)

    # Szimmetria megőrzés (ha bekapcsolt)
    if preserve_symmetry and q_vals is not None:
        for i in range(n):
            q = q_vals[i]
            if not mask[i]:  # ha ez a q kiesett...
                for j in range(n):
                    if np.isclose(q_vals[j], -q) and mask[j]:
                        mask[j] = False  # ... dobjuk ki a -q párját is
                        break

    # Mindig megtartjuk az f(α) maximumát
    if keep_peak:
        peak_idx = np.argmax(falpha)
        mask[peak_idx] = True

    result = {
        'alpha': alpha[mask],
        'falpha': falpha[mask],
        'indices': np.nonzero(mask)[0]
    }

    if q_vals is not None:
        result['q_vals'] = q_vals[mask]
    if Rsqr_falpha is not None:
        result['Rsqr_falpha'] = Rsqr_falpha[mask]
    if compute_width and np.sum(mask) > 1:
        result['alpha_width'] = np.max(result['alpha']) - np.min(result['alpha'])
    
    #st.write (f"Szűrés: {result}")

    return result

# --- 4. f(α) spektrum ábrázolása ---
def plot_falpha_spectrum(alpha, falpha, Rsqr_falpha=None, q_vals=None,
                         r2_threshold=0.95, exclude_edges=2, title='f(α) spektrum'):
    result = filter_multifractal_spectrum(alpha, falpha, Rsqr_falpha, q_vals,
                                          r2_threshold, exclude_edges)
    plt.figure(figsize=(7, 5))
    plt.plot(alpha, falpha, 'o--', alpha=0.3, label='Eredeti')
    plt.plot(result['alpha'], result['falpha'], 'o-', label='Szűrt')
    plt.xlabel('α')
    plt.ylabel('f(α)')
    plt.title(title + (f"\nα-szélesség: {result.get('alpha_width', 0):.3f}"))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return result, plt

# --- 5. D(q) spektrum ábrázolása ---
def plot_dq_curve(q_vals, Dq, title="D(q) spektrum"):
    plt.figure(figsize=(7, 4))
    plt.plot(q_vals, Dq, 'o-', label='D(q)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('q')
    plt.ylabel('D(q)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt

# --- 6. μ(q, i) mátrix számítása ---
@njit(parallel=True)
def compute_mu_q_matrix(signal, q_vals, window_size):
    nq = len(q_vals)
    valid_length = (len(signal) // window_size) * window_size
    reshaped = signal[:valid_length].reshape(-1, window_size).T
    window_sums = np.sum(reshaped, axis=0)
    total = np.sum(window_sums)
    if total <= 0:
        return np.zeros((nq, reshaped.shape[1]))
    p = window_sums / total
    safe_p = np.where(p > 0, p, 1e-12)

    mu_matrix = np.zeros((nq, len(safe_p)))
    for i in prange(nq):
        p_q = safe_p ** q_vals[i]
        Zq = np.sum(p_q)
        mu_matrix[i] = p_q / Zq if Zq > 0 else 0.0
    return mu_matrix

# --- 7. μ(q, i) hőtérkép ---
def plot_mu_q_heatmap(signal, q_vals, scale=32, colormap='plasma', save_path=None, dpi=300):
    length = (len(signal) // scale) * scale
    reshaped = signal[:length].reshape(-1, scale).T
    window_sums = np.sum(reshaped, axis=0)
    p = window_sums / np.sum(window_sums)
    safe_p = np.where(p > 0, p, 1e-12)

    mu_map = []
    for q in q_vals:
        p_q = safe_p ** q
        Zq = np.sum(p_q)
        mu_q = p_q / Zq if Zq > 0 else np.zeros_like(p_q)
        mu_map.append(mu_q)

    mu_map = np.array(mu_map)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mu_map, aspect='auto', cmap=colormap,
                   extent=[0, mu_map.shape[1], q_vals[-1], q_vals[0]])
    ax.set_title(f"μ(q, i) hőtérkép (skála={scale})")
    ax.set_xlabel("Ablak index")
    ax.set_ylabel("q érték")
    plt.colorbar(im, ax=ax, label="μ(q, i)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, ax, mu_map

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

# --- Z_q skálázódás vizsgálata egyes q értékekre külön ábrákon ---
def plot_zq_scaling_per_q(signal, q_values, window_sizes):
    """
    Minden q értékre külön-külön kirajzolja a log2(Z_q) vs log2(scale) görbét,
    valamint kiírja az illesztett egyenes R² értékét.
    """
    log_scales = np.log2(window_sizes.astype(np.float64))
    for q in q_values:
        log_zq = []
        for scale in window_sizes:
            valid_len = (len(signal) // scale) * scale
            if valid_len == 0:
                log_zq.append(np.nan)
                continue
            reshaped = signal[:valid_len].reshape(-1, scale).T
            p = np.sum(reshaped, axis=0)
            total = np.sum(p)
            if total == 0:
                log_zq.append(np.nan)
                continue
            p = p / total
            safe_p = np.where(p > 0, p, 1e-12)
            Zq = np.sum(safe_p ** q)
            log_zq.append(np.log2(Zq) if Zq > 0 else np.nan)

        log_zq = np.array(log_zq)
        valid = ~np.isnan(log_zq)
        if np.sum(valid) >= 2:
            slope, intercept, r2 = linear_regression_loglog(log_scales[valid], log_zq[valid])
            plt.figure(figsize=(6, 4))
            plt.plot(log_scales[valid], log_zq[valid], 'o-', label=f'q = {q:.2f}')
            plt.plot(log_scales[valid], slope * log_scales[valid] + intercept, '--', label=f'Illesztés (R²={r2:.3f})')
            plt.xlabel('log₂(scale)')
            plt.ylabel('log₂(Z_q)')
            plt.title(f'Skálázódás Z_q, q = {q:.2f}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            st.pyplot (plt)
            plt.close()
            
    return plt

def prepare_signal(signal, min_window=4,  padding_len=0, smoothing=0.1):
    #st.write (f"CHHB Jel előkészítése: {len(signal)}")
    #signal_max = np.max(np.abs(signal))
    #signal = signal / signal_max if signal_max > 0 else signal
    # egységmi szórású normalizálás
    #signal = (signal - np.mean(signal)) / np.std(signal) if np.std(signal) > 0 else signal
    #signal.astype(np.float32)
    # normalizálunk és eltávolítjuk a negatív értékeket
    signal = moving_average_with_interpolation (signal, window=3)
    #signal = UnivariateSpline(np.arange(len(signal)), signal, s=smoothing)(np.arange(len(signal)))
    signal = np.abs (signal)+1e-12
    valid_length = (len(signal) // min_window) * min_window
    signal = signal[:valid_length]
    #st.write (f"CHHB Jel előkészítve: {len(signal)}")
    #st.line_chart (signal, use_container_width=True)
    return signal


def analyze_multifractal_over_time(signal, sampling_rate, window_sec, window_sizes, q_values,  r2_threshold=0.95):
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
    segment_size = int(window_sec * sampling_rate)
    num_segments = int ((len(signal) - segment_size) // segment_size + 1)
    

    times_sec = []
    widths = []
    min_window_size = np.min (window_sizes) 

    for seg_idx in range(num_segments):
        start = seg_idx * segment_size
        end = start + segment_size
        segment = signal[start:end]
        print (f"Szegmens {seg_idx}: {start/sampling_rate:.2f} - {end/sampling_rate:.2f} másodperc len: {len(segment)}")
        print (f" min.window_size: {min_window_size}, min valid len: {len(segment) // min_window_size * min_window_size}")

        try:
        #    st.line_chart (segment, use_container_width=True)
            #segment = prepare_signal(segment, min_window=min_window_size)
            #print (f"  Előkészített szegmens: {len(segment)}")
            alpha, falpha, Dq, Rsqr_falpha  = chhabra_jensen_multifractal(segment, q_values=q_values,  window_sizes=window_sizes, keep_ratio=0.95)

            result = filter_multifractal_spectrum(alpha, falpha, Rsqr_falpha=Rsqr_falpha, q_vals=q_values,
                                      r2_threshold=r2_threshold, exclude_edges=2, keep_peak=True,
                                       compute_width=True )
            
            alpha = result['alpha']
        
            valid_alpha = alpha[np.isfinite(alpha)]

            if valid_alpha.size >= 2:
                width = np.max(valid_alpha) - np.min(valid_alpha)
                #st.write(f"Alpha width: {width:.4f}")
            else:
                width = np.nan
                #st.warning("Nem volt elég érvényes α érték a szélesség számításhoz.")

        except Exception as e:
            print(f"❌ Szegmens {seg_idx}: hiba a CHHB-ben: {e}")
            width = np.nan

        center_time = (start + end) / 2 / sampling_rate
        times_sec.append(center_time)
        widths.append(width)

    return np.array(times_sec), np.array(widths)
