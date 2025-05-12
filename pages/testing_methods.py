import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from numpy.polynomial import Polynomial
from scipy.stats import skew
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA



from test_signal import generate_test_signal, generate_1_over_f_noise
from filters import (
    butter_lowpass_filter,
    normalize_signal,
    gpr_fill,
    gpr_fill_local,
    moving_average,
    savitzky_golay_filter,
    log_scale_transform,
    scale_invariant_transform,

)
import chhabrajensen
from mfdfa import mfdfa_multifractal_spectrum
filename = "multifractal_analysis _test_signal"
dir = os.path.dirname(__file__)
st.set_page_config(layout="wide")

def pca_phase_plot (signal, tau): 

    # Mean-centered cumsum
    centered = signal - np.mean(signal)
    cumsum = np.cumsum(centered)

    # Eredeti + cumsum összefűzése mint 2D pontok
    X = np.vstack((signal, cumsum)).T

    # PCA vetítés
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Ábra
    plt.figure(figsize=(6,6))
    plt.plot(X[:,0], X[:,1], '.', markersize=1, alpha=0.6)
    plt.xlabel('x(t)')
    plt.ylabel('∑(x - mean)')
    plt.title('Fázisdiagram: x(t) vs. mean-centered cumsum')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # PCA komponensek megjelenítése (opcionális)
    plt.figure(figsize=(6,6))
    plt.plot(X_pca[:,0], X_pca[:,1], '.', markersize=1, alpha=0.6)
    plt.title("Fázistér PCA komponensekkel")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
    
   


def phase_plot (signal, tau):
    # Paraméterek
    if tau is None:
        tau = 20  # időeltolás mintában (pl. 20 ms, ha 1 kHz a mintavételezés)
    
    signal = signal - np.mean(signal) / np.std(signal)
    # Eltolásos fázistér
    #x_t = signal[:-tau]
    #x_t_tau = signal[tau:]
    x_t = signal[:]
    x_t_tau_2 = np.diff(signal[:], prepend=0) 
    x_t_tau_2 = (x_t_tau_2 - np.mean(x_t_tau_2)) / np.std(x_t_tau_2)
    x_t_tau_3 = np.diff(x_t_tau_2, prepend=0) 
    x_t_tau_3 = (x_t_tau_3 - np.mean(x_t_tau_3)) / np.std(x_t_tau_3)
    x_t_tau_4 = np.diff(x_t_tau_3, prepend=0)
    x_t_tau_4 = (x_t_tau_4 - np.mean(x_t_tau_4)) / np.std(x_t_tau_4)
   


    x_t_tau = np.cumsum(signal - np.mean(signal))
    x_t_tau = (x_t_tau - np.mean(x_t_tau)) / np.std(x_t_tau)
    x_t_tau2 = np.cumsum(x_t_tau - np.mean(x_t_tau))
    x_t_tau2 = (x_t_tau2 - np.mean(x_t_tau2)) / np.std(x_t_tau2)
    x_t_tau3 = np.cumsum(x_t_tau2 - np.mean(x_t_tau2))
    x_t_tau3 = (x_t_tau3 - np.mean(x_t_tau3)) / np.std(x_t_tau3)
    x_t_tau4 = np.cumsum(x_t_tau3 - np.mean(x_t_tau3))
    x_t_tau4 = (x_t_tau4 - np.mean(x_t_tau4)) / np.std(x_t_tau4)
    st.line_chart({"x(t)": x_t, "x(t+tau)": x_t_tau, "x(t+2*tau)": x_t_tau2, "x(t+3*tau)": x_t_tau3, "x(t+4*tau)": x_t_tau4})
    
    st.line_chart({"x(t)": x_t, "x(t+tau)": x_t_tau_2, "x(t+2*tau)": x_t_tau_3, "x(t+3*tau)": x_t_tau_4} )
    #x_t_tau = x_t_tau - np.mean(x_t_tau)

    
    
    # Ábra
    #x_t = x_t - np.mean(x_t)
    #x_t_tau = x_t_tau - np.mean(x_t_tau)
    x_t = x_t
    plt.figure(figsize=(6, 6))
    plt.plot(x_t, x_t_tau, '-o', markersize=1, color='blue', alpha=0.5)
    plt.plot(x_t_tau, x_t_tau2, '-o', markersize=1, color='red', alpha=0.5)
    plt.plot(x_t_tau2, x_t_tau3, '-o', markersize=1, color='green', alpha=0.5)
    plt.plot(x_t_tau3, x_t_tau4, '-o', markersize=1, color='orange', alpha=0.5)
    plt.plot(x_t, x_t_tau_2, '-o', markersize=1, color='purple', alpha=0.5)
    #plt.plot(x_t_tau_2, x_t_tau_3, '-o', markersize=1, color='brown', alpha=0.5)
    #plt.plot(x_t_tau_3, x_t_tau_4, '-o', markersize=1, color='pink', alpha=0.5)
    plt.xlabel('x(t)')
    plt.ylabel(f'x(t) cummsum )')
    plt.title('2D fázisdiagram (idősor)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot(plt)  
    plt.close()
   
    # distribution hystogram of x_t and x_t_tau
    plt.figure(figsize=(6, 6))
    plt.hist(x_t, bins=50, density=True, alpha=0.5, color='blue', label='x(t)')
    plt.hist(x_t_tau, bins=50, density=True, alpha=0.5, color='red', label=f'x(t+{tau})')
    plt.xlabel('x(t)')
    plt.ylabel('Density')
    plt.title('Eloszlás hisztogram (x(t) és x(t+tau))')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# --- Debug kapcsoló ---
DEBUG = st.sidebar.checkbox("Debug mód", value=False)

def analyze_spectrum(alpha, f_alpha, q_values=None, tau_q=None, Dq=None):
    """
    Multifraktál spektrum kvantitatív jellemzése.

    Paraméterek:
        alpha : np.ndarray
            Szingularitás-exponensek (α)
        f_alpha : np.ndarray
            f(α) spektrum
        q_values : np.ndarray or None
            A használt q értékek (τ(q) analízishez szükséges)
        tau_q : np.ndarray or None
            τ(q) értékek, ha elérhetők
        Dq : np.ndarray or None
            D(q) értékek, ha elérhetők

    Visszatérés:
        dict : minden jellemző számszerűsítve
    """
    result = {}

    # Tisztítás
    valid = (~np.isnan(alpha)) & (~np.isnan(f_alpha)) & (~np.isinf(alpha)) & (~np.isinf(f_alpha))
    a = alpha[valid]
    f = f_alpha[valid]

    if len(a) < 2:
        return {"error": "Not enough valid data to analyze spectrum."}

    # Alap jellemzők
    result['alpha_min'] = np.min(a)
    result['alpha_max'] = np.max(a)
    result['spectrum_width'] = result['alpha_max'] - result['alpha_min']
    result['area_under_curve'] = np.trapz(f, x=a)

    # Csúcs (domináns szingularitás)
    idx_peak = np.argmax(f)
    result['alpha_peak'] = a[idx_peak]
    result['f_alpha_peak'] = f[idx_peak]

    # Aszimmetria
    result['alpha_skewness'] = skew(a)
    result['f_alpha_skewness'] = skew(f)

    # Konkavitás (f(α) második deriváltja spline alapján)
    try:
        spline = UnivariateSpline(a, f, s=0.5, k=4)
        second_deriv = spline.derivative(n=2)(a)
        result['f_alpha_concavity_median'] = np.median(second_deriv)
    except Exception as e:
        result['f_alpha_concavity_median'] = np.nan

    # τ(q) görbület (csak ha elérhető)
    if q_values is not None and tau_q is not None:
        valid_tau = (~np.isnan(tau_q)) & (~np.isinf(tau_q))
        if np.sum(valid_tau) > 4:
            coeffs = np.polyfit(q_values[valid_tau], tau_q[valid_tau], 2)
            result['tau_q_curvature'] = coeffs[0]
        else:
            result['tau_q_curvature'] = np.nan
    else:
        result['tau_q_curvature'] = np.nan

    # D(q) dimenziók (ha elérhetők)
    if Dq is not None and q_values is not None:
        for q0 in [0, 1, 2]:
            idx = np.argmin(np.abs(q_values - q0))
            result[f'D({q0})'] = Dq[idx]
    else:
        result['D(0)'] = result['D(1)'] = result['D(2)'] = np.nan

    return result


def plot_f_alpha_spectrum_with_metrics(
    alpha,
    f_alpha,
    q_values=None,
    tau_q=None,
    Dq=None,
    title="Multifraktál spektrum (f(α))",
    save_path=None,
    dpi=300
    ):
    from textwrap import dedent
    from matplotlib.ticker import MaxNLocator

    # Spektrum jellemzése
    metrics = analyze_spectrum(alpha, f_alpha, q_values=q_values, tau_q=tau_q, Dq=Dq)

    # Érvényes adatok szűrése
    valid = (~np.isnan(alpha)) & (~np.isnan(f_alpha)) & (~np.isinf(alpha)) & (~np.isinf(f_alpha))
    alpha_clean = alpha[valid]
    f_clean = f_alpha[valid]

    if len(alpha_clean) < 2:
        print("⚠️ Nem elég adat az f(α) spektrum kirajzolásához.")
        return

    # Ábra
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(alpha_clean, f_clean, '-o', color='navy', label="f(α)")

    # Szövegdoboz (annotáció)
    annotation = dedent(f"""
        Δα = {metrics['spectrum_width']:.4f}
        ∫f(α)dα = {metrics['area_under_curve']:.4f}
        α_peak = {metrics['alpha_peak']:.4f}
        skew(α) = {metrics['alpha_skewness']:.4f}
        skew(f(α)) = {metrics['f_alpha_skewness']:.4f}
        τ(q) görbület ≈ {metrics['tau_q_curvature']:.4f}
    """).strip()

    ax.text(0.05, 0.95, annotation, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # Tengelyek, rács, stílus
    ax.set_title(title)
    ax.set_xlabel("α (szingularitás erőssége)")
    ax.set_ylabel("f(α)")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"📁 f(α) spektrum mentve: {save_path}")

    #plt.show()
    return fig, metrics


# --- SessionState inicializálás ---
if 'original_signal' not in st.session_state:
    st.session_state.original_signal = None
    st.session_state.time = None

# --- Oldalsáv: Jelforrás kiválasztás ---
st.sidebar.header("1. Jelforrás kiválasztás")
source = st.sidebar.radio("Jelforrás", [ "Szintetikus tesztjel","Feltöltött fájl"])

available_signals = [
    "white", "pink", "brownian", "chirp", "sinus+noise", "step",
    "gauss", "cantor", "cascade", "fBm", "multifractal", "binomial"
]

uploaded_df = None

if source == "Feltöltött fájl":
    file = st.sidebar.file_uploader("CSV fájl feltöltése (idő, jel)", type=["csv"])
    if file is not None:
        uploaded_df = pd.read_csv(file)
        if uploaded_df.shape[1] == 2:
            uploaded_df.columns = ["time", "value"]
            st.session_state.time = uploaded_df["time"].values
            st.session_state.original_signal = uploaded_df["value"].values
        else:
            st.session_state.original_signal = uploaded_df.iloc[:, 0].dropna().values
            st.session_state.time = np.arange(len(st.session_state.original_signal))
else:
    
    signal_type = st.sidebar.selectbox("Szintetikus jel típusa", available_signals, key="sigtype")
    N = st.sidebar.slider("Jelhossz", 1024, 1048576, 4096, step=1024, key="siglen")
    if signal_type in ["pink", "brownian", "multifractal"]: 
        alpha = st.sidebar.slider("Alpha ", 0.1, 2.0, 1.0, step=0.1, key="sigalpha")
    else:
        alpha = 1.0
        
    if st.sidebar.button("Jel generálása") or st.session_state.original_signal is None:
        st.session_state.original_signal = generate_1_over_f_noise(N, alpha) if signal_type == "pink" else generate_test_signal(signal_type, N, alpha)
        st.session_state.time = np.arange(len(st.session_state.original_signal))

# --- Szűrés input (mindig újra lefut) ---
signal = st.session_state.original_signal.copy()
time = st.session_state.time.copy()

# --- Oldalsáv: Előfeldolgozási pipeline ---
st.sidebar.header("2. Előfeldolgozás")
st.sidebar.write("Több lépésből álló szűrési lánc kiválasztása:")
pipeline = []

if st.sidebar.checkbox("Időbeli jitter kiegyenlítés", value=False):
    if uploaded_df is not None and "time" in uploaded_df.columns:
        resample_len = len(signal)
        def jitter_correction(sig):
            t_uniform = np.linspace(time[0], time[-1], resample_len)
            return np.interp(t_uniform, time, sig)
        pipeline.append(("de-jitter", jitter_correction))
    else:
        st.sidebar.warning("Ez a szűrő csak akkor használható, ha időoszlop is van a bemeneti fájlban.")

for step_label in [
        "Hiányzó értékek kitöltése (globális GPR)",
    "Hiányzó értékek kitöltése (lokális GPR)",
    "Logaritmikus skálázás",
    "Polinom detrend",
    "Mean-centered cumsum",
    "Savitzky-Golay szűrő",
    "Mozgóátlag szűrő",
    "Butterworth aluláteresztő szűrő",
    "Normalizálás (z-score)",
    "Skálafüggetlen transzformáció",
    "FFT spektrum (abs)",
]:

    if st.sidebar.checkbox(step_label):
        if step_label == "Butterworth aluláteresztő szűrő":
            cutoff = st.sidebar.number_input("Vágási frekvencia (Hz)", value=30.0)
            fs = st.sidebar.number_input("Mintavételezési frekvencia (Hz)", value=250.0)
            pipeline.append(("butter", lambda s: butter_lowpass_filter(s, cutoff=cutoff, fs=fs)))

        elif step_label == "Normalizálás (z-score)":
            pipeline.append(("normalize", normalize_signal))

        elif step_label == "FFT spektrum (abs)":
            pipeline.append(("fft_abs", lambda s: np.abs(np.fft.fft(s))))

        elif step_label == "Hiányzó értékek kitöltése (globális GPR)":
            pipeline.append(("gpr_global", lambda s: gpr_fill(s).values))

        elif step_label == "Hiányzó értékek kitöltése (lokális GPR)":
            pipeline.append(("gpr_local", lambda s: gpr_fill_local(s).values))

        elif step_label == "Mozgóátlag szűrő":
            ma_window = st.sidebar.slider("Mozgóátlag ablakméret", min_value=3, max_value=101, value=5, step=2)
            pipeline.append(("moving_avg", lambda s: moving_average(s, window_size=ma_window)))

        elif step_label == "Savitzky-Golay szűrő":
            sg_window = st.sidebar.slider("Savitzky-Golay ablakméret", min_value=5, max_value=101, value=11, step=2)
            sg_order = st.sidebar.slider("Polinomfok", min_value=1, max_value=5, value=2)
            pipeline.append(("savgol", lambda s: savitzky_golay_filter(s, window_length=sg_window, polyorder=sg_order)))

        elif step_label == "Logaritmikus skálázás":
            pipeline.append(("logscale", log_scale_transform))

        elif step_label == "Mean-centered cumsum":
            pipeline.append(("mean_cumsum", lambda s: np.cumsum(s - np.mean(s))))

        elif step_label == "Polinom detrend":
            deg = st.sidebar.slider("Polinom fokszám", min_value=1, max_value=5, value=2)
            pipeline.append(("poly_detrend", lambda s: s - np.polyval(np.polyfit(np.arange(len(s)), s, deg), np.arange(len(s)))))
        elif step_label == "Skálafüggetlen transzformáció":
            pipeline.append(("scale_invariant", scale_invariant_transform))

# --- Főablak: eredeti jel és szűrés ---
st.title("Multifraktál Analízis Platform")
st.subheader("Eredeti vagy szintetikusan generált jel")
resampled = np.interp(
    np.linspace(0, 1, 1024),
    np.linspace(0, 1, len(st.session_state.original_signal)),
    st.session_state.original_signal
)
st.line_chart(resampled)


if DEBUG:
    st.subheader("Szűrési lépések előnézete")

for name, step_func in pipeline:
    prev_signal = signal.copy()
    signal = step_func(signal)
    if DEBUG:
        resampled_debug = np.interp(
            np.linspace(0, 1, 1024),
            np.linspace(0, 1, len(signal)),
            signal
        )
        st.markdown(f"**{name} után:**")
        st.line_chart({"Szűrt jel": resampled_debug})

phase_plot(resampled, 20)
pca_phase_plot (resampled, 20)

# --- Multifraktál elemzés ---
st.header("3. Multifraktál elemzés és összehasonlítás")
q_values = np.linspace(-5, 5, 41)
q_values = q_values[q_values != 1.0]
scales = np.unique(np.round(np.logspace(2, np.log2(len(signal) // 8), num=20, base=2)).astype(int))

# --- Chhabra–Jensen ---
r2_values = chhabrajensen.compute_zq_r2_vs_q(signal, q_values, scales)
plt = chhabrajensen.plot_r2_vs_q( q_values, r2_values, title="R²(q) görbe", )
#plt.savefig(os.path.join(dir, f"{filename}_CHHB_r2_vs_q.png"), dpi=300)
st.pyplot(plt)
plt.close()
alpha_cj, f_cj,Dq, Rsqr_falpha = chhabrajensen.chhabra_jensen_multifractal(signal, q_values, scales)

try:
    fig, chhb_metrics = plot_f_alpha_spectrum_with_metrics ( alpha_cj, f_cj, q_values=q_values, tau_q=None, title="CHHB Multifraktál spektrum (f(α))", save_path=os.path.join(dir, f"{filename}__CHHB_spectrum.png"), dpi=300)
    st.pyplot(fig)
except Exception as e:
    st.write (f"Error in plot_f_alpha_spectrum_with_metrics: {e}")
    st.write (f"Alpha: {alpha_cj}")
    st.write (f"F_alpha: {f_cj}")
    st.write (f"Q_values: {q_values}")
    st.write (f"Tau_q: None")
    st.write (f"Title: CHHB Multifraktál spektrum (f(α))")
    st.write (f"Save path: {os.path.join(dir, f'{filename}__CHHB_spectrum.png')}")
#fig.close()

# --- MFDFA ---
alpha_mf, f_mf, tau_q, F_q, log_scales, *_ = mfdfa_multifractal_spectrum(signal, scales, q_values)
fig_mf, ax_mf = plt.subplots()
ax_mf.plot(alpha_mf, f_mf, label="MFDFA", color='orange')
ax_mf.set_title("MFDFA spektrum")
ax_mf.set_xlabel("alpha")
ax_mf.set_ylabel("f(alpha)")
ax_mf.grid()
st.pyplot(fig_mf)

# --- Debug kimenetek ---
if DEBUG:
    st.subheader("Debug: F(q, s) és Tau(q)")

    fig_debug, ax_debug = plt.subplots()
    cax = ax_debug.imshow(F_q, aspect='auto', origin='lower', extent=[log_scales[0], log_scales[-1], q_values[0], q_values[-1]])
    fig_debug.colorbar(cax, ax=ax_debug)
    ax_debug.set_title("F(q, s) hőtérkép")
    st.pyplot(fig_debug)

    fig_tau, ax_tau = plt.subplots()
    ax_tau.plot(q_values, tau_q, 'bo-')
    ax_tau.set_title("Tau(q) görbe")
    st.pyplot(fig_tau)
