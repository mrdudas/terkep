import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import UnivariateSpline
import chhabrajensen
import mfdfa
# --- Chhabra–Jensen multifraktál ---
def chhabra_jensen_multifractal(signal, q_values, window_sizes):
    nq, ns = q_values.shape[0], window_sizes.shape[0]
    Ma, Mf = np.zeros((nq, ns)), np.zeros((nq, ns))

    for i, q in enumerate(q_values):
        for j, scale in enumerate(window_sizes):
            valid_len = (len(signal) // scale) * scale
            reshaped = signal[:valid_len].reshape(-1, scale).T
            p = np.sum(np.abs(reshaped), axis=0)
            total = np.sum(p)
            if total == 0:
                continue
            p = p / total
            safe_p = np.where(p > 0, p, 1e-12)
            log_p = np.log2(safe_p)

            if q == 1.0:
                Ma[i, j] = np.sum(p * log_p)
                Mf[i, j] = np.sum(p * log_p)
            else:
                Ma[i, j] = np.sum(p ** q * log_p)
                Mf[i, j] = np.log2(np.sum(p ** q))

    alpha, f_alpha = [], []
    for i in range(nq):
        coeffs_a = np.polyfit(np.log2(window_sizes), Ma[i, :], 1)
        coeffs_f = np.polyfit(np.log2(window_sizes), Mf[i, :], 1)
        alpha.append(-coeffs_a[0])
        f_alpha.append(-coeffs_f[0])

    return np.array(alpha), np.array(f_alpha), Ma, Mf

def mfdfa_multifractal_spectrum(signal, scales=None, q_values=np.linspace(-5, 5, 41), m=2):
    N = len(signal)
    Y = np.cumsum(signal - np.mean(signal))
    if scales is None:
        max_exp = int(np.floor(np.log2(N // 4)))
        exponents = np.linspace(2, max_exp, num=15)  # legyen több skála
        scales = np.unique(np.round(2 ** exponents).astype(int))
    q_values = np.array(q_values)

    F_q = np.zeros((len(q_values), len(scales)))
    for idx_s, s in enumerate(scales):
        n_segments = N // s
        if n_segments < 2:
            continue
        segments = Y[:n_segments * s].reshape(n_segments, s)

        local_flucts = []
        for seg in segments:
            x = np.arange(s)
            coeffs = np.polyfit(x, seg, m)
            trend = np.polyval(coeffs, x)
            local_flucts.append(np.mean((seg - trend) ** 2))
        F_s = np.array(local_flucts)

        for idx_q, q in enumerate(q_values):
            if q == 0:
                F_q[idx_q, idx_s] = np.exp(0.5 * np.mean(np.log(F_s + 1e-12)))
            else:
                F_q[idx_q, idx_s] = (np.mean(F_s ** (q / 2))) ** (1.0 / q)
    st.write(F_q)
    log_scales = np.log2(scales)
    tau_q = []
    
    
    for idx_q in range(len(q_values)):
        fq_row = F_q[idx_q, :]
        if np.any(fq_row <= 0) or np.any(np.isnan(fq_row)):
            tau_q.append(np.nan)
        else:
            coeffs = np.polyfit(log_scales, np.log2(fq_row + 1e-12), 1)
            tau_q.append(coeffs[0])
    tau_q = np.array(tau_q)

    valid_mask = ~np.isnan(tau_q)
    if np.sum(valid_mask) < 3:
        return np.array([]), np.array([]), tau_q, F_q, log_scales

    q_valid = q_values[valid_mask]
    tau_valid = tau_q[valid_mask]
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            dq = q_valid[1] - q_valid[0] 
            spline = UnivariateSpline(q_valid, tau_valid, k=3, s=0)
            alpha = spline.derivative()(q_valid)
            f_alpha = q_valid * alpha - tau_valid

            # Érvénytelen értékek kiszűrése
            mask = ~(np.isnan(alpha) | np.isinf(alpha) | np.isnan(f_alpha) | np.isinf(f_alpha))
            alpha = alpha[mask]
            f_alpha = f_alpha[mask]
            #st.write ("alpha", alpha)
            #st.write ("f_alpha", f_alpha)

    except Exception as e:
        print(f"Gradient calculation failed: {e}")
        return np.array([]), np.array([]), tau_q, F_q, log_scales

    return alpha, f_alpha, tau_q, F_q, log_scales

def generate_test_signal(signal_type="pink", N=4096, alpha=1.0):
    if signal_type == "white":
        return np.random.normal(0, 1, N)

    elif signal_type == "pink":  # 1/f
        freqs = np.fft.rfftfreq(N)
        freqs[0] = 1e-6
        amplitude = 1 / freqs**(alpha / 2)
        phase = np.exp(2j * np.pi * np.random.rand(len(freqs)))
        spectrum = amplitude * phase
        return np.fft.irfft(spectrum, n=N)

    elif signal_type == "brownian":
        return np.cumsum(np.random.randn(N))

    elif signal_type == "chirp":
        t = np.linspace(0, 1, N)
        return np.sin(2 * np.pi * t * t * 50)

    elif signal_type == "sinus+noise":
        t = np.linspace(0, 1, N)
        return np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(N)

    elif signal_type == "step":
        signal = np.zeros(N)
        signal[N//4:N//2] = 1
        signal[3*N//4:] = -1
        return signal + 0.1 * np.random.randn(N)

    else:
        raise ValueError("Ismeretlen signal_type")

# --- 1/f zaj generátor ---
def generate_1_over_f_noise(N, alpha=1.0):
    freqs = np.fft.rfftfreq(N)
    freqs[0] = 1e-10
    amplitude = 1 / freqs**(alpha / 2)
    phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))
    spectrum = amplitude * phases
    signal = np.fft.irfft(spectrum)
    return signal

# --- Szűrés és normalizálás ---
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

# --- Streamlit app ---
st.title("Multifraktál spektrum elemzés - EEG vagy 1/f jel")
st.markdown("""
Ez az alkalmazás lehetővé teszi saját EEG vagy szintetikus 1/f jel multifraktális spektrumának meghatározását **Chhabra–Jensen** és **MFDFA** módszerekkel.
""")

uploaded_file = st.file_uploader("Tölts fel EEG vagy bármilyen idősor fájlt (.csv, .txt)", type=["csv", "txt"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    signal = df.iloc[:, 0].dropna().values
    st.success("Sikeres fájlbetöltés. A jel hossza: {} pont".format(len(signal)))
else:
    #st.selectbox("Válassz szintetikus jelet", ("1/f zaj", "Brownian zaj", "Chirp zaj", "Sinus + zaj", "Lépés jel"))
    signal_type = st.selectbox("Jel típusa", ("pink", "white", "brownian", "chirp", "sinus+noise", "step"))
    N = st.slider("Szintetikus 1/f jel hossza", 1024, 16384, 4096, step=1024)
    alpha = st.slider("Alpha érték (1/f zaj)", 0.1, 2.0, 1.0, step=0.1)
    if signal_type == "pink":

        signal = generate_1_over_f_noise(N, alpha=alpha)
    else:  
        signal = generate_test_signal(signal_type=signal_type, N=N, alpha=alpha)
   

# --- Szűrés és normalizálás opció ---
if st.checkbox("Alacsonyfrekvenciás szűrés (cutoff = 30 Hz, fs = 250 Hz)"):
    signal = butter_lowpass_filter(signal, cutoff=30, fs=250)
    st.success("Szűrés alkalmazva.")

if st.checkbox("Normalizálás (z-score)"):
    signal = normalize_signal(signal)
    st.success("Normalizálás alkalmazva.")

# Eredeti jel kirajzolása
fig, ax = plt.subplots()
ax.plot(signal, color='black')
ax.set_title("Előkészített bemenő jel")
ax.set_xlabel("Idő (mintavételi pont)")
ax.set_ylabel("Amplitúdó")
st.pyplot(fig)

# Paraméterek
q_values = np.linspace(-5, 5, 41)
scales = np.unique(np.round(np.logspace(2, np.log2(len(signal)//8), num=20, base=2)).astype(int))

# CJ elemzés
def run_chhabra_jensen():
    alpha, f_alpha, *_ = chhabra_jensen_multifractal(signal, q_values, scales)
    fig, ax = plt.subplots()
    ax.plot(alpha, f_alpha, label="Chhabra–Jensen")
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("Multifraktál spektrum - CJ")
    ax.grid(True)
    st.pyplot(fig)
    
    alpha, falpha, Dq, Rsqr_falpha = chhabrajensen.chhabra_jensen_multifractal (signal, q_values=q_values, window_sizes=scales)
    fig, ax = plt.subplots()
    ax.plot(alpha, f_alpha, label="Chhabra–Jensen")
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("Multifraktál spektrum - CJ")
    ax.grid(True)
    st.pyplot(fig)

# MFDFA elemzés
def run_mfdfa():
    alpha, f_alpha, tau_q, F_q, log_scales = mfdfa_multifractal_spectrum(signal, scales=scales, q_values=q_values)

    # Debug: F_q hőtérkép
    fig1, ax1 = plt.subplots()
    c = ax1.imshow(F_q, aspect='auto', origin='lower', interpolation='none', extent=[log_scales[0], log_scales[-1], q_values[0], q_values[-1]])
    fig1.colorbar(c, ax=ax1)
    ax1.set_title("F_q skálázás (log-log térben)")
    ax1.set_xlabel("log2(skála)")
    ax1.set_ylabel("q")
    st.pyplot(fig1)

    # Debug: tau(q) görbe
    fig2, ax2 = plt.subplots()
    ax2.plot(q_values, tau_q, marker='o')
    ax2.set_title("Tau(q) görbe")
    ax2.set_xlabel("q")
    ax2.set_ylabel("tau(q)")
    ax2.grid(True)
    st.pyplot(fig2)

    # Debug: tau_valid vs q_valid
    valid_mask = ~np.isnan(tau_q)
    q_valid = q_values[valid_mask]
    tau_valid = tau_q[valid_mask]

    fig3, ax3 = plt.subplots()
    ax3.plot(q_valid, tau_valid, 'bo-', label="tau_valid")


    # Debug: alpha és f_alpha értékek
    st.subheader("Alpha és f(Alpha) értékek")
    debug_df = pd.DataFrame({
        "alpha": alpha,
        "f_alpha": f_alpha
    })
    #st.dataframe(debug_df)

    if alpha.size == 0 or f_alpha.size == 0 or np.all(np.isnan(alpha)) or np.all(np.isnan(f_alpha)):
        st.warning("Nincs elég érvényes adat a MFDFA spektrum megjelenítéséhez.")
        return

    # Spektrum maximum megjelölése
    max_idx = np.argmax(f_alpha)
    alpha_max = alpha[max_idx]
    f_alpha_max = f_alpha[max_idx]

    
    fig, ax = plt.subplots()
    ax.plot(alpha, f_alpha, label="MFDFA", color='orange')
    ax.plot(alpha_max, f_alpha_max, 'ro', label=f"max: α={alpha_max:.3f}, f(α)={f_alpha_max:.3f}")
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("Multifraktál spektrum - MFDFA")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    alpha, f_alpha, mfdfa_tau_q, mfdfa_Z_q_s, mfdfa_coef, mfdfa_maxima, log_scales2 = mfdfa.mfdfa_multifractal_spectrum (signal, scales=scales, q_values=q_values)
      # Spektrum maximum megjelölése
    max_idx = np.argmax(f_alpha)
    alpha_max = alpha[max_idx]
    f_alpha_max = f_alpha[max_idx]
    
    fig, ax = plt.subplots()
    ax.plot(alpha, f_alpha, label="MFDFA", color='orange')
    ax.plot(alpha_max, f_alpha_max, 'ro', label=f"max: α={alpha_max:.3f}, f(α)={f_alpha_max:.3f}")
    ax.set_xlabel("α")
    ax.set_ylabel("f(α)")
    ax.set_title("Multifraktál spektrum - MFDFA")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
            

    

# Gombok
col1, col2 = st.columns(2)
with col1:
    if st.button("Chhabra–Jensen elemzés"):
        run_chhabra_jensen()

with col2:
    if st.button("MFDFA elemzés"):
        run_mfdfa()
