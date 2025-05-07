import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chhabrajensen
import plotly.express as px
import wtmm
import cv2
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d

from scipy.stats import skew
from scipy.interpolate import UnivariateSpline
import numpy as np
import test_signal as ts 
import seaborn as sns
import filters as flt
import mfdfa
import matplotlib.pyplot as plt
import numpy as np



def plot_log_zq_vs_scale(signal, q, window_sizes):
    """
    Egy adott q értékre megmutatja, hogy Z_q hogyan skálázódik a scale függvényében.
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
        p_q = safe_p ** q
        Z_q = np.sum(p_q)
        log_zq.append(np.log2(Z_q) if Z_q > 0 else np.nan)

    log_zq = np.array(log_zq)
    valid = ~np.isnan(log_zq)

    plt.figure(figsize=(6, 4))
    plt.plot(log_scales[valid], log_zq[valid], 'o-', label=f'q = {q}')
    plt.xlabel('log₂(scale)')
    plt.ylabel('log₂(Z_q)')
    plt.title('Skálázódás vizsgálat')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    

    return log_scales[valid], log_zq[valid], plt


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

def compress_dynamics_logscale(x, epsilon=1e-8):
    x = np.clip(x, epsilon, None)            # hogy ne legyen log(0)
    x_log = np.log(x)                        # log csökkenti a dinamikát
    x_norm = (x_log - x_log.min()) / (x_log.max() - x_log.min())  # skálázás 0-1 közé
    return x_norm

def plot_alpha_falpha(alpha, falpha, ):
    plt.figure(figsize=(8, 6))
    plt.scatter(alpha, falpha, marker='o', color='blue', label='Multifractal Spectrum')
    plt.xlabel('Alpha')
    plt.ylabel('f(Alpha)')
    plt.title('Multifractal Spectrum (Alpha vs f(Alpha))')
    plt.grid(True)

    return plt

def lowpass_filter(data, sample_rate_hz , cutoff ):
    nyquist = 0.5 * sample_rate_hz
    #cutoff = sample_rate_hz / 4
    b, a = butter(N=2, Wn=cutoff / nyquist, btype='low')
    return filtfilt(b, a, data)

def shift_to_positive(signal, epsilon=1e-6):
    """
    Ha a bemeneti jel tartalmaz 0 alatti értékeket, eltolja úgy, hogy minden érték > 0 legyen.
    """
    min_val = np.min(signal)
    
    if min_val <= 0:
        shift = abs(min_val) + epsilon
        return signal + shift
    else:
        return signal.copy()

def resample_and_filter_measurements(df, time_col='TIME', m_cols=None, factor=4, apply_filter=True):
    #m_cols = ["FPOGX", "FPOGY", "FPOGD", "FPOGID", "FPOGV", "BPOGX", "BPOGY", "BPOGV"]
    m_cols = ["FPOGX", "FPOGY", "RPD", "LPD","LPMM", "RPMM"]
    m_cols += ["CNT", "TIMETICK", "FPOGS", "FPOGD","BPOGX", "BPOGY", "CX", "CY", "GSR", "HR", "LPCX", "LPCY", "RPCX", "RPCY"]
    

    for col in m_cols:
        df[col].loc[df[col] == 0.0] = np.nan 
    
    # 2. Időlépcső számítás
    df = df.sort_values(by=time_col)
    time_diffs = df[time_col].diff().dropna()
    median_diff = time_diffs.median()
    maxfrq= 1/ (median_diff*2)

    new_time_step = median_diff / factor
    sample_rate_hz = 1.0 / new_time_step
    st.write (f"Old Sample rate: {1.0 / median_diff} Hz")
    st.write (f"New Sample rate: {sample_rate_hz} Hz")
    st.write (f"Max (Cutoff) frequency: {maxfrq} Hz")

    # 3. Új időskála
    t_min = df[time_col].min()
    t_max = df[time_col].max()
    new_times = np.arange(t_min, t_max + new_time_step, new_time_step)
    df_resampled = pd.DataFrame({time_col: new_times})

    # 4. Interpoláció m_cols oszlopokra
    for col in df.columns:
        if col == time_col:
            continue
        if m_cols and col in m_cols:
            print ("Interpolating column:", col)
            valid = df[col].notna()
            if valid.sum() < 2:
                df_resampled[col] = np.nan
                continue
            f = interp1d(df[time_col][valid], df[col][valid], kind='linear', bounds_error=False, fill_value=np.nan)
            df_resampled[col] = f(new_times)
            if apply_filter:
                valid = df_resampled[col].notna()
                if valid.sum() >= 3:
                    filtered = df_resampled[col].copy()
                    filtered[valid] = lowpass_filter(df_resampled[col][valid], sample_rate_hz, maxfrq)
                    df_resampled[col] = filtered

        else:
            # Nearest érték hozzárendelés
            print ("Nearest column:", col)
            temp = df[[time_col, col]].sort_values(by=time_col)
            
            df_resampled = pd.merge_asof(df_resampled, temp, on=time_col, direction='nearest', suffixes=('', f'_{col}'))
            df_resampled[col] = df_resampled[f'{col}']
            #df_resampled.drop(columns=[f'{col}'], inplace=True)

    # 5. NaN átvitel (ahol az eredetiben NaN volt, ott az interpoláltban is legyen az)
    
        
        #nan_times = df[df[col].isna()][time_col]
        #df_resampled.loc[df_resampled[time_col].isin(nan_times), col] = np.nan



    return df_resampled

def IQR (data):
    #st.write ("IRQ", len(data))
    #st.line_chart(data)
    # IQR kiszámítása
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    #st.write (f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR} MAX: {Q1 - 1. * IQR} MIN: {Q3 + 1. * IQR}") 

    # Szűrés: legyen nan minden ami az elfogadási tartományon kívül van
    filtered_data = data.copy()
    filtered_data[(data < (Q1 - 1. * IQR)) | (data > (Q3 + 1. * IQR))] = np.nan
    #st.line_chart(data)
    #st.writew ("IRQ filtered", len(filtered_data))

    return filtered_data

def eloszlasgorbe (data):
    fig2, ax2 = plt.subplots()
    combined_output = data
    sns.histplot(combined_output, kde=True, bins=20, ax=ax2, color='steelblue')
    ax2.set_title("Válasz eloszlása (histogram + KDE)")
    ax2.set_xlabel("Válasz érték")
    ax2.set_ylabel("Gyakoriság / sűrűség")
    st.pyplot(fig2)


def filter_derivative_percentile(y, top_percent=1.0):
    y = pd.Series(y)
    dy = y.diff().abs()

    # Küszöb a derivált top X százalékához
    threshold = np.percentile(dy.dropna(), 100 - top_percent)

    # Csak azokat hagyjuk meg, amelyek a küszöb alatt vannak
    mask = (dy < threshold) | dy.isna()
    return y.where(mask)  # a többi NaN lesz


def remove_short_valid_sequences(y, min_length=5):
    y = pd.Series(y)
    is_valid = y.notna()

    # Az érvényes szakaszokat megszámozzuk (mint egy label)
    group = (is_valid != is_valid.shift()).cumsum()
    
    # Minden csoporthoz megnézzük, mennyi a hossza
    lengths = is_valid.groupby(group).transform('sum')

    # Ha érvényes (True) és rövidebb, mint minimum hossz: NaN lesz
    y[(is_valid) & (lengths < min_length)] = np.nan
    return y
def remove_low_variability_segments(y, window=10, std_thresh=1e-3):
    y = pd.Series(y)
    # Mozgó szórás
    rolling_std = y.rolling(window=window, center=True).std()
    
    # Alacsony varianciájú pontokat NaN-ra állítjuk
    return y.where(rolling_std > std_thresh)

def filter_high_acceleration(y, top_percent=1.0):
    y = pd.Series(y)

    # Első derivált (sebesség)
    dy = y.diff()

    # Második derivált (gyorsulás)
    ddy = dy.diff().abs()

    # Küszöb a gyorsulás top X százalékához
    threshold = np.percentile(ddy.dropna(), 100 - top_percent)

    # Elfogadjuk, ahol a gyorsulás alacsonyabb a küszöbnél
    mask = (ddy < threshold) | ddy.isna()
    return y.where(mask)

def rolling_std_filter_within_nan_segments(y, window=5, std_thresh=1e-2):
    y = pd.Series(y)
    result = y.copy()

    # Szakaszok beazonosítása: NaN → False, érték → True
    is_valid = y.notna()
    group_id = (is_valid != is_valid.shift()).cumsum()

    # Minden True szakasz (azaz nem-NaN szegmensek)
    for gid, segment in y.groupby(group_id):
        if segment.isna().all():
            continue  # ez egy NaN szakasz, kihagyjuk

        # Csak akkor szűrjük, ha a szakasz NaN-nal van körülvéve (előtte és utána)
        start = segment.index[0]
        end = segment.index[-1]

        is_bounded_by_nan = (
            (start > 0 and y.iloc[start - 1] != y.iloc[start - 1]) and
            (end < len(y) - 1 and y.iloc[end + 1] != y.iloc[end + 1])
        )

        if is_bounded_by_nan:
            # Görgetett szórás kiszámítása a szakaszon belül
            local = segment.copy()
            rolling_std = local.rolling(window=window, center=True).std()
            result.loc[segment.index] = local.where(rolling_std > std_thresh)

    return result


def prepare_measurements (Measurements):
    # XY preparation

    for col in ["FPOGX", "FPOGY", "BPOGX", "BPOGY"]:
        
        Measurements[col] = IQR(Measurements[col])
        Measurements[col] = filter_high_acceleration(Measurements[col], top_percent=1.0)
        Measurements[col] = filter_derivative_percentile(Measurements[col], top_percent=0.5)
        Measurements[col] = remove_short_valid_sequences (Measurements[col], min_length=10)
        #Measurements[col] = flt.gpr_fill (Measurements[col] )
        #Measurements[col] = flt.gpr_fill_local (Measurements[col] )
        #Measurements[col] = rolling_std_filter_within_nan_segments(Measurements[col], window=10, std_thresh=1e-4)

    Measurements["FPOGX"][Measurements["FPOGY"].isna()] = np.nan
    Measurements["FPOGY"][Measurements["FPOGX"].isna()] = np.nan
    Measurements["BPOGX"][Measurements["BPOGY"].isna()] = np.nan
    Measurements["BPOGY"][Measurements["BPOGX"].isna()] = np.nan
    
    for col in ["FPOGX", "FPOGY", "BPOGX", "BPOGY"]:
        #pass
        Measurements[col] = flt.gpr_fill_local (Measurements[col] )


    Measurements["displacement"] = np.sqrt (Measurements["FPOGX"].diff()**2 + Measurements["FPOGY"].diff()**2)
    #Measurements["displacement"] = IQR(Measurements["displacement"])
    #Measurements["displacement"] = filter_derivative_percentile(Measurements["displacement"], top_percent=0.5)
    #Measurements["displacement"] = filter_derivative_percentile(Measurements["displacement"], top_percent=0.5)
    #Measurements["displacement"] = remove_short_valid_sequences (Measurements["displacement"], min_length=5)
    Measurements["displacement"] = Measurements["displacement"].interpolate(method='linear', limit_direction='both')    
    
    # normalizálás
    Measurements["displacement"] = Measurements["displacement"] - Measurements["displacement"].min()+1e-6
    Measurements["displacement"] = Measurements["displacement"] / Measurements["displacement"].max()
    #Measurements["displacement"] = flt.gpr_fill_local (Measurements["displacement"] )
    # EYE energy 

    Measurements["x_energy"] = Measurements["FPOGX"]- Measurements["FPOGX"].mean()
    #st.line_chart(Measurements["x_energy"])
    Measurements["x_energy"] = Measurements["x_energy"].abs() 
    #st.line_chart(Measurements["x_energy"])
    Measurements["y_energy"] = Measurements["FPOGY"]- Measurements["FPOGY"].mean()
    Measurements["y_energy"] = Measurements["y_energy"].abs() 
    Measurements["total_eye_energy"] = Measurements["x_energy"] + Measurements["y_energy"]
    Measurements["total_eye_energy"] = Measurements["total_eye_energy"].interpolate(method='linear', limit_direction='both')    
   
    #st.line_chart(Measurements["x_energy"])
    
    
    
    # pupill preparation
    Measurements["RPD"][Measurements["RPV"]!=1] = np.nan
    Measurements["LPD"][Measurements["LPV"]!=1] = np.nan
    l_avg = Measurements["LPD"].mean()
    r_avg = Measurements["RPD"].mean()
    l_min = Measurements["LPD"].min()
    r_min = Measurements["RPD"].min()
    l_max = Measurements["LPD"].max()
    r_max = Measurements["RPD"].max()

    # pupilla változási sebesség átlag szoras. 
    lpd_speed_mean = Measurements["LPD"].diff().mean()
    lsp_speed_stdev = Measurements["LPD"].diff().std()
    Measurements["LPD_Change"] = Measurements["LPD"].diff()
    Measurements["LPD_Change"][Measurements["LPD_Change"] > (lpd_speed_mean + lsp_speed_stdev)] = np.nan
    Measurements["LPD_Change"][Measurements["LPD_Change"] < (lpd_speed_mean - lsp_speed_stdev)] = np.nan
    Measurements["LPD"][Measurements["LPD_Change"].shift(-1).isna()] = np.nan
    Measurements["LPD"][(Measurements["LPD"].shift(1).isna()) | (Measurements["LPD"].shift(-1).isna())] = np.nan
    Measurements["RPD_Change"] = Measurements["RPD"].diff()
    rpd_speed_mean = Measurements["RPD"].diff().mean()
    rsp_speed_stdev = Measurements["RPD"].diff().std()
    Measurements["RPD_Change"][Measurements["RPD_Change"] > (rpd_speed_mean + rsp_speed_stdev)] = np.nan
    Measurements["RPD_Change"][Measurements["RPD_Change"] < (rpd_speed_mean - rsp_speed_stdev)] = np.nan
    Measurements["RPD"][Measurements["RPD_Change"].shift(-1).isna()] = np.nan
    Measurements["RPD"][(Measurements["RPD"].shift(1).isna()) | (Measurements["RPD"].shift(-1).isna())] = np.nan

    Measurements["PUPIL_L_NORM"] = (Measurements["LPD"] - l_min) / (l_max - l_min)
    Measurements["PUPIL_R_NORM"] = (Measurements["RPD"] - r_min) / (r_max - r_min)

    Measurements["PUPIL_L_NORM"] = Measurements["PUPIL_L_NORM"].interpolate(method='linear', limit_direction='both')
    Measurements["PUPIL_R_NORM"] = Measurements["PUPIL_R_NORM"].interpolate(method='linear', limit_direction='both')
    Measurements["PUPIL_NORM"] = (Measurements["PUPIL_L_NORM"] + Measurements["PUPIL_R_NORM"]) / 2
    

    
    return Measurements


def do_multifractal (Measurements,  dir ="", filename = "tmp", kernel_size=1, wavelet_type = "ricker"):
    # Placeholder for the multifractal analysis function
    # Replace with actual implementation
    plot_graphs = False
    #st.checkbox("Plot graphs", value=plot_graphs)
    #Measurements = prepare_measurements (Measurements)
    #st.write (f"Measurements: {len(Measurements)}")

    metrics_txt = ""
    wtmm_metrics,chhb_metrics, mfdfa_metrics = {}, {}, {}
    if st.session_state.get("metrics") == "Displacement":
        metrics_txt = "Displacement"
        data = Measurements["displacement"].to_numpy()
    elif st.session_state.get("metrics") == "Tekintet X":
        metrics_txt = "Tekintet X"
        data = Measurements["FPOGX"].to_numpy()
    elif st.session_state.get("metrics") == "Pupilla":
        metrics_txt = "Pupilla"
        data = Measurements["PUPIL_NORM"].to_numpy()
    elif st.session_state.get("metrics") == "Total Eye Energy":
        metrics_txt = "Eye Energy"
        data = Measurements["total_eye_energy"].to_numpy()


    #"Teszt_Gauss", "Teszt_cantor_measure", "Teszt_multiplicative_cascade", "Teszt_Fractional_Brownian_Motion", "Teszt_generate_multifractal_signal"
    elif st.session_state.get("metrics") == "Teszt_Gauss":
        data = ts.generate_gauss (length=len(Measurements))
    elif st.session_state.get("metrics") == "Teszt_cantor_measure":
        data = ts.cantor_measure(length=len(Measurements), N=10)
    elif st.session_state.get("metrics") == "Teszt_multiplicative_cascade":
        data = ts.multiplicative_cascade(target_length=len(Measurements), weights=(0.6, 0.4), seed=None)
    elif st.session_state.get("metrics") == "Teszt_Fractional_Brownian_Motion":
        data = ts.Fractional_Brownian_Motion (length=len(Measurements), H=0.7)
    elif st.session_state.get("metrics") == "Teszt_generate_multifractal_signal":
        data = ts.generate_multifractal_signal(length=len(Measurements), H=0.7)
    elif st.session_state.get("metrics") == "Teszt_generate_binomial_measure":
        data = ts.generate_binomial_measure(length=len(Measurements))
    
    filename = filename+"_"+ metrics_txt

                        #Modes:
                    #- "log":     logaritmikus skálák: base^exp
                    #- "linear":  lineáris skálák: exp ∈ [min_exp, max_exp]
                    #- "dyadic":  skálák: 2^n (ha num_scales is None)
    scales, window_sizes, scale_exponents = generate_multifractal_scales(len(Measurements), num_scales= st.session_state["nmbr_scales"], base=2, mode= st.session_state["scale_mode"])
    q_values = generate_log_q_values (0.1, st.session_state["minmax_q_values"], st.session_state["nmbr_q_values"])
    q_values = np.array(q_values)
    scales= scales.astype(int)
    

    signal = data.copy()
    display_coluns = 0
    for todo in ["do_chhb", "do_wtmm", "do_mfdfa"]:
        if st.session_state[todo] :
            display_coluns += 1

    columns = st.columns(display_coluns)
    display_coluns = 0
    
    if st.session_state["do_chhb"]: 
        with columns[display_coluns] :
            display_coluns += 1

            data = chhabrajensen.prepare_signal(data, min(window_sizes), smoothing=0.1)
            st.write ("CHHB Multifraktál spektrum")
            r2_values = chhabrajensen.compute_zq_r2_vs_q(data, q_values, window_sizes)
            plt = chhabrajensen.plot_r2_vs_q( q_values, r2_values, title="R²(q) görbe", )
            plt.savefig(os.path.join(dir, f"{filename}_CHHB_r2_vs_q.png"), dpi=300)
            st.pyplot(plt)
            plt.close()
            
            q_values = np.array(q_values)

            alpha, falpha, Dq, Rsqr_falpha = chhabrajensen.chhabra_jensen_multifractal (data, q_values=q_values, window_sizes=window_sizes)
            fig = chhabrajensen.plot_dq_curve(q_values, Dq, title="D(q) spektrum") 
            fig.savefig(os.path.join(dir, f"{filename}_CHHB_Dq.png"), dpi=300)
            st.pyplot(fig)
            #fig.close()

            r2_threshold = 0.95
            exclude_extremes = True
            #st.write (len(alpha), "Before filtering" )
            result = chhabrajensen.filter_multifractal_spectrum(alpha, falpha, Rsqr_falpha=Rsqr_falpha, q_vals=q_values,
                                        r2_threshold=r2_threshold, exclude_edges=2, keep_peak=True,
                                        compute_width=True )
        
            #st.write (len(result["alpha"]), "After filtering" )
            try:
                fig, chhb_metrics = plot_f_alpha_spectrum_with_metrics ( result["alpha"], result["falpha"], q_values=q_values, tau_q=None, title="CHHB Multifraktál spektrum (f(α))", save_path=os.path.join(dir, f"{filename}__CHHB_spectrum.png"), dpi=300)
                st.pyplot(fig)
            except Exception as e:
                st.write (f"Error in plot_f_alpha_spectrum_with_metrics: {e}")
                st.write (f"Alpha: {result['alpha']}")
                st.write (f"F_alpha: {result['falpha']}")
                st.write (f"Q_values: {q_values}")
                st.write (f"Tau_q: None")
                st.write (f"Title: CHHB Multifraktál spektrum (f(α))")
                st.write (f"Save path: {os.path.join(dir, f'{filename}__CHHB_spectrum.png')}")
            #fig.close()
            chhb_segments, chhb_avalues = {}, {}
        if st.session_state["do_10sec"]:
            sampling_rate = 1.0 / (Measurements["TIME"].diff().median())
            window_sec = 10.0
            chhb_segments, chhb_avalues = chhabrajensen.analyze_multifractal_over_time (data, sampling_rate, window_sec, window_sizes=window_sizes, q_values=q_values, r2_threshold=0.95)
            #st.write (f"CHHB α-értékek időfüggése: {chhb_segments}, {chhb_avalues}")    
            plt.figure(figsize=(10, 4))
            plt.plot(chhb_segments, chhb_avalues, marker='o', linestyle='-')
            plt.xlabel("Idő (s)")
            plt.ylabel("CHHB α-értékek")
            plt.title("CHHB α-értékek időfüggése")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(dir, f"{filename}_CHHB_avalues.png"), dpi=300)
            st.pyplot (plt)
            plt.close()
            i=0
            for segment in chhb_segments:
                i+=1
                chhb_metrics["alpha_segment("+str(i)+")"] = chhb_avalues[i-1]


    if st.session_state["do_wtmm"]: 
        with columns[display_coluns] :
            display_coluns += 1
            st.write ("WTMM Multifraktál spektrum")
            #st.write (f"Jel Hossza: {len(signal)}")
            signal = wtmm.prepare_signal(signal, min(window_sizes), smoothing=0.1)  

            # alpha, f_alpha, tau_q, Z_q_s, coef, maxima, log_scales_trimmed, r2
            wtmm_alpha, wtmm_f_alpha, wtmm_tau_q, wtmm_Z_q_s, wtmm_coef, wtmm_maxima, log_scales, r2 = wtmm.wtmm_multifractal_spectrum (signal.copy(), scales=scales, q_values=q_values, kernel_size=kernel_size, wavelet_type = wavelet_type)
            
            
            r2_array = r2
            plot= wtmm.plot_wtmm_r2_vs_q(q_values, r2_array)
            plot.savefig(os.path.join(dir, f"{filename}_WTMM_r2_vs_q.png"), dpi=300)
            st.pyplot(plot)
            plot.close()

            dq, plot = wtmm.plot_wtmm_dq_spectrum (q_values, wtmm_tau_q, title="WTMM D(q) spektrum")
            plot.savefig(os.path.join(dir, f"{filename}_WTMM_Dq.png"), dpi=300)
            st.pyplot(plot)
            plot.close()
            
            #  alpha, f_alpha, tau_q, Z_q_s, coef, maxima

            if wtmm_alpha is not None and wtmm_f_alpha is not None:
                try:
                    fig, wtmm_metrics = plot_f_alpha_spectrum_with_metrics (wtmm_alpha, wtmm_f_alpha, q_values=q_values, tau_q=wtmm_tau_q, Dq=None, title="WTMM Multifraktál spektrum (f(α))", save_path=os.path.join(dir, f"{filename}_WTMM_spectrum.png"), dpi=300)
                    st.pyplot(fig)
                except Exception as e: 
                    st.write (f"Error in plot_f_alpha_spectrum_with_metrics: {e}")
                    st.write (f"Alpha: {wtmm_alpha}")
                    st.write (f"F_alpha: {wtmm_f_alpha}")
                    st.write (f"Q_values: {q_values}")
                    st.write (f"Tau_q: {wtmm_tau_q}")
                    st.write (f"Title: WTMM Multifraktál spektrum (f(α))")
                    st.write (f"Save path: {os.path.join(dir, f'{filename}_WTMM_spectrum.png')}")
                # signal, scales, q_values, alpha, f_alpha, tau_q, Z_q_s, coef, maxima, output_dir="output", filename_prefix="wtmm_result", dpi=300
                fig, axis = wtmm.plot_wtmm_all_outputs(signal, scales, q_values, wtmm_alpha, wtmm_f_alpha, wtmm_tau_q, wtmm_Z_q_s, wtmm_coef, wtmm_maxima, output_dir=dir, filename_prefix=f"{filename}_WTMM_result_", dpi=300)
   
        
        wtmm_segments, wtmm_avalues = {}, {}
        if st.session_state["do_10sec"]:
            sampling_rate = 1.0 / (Measurements["TIME"].diff().median())
            window_sec = 10.0
            #st.write (f"Jel Hossza: {len(signal)}") 
            wtmm_segments, wtmm_avalues = wtmm.analyze_multifractal_over_time(signal.copy(), sampling_rate, window_sec, scales, q_values, keep_ratio=0.3)
            # plot segments and a values
            import matplotlib.pyplot as plt 
            plt.figure(figsize=(10, 4))
            plt.plot(wtmm_segments, wtmm_avalues, marker='o', linestyle='-')
            plt.xlabel("Idő (s)")
            plt.ylabel("wtmm α-értékek")
            plt.title("wtmm α-értékek időfüggése")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(dir, f"{filename}_WTMM_avalues.png"), dpi=300)
            st.pyplot (plt)
            plt.close()
            #st.write (wtmm_segments)
            i=0
            for segment in wtmm_segments:
                i+=1
                wtmm_metrics["alpha_segment("+str(i)+")"] = wtmm_avalues[i-1]
        
        if wtmm_alpha is not None and wtmm_f_alpha is not None:    
            fig, axis = wtmm.plot_wtmm_results(signal, scales, q_values, wtmm_alpha, wtmm_f_alpha, wtmm_tau_q, wtmm_Z_q_s, wtmm_coef, wtmm_maxima, dpi=300)
            st.pyplot(fig)
            fig.savefig(os.path.join(dir, f"{filename}_WTMM_Display_results.png"), dpi=300)
        
    if st.session_state["do_mfdfa"]: 
        with columns[display_coluns] :
            display_coluns += 1
            st.write ("MFDFA Multifraktál spektrum")
            #st.write (f"Jel Hossza: {len(signal)}")
            signal = mfdfa.prepare_signal(signal, min(window_sizes), smoothing=0.1)  

            # alpha, f_alpha, tau_q, Z_q_s, coef, maxima, log_scales_trimmed, r2
            mfdfa_alpha, mfdfa_f_alpha, mfdfa_tau_q, mfdfa_Z_q_s, mfdfa_coef, mfdfa_maxima, log_scales2, r2 = mfdfa.mfdfa_multifractal_spectrum (signal.copy(), scales=scales, q_values=q_values)
            
            
            r2_array = r2
            plot= mfdfa.plot_mfdfa_r2_vs_q(q_values, r2_array)
            plot.savefig(os.path.join(dir, f"{filename}_mfdfa_r2_vs_q.png"), dpi=300)
            st.pyplot(plot)
            plot.close()

            dq, plot = mfdfa.plot_mfdfa_dq_spectrum (q_values, mfdfa_tau_q, title="mfdfa D(q) spektrum")
            plot.savefig(os.path.join(dir, f"{filename}_mfdfa_Dq.png"), dpi=300)
            st.pyplot(plot)
            plot.close()
            
            #  alpha, f_alpha, tau_q, Z_q_s, coef, maxima

            if mfdfa_alpha is not None and mfdfa_f_alpha is not None:
                try:
                    fig, mfdfa_metrics = plot_f_alpha_spectrum_with_metrics (mfdfa_alpha, mfdfa_f_alpha, q_values=q_values, tau_q=mfdfa_tau_q, Dq=None, title="mfdfa Multifraktál spektrum (f(α))", save_path=os.path.join(dir, f"{filename}_mfdfa_spectrum.png"), dpi=300)
                    st.pyplot(fig)
                except Exception as e: 
                    st.write (f"Error in plot_f_alpha_spectrum_with_metrics: {e}")
                    st.write (f"Alpha: {mfdfa_alpha}")
                    st.write (f"F_alpha: {mfdfa_f_alpha}")
                    st.write (f"Q_values: {q_values}")
                    st.write (f"Tau_q: {mfdfa_tau_q}")
                    st.write (f"Title: mfdfa Multifraktál spektrum (f(α))")
                    st.write (f"Save path: {os.path.join(dir, f'{filename}_mfdfa_spectrum.png')}")
                # signal, scales, q_values, alpha, f_alpha, tau_q, Z_q_s, coef, maxima, output_dir="output", filename_prefix="mfdfa_result", dpi=300
                fig, axis = mfdfa.plot_mfdfa_all_outputs(signal, scales, q_values, mfdfa_alpha, mfdfa_f_alpha, mfdfa_tau_q, mfdfa_Z_q_s, mfdfa_coef, mfdfa_maxima, output_dir=dir, filename_prefix=f"{filename}_mfdfa_result_", dpi=300)

            mfdfa_segments, mfdfa_avalues = {}, {}
            if st.session_state["do_10sec"]:
                sampling_rate = 1.0 / (Measurements["TIME"].diff().median())
                window_sec = 10.0
                #st.write (f"Jel Hossza: {len(signal)}") 
                mfdfa_segments, mfdfa_avalues = mfdfa.analyze_multifractal_over_time(signal.copy(), sampling_rate, window_sec, scales, q_values, keep_ratio=0.3)
                # plot segments and a values
                import matplotlib.pyplot as plt 
                plt.figure(figsize=(10, 4))
                plt.plot(mfdfa_segments, mfdfa_avalues, marker='o', linestyle='-')
                plt.xlabel("Idő (s)")
                plt.ylabel("mfdfa α-értékek")
                plt.title("mfdfa α-értékek időfüggése")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{filename}_mfdfa_avalues.png"), dpi=300)
                st.pyplot (plt)
                plt.close()
                #st.write (mfdfa_segments)
                i=0
                for segment in mfdfa_segments:
                    i+=1
                    mfdfa_metrics["alpha_segment("+str(i)+")"] = mfdfa_avalues[i-1]
            
        if mfdfa_alpha is not None and mfdfa_f_alpha is not None:    
            fig, axis = mfdfa.plot_mfdfa_results(signal, scales, q_values, mfdfa_alpha, mfdfa_f_alpha, mfdfa_tau_q, mfdfa_Z_q_s, mfdfa_coef, mfdfa_maxima, dpi=300)
            st.pyplot(fig)
            fig.savefig(os.path.join(dir, f"{filename}_mfdfa_Display_results.png"), dpi=300)
            pass




        



    #st.write (f"Scales: {scales}")

    return wtmm_metrics, chhb_metrics, mfdfa_metrics, metrics_txt

def create_graps_for_screens (data):


    fig = plt.figure(figsize=(10, 6))
    return  []

def flattening_table (screens):
    flat_data = {}
    for idx, row in screens.iterrows():
        for col in screens.columns:
            flat_data[f"{col}_{idx}"] = row[col]
    flat_df = pd.DataFrame([flat_data])
    return flat_df


def get_screens(datas):
    result_df = pd.DataFrame()
    if len(datas) != 0:
        mediaids = datas["MediaId"].unique()
        st.write (f"MediaIds: {mediaids}")
        result_df = pd.DataFrame()
        for mediaid in mediaids:
            data = datas[datas["MediaId"] == mediaid]
            screenname = "NA"
            if mediaid==0 : screenname = "H"
            elif mediaid==1 : screenname = "A"

            data = datas[datas["MediaId"] == mediaid]
            screenids = data["ScreenId"].unique()
            screenids = screenids[~pd.isna(screenids)].astype(int)
            screens = pd.DataFrame({
                "ScreenId": screenids
            })
            screens["ScreenId"] = screens["ScreenId"]
            screens = screens.dropna(axis=0, how='all')
            # Új üres oszlopok létrehozása

            screens["Name"] = screenname+screens["ScreenId"].astype(str)
            screens["MediaId"] = mediaid
            screens["start"] = None
            screens["end"] = None
            screens["duration"] = None
            screens["avg_eye_speed"] = None
            screens["avg_fixation_time"] = None
            screens["nbr_of_fixations"] = None
            screens["fix/sec"] = None
            screens["avg_left_pupil"] = None
            screens["std_left_pupil"] = None
            screens["avg_right_pupil"] = None
            screens["std_right_pupil"] = None
            
            
            for idx, row in screens.iterrows():
                screen_id = row["ScreenId"]
                #"FPOGD":"The duration of the fixation POG in seconds.",
                #"FPOGID":"The fixation POG ID number.",
                #"FPOGV":"The FPOG valid flag is 1 for valid and 0 for not valid.",
                #"RPD":"The diameter of the right eye pupil in pixels.",
                #"RPS":"The scale factor of the right eye pupil, normalized to 1 at the head depth at calibration.",
                #"RPV":"The right pupil valid flag is 1 for valid and 0 for not valid.",
                
                start_time = data.loc[data["ScreenId"] == screen_id, "TIME"].min()
                end_time = data.loc[data["ScreenId"] == screen_id, "TIME"].max()
                avg_eye_speed = data.loc[data["ScreenId"] == screen_id, "eye_speed"].mean()
                avg_fixation_time = data.loc[(data["ScreenId"] == screen_id)&(data["FPOGV"] == 1 )  , "FPOGD"].mean()
                # count the number of fixations
                nbr_of_fixations = data.loc[(data["ScreenId"] == screen_id)&(data["FPOGV"] == 1 )  , "FPOGID"].nunique()
                fix_sec = data.loc[(data["ScreenId"] == screen_id)&(data["FPOGV"] == 1 )  , "FPOGID"].count() / (end_time - start_time)
                avg_left_pupil = data.loc[(data["ScreenId"] == screen_id)&(data["LPMMV"] == 1 )  , "LPMM"].mean()
                avg_right_pupil = data.loc[(data["ScreenId"] == screen_id)&(data["RPMMV"] == 1 )  , "RPMM"].mean()
                std_left_pupil = data.loc[(data["ScreenId"] == screen_id)&(data["LPMMV"] == 1 )  , "LPMM"].std()
                std_right_pupil = data.loc[(data["ScreenId"] == screen_id)&(data["RPMMV"] == 1 )  , "RPMM"].std()
                
                
                duration = end_time - start_time
                
                screens.at[idx, "start"] = start_time
                screens.at[idx, "end"] = end_time
                screens.at[idx, "duration"] = duration
                screens.at[idx, "avg_eye_speed"] = avg_eye_speed
                screens.at[idx, "avg_fixation_time"] = avg_fixation_time
                screens.at[idx, "nbr_of_fixations"] = nbr_of_fixations
                screens.at[idx, "fix/sec"] = fix_sec
                screens.at[idx, "avg_left_pupil"] = avg_left_pupil
                screens.at[idx, "std_left_pupil"] = std_left_pupil
                screens.at[idx, "avg_right_pupil"] = avg_right_pupil
                screens.at[idx, "std_right_pupil"] = std_right_pupil
                
            result_df = pd.concat([result_df, screens], ignore_index=True)
        #st.write(screens)
    return result_df


def generate_multifractal_scales(
    signal_length,
    min_exp=2,
    max_exp=None,
    num_scales=None,
    base=2,
    min_repeats=8,
    mode="log",  # "log", "linear", "dyadic"
    show_debug=False,
):
    """
    Skálák generálása multifraktál elemzéshez (WTMM / Chhabra–Jensen)

    Modes:
        - "log":     logaritmikus skálák: base^exp
        - "linear":  lineáris skálák: exp ∈ [min_exp, max_exp]
        - "dyadic":  skálák: 2^n (ha num_scales is None)

    Csak olyan skálák maradnak meg, amik legalább min_repeats-szer beleférnek a jelbe.

    Returns:
        scales, window_sizes, scale_exponents (log2)
    """

    if max_exp is None:
        max_exp = int(np.floor(np.log2(signal_length // min_repeats)))

    if show_debug:
        print(f"[DEBUG] max_exp = {max_exp}")

    if mode == "dyadic" or (mode == "log" and num_scales is None):
        # Duplázódó skálák: 2^n
        exponents = np.arange(min_exp, max_exp + 1)
        scale_vals = base ** exponents

    elif mode == "log":
        # Logaritmikus skálák: base^exp
        exponents = np.linspace(min_exp, max_exp, num=num_scales)
        scale_vals = base ** exponents
        scale_vals = np.round(scale_vals).astype(int)
        scale_vals = np.unique(scale_vals)

    elif mode == "linear":
        # Lineáris skálák: min_exp → max_exp
        scale_vals = np.linspace(base**min_exp, base**max_exp, num=num_scales)
        scale_vals = np.round(scale_vals).astype(int)
        scale_vals = np.unique(scale_vals)

    else:
        raise ValueError(f"Ismeretlen mode: {mode}. Használható: 'log', 'linear', 'dyadic'")

    # Csak olyan skálák, amik min_repeats-szer beleférnek az idősorba
    scale_vals = scale_vals[scale_vals * min_repeats <= signal_length]

    # Kimenetek
    scales = scale_vals.astype(float)
    window_sizes = scale_vals.copy()
    scale_exponents = np.log2(scales)
    #st.write (f"Scales: {scales}")
    if show_debug:
        print(f"[DEBUG] scales: {scales}")
        print(f"[DEBUG] window_sizes: {window_sizes}")
        print(f"[DEBUG] scale_exponents: {scale_exponents}")

    return scales, window_sizes, scale_exponents


def generate_log_q_values_no_q1(
    q_min_log=0.1,
    q_max=10,
    num_below=20,
    num_above=20
):
    """
    Logaritmikusan elosztott q értékeket generál, úgy hogy q=1.0 SOHA ne legyen benne.

    Parameters:
        q_min_log : float
            Legkisebb pozitív abs(q), pl. 0.1
        q_max : float
            Legnagyobb abs(q), pl. 10
        num_below : int
            q < 1 tartomány logaritmikus felosztása (0.1 – 0.99)
        num_above : int
            q > 1 tartomány logaritmikus felosztása (1.01 – 10)

    Returns:
        np.ndarray:
            Szimmetrikusan logaritmikusan elosztott q értékek, q=1 nélkül
    """
    # q < 1 (pl. 0.1 – 0.99)
    q_below_1 = np.logspace(np.log10(q_min_log), np.log10(0.99), num=num_below, endpoint=True)

    # q > 1 (pl. 1.01 – 10)
    q_above_1 = np.logspace(np.log10(1.01), np.log10(q_max), num=num_above, endpoint=True)

    # Összefűzés: negatív oldal és pozitív oldal
    q_vals = np.concatenate((-q_above_1[::-1], -q_below_1[::-1], q_below_1, q_above_1))
    return q_vals


def generate_log_q_values(
    q_min_log=0.1,
    q_max=10,
    num_per_side=20,
    exclude_exact=(1.0,),
):
    """
    Generál logaritmikusan sűrített q értékeket -q_max...-q_min_log, q_min_log...q_max között.
    q = 1.0 kizárható (alapértelmezetten ki is van zárva).

    Parameters:
        q_min_log : float
            A logaritmikus skála legkisebb abszolút q értéke (> 0), pl. 0.1
        q_max : float
            A legnagyobb q abszolút érték
        num_per_side : int
            Hány q értéket generálunk az egyik oldalon (pozitív vagy negatív)
        exclude_exact : tuple
            Pontosan kizárandó értékek, pl. (1.0,)

    Returns:
        np.ndarray:
            Stabil, logaritmikusan elosztott q értékek
    """
    # Log skála: pozitív oldalon
    pos_q = np.logspace(np.log10(q_min_log), np.log10(q_max), num=num_per_side)
    neg_q = -pos_q[::-1]  # negatív tükörkép
    raw_q = np.concatenate((neg_q, pos_q))

    # Kizárás: q == 1.0 (vagy amit megadsz)
    mask = np.ones_like(raw_q, dtype=bool)
    for val in exclude_exact:
        mask &= np.abs(raw_q - val) > 1e-8

    return raw_q[mask]


def generate_safe_q_values(
    q_min=-5,
    q_max=5,
    num=101,
    exclude_exact=(1.0,),
    exclude_range=None
):
    """
    Generál lineárisan elosztott q értékeket, kizárva specifikus értékeket és opcionálisan egy intervallumot is.

    Parameters:
        q_min : float
            Minimum q érték
        q_max : float
            Maximum q érték
        num : int
            A generált értékek száma (a szűrés előtti)
        exclude_exact : tuple of float
            Pontosan kizárandó q értékek (pl. (1.0,))
        exclude_range : tuple of float or None
            Opcionálisan kizárt tartomány: (alsó, felső), pl. (0.0, 1.0)

    Returns:
        np.ndarray:
            Stabil q értékek
    """
    raw_q = np.linspace(q_min, q_max, num)
    mask = np.ones_like(raw_q, dtype=bool)

    # Pontos kizárás (lebegőpontos toleranciával)
    for val in exclude_exact:
        mask &= np.abs(raw_q - val) > 1e-8

    # Tartomány kizárás, ha meg van adva
    if exclude_range is not None:
        lower, upper = exclude_range
        mask &= ~((raw_q > lower) & (raw_q < upper))

    return raw_q[mask]

def do_prepare_measurement (measurement_file,factor, apply_filter, prepared_file ):
    df = pd.read_parquet(measurement_file)
    st.markdown("---")
    st.subheader (f"Preparing  {measurement_file}")
    mediaids = df["MediaId"].unique()
    df_prepared = pd.DataFrame()
    for mediaid in mediaids:
        try: 
            st.write (f"Opening {mediaid}")
            data = df[df["MediaId"] == mediaid]
            st.write("Preparing data ...", len(data))
            data = prepare_measurements (data)
            st.write("Resampling ... ", len(data))
            data = resample_and_filter_measurements(data, time_col='TIME', factor=factor, apply_filter=apply_filter)
            df_prepared = pd.concat([df_prepared, data], ignore_index=True)
        except:
            st.warning ("Nem sikerült a preparáció 943.sor")
    df_prepared.to_parquet(prepared_file, index=False)
    st.write (f"Prepared data saved.{prepared_file}")


def main():
    st.title("Fractal Analysis")
    with st.sidebar:
        col1, col2 = st.columns(2)  
        files = os.listdir("import")
        teams = [d for d in files if os.path.isdir(os.path.join("import", d))]
        with col1:
            selected_team= st.selectbox("Select team", teams, key="team")
            if selected_team:
                samples =  os.listdir(os.path.join("import", selected_team))
                samples.sort()
                # remove files in list, which is starting with .
                samples = [s for s in samples if not s.startswith('.')]
                with col2:
                    selected_sample = st.selectbox("Select sample", samples)
                file_path = os.path.join("import", selected_team, selected_sample)
    # if file /result/Measurements.parquet 
    measurement_file = os.path.join(file_path, "result", "Measurements.parquet")
    prepared_file = os.path.join(file_path, "result", "Measurements_prepared.parquet")
    subject_id = selected_team + "_" + selected_sample 
    st.header ("Subject "+subject_id)

    if os.path.exists(measurement_file):
        factor = st.sidebar.number_input("Resample factor", min_value=1, max_value=16, value=4) 
        apply_filter = st.sidebar.checkbox("Apply lowpass filter", value=False)
        plot_graphs = st.sidebar.checkbox("Plot graphs", value=False)
        
        if st.sidebar.button("Preparation", use_container_width=True):
            if selected_sample:
                do_prepare_measurement (measurement_file,factor, apply_filter, prepared_file )
        if st.sidebar.button ("PREPARE_ALL", use_container_width=True):
                
                for selected_sample in samples:
                    try:
                        file_path = os.path.join("import", selected_team, selected_sample)
                        measurement_file = os.path.join(file_path, "result", "Measurements.parquet")
                        prepared_file = os.path.join(file_path, "result", "Measurements_prepared.parquet")
                        do_prepare_measurement (measurement_file,factor, apply_filter, prepared_file )
                    except Exception as e: 
                        st.warning (f"Error metrics preparation: at  {file_path} {e}")
                       
                
        if os.path.exists(prepared_file):
            df = pd.read_parquet(prepared_file)
            screens = get_screens (df)
            mediaids = df["MediaId"].unique()
            for mediaid in mediaids:
                data = df[df["MediaId"] == mediaid]
                
                if plot_graphs:           
                    fig = px.line(data, x="TIME", y=["FPOGX", "FPOGY", "displacement", "total_eye_energy"], title="Prepared Eye Movement")
                    fig.update_traces(mode='lines')
                    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Displacement (px)")
                    st.plotly_chart(fig, use_container_width=True)
            
            
            with st.sidebar.form(key="Multifractal_Analysis"):
                metrics = st.selectbox ("Mit vizsgáljunk",options=["Displacement", "Tekintet X", "Pupilla", "Total Eye Energy", "Teszt_Gauss", "Teszt_cantor_measure", "Teszt_multiplicative_cascade", "Teszt_Fractional_Brownian_Motion", "Teszt_generate_multifractal_signal", "Teszt_generate_binomial_measure" ] , key="metrics")
                cut_60s = st.checkbox("Csak az első 60s vizsgáljuk? ", value=True, key="cut_60s")
                do_10sec = st.checkbox("10 sec ablakokban vizsgáljuk? ", value=False, key="do_10sec")
                
                
                ccol1,ccol2,= st.columns(2)
                with ccol1: 
                    st.number_input("q_values", min_value=1, max_value=100, value=5, key="minmax_q_values") 
                with ccol2: 
                    st.number_input("q_resolution", min_value=1, max_value=100, value=51, key="nmbr_q_values")
                with ccol1: 
                    st.number_input("nmbr of scales", min_value=1, max_value=300, value=100, key="nmbr_scales")
                with ccol2: 
                    scale_mode = st.selectbox ("Scaling Mode", options=["log", "linear", "dyadic"], key="scale_mode")
                
                st.write ("WTMM")
                #st.markdown ("---")
                ccol3,ccol4,= st.columns(2)
               
                with ccol3:
                    wavelet_type = st.selectbox("wavelet", options=["ricker", "morlet"], index=0, key="wavelet type")
                with ccol4:
                    st.number_input("Kernel Size", min_value=0.0001, max_value=5.0, value=1.0, key="kernel_size")
                do_WTTM = st.checkbox("Do WTMM", value=True, key="do_wtmm")
                
            
                
                
                do_CHHB = st.checkbox("Do Chhabra-Jensen", value=True, key="do_chhb")
                do_mfdfa= st.checkbox("Do MFDFA", value=True, key="do_mfdfa")
               
                screen_names = screens["Name"].unique()
                do_on_screens = st.multiselect("Do on screens", screen_names, default=screen_names)
                #if st.checkbox("Do on all screens", value=True, key="do_on_all_screens"):
                #    do_on_screens = screen_names
                screens = screens[screens["Name"].isin(do_on_screens)]
                do_fractal = st.form_submit_button("Do Multifractal analysis", use_container_width=True)
            
            do_fractal_all = st.sidebar.button(f"Do Multifractal analysis \n FOR ALL SAMPLES")
            
            
            
            if do_fractal:

                #st.write(subject_id)
                mediaids = screens["MediaId"].unique()
                directory_path = os.path.join(file_path, "result", "multifractal")
                


                for idx, row in screens.iterrows():
                #for screenid in screenids:
                    screenid = row["ScreenId"]
                    mediaid = row["MediaId"]
                    task_name = row["Name"]
                    st.markdown ("---")
                    texti = f"MediaId: {mediaid} ScreenId: {screenid}", "task_name: ", task_name
                    st.subheader (texti)
                    screen_data = df[(df["ScreenId"] == screenid ) & (df["MediaId"] == mediaid)]
                    if len (screen_data) < 1000:
                        st.warning(f"Screen data is too short for multifractal analysis: {len(screen_data)}")
                        continue
                    start_time = screen_data["TIME"].min()
                    max_time = screen_data["TIME"].max()

                    first60s = "full_length"
                    if cut_60s:
                        #st.write ("Cutting 60s")
                        end_time = start_time + 60
                        # cut firs 1 sec
                        if end_time > max_time: 
                            end_time = max_time
                        screen_data = screen_data[screen_data["TIME"] >= start_time+1]
                        screen_data = screen_data[screen_data["TIME"] <= end_time+1]
                        first60s = "first60sec"
                    st.write ("Cutting ", first60s, "Start:", start_time, " End:", end_time, "Len:", end_time-start_time)  
                        
                    
                    if mediaid == 1: a_or_h = "A"
                    else: a_or_h = "H"
                    filename = f"{task_name}_{first60s}"
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)

                #    # Perform multifractal analysis
                    #screen_data = prepare_for_multifractal(screen_data)
                    #Modes:
                    #- "log":     logaritmikus skálák: base^exp
                    #- "linear":  lineáris skálák: exp ∈ [min_exp, max_exp]
                    #- "dyadic":  skálák: 2^n (ha num_scales is None)
                    scales, window_sizes, scale_exponents = generate_multifractal_scales(len(screen_data), num_scales= st.session_state["nmbr_scales"], base=2, mode= st.session_state["scale_mode"])
    
                    if len (screen_data) > (window_sizes.max() *4): 
                        wtmm_metrics, chhb_metrics, mfdfa_metrics, metrics_txt = do_multifractal(screen_data, dir= directory_path, filename=filename, kernel_size= st.session_state["kernel_size"], wavelet_type= wavelet_type) 
                    else:
                        st.warning(f"Screen data is too short for multifractal analysis: {len(screen_data)}")
                        continue
                    Measurement_id = filename
                    
                    for metrics in wtmm_metrics: 
                        screens.at[idx, metrics_txt+"_WTMM_"+metrics] = wtmm_metrics[metrics]
                    for metrics in chhb_metrics: 
                        screens.at[idx, metrics_txt+"_CHHB_"+metrics] = chhb_metrics[metrics]
                    for metrics in mfdfa_metrics: 
                        screens.at[idx, metrics_txt+"_MFDFA_"+metrics] = mfdfa_metrics[metrics]
                    

                    #return
                # save screens to file 
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                #screens = flattening_table (screens)
                screens.to_parquet(os.path.join(directory_path, f"fractal_results.parquet"), index=False)
                screens.to_csv(os.path.join(directory_path, f"fractal_results.csv"), index=False)

                st.write ("Screens: ", screens)
            #flattened = flattening_table (screens)
            #st.write (flattened)


            
    else:
        st.warning("No measurement file found. Please check the selected sample.") 
        st.write(f"File path: {file_path}")
        st.write(f"Measurement file: {measurement_file}")
        st.write ("DO IMPORT FIRSTS")
if __name__ == "__main__":
    main()

