import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st

# --- Numba gyorsítás ---

def compress_dynamics_logscale(x, epsilon=1e-8):
    x = np.clip(x, epsilon, None)            # hogy ne legyen log(0)
    x_log = np.log(x)                        # log csökkenti a dinamikát
    x_norm = (x_log - x_log.min()) / (x_log.max() - x_log.min())  # skálázás 0-1 közé
    return x_norm

@njit
def fast_ricker_cwt(signal, scales):
    N = len(signal)
    num_scales = len(scales)
    cwt_matrix = np.zeros((num_scales, N))
    for idx in prange(num_scales):
        scale = scales[idx]
        wavelet_length = min(10 * scale, N)
        if wavelet_length % 2 == 0:
            wavelet_length += 1
        wavelet = generate_ricker_wavelet(wavelet_length, scale)
        conv_result = convolve_same(signal, wavelet)
        cwt_matrix[idx, :] = pad_or_truncate(conv_result, N)
    return cwt_matrix
# "RPV":"The right pupil valid flag is 1 for valid and 0 for not valid.",
# "LPMM":"The left pupil diameter in millimeters.",
# "LPMMV":"The left pupil diameter valid flag is 1 for valid and 0 for not valid.",
# "RPMM":"The right pupil diameter in millimeters.",
# "RPMMV":"The right pupil diameter valid flag is 1 for valid and 0 for not valid.",
# "DIAL":"The position of the user dial (0 to 1).",
# "DIALV":"The dial valid flag is 1 for valid (connected) and 0 for not valid.",
# "GSR":"The galvanic skin response value (ohms).",
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

@njit(parallel=True)
def fast_ricker_cwt_numba(signal, scales):
    N = len(signal)
    num_scales = len(scales)
    cwt_matrix = np.zeros((num_scales, N))
    for idx in prange(num_scales):
        scale = scales[idx]
        wavelet_length = min(10 * scale, N)
        if wavelet_length % 2 == 0:
            wavelet_length += 1
        wavelet = generate_ricker_wavelet(wavelet_length, scale)
        conv_result = convolve_same(signal, wavelet)
        cwt_matrix[idx, :] = pad_or_truncate(conv_result, N)
    return cwt_matrix

@njit
def find_modulus_maxima_numba(coef_matrix):
    rows, cols = coef_matrix.shape
    maxima = np.zeros((rows, cols))
    window = 2  # lehet 1 vagy 2, érzékenységtől függően

    for i in range(rows):
        for j in range(window, cols - window):
            center = coef_matrix[i, j]
            is_max = True
            for k in range(-window, window + 1):
                if k == 0:
                    continue
                if center < coef_matrix[i, j + k]:
                    is_max = False
                    break
            if is_max:
                maxima[i, j] = abs(center)
    return maxima

@njit (parallel=True)
def compute_Z_q_s(maxima, q_values):
    nq = len(q_values)
    ns = maxima.shape[0]
    Z_q_s = np.zeros((nq, ns))
    for i in range(nq):
        q = q_values[i]
        for j in range(ns):
            m = maxima[j]
            nonzero = m[m > 0]
            if len(nonzero) > 0:
                Z_q_s[i, j] = np.sum(nonzero ** q)
            else:
                Z_q_s[i, j] = np.nan
    return Z_q_s

@njit
def linear_fit(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sum((x - mean_x) ** 2)
    if den == 0:
        return 0.0, 0.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


@njit (parallel=True)
def calc_tau_q(q_values, tau_q, Z_q_s, log_scales):
    for i in range(len(q_values)):
        valid = ~np.isnan(Z_q_s[i, :]) & (Z_q_s[i, :] > 0)
        if np.sum(valid) > 2:
            y = np.log2(Z_q_s[i, valid] + 1e-12)
            x = log_scales[valid]
            slope, _ = linear_fit(x, y)
            tau_q[i] = slope
        else:
            tau_q[i] = np.nan
    return tau_q


# --- WTMM multifraktál spektrum ---
def wtmm_multifractal_spectrum(signal, scales=None, q_values=None, plot=True, streamlit_mode=True, directoy="", filename="tmp"):
    signal = np.asarray(signal)
    N = len(signal)
    time = np.linspace(0, 1, N)


    st.write (f"Optimal scales: {scales}")
    st.write (f"Optimal q values: {q_values}")
    st.write (f"Signal length: {N}")
    
    st.write ("Calculating CWT...")
    coef = fast_ricker_cwt_numba(signal, scales)
    
    st.write ("Calculating modulus maxima...")
    maxima = find_modulus_maxima_numba(coef)
    st.write (f"Maxima shape: {maxima.shape}")
    st.write("Nem nulla maximumok:", np.sum(maxima > 0))
    st.write("Maximum érték:", np.max(maxima))
    
    st.write ("Calculating Z_q_s...")
    Z_q_s = compute_Z_q_s(maxima, q_values)
    
    tau_q = np.zeros(len(q_values))
    log_scales = np.log2(scales)
    
    st.write ("Calculating tau_q...")  
    calc_tau_q (q_values, tau_q, Z_q_s, log_scales)
    #for i in range(len(q_values)):
    #    valid = ~np.isnan(Z_q_s[i, :]) & (Z_q_s[i, :] > 0)
    #    if np.sum(valid) > 2:
    #        y = np.log2(Z_q_s[i, valid] + 1e-12)
    #        x = log_scales[valid]
    #        slope, _ = linear_fit(x, y)
    #        tau_q[i] = slope
    #    else:
    #        tau_q[i] = np.nan

    alpha = np.diff(tau_q) / np.diff(q_values)
    q_mid = (q_values[1:] + q_values[:-1]) / 2
    tau_q_mid = (tau_q[1:] + tau_q[:-1]) / 2
    f_alpha = q_mid * alpha - tau_q_mid

    # --- 1. Eredeti jel ---
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(time, signal, color='black')
    ax1.axis('off')  # minden felirat, tengely eltüntetése
    fig1.savefig(f"{directoy}/{filename}_original_signal.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig1)

    # --- 2. Continuous Wavelet Transform (CWT) ---
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    extent = [time[0], time[-1], scales[-1], scales[0]]
    ax2.imshow(np.abs(coef), extent=extent, cmap='jet', aspect='auto')
    ax2.axis('off')
    fig2.savefig(f"{directoy}/{filename}_cwt.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig2)

    # --- 3. WTMM modulus maxima ---
    fig3, ax3 = plt.subplots(figsize=(12, 4))

    nonzero_vals = maxima[maxima > 0]
    vmin = np.percentile(nonzero_vals, 5) if len(nonzero_vals) > 0 else 1e-12
    vmax = np.max(maxima)
    ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)  

    #ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto')
    #ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto', vmin=1e-12, vmax=np.max(maxima))
    ax3.axis('off')
    fig3.savefig(f"{directoy}/{filename}_modulus_maxima.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig3)



    if plot:    
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # --- 1. Eredeti jel ---
        ax1.plot(time, signal, color='black')
        ax1.set_title('Eredeti jel')
        ax1.set_ylabel('Érték')

        # Mentés csak képként (keret nélkül)
        #ax1.axis('off')
        #fig.savefig(f"{directoy}/{filename}_original_signal.png", bbox_inches='tight', pad_inches=0)
        #ax1.axis('on')  # visszakapcsoljuk, ha kell

        # --- 2. CWT ---
        extent = [time[0], time[-1], scales[-1], scales[0]]
        ax2.imshow(np.abs(coef), extent=extent, cmap='jet', aspect='auto')
        ax2.set_title('Continuous Wavelet Transform (Ricker)')
        ax2.set_ylabel('Skálák')

        #ax2.axis('off')
        #fig.savefig(f"{directoy}/{filename}_cwt.png", bbox_inches='tight', pad_inches=0)
        #ax2.axis('on')

        # --- 3. WTMM ---
        #ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto')
        nonzero_vals = maxima[maxima > 0]
        vmin = np.percentile(nonzero_vals, 5) if len(nonzero_vals) > 0 else 1e-12
        vmax = np.max(maxima)
        ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)  

        #ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto')
        #
        
        #ax3.imshow(maxima, extent=extent, cmap='hot', aspect='auto', vmin=1e-12, vmax=np.max(maxima))
        ax3.set_title('WTMM modulus maxima')
        ax3.set_ylabel('Skálák')
        ax3.set_xlabel('Idő')

        #ax3.axis('off')
        #fig.savefig(f"{directoy}/{filename}_modulus_maxima.png", bbox_inches='tight', pad_inches=0)
        #ax3.axis('on')

        plt.tight_layout()
        
        if streamlit_mode:
            st.pyplot(fig)
        else:
            plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(alpha, f_alpha, '-o', color='navy')
    plt.xlabel('α (Szingularitás erőssége)')
    plt.ylabel('f(α) (Spektrum)')
    plt.title('Multifraktál spektrum (WTMM + CWT)')
    # save the multifractal spectrum to file
    #plt.axis('off')
    plt.savefig(f"{directoy}/{filename}_multifractal_spectrum.png")
    #plt.axis('on')
    plt.grid(True)
    if plot:
        if streamlit_mode:
            st.pyplot(plt)
        else:
            plt.show()

    return alpha, f_alpha, tau_q, Z_q_s, q_values, scales, coef, maxima
