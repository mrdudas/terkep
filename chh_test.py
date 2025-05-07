import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from numpy.polynomial.polynomial import Polynomial

def lognormal_cascade(n_iter=14, mu=0.0, sigma=0.5):
    """Multifraktális lognormális kaszkád generálása determinisztikusan"""
    N = 2 ** n_iter
    measure = np.ones(1)

    for _ in range(n_iter):
        measure = np.repeat(measure, 2)
        # lognormális szorzók (log-Gauss)
        log_weights = np.random.normal(loc=mu, scale=sigma, size=len(measure) // 2)
        w1 = np.exp(log_weights)
        w2 = np.exp(-log_weights)
        measure[0::2] *= w1
        measure[1::2] *= w2

    # Normalizálás
    measure /= np.sum(measure)
    return measure

def chhabra_jensen(signal, q_vals=np.linspace(-5, 5, 51), epsilons=None):
    N = len(signal)
    if epsilons is None:
        epsilons = [2**i for i in range(1, int(np.log2(N)) - 2)]

    alpha_all = []
    f_alpha_all = []

    for q in q_vals:
        alpha_eps = []
        f_eps = []

        for eps in epsilons:
            n_boxes = N // eps
            mu = []

            for i in range(n_boxes):
                box = signal[i * eps:(i + 1) * eps]
                mu.append(np.sum(np.abs(box)))

            mu = np.array(mu)
            mu = mu / np.sum(mu)

            if q == 1:
                tau_q = -np.sum(mu * np.log(mu + 1e-12))
            else:
                tau_q = (np.log(np.sum(mu ** q + 1e-12))) / np.log(1 / eps)

            # Local Hölder exponent alpha(q)
            alpha_val = np.sum(mu ** q * np.log(mu + 1e-12)) / np.sum(mu ** q + 1e-12)
            alpha_val /= np.log(1 / eps)

            # Multifractal spectrum f(α)
            f_val = q * alpha_val - tau_q

            alpha_eps.append(alpha_val)
            f_eps.append(f_val)

        alpha_all.append(np.mean(alpha_eps))
        f_alpha_all.append(np.mean(f_eps))

    return np.array(alpha_all), np.array(f_alpha_all)


def chhabra_jensen_fixed(signal, q_vals=np.linspace(-5, 5, 51), epsilons=None):
    N = len(signal)
    if epsilons is None:
        epsilons = [2**i for i in range(1, int(np.log2(N)) - 3)]

    alpha_list = []
    f_alpha_list = []

    for q in q_vals:
        log_eps = []
        log_sum_mu_q = []
        log_weighted_avg = []

        for eps in epsilons:
            n_boxes = N // eps
            mu = []

            for i in range(n_boxes):
                box = signal[i * eps : (i + 1) * eps]
                mu.append(np.sum(np.abs(box)))

            mu = np.array(mu)
            mu = mu / np.sum(mu)

            mu_q = mu ** q
            sum_mu_q = np.sum(mu_q)
            weighted_log_mu = np.sum(mu_q * np.log(mu + 1e-12)) / sum_mu_q

            log_eps.append(np.log(eps))
            log_sum_mu_q.append(np.log(sum_mu_q + 1e-12))
            log_weighted_avg.append(weighted_log_mu)

        tau_q_slope, _, _, _, _ = linregress(log_eps, log_sum_mu_q)
        alpha_q_slope, _, _, _, _ = linregress(log_eps, log_weighted_avg)

        alpha_list.append(-alpha_q_slope)
        f_alpha_list.append(q * (-alpha_q_slope) - tau_q_slope)

    return np.array(alpha_list), np.array(f_alpha_list)

def generate_parabolic_spectrum(h=1.0, sigma=0.5, q_vals=np.linspace(-5, 5, 100)):
    """
    Elméleti parabolikus multifraktális spektrum generálása.
    """
    tau_q = h * q_vals - 0.5 * sigma**2 * q_vals**2
    # Derivált tau'(q) → alpha(q)
    alpha = np.gradient(tau_q, q_vals)
    # f(α) = q * α(q) - τ(q)
    f_alpha = q_vals * alpha - tau_q
    return alpha, f_alpha



def lognormal_cascade1(n_iter=14, mu=0.0, sigma=0.4):
    """Lognormális multiplikatív kaszkád mérték"""
    measure = np.ones(1)
    for _ in range(n_iter):
        measure = np.repeat(measure, 2)
        log_weights = np.random.normal(loc=mu, scale=sigma, size=len(measure) // 2)
        w1 = np.exp(log_weights)
        w2 = np.exp(-log_weights)
        measure[0::2] *= w1
        measure[1::2] *= w2
    measure /= np.sum(measure)
    return measure

def chhabra_jensen2(signal, q_vals=np.linspace(-5, 5, 51), epsilons=None):
    N = len(signal)
    if epsilons is None:
        epsilons = [2**i for i in range(1, int(np.log2(N)) - 2)]

    alpha_all = []
    f_alpha_all = []

    for q in q_vals:
        alpha_eps = []
        f_eps = []

        for eps in epsilons:
            n_boxes = N // eps
            mu = []

            for i in range(n_boxes):
                box = signal[i * eps:(i + 1) * eps]
                mu.append(np.sum(np.abs(box)))

            mu = np.array(mu)
            mu = mu / np.sum(mu)

            if q == 1:
                tau_q = -np.sum(mu * np.log(mu + 1e-12))
            else:
                tau_q = (np.log(np.sum(mu ** q + 1e-12))) / np.log(1 / eps)

            alpha_val = np.sum(mu ** q * np.log(mu + 1e-12)) / np.sum(mu ** q + 1e-12)
            alpha_val /= np.log(1 / eps)
            f_val = q * alpha_val - tau_q

            alpha_eps.append(alpha_val)
            f_eps.append(f_val)

        alpha_all.append(np.mean(alpha_eps))
        f_alpha_all.append(np.mean(f_eps))

    return np.array(alpha_all), np.array(f_alpha_all)

def MFDFA(signal, q_vals=np.linspace(-5, 5, 41), scale_vals=None, m=1):
    """
    Egyszerű MFDFA implementáció.
    - signal: 1D numpy array
    - q_vals: list of q exponents
    - scale_vals: list of window sizes (e.g. [16, 32, 64, ...])
    - m: detrending polinom fokszáma (m = 1 → lineáris detrend)
    """
    N = len(signal)
    Y = np.cumsum(signal - np.mean(signal))  # profil

    if scale_vals is None:
        scale_vals = np.unique(np.logspace(2, np.log2(N//4), num=20, base=2, dtype=int))

    F_q = []

    for s in scale_vals:
        Ns = N // s
        F_s = []

        for start in range(0, Ns * s, s):
            segment = Y[start:start + s]
            x = np.arange(s)
            coefs = Polynomial.fit(x, segment, m).convert().coef
            fit = np.polyval(coefs[::-1], x)
            F_s.append(np.mean((segment - fit) ** 2))

        F_s = np.array(F_s)
        F_q_s = []

        for q in q_vals:
            if q == 0:
                Fq = np.exp(0.5 * np.mean(np.log(F_s + 1e-12)))
            else:
                Fq = (np.mean(F_s ** (q / 2))) ** (1 / q)
            F_q_s.append(Fq)

        F_q.append(F_q_s)

    F_q = np.array(F_q)  # shape: (len(scales), len(q))
    log_scales = np.log2(scale_vals)

    h_q = []
    for i, q in enumerate(q_vals):
        y = np.log2(F_q[:, i] + 1e-12)
        slope, _ = np.polyfit(log_scales, y, 1)
        h_q.append(slope)

    h_q = np.array(h_q)
    tau_q = q_vals * h_q - 1
    alpha = np.gradient(tau_q, q_vals)
    f_alpha = q_vals * alpha - tau_q

    return q_vals, h_q, alpha, f_alpha



# --- Generálás és spektrum ---
measure = lognormal_cascade1(n_iter=14, mu=0.0, sigma=0.4)

measure = measure / np.std(measure)  # Normalizálás
measure = measure-np.mean(measure)  # Középpontba állítás

#measure = measure[:1024]  # Korlátozás 1024 elemre
# --- Spektrum számítása ---
print ("Chhabra-Jensen spektrum számítása...",measure.min(), measure.max())
alpha, f_alpha = chhabra_jensen_fixed(measure)

# --- Fraktálszélesség és maximum ---
alpha_0 = alpha[np.argmax(f_alpha)]
delta_alpha = np.max(alpha) - np.min(alpha)

# --- Ábra ---
plt.plot(alpha, f_alpha, marker='o', label='f(α)')
plt.title(f"Fixed Multifraktális spektrum\nα₀ ≈ {alpha_0:.3f}, Δα ≈ {delta_alpha:.3f}")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$f(\alpha)$")
plt.grid()
plt.legend()
plt.show()

alpha, f_alpha = chhabra_jensen2(measure)

# --- Fraktálszélesség és maximum ---
alpha_0 = alpha[np.argmax(f_alpha)]
delta_alpha = np.max(alpha) - np.min(alpha)

# --- Ábra ---
plt.plot(alpha, f_alpha, marker='o', label='f(α)')
plt.title(f"Multifraktális spektrum\nα₀ ≈ {alpha_0:.3f}, Δα ≈ {delta_alpha:.3f}")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$f(\alpha)$")
plt.grid()
plt.legend()
plt.show()

# MFDFA
q_vals, h_q, alpha, f_alpha = MFDFA(measure)

# Ábra
plt.plot(alpha, f_alpha, marker='o')
plt.title("MFDFA: Multifraktális spektrum")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$f(\alpha)$")
plt.grid()
plt.show()