import numpy as np
from scipy.stats import norm
from fbm import FBM


def generate_gauss(length=1024):
    """
    Gauss-görbére illesztett nem-multifraktális tesztjel generálása.
    """
    return norm.pdf(np.linspace(-5, 5, length))


def cantor_measure(length=1024):
    """
    Zajjal kombinált szinusz hullám, amely pszeudo-kvantált mintázatot ad.
    """
    t = np.linspace(0, 1, length)
    return np.sin(5 * np.pi * t) + np.random.normal(0, 0.1, len(t))


def generate_test_signal(kind: str, length: int, alpha: float = 1.0):
    """
    Általános jelfeldolgozási benchmarkhoz használt szintetikus jelek.
    """
    if kind == "white":
        return np.random.normal(0, 1, length)
    elif kind == "brownian":
        return np.cumsum(np.random.normal(0, 1, length))
    elif kind == "chirp":
        t = np.linspace(0, 1, length)
        return np.sin(2 * np.pi * t * (1 + 10 * t))
    elif kind == "sinus+noise":
        t = np.linspace(0, 2 * np.pi, length)
        return np.sin(5 * t) + 0.5 * np.random.normal(0, 1, length)
    elif kind == "step":
        return np.concatenate([np.ones(length // 2), -np.ones(length - length // 2)])
    elif kind == "gauss":
        return generate_gauss(length)
    elif kind == "cantor":
        return cantor_measure(length)
    elif kind == "cascade":
        return multiplicative_cascade(length)
    elif kind == "fBm":
        return Fractional_Brownian_Motion(length)
    elif kind == "multifractal":
        return generate_multifractal_signal(length)
    elif kind == "binomial":
        return generate_binomial_measure(length)
    else:
        raise ValueError(f"Ismeretlen jeltípus: {kind}")


def generate_1_over_f_noise(N, alpha=1.0):
    """
    1/f^alpha típusú zaj generálása spektrális módszerrel (FFT).
    """
    rng = np.random.default_rng()
    phases = rng.uniform(0, 2 * np.pi, N // 2 - 1)
    amplitudes = 1.0 / np.power(np.arange(2, N // 2 + 1), alpha / 2)
    re = amplitudes * np.cos(phases)
    im = amplitudes * np.sin(phases)
    spec = np.zeros(N, dtype=np.complex64)
    spec[1:N // 2] = re + 1j * im
    spec[-(N // 2) + 1:] = np.conj(spec[1:N // 2][::-1])
    signal = np.fft.ifft(spec).real
    return (signal - np.mean(signal)) / np.std(signal)


def multiplicative_cascade(target_length=1024, weights=(0.6, 0.4), seed=None):
    """
    Multiplikatív kaszkád alapú multifraktális jel generálása.
    """
    if seed is not None:
        np.random.seed(seed)

    levels = int(np.ceil(np.log2(target_length)))
    signal = np.array([1.0])

    for _ in range(levels):
        left = signal * weights[0]
        right = signal * weights[1]
        signal = np.concatenate([left, right])

    return signal[:target_length]


def Fractional_Brownian_Motion(length=1024, H=0.7):
    """
    Fractional Brownian motion (fBm) jel generálása megadott Hurst-paraméterrel.
    """
    t = np.arange(1, length + 1)
    return np.cumsum(np.random.normal(size=length)) * (t ** H)


def generate_multifractal_signal(length=1024, H=0.7, p=0.3):
    """
    Multiplikatív binomiális kaszkádon alapuló multifraktális jel generálása.
    """
    n_iter = int(np.ceil(np.log2(length)))
    N = 2 ** n_iter
    measure = np.ones(1)

    for _ in range(n_iter):
        measure = np.repeat(measure, 2)
        rand_mask = np.random.rand(len(measure) // 2) < 0.5
        weights = np.where(rand_mask, p, 1 - p)
        measure[0::2] *= weights
        measure[1::2] *= 1 - weights

    measure /= np.sum(measure)
    return measure[:length]


def generate_binomial_measure(length, p=0.7):
    """
    Klasszikus binomiális multifraktál mérő generálása.
    """
    power = int(np.ceil(np.log2(length)))
    n = 2 ** power

    measure = np.array([1.0])
    for _ in range(power):
        measure = np.repeat(measure, 2)
        measure[::2] *= p
        measure[1::2] *= (1 - p)

    if n > length:
        measure = measure[:length]

    return measure / np.sum(measure)
