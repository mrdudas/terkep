

import numpy as np
from scipy.stats import norm
from fbm import FBM
import streamlit as st    

def generate_gauss (length=1024):
    """
    Generate a Gaussian signal.
    """
    # Mű jel: Gauss-jel (nem multifraktál)
    signal = norm.pdf(np.linspace(-5, 5, length))
    return signal

def cantor_measure(length=1024, N=10):
    t = np.linspace(0, 1, length)
    signal = np.sin(5 * np.pi * t) + np.random.normal(0, 0.1, len(t))
    return signal



def multiplicative_cascade(target_length=1024, weights=(0.6, 0.4), seed=None):
    """
    Random multiplicative cascade szintetikus multifraktál jel generálása adott hosszra.

    Parameters:
        target_length : int
            A kívánt jelsorozat hossza (a kimenet legfeljebb ez lesz, de pontosan ennyi, ha le is van vágva).
        weights : tuple
            A szorzótényezők minden lépésben (ált. (0.6, 0.4)).
        seed : int or None
            Véletlenszerűség szabályozásához.

    Returns:
        np.ndarray
            A generált multifraktális jel (1D tömb, hossz = target_length).
    """
    if seed is not None:
        np.random.seed(seed)

    levels = int(np.ceil(np.log2(target_length)))
    signal = np.array([1.0])

    for _ in range(levels):
        left = signal * weights[0]
        right = signal * weights[1]
        signal = np.concatenate([left, right])
    st.write( signal[:target_length].max() )
    return signal[:target_length]

def Fractional_Brownian_Motion (length=1024, H=0.7):
    """
    Generate a Fractional Brownian Motion (fBm) signal.
    """
    # fBm generálás
    t = np.arange(length)
    fBm = np.cumsum(np.random.normal(size=length)) * (t ** H)
    return fBm

def generate_multifractal_signal(length =1024, H=0.7, p=0.3):
#def multifractal_cascade(n_iter=14, p=0.3):
    """Generate a 1D multifractal measure using a multiplicative binomial cascade."""
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
    #signal = np.cumsum(measure)  # Cumulative signal
    signal = measure.copy()
    return signal

def generate_binomial_measure(length, p=0.7):
    """
    Multiplikatív binomiális multifraktál mérő generálása egy adott hosszra.

    Parameters:
        length : int
            A generált jelsorozat hossza. A legközelebbi 2^N hosszra lesz kerekítve.
        p : float
            Az elosztási paraméter (0 < p < 1), pl. 0.7

    Returns:
        np.ndarray:
            Normalizált binomiális mérő [0, 1] intervallumban.
    """
    # Legkisebb 2^N hossz, ami >= length
    power = int(np.ceil(np.log2(length)))
    n = 2 ** power

    measure = np.array([1.0])
    for _ in range(power):
        measure = np.repeat(measure, 2)
        measure[::2] *= p
        measure[1::2] *= (1 - p)

    # Ha kell, vágjuk le
    if n > length:
        measure = measure[:length]

    # Normalizálás
    measure = measure / np.sum(measure)
    return measure
