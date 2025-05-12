
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import json
import os

st.set_page_config(layout="wide")
st.title("Fázistér • Multifraktál • Annotáció")

# --- Oldalsáv beállítások ---
st.sidebar.header("Adatbevitel")
use_custom = st.sidebar.checkbox("Saját adat feltöltése", value=False)

if use_custom:
    uploaded_file = st.sidebar.file_uploader("CSV fájl feltöltése", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        col = st.sidebar.selectbox("Válassz oszlopot", df.columns)
        signal = df[col].dropna().values
    else:
        st.stop()
else:
    t = np.linspace(0, 10, 2000)
    signal = np.sin(4 * np.pi * t) + 0.3 * np.random.randn(len(t))

# --- Paraméterek ---
st.sidebar.header("Fázistér beállítások")
m = st.sidebar.slider("Dimenzió (m)", 2, 10, 3)
tau = st.sidebar.slider("Időkésés (τ)", 1, 100, 10)
use_cumsum = st.sidebar.checkbox("Használj mean-centered cumsum-ot", value=False)
use_pca = st.sidebar.checkbox("PCA vetítés 2D-re", value=True)

st.sidebar.header("Annotáció")
segment_half = st.sidebar.slider("Kiemelt szakasz hossza (fél oldal)", 10, 500, 100)
label_options = ["ritmikus", "nyugtalan", "artefaktum"]
annotation_label = st.sidebar.selectbox("Címke", label_options + ["Egyéni"])
custom_label = ""
if annotation_label == "Egyéni":
    custom_label = st.sidebar.text_input("Add meg a saját címkét")

# --- Előfeldolgozás ---
if use_cumsum:
    signal = np.cumsum(signal - np.mean(signal))

def embed(signal, m, tau):
    N = len(signal)
    M = N - (m - 1) * tau
    if M <= 0:
        return None
    return np.array([signal[i:i + M] for i in range(0, m * tau, tau)]).T

X = embed(signal, m, tau)
if X is None:
    st.error("Túl nagy m vagy τ.")
    st.stop()

# PCA
if use_pca and m > 2:
    X_vis = PCA(n_components=2).fit_transform(X)
    xlabel, ylabel = "PC1", "PC2"
else:
    X_vis = X[:, :2]
    xlabel, ylabel = f"x(t)", f"x(t+{tau})"

# --- Ábrák és annotáció ---
st.subheader("Fázistér és annotáció")
click_info = st.empty()
annot_list = []

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(X_vis[:, 0], X_vis[:, 1], s=1, alpha=0.5)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title("Kattints egy pontra annotációhoz")
ax.grid(True)

def onclick(event):
    if event.inaxes != ax:
        return
    x_click, y_click = event.xdata, event.ydata
    dists = np.linalg.norm(X_vis - np.array([x_click, y_click]), axis=1)
    nearest_idx = np.argmin(dists)
    time_idx = nearest_idx
    label = annotation_label if annotation_label != "Egyéni" else custom_label
    if label:
        annot_list.append({
            "phase_index": nearest_idx,
            "time_index": time_idx,
            "label": label
        })
        click_info.info(f"Annotáció: index={time_idx}, címke='{label}'")

fig.canvas.mpl_connect("button_press_event", onclick)
st.pyplot(fig)

# --- Eredeti szakasz megjelenítése ---
if annot_list:
    st.subheader("Kiemelt szakasz")
    last = annot_list[-1]
    idx = last["time_index"]
    start = max(0, idx - segment_half)
    end = min(len(signal), idx + segment_half + 1)
    seg = signal[start:end]
    t_seg = np.arange(start, end)

    fig2, ax2 = plt.subplots()
    ax2.plot(t_seg, seg, color='black')
    ax2.axvline(idx, color='red', linestyle='--')
    ax2.set_title(f"Eredeti jel: index={idx}, címke='{last['label']}'")
    st.pyplot(fig2)

# --- Mentés / letöltés ---
st.subheader("Annotációk mentése")
fname = st.text_input("Fájlnév", value="annotations.json")
save_btn = st.button("Mentés fájlba")

if save_btn:
    with open(fname, "w") as f:
        json.dump(annot_list, f, indent=4)
    st.success(f"{fname} elmentve.")

# --- Betöltés ---
st.subheader("Korábbi annotáció betöltése")
upload_annot = st.file_uploader("Tölts fel egy JSON fájlt", type=["json"])
if upload_annot:
    loaded = json.load(upload_annot)
    st.json(loaded)
    st.info(f"{len(loaded)} annotáció betöltve.")
