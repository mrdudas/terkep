import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import matplotlib.animation as animation

# Paraméterek
fs = 1000
t = np.linspace(0, 1.0, fs, endpoint=False)
f0, f1 = 5, 20
x = np.sin(2 * np.pi * (f0 + (f1 - f0) * t) * t)

# Hilbert-transzformáció
analytic_signal = hilbert(x)
envelope = np.abs(analytic_signal)
phase = np.unwrap(np.angle(analytic_signal))
instantaneous_freq = np.diff(phase) / (2.0 * np.pi) * fs

# Animációs ábra előkészítése
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1.0)
ax.set_ylim(-1.5, 25)
line1, = ax.plot([], [], label="Jel", color='blue')
line2, = ax.plot([], [], label="Burkoló", color='green', linestyle='--')
line3, = ax.plot([], [], label="Azonnali frekvencia", color='red')
ax.set_xlabel("Idő [s]")
ax.set_ylabel("Amplitúdó / Frekvencia [Hz]")
ax.legend(loc="upper right")
ax.set_title("Jel, Burkoló és Azonnali Frekvencia időben")

# Animáció frissítő függvény
def update(frame):
    i = frame + 2
    line1.set_data(t[:i], x[:i])
    line2.set_data(t[:i], envelope[:i])
    line3.set_data(t[1:i], instantaneous_freq[:i-1])
    return line1, line2, line3

ani = animation.FuncAnimation(
    fig, update, frames=np.arange(0, len(t)-2, 5),
    blit=True, interval=50, repeat=False
)

# Animáció mentése
ani_path = "mnt/data/hilbert_phase_freq_animation.avi"
ani.save(ani_path, writer='ffmpeg')

ani_path
