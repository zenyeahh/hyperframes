"""Synthesize episode 3's sound bed. Deterministic (fixed seeds).

Output: assets/audio/bed.wav (44.1 kHz stereo).
Run: python3 tools/make_audio.py
"""

import os
import wave

import numpy as np

SR = 44100
DUR = 90.0
N = int(SR * DUR)
rng = np.random.default_rng(3033)
t = np.arange(N) / SR


def env(points):
    xs, ys = zip(*points)
    return np.interp(t, xs, ys)


def lowpass(x, cutoff):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1 / (1 + (freqs / cutoff) ** 4)
    return np.fft.irfft(spec, len(x))


def bandpass(x, lo, hi):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= (freqs > lo) & (freqs < hi)
    return np.fft.irfft(spec, len(x))


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def place(buf, clip, at, gain=1.0):
    i = int(at * SR)
    j = min(N, i + len(clip))
    buf[i:j] += clip[: j - i] * gain


def seg(length):
    n = int(length * SR)
    return n, np.arange(n) / SR


out = np.zeros(N)

# Drone bed with silences at the key beats.
drone = (
    np.sin(2 * np.pi * 39 * t)
    + 0.7 * np.sin(2 * np.pi * 39.6 * t)
    + 0.3 * np.sin(2 * np.pi * 58.6 * t)
    + 0.12 * np.sin(2 * np.pi * 117.4 * t + 0.5 * np.sin(2 * np.pi * 0.06 * t))
)
out += drone * 0.35 * env(
    [
        (0, 0), (0.3, 0.25), (8, 0.22), (15.3, 0.3), (15.45, 0.05), (16, 0.2), (22, 0.26),
        (34.15, 0.4), (34.3, 0.12), (44, 0.12), (44.2, 0.18), (56, 0.2), (61.2, 0.4),
        (63.5, 0.55), (70, 0.5), (73, 0.08), (80, 0.3), (82, 0.6), (83.95, 0.6),
        (84.05, 0.0), (84.4, 0.45), (89, 0), (90, 0),
    ]
)
room = norm(lowpass(rng.standard_normal(N), 380))
out += room * 0.08 * env([(0, 1), (83.95, 1), (84.05, 0.05), (90, 0.05)])


def knock(f=95):
    n, tt = seg(0.35)
    body = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 22)
    wood = norm(lowpass(rng.standard_normal(n), 900)) * np.exp(-tt * 40)
    return norm(body + 0.7 * wood)


for at in [0.4, 1.1, 1.8]:
    place(out, knock(), at, 0.9)
for at in [80.2, 80.9, 81.6]:
    place(out, knock(85), at, 1.0)


def click(seed=7):
    n = int(0.03 * SR)
    r = np.random.default_rng(seed)
    return bandpass(r.standard_normal(n), 2000, 7000) * np.exp(-np.arange(n) / SR * 200)


# Phone counter ticks, then they stop dead.
for i in range(int((15.4 - 8.5) / 0.25)):
    place(out, click(), 8.5 + i * 0.25, 0.2)
# Rapid countdown to 1.
tick_t = 56.2
gap = 0.2
while tick_t < 60.9:
    place(out, click(9), tick_t, 0.22)
    tick_t += gap
    gap = max(0.035, gap * 0.9)


def thump(f=55):
    n, tt = seg(0.18)
    return np.sin(2 * np.pi * f * tt) * np.exp(-tt * 28)


bt, g = 15.6, 0.85
while bt < 22:
    place(out, thump(), bt, 0.8)
    place(out, thump(50), bt + 0.2, 0.55)
    bt += g
    g = max(0.5, g * 0.95)

# Door handle creak.
n_c, tc = seg(2.0)
creak = bandpass(rng.standard_normal(n_c), 300, 1400) * (0.5 + 0.5 * np.sin(2 * np.pi * 23 * tc))
place(out, norm(creak) * np.sin(np.pi * tc / 2.0), 22.5, 0.18)


def clink(f1=1850, f2=3120, length=0.5):
    n, tt = seg(length)
    return norm((np.sin(2 * np.pi * f1 * tt) + 0.6 * np.sin(2 * np.pi * f2 * tt)) * np.exp(-tt * 12))


place(out, clink(1300, 2400, 0.6), 24.5, 0.35)
place(out, clink(1500, 2800, 0.4), 24.62, 0.2)


def boom(length=3.0, f=48):
    n, tt = seg(length)
    y = np.sin(2 * np.pi * f * tt * (1 - 0.25 * tt / length)) * np.exp(-tt * 1.6)
    hit = norm(lowpass(rng.standard_normal(n), 300)) * np.exp(-tt * 9)
    return norm(y + 0.6 * hit)


place(out, boom(1.5, 70), 28.5, 0.35)
# Cans rolling.
n_r, tr = seg(1.5)
roll = lowpass(rng.standard_normal(n_r), 600) * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * tr)) * np.exp(-tr * 1.5)
place(out, norm(roll), 28.7, 0.15)
for at in [29.0, 29.4, 29.8]:
    place(out, clink(1700, 2900, 0.3), at, 0.08)
place(out, boom(2.5, 52), 34.2, 0.8)


def step(f=70, seed=0):
    n, tt = seg(0.12)
    r = np.random.default_rng(seed + 500)
    return norm(np.sin(2 * np.pi * f * tt) * np.exp(-tt * 45) + 0.4 * lowpass(r.standard_normal(n), 1500) * np.exp(-tt * 60))


for i, at in enumerate(np.arange(35.5, 39.6, 0.45)):
    place(out, step(65, i), at, 0.32 * (1 - 0.6 * (at - 35.5) / 4.1))


def glitch(length=0.3, seed=0):
    r = np.random.default_rng(seed + 100)
    n = int(length * SR)
    x = bandpass(r.standard_normal(n), 1500, 9000)
    gate = np.repeat(r.random(n // 441 + 1) > 0.45, 441)[:n]
    tone = np.sign(np.sin(2 * np.pi * r.choice([880, 1320, 1760]) * np.arange(n) / SR))
    return norm((0.6 * x + 0.25 * tone) * gate * np.linspace(1, 0.2, n))


for k, at in enumerate([38.5, 47.5, 61.2, 61.5]):
    place(out, glitch(0.3, k), at, 0.22)
for at in [40.2, 41.2, 42.2]:
    place(out, click(11), at, 0.6)


def scratch(seed):
    r = np.random.default_rng(seed)
    n, tt = seg(0.5)
    x = bandpass(r.standard_normal(n), 2200, 7000)
    return norm(x * (0.6 + 0.4 * np.sin(2 * np.pi * 38 * tt)) * np.sin(np.pi * tt / 0.5))


place(out, scratch(4), 44.6, 0.45)


def rustle(seed):
    r = np.random.default_rng(seed + 300)
    n, tt = seg(0.6)
    x = bandpass(r.standard_normal(n), 1200, 6000)
    return norm(x * np.abs(np.sin(2 * np.pi * 9 * tt)) * np.sin(np.pi * tt / 0.6))


place(out, rustle(1), 50.6, 0.18)
place(out, boom(3.0, 44), 63.5, 0.7)
out += np.sin(2 * np.pi * 5800 * t) * env([(66.3, 0), (66.6, 0.014), (72.5, 0.014), (73, 0)])
sw = np.clip(t - 66.5, 0, 6.5)
out += np.sin(2 * np.pi * (120 * sw + 40 * sw**2)) * env([(66.5, 0), (70, 0.05), (72.9, 0.1), (73.0, 0)])
place(out, click(5), 73.0, 0.6)
place(out, boom(5.0, 34), 84.0, 1.0)

out *= env([(0, 1), (88.8, 1), (90, 0)])
out /= np.max(np.abs(out)) * 1.12
right = np.roll(out, int(0.011 * SR))
stereo = np.stack([out, 0.92 * out + 0.08 * right], axis=1)
pcm = (stereo * 32767).astype(np.int16)
os.makedirs("assets/audio", exist_ok=True)
with wave.open("assets/audio/bed.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote assets/audio/bed.wav", pcm.shape)
