"""Synthesize episode 2's sound bed. Deterministic (fixed seeds).

Output: assets/audio/bed.wav (44.1 kHz stereo).
Run: python3 tools/make_audio.py
"""

import os
import wave

import numpy as np

SR = 44100
DUR = 90.0
N = int(SR * DUR)
rng = np.random.default_rng(2024)
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

# Drone bed.
drone = (
    np.sin(2 * np.pi * 41 * t)
    + 0.7 * np.sin(2 * np.pi * 41.5 * t)
    + 0.3 * np.sin(2 * np.pi * 61.7 * t)
    + 0.12 * np.sin(2 * np.pi * 123.1 * t + 0.5 * np.sin(2 * np.pi * 0.05 * t))
)
out += drone * 0.35 * env(
    [
        (0, 0), (1, 0.2), (7, 0.2), (26, 0.18), (27.5, 0.34), (32, 0.2), (48, 0.2),
        (56, 0.24), (66.4, 0.3), (72.4, 0.65), (74.75, 0.65), (74.85, 0.05), (76, 0.2),
        (81, 0.26), (85.55, 0.55), (85.65, 0.0), (86.5, 0.0), (86.6, 0.45), (89, 0), (90, 0),
    ]
)
room = norm(lowpass(rng.standard_normal(N), 380))
out += room * 0.08 * env([(0, 0), (1, 1), (85.55, 1), (85.65, 0.1), (90, 0.1)])

# Knife scratches on the wall.
def scratch(seed):
    r = np.random.default_rng(seed)
    n, tt = seg(0.5)
    x = bandpass(r.standard_normal(n), 2200, 7000)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 38 * tt)
    return norm(x * am * np.sin(np.pi * tt / 0.5))


for k, at in enumerate([0.5, 1.3, 2.1, 2.9]):
    place(out, scratch(k), at, 0.35 if k < 3 else 0.5)


def glitch(length=0.3, seed=0):
    r = np.random.default_rng(seed + 100)
    n = int(length * SR)
    x = bandpass(r.standard_normal(n), 1500, 9000)
    gate = np.repeat(r.random(n // 441 + 1) > 0.45, 441)[:n]
    tone = np.sign(np.sin(2 * np.pi * r.choice([880, 1320, 1760]) * np.arange(n) / SR))
    return norm((0.6 * x + 0.25 * tone) * gate * np.linspace(1, 0.2, n))


for k, at in enumerate([4.2, 72.4, 72.75, 79.5, 79.85]):
    place(out, glitch(0.3, k), at, 0.22)


def boom(length=3.0, f=48):
    n, tt = seg(length)
    y = np.sin(2 * np.pi * f * tt * (1 - 0.25 * tt / length)) * np.exp(-tt * 1.6)
    hit = norm(lowpass(rng.standard_normal(n), 300)) * np.exp(-tt * 9)
    return norm(y + 0.6 * hit)


place(out, boom(), 5.0, 0.8)
place(out, boom(2.0, 55), 27.5, 0.7)
place(out, boom(2.5, 42), 66.4, 0.55)
place(out, boom(2.5, 50), 74.8, 0.9)
place(out, boom(4.0, 36), 85.6, 1.0)


def knock(f=110):
    n, tt = seg(0.35)
    body = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 22)
    wood = norm(lowpass(rng.standard_normal(n), 900)) * np.exp(-tt * 40)
    return norm(body + 0.7 * wood)


# Items landing on the table.
for at in [7.8, 9.2, 10.6, 12.0, 13.4]:
    place(out, knock(180), at, 0.35)

# Calorie counter ticks.
def click(seed=7):
    n = int(0.03 * SR)
    r = np.random.default_rng(seed)
    return bandpass(r.standard_normal(n), 2000, 7000) * np.exp(-np.arange(n) / SR * 200)


for i in range(20):
    place(out, click(), 14.0 + i * 0.12, 0.22)

# Water glug and fridge hum.
n_w = int(2.5 * SR)
tw = np.arange(n_w) / SR
glug = np.sin(2 * np.pi * (140 + 60 * np.sin(2 * np.pi * 5 * tw)) * tw) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 3 * tw)))
place(out, lowpass(glug, 500) * np.exp(-tw * 0.4), 18.5, 0.25)
place(out, np.sign(np.sin(2 * np.pi * 100 * tw[: int(4 * SR) if len(tw) > 4 * SR else len(tw)])) * 0.0, 22, 0)
hum_len = int(4.0 * SR)
th = np.arange(hum_len) / SR
hum = lowpass(np.sign(np.sin(2 * np.pi * 100 * th)), 400) * np.minimum(1, th * 4) * np.minimum(1, (4 - th) * 4)
place(out, hum, 22.2, 0.06)

# Flashlight clicks and paper rustle.
for at in [33.0, 33.4, 35.0]:
    place(out, click(3), at, 0.45)


def rustle(seed):
    r = np.random.default_rng(seed + 300)
    n, tt = seg(0.6)
    x = bandpass(r.standard_normal(n), 1200, 6000)
    am = np.abs(np.sin(2 * np.pi * 9 * tt)) * np.sin(np.pi * tt / 0.6)
    return norm(x * am)


for k, at in enumerate([36.0, 38.5, 41.0, 43.5, 44.3]):
    place(out, rustle(k), at, 0.18)


def step(f=70, seed=0):
    n, tt = seg(0.12)
    r = np.random.default_rng(seed + 500)
    return norm(np.sin(2 * np.pi * f * tt) * np.exp(-tt * 45) + 0.4 * lowpass(r.standard_normal(n), 1500) * np.exp(-tt * 60))


for i, at in enumerate(np.arange(45.2, 47.8, 0.18)):
    place(out, step(90, i), at, 0.12)

# Phone vibration.
for a, b in [(48.5, 49.3), (49.8, 50.6)]:
    n, tt = seg(b - a)
    buzz = lowpass(np.sign(np.sin(2 * np.pi * 175 * tt)), 600) * np.minimum(1, tt * 30)
    place(out, buzz, a, 0.14)

# Footsteps down the stairs, then running up.
for i, at in enumerate(np.arange(56.4, 60.6, 0.4)):
    place(out, step(65, i), at, 0.3)
for i, at in enumerate(np.arange(67.0, 72.4, 0.2)):
    place(out, step(75, i + 50), at, 0.36)

# Cans clinking into the backpack.
def clink():
    n, tt = seg(0.5)
    return norm((np.sin(2 * np.pi * 1850 * tt) + 0.6 * np.sin(2 * np.pi * 3120 * tt)) * np.exp(-tt * 12))


for at in [62.5, 63.3, 64.1, 64.9]:
    place(out, clink(), at, 0.16)

# Rising sweep during the run.
sw = np.clip(t - 66.4, 0, 6.0)
out += np.sin(2 * np.pi * (160 * sw + 70 * sw**2)) * env([(66.4, 0), (70, 0.06), (72.4, 0.15), (72.5, 0)])


def thump(f=55):
    n, tt = seg(0.18)
    return np.sin(2 * np.pi * f * tt) * np.exp(-tt * 28)


bt, gap = 67.0, 0.75
while bt < 74.6:
    place(out, thump(), bt, 0.8)
    place(out, thump(50), bt + 0.2, 0.55)
    bt += gap
    gap = max(0.4, gap * 0.95)

# Night wind.
wind = norm(bandpass(rng.standard_normal(N), 200, 900))
out += wind * env([(75, 0), (76, 0.08), (79, 0.08), (79.3, 0)])

# Three knocks.
for at in [81.0, 81.7, 82.4]:
    place(out, knock(95), at, 1.0)

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
