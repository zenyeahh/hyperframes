"""Synthesize the episode's sound bed (drone, glitches, heartbeat, stings).

Deterministic: fixed RNG seed. Output: assets/audio/bed.wav (44.1 kHz stereo).
Run: python3 tools/make_audio.py
"""

import os
import wave

import numpy as np

SR = 44100
DUR = 90.0
N = int(SR * DUR)
rng = np.random.default_rng(1337)
t = np.arange(N) / SR


def env(points):
    """Piecewise-linear envelope from [(time, gain), ...]."""
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


def place(buf, clip, at, gain=1.0):
    i = int(at * SR)
    j = min(N, i + len(clip))
    buf[i:j] += clip[: j - i] * gain


out = np.zeros(N)

# Drone: detuned low sines + slow beating, swelling at sunset and before the snap.
drone = (
    np.sin(2 * np.pi * 43 * t)
    + 0.7 * np.sin(2 * np.pi * 43.6 * t)
    + 0.35 * np.sin(2 * np.pi * 64.5 * t)
    + 0.15 * np.sin(2 * np.pi * 129.3 * t + 0.4 * np.sin(2 * np.pi * 0.07 * t))
)
drone_env = env(
    [
        (0, 0), (1.5, 0.22), (9, 0.25), (33, 0.2), (46, 0.3), (58, 0.38),
        (67.3, 0.7), (67.45, 0.0), (69.6, 0.0), (71, 0.25), (84.3, 0.45),
        (84.5, 0.6), (89, 0.0), (90, 0),
    ]
)
out += drone * drone_env * 0.35

# Room tone / wind: band-limited noise.
room = lowpass(rng.standard_normal(N), 420)
room /= np.max(np.abs(room))
out += room * env([(0, 0), (2, 0.18), (67.3, 0.3), (67.45, 0.03), (70, 0.03), (71, 0.15), (90, 0.1)]) * 0.5

# Silence ringing (tinnitus) over the empty city.
out += np.sin(2 * np.pi * 5800 * t) * env([(33, 0), (36, 0.012), (45, 0.012), (47, 0)])

# Rising sweep before the shadow scene climax.
sweep_t = np.clip(t - 58, 0, 9.3)
phase = 2 * np.pi * (180 * sweep_t + (720 / (2 * 9.3)) * sweep_t**2)
out += np.sin(phase) * env([(58, 0), (62, 0.05), (67.3, 0.16), (67.45, 0)])


def glitch(length=0.35, seed=0):
    r = np.random.default_rng(seed)
    n = int(length * SR)
    x = bandpass(r.standard_normal(n), 1500, 9000)
    gate = np.repeat(r.random(n // 441 + 1) > 0.45, 441)[:n]
    tone = np.sign(np.sin(2 * np.pi * r.choice([880, 1320, 1760]) * np.arange(n) / SR))
    y = (0.6 * x + 0.25 * tone) * gate
    y *= np.linspace(1, 0.2, n)
    return y / (np.max(np.abs(y)) + 1e-9)


for k, at in enumerate([0.9, 1.3, 9.1, 22.2, 24.9, 27.4, 29.9, 70.2, 70.6, 71.0]):
    place(out, glitch(0.3 + 0.1 * (k % 3), seed=k), at, 0.22)


def boom(length=3.0, f=48):
    n = int(length * SR)
    tt = np.arange(n) / SR
    y = np.sin(2 * np.pi * f * tt * (1 - 0.25 * tt / length)) * np.exp(-tt * 1.6)
    hit = lowpass(rng.standard_normal(n), 300) * np.exp(-tt * 9)
    y += hit / (np.max(np.abs(hit)) + 1e-9) * 0.6
    return y / np.max(np.abs(y))


place(out, boom(), 6.0, 0.8)
place(out, boom(2.5, 40), 46.0, 0.45)
place(out, boom(3.5, 38), 67.35, 0.95)
place(out, boom(4.5, 36), 84.5, 1.0)


def thump(f=55):
    n = int(0.18 * SR)
    tt = np.arange(n) / SR
    return np.sin(2 * np.pi * f * tt) * np.exp(-tt * 28)


# Heartbeat accelerating 54 -> 67s.
beat_t, gap = 54.0, 1.0
while beat_t < 67.2:
    place(out, thump(), beat_t, 0.9)
    place(out, thump(50), beat_t + 0.22, 0.6)
    beat_t += gap
    gap = max(0.42, gap * 0.93)


def click():
    n = int(0.03 * SR)
    r = np.random.default_rng(7)
    return bandpass(r.standard_normal(n), 2000, 7000) * np.exp(-np.arange(n) / SR * 200)


# Counter ticks.
for i in range(int((84.2 - 73.5) / 0.25)):
    place(out, click(), 73.5 + i * 0.25, 0.25)

# Final fade and normalize.
out *= env([(0, 1), (88.5, 1), (90, 0)])
out /= np.max(np.abs(out)) * 1.12

# Light stereo widening via a short delay on the right channel.
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
