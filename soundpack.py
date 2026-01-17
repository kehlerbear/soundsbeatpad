import os, math, random
import numpy as np
from scipy.io import wavfile

# ---------------------------
# Config
# ---------------------------
SR = 44100
OUT_DIR = "ABSTRUCT_Beatpack_Python"
random.seed(7)
np.random.seed(7)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def write_wav(path, x, sr=SR):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (x * 32767).astype(np.int16))

def db_to_amp(db):
    return 10 ** (db / 20)

def normalize(x, peak=0.98):
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)

# ---------------------------
# Basic building blocks
# ---------------------------
def sine(freq, dur, sr=SR, phase=0.0):
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2*np.pi*freq*t + phase)

def saw(freq, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    # naive saw
    return 2.0 * ((t * freq) % 1.0) - 1.0

def square(freq, dur, sr=SR):
    return np.sign(sine(freq, dur, sr))

def noise(dur, sr=SR):
    return np.random.uniform(-1.0, 1.0, int(dur * sr))

def env_adsr(dur, a=0.005, d=0.08, s=0.25, r=0.08, sr=SR):
    n = int(dur * sr)
    aN = max(1, int(a * sr))
    dN = max(1, int(d * sr))
    rN = max(1, int(r * sr))
    sN = max(1, n - (aN + dN + rN))
    a_env = np.linspace(0, 1, aN, endpoint=False)
    d_env = np.linspace(1, s, dN, endpoint=False)
    s_env = np.full(sN, s)
    r_env = np.linspace(s, 0, rN, endpoint=True)
    env = np.concatenate([a_env, d_env, s_env, r_env])
    if len(env) < n:
        env = np.pad(env, (0, n-len(env)))
    return env[:n]

def env_exp_decay(dur, tau=0.15, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return np.exp(-t / max(1e-6, tau))

def one_pole_lpf(x, cutoff_hz, sr=SR):
    # simple one-pole lowpass
    x = np.asarray(x, dtype=np.float32)
    c = math.exp(-2.0 * math.pi * cutoff_hz / sr)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = (1 - c) * x[i] + c * y[i-1]
    return y

def one_pole_hpf(x, cutoff_hz, sr=SR):
    # simple one-pole highpass via lowpass subtraction
    lp = one_pole_lpf(x, cutoff_hz, sr)
    return x - lp

def soft_clip(x, drive=2.5):
    x = np.asarray(x, dtype=np.float32) * drive
    return np.tanh(x)

def bitcrush(x, bits=8):
    x = np.asarray(x, dtype=np.float32)
    levels = 2 ** bits
    return np.round((x + 1) * (levels/2)) / (levels/2) - 1

def tremolo(x, rate_hz=6.0, depth=0.6, sr=SR):
    t = np.arange(len(x)) / sr
    lfo = (1 - depth) + depth * (0.5 * (1 + np.sin(2*np.pi*rate_hz*t)))
    return x * lfo

# ---------------------------
# Drum synths
# ---------------------------
def synth_kick(name="kick", dur=0.6, f0=90, f1=35, click=0.02):
    n = int(dur * SR)
    t = np.arange(n) / SR
    # pitch drop
    f = np.linspace(f0, f1, n)
    phase = 2*np.pi*np.cumsum(f)/SR
    body = np.sin(phase) * env_exp_decay(dur, tau=0.18)
    # click
    c = noise(click)
    c = one_pole_hpf(c, 3000)
    c = c * env_exp_decay(click, tau=0.01)
    x = body
    x[:len(c)] += 0.6 * c
    x = soft_clip(x, drive=2.0)
    return normalize(x)

def synth_snare(name="snare", dur=0.35, tone=180):
    t = np.arange(int(dur*SR))/SR
    # noise burst + tonal pop
    n = noise(dur)
    n = one_pole_hpf(n, 1200)
    n = n * env_exp_decay(dur, tau=0.09)
    pop = sine(tone, dur) * env_exp_decay(dur, tau=0.05)
    x = 0.9*n + 0.35*pop
    x = soft_clip(x, drive=2.0)
    return normalize(x)

def synth_clap(dur=0.32):
    n = noise(dur)
    n = one_pole_hpf(n, 1800)
    # multi-burst envelope (clap "hands")
    env = np.zeros_like(n)
    bursts = [0.0, 0.02, 0.045, 0.075]
    for b in bursts:
        start = int(b*SR)
        end = min(len(env), start + int(0.06*SR))
        env[start:end] += env_exp_decay((end-start)/SR, tau=0.02)
    env = np.clip(env, 0, 1)
    x = n * env
    x = soft_clip(x, drive=2.3)
    return normalize(x)

def synth_hat(dur=0.12, open_hat=False):
    n = noise(dur)
    n = one_pole_hpf(n, 6000)
    tau = 0.09 if open_hat else 0.03
    x = n * env_exp_decay(dur, tau=tau)
    x = bitcrush(x, bits=10 if open_hat else 8)
    return normalize(x, peak=0.9)

def synth_rim(dur=0.15):
    t = np.arange(int(dur*SR))/SR
    x = (0.6*sine(2200, dur) + 0.35*sine(1800, dur)) * env_exp_decay(dur, tau=0.03)
    x += 0.15 * one_pole_hpf(noise(dur), 4000) * env_exp_decay(dur, tau=0.02)
    x = soft_clip(x, drive=2.2)
    return normalize(x, peak=0.9)

def synth_crash(dur=1.8):
    n = noise(dur)
    n = one_pole_hpf(n, 4500)
    x = n * env_exp_decay(dur, tau=0.6)
    x = soft_clip(x, drive=1.6)
    return normalize(x, peak=0.9)

# ---------------------------
# Bass synths
# ---------------------------
NOTE_FREQ = {
    "C": 16.3516, "C#": 17.3239, "D": 18.3540, "D#": 19.4454, "E": 20.6017,
    "F": 21.8268, "F#": 23.1247, "G": 24.4997, "G#": 25.9565, "A": 27.5000,
    "A#": 29.1352, "B": 30.8677
}
def note_to_hz(note="A", octave=2):
    return NOTE_FREQ[note] * (2 ** octave)

def synth_808(dur=1.2, note="A", octave=1, drive=2.8):
    f = note_to_hz(note, octave)
    n = int(dur*SR)
    t = np.arange(n)/SR
    # slight pitch drop for punch
    f_env = np.linspace(f*1.6, f, n)
    phase = 2*np.pi*np.cumsum(f_env)/SR
    x = np.sin(phase)
    x *= env_exp_decay(dur, tau=0.55)
    x = soft_clip(x, drive=drive)
    x = one_pole_lpf(x, 180)  # keep it subby
    return normalize(x)

def synth_reese(dur=1.5, f=55):
    x = 0.55*saw(f, dur) + 0.55*saw(f*1.01, dur)
    x *= env_adsr(dur, a=0.01, d=0.25, s=0.45, r=0.15)
    x = one_pole_lpf(x, 900)
    x = soft_clip(x, drive=2.3)
    return normalize(x)

def synth_wub(dur=2.0, f=55, wobble_hz=6.0):
    # Wub: harmonic source -> moving lowpass (LFO) + distortion
    src = 0.6*saw(f, dur) + 0.25*square(f*0.5, dur) + 0.15*sine(f*2, dur)
    src *= env_adsr(dur, a=0.01, d=0.2, s=0.75, r=0.12)
    t = np.arange(len(src))/SR
    lfo = 0.5*(1 + np.sin(2*np.pi*wobble_hz*t))
    # cutoff swings between 120 and 1200 Hz
    cutoff = 120 + lfo*(1200-120)
    y = np.zeros_like(src)
    # cheap time-varying LPF by chunking
    chunk = 256
    for i in range(0, len(src), chunk):
        c = float(np.mean(cutoff[i:i+chunk]))
        y[i:i+chunk] = one_pole_lpf(src[i:i+chunk], c)
    y = soft_clip(y, drive=3.0)
    y = bitcrush(y, bits=10)
    return normalize(y)

# ---------------------------
# FX / textures
# ---------------------------
def synth_riser(dur=2.5, start=200, end=2200):
    n = int(dur*SR)
    freqs = np.linspace(start, end, n)
    phase = 2*np.pi*np.cumsum(freqs)/SR
    x = np.sin(phase) + 0.25*np.sin(2*phase)
    x *= np.linspace(0, 1, n)
    x = one_pole_hpf(x, 120)
    x = soft_clip(x, drive=1.8)
    return normalize(x, peak=0.9)

def synth_downlifter(dur=1.6, start=1800, end=120):
    n = int(dur*SR)
    freqs = np.linspace(start, end, n)
    phase = 2*np.pi*np.cumsum(freqs)/SR
    x = np.sin(phase) * np.linspace(1, 0, n)
    x += 0.35*one_pole_hpf(noise(dur), 2000)*env_exp_decay(dur, tau=0.35)
    x = soft_clip(x, drive=2.0)
    return normalize(x, peak=0.9)

def synth_impact(dur=1.0):
    x = one_pole_lpf(noise(dur), 220) * env_exp_decay(dur, tau=0.22)
    x += 0.4*sine(55, dur)*env_exp_decay(dur, tau=0.35)
    x = soft_clip(x, drive=2.6)
    return normalize(x)

def synth_vinyl_noise(dur=4.0):
    x = noise(dur)
    x = one_pole_hpf(x, 600)
    x = one_pole_lpf(x, 9000)
    x = tremolo(x, rate_hz=0.4, depth=0.4)
    x = x * db_to_amp(-18)
    return normalize(x, peak=0.6)

# ---------------------------
# Loops
# ---------------------------
def mix_at(x, y, start):
    end = min(len(x), start + len(y))
    x[start:end] += y[:end-start]
    return x

def make_loop(bpm=140, bars=2):
    beats = bars * 4
    sec = (60.0 / bpm) * beats
    n = int(sec * SR)
    return np.zeros(n, dtype=np.float32), sec

def place_step(loop, sample, bpm, step_idx, steps_per_bar=16, bars=2, gain=1.0):
    steps_total = steps_per_bar * bars
    step_dur = (60.0 / bpm) * 4 / steps_per_bar
    start = int(step_idx * step_dur * SR)
    return mix_at(loop, sample*gain, start)

def gen_loops():
    # One-shots to use in loops (slightly randomized)
    k = synth_kick(dur=0.55, f0=95, f1=35)
    s = synth_snare(dur=0.33, tone=190)
    c = synth_clap(dur=0.30)
    hh = synth_hat(dur=0.10, open_hat=False)
    oh = synth_hat(dur=0.22, open_hat=True)
    rim = synth_rim(dur=0.12)

    loops = []
    # Trap 140 bpm, 2 bars
    bpm = 140
    loop, _ = make_loop(bpm, bars=2)
    # kick pattern
    for st in [0, 6, 8, 14, 16, 22, 24, 30]:
        loop = place_step(loop, k, bpm, st, gain=0.95)
    # snare on 2 and 4 (step 8, 24 in 16-step bar)
    for st in [8, 24]:
        loop = place_step(loop, s, bpm, st, gain=0.9)
        loop = place_step(loop, c, bpm, st, gain=0.5)
    # hats (with some rolls)
    for st in range(0, 32):
        if st % 2 == 0:
            loop = place_step(loop, hh, bpm, st, gain=0.35)
    for st in [13, 13.5, 13.75, 29, 29.5, 29.75]:
        # micro-rolls (substeps)
        step_dur = (60.0 / bpm) * 4 / 16
        start = int(st * step_dur * SR)
        loop = mix_at(loop, hh*0.25, start)
    loop = normalize(soft_clip(loop, drive=1.4))
    loops.append(("Loop_Trap_140bpm_2bar.wav", loop))

    # Grime 140 bpm, 2 bars (more space + rim + open hat)
    loop, _ = make_loop(140, bars=2)
    for st in [0, 3, 10, 16, 19, 26]:
        loop = place_step(loop, k, 140, st, gain=0.95)
    for st in [8, 24]:
        loop = place_step(loop, s, 140, st, gain=0.85)
    for st in [7, 15, 23, 31]:
        loop = place_step(loop, rim, 140, st, gain=0.55)
    for st in range(0, 32, 2):
        loop = place_step(loop, hh, 140, st, gain=0.28)
    for st in [12, 28]:
        loop = place_step(loop, oh, 140, st, gain=0.32)
    loop = normalize(soft_clip(loop, drive=1.5))
    loops.append(("Loop_Grime_140bpm_2bar.wav", loop))

    # Boom bap 90 bpm, 2 bars
    bpm = 90
    k2 = synth_kick(dur=0.70, f0=80, f1=38)
    s2 = synth_snare(dur=0.38, tone=170)
    hh2 = synth_hat(dur=0.12, open_hat=False)
    loop, _ = make_loop(bpm, bars=2)
    for st in [0, 5, 12, 16, 21, 28]:
        loop = place_step(loop, k2, bpm, st, gain=0.95)
    for st in [8, 24]:
        loop = place_step(loop, s2, bpm, st, gain=0.9)
    for st in range(0, 32):
        if st % 2 == 0:
            loop = place_step(loop, hh2, bpm, st, gain=0.26)
    loop = normalize(soft_clip(loop, drive=1.35))
    loops.append(("Loop_BoomBap_90bpm_2bar.wav", loop))

    return loops

# ---------------------------
# Main: build beatpack
# ---------------------------
def main():
    ensure_dir(OUT_DIR)
    subdirs = ["Drums", "Bass", "Wubs", "FX", "Loops", "Textures"]
    for sd in subdirs:
        ensure_dir(os.path.join(OUT_DIR, sd))

    # Drums
    drums = [
        ("Kick_GrimeA.wav", synth_kick(dur=0.55, f0=110, f1=35)),
        ("Kick_ThumpB.wav", synth_kick(dur=0.65, f0=85, f1=32)),
        ("Snare_CrackA.wav", synth_snare(dur=0.33, tone=200)),
        ("Snare_DirtyB.wav", normalize(soft_clip(synth_snare(dur=0.36, tone=170) + 0.15*noise(0.36), drive=1.7))),
        ("Clap_WideA.wav", synth_clap(dur=0.30)),
        ("Hat_ClosedA.wav", synth_hat(dur=0.10, open_hat=False)),
        ("Hat_ClosedB.wav", synth_hat(dur=0.08, open_hat=False)),
        ("Hat_OpenA.wav", synth_hat(dur=0.22, open_hat=True)),
        ("Rim_SnapA.wav", synth_rim(dur=0.12)),
        ("Crash_A.wav", synth_crash(dur=1.8)),
    ]
    for fn, x in drums:
        write_wav(os.path.join(OUT_DIR, "Drums", fn), x)

    # Bass
    bass = [
        ("808_A1.wav", synth_808(dur=1.3, note="A", octave=1, drive=3.1)),
        ("808_F1.wav", synth_808(dur=1.3, note="F", octave=1, drive=3.0)),
        ("808_C2.wav", synth_808(dur=1.1, note="C", octave=2, drive=2.9)),
        ("Reese_55hz.wav", synth_reese(dur=1.6, f=55)),
        ("Reese_46hz.wav", synth_reese(dur=1.6, f=46.25)),
    ]
    for fn, x in bass:
        write_wav(os.path.join(OUT_DIR, "Bass", fn), x)

    # Wubs
    wubs = [
        ("Wub_55hz_4hz.wav", synth_wub(dur=2.0, f=55, wobble_hz=4.0)),
        ("Wub_55hz_6hz.wav", synth_wub(dur=2.0, f=55, wobble_hz=6.0)),
        ("Wub_46hz_8hz.wav", synth_wub(dur=2.0, f=46.25, wobble_hz=8.0)),
    ]
    for fn, x in wubs:
        write_wav(os.path.join(OUT_DIR, "Wubs", fn), x)

    # FX
    fx = [
        ("Riser_A.wav", synth_riser(dur=2.5, start=220, end=2400)),
        ("Downlifter_A.wav", synth_downlifter(dur=1.6, start=2000, end=120)),
        ("Impact_A.wav", synth_impact(dur=1.0)),
    ]
    for fn, x in fx:
        write_wav(os.path.join(OUT_DIR, "FX", fn), x)

    # Textures
    tex = [
        ("Vinyl_Noise_4s.wav", synth_vinyl_noise(dur=4.0)),
    ]
    for fn, x in tex:
        write_wav(os.path.join(OUT_DIR, "Textures", fn), x)

    # Loops
    for fn, x in gen_loops():
        write_wav(os.path.join(OUT_DIR, "Loops", fn), x)

    print(f"Done. Beatpack created in: {OUT_DIR}")

if __name__ == "__main__":
    main()
