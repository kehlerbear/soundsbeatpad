import os, json, math, random
import numpy as np
import soundfile as sf
from scipy import signal

# -----------------------------
# Settings
# -----------------------------
SR = 44100
OUT = "ABSTRUCT_ALL_SAMPLES"

# How many variations (crank these up if you want MORE)
N_KICKS = 120
N_SNARES = 120
N_CLAPS = 90
N_HATS_C = 140
N_HATS_O = 90
N_PERCS = 120
N_RIMS = 60
N_CRASH = 40

N_IMPACTS = 60
N_RISERS = 60
N_DROPS = 60
N_TEXTURES = 50

N_WUBS = 120
N_REESE = 120
N_STABS = 120
N_DRONES = 60

# 808 notes (MIDI range)
MIDI_MIN = 28   # E0-ish
MIDI_MAX = 55   # G3-ish
SLIDES_PER_NOTE = 2  # slide variants per note

# Global seed (change this for a totally new pack)
SEED = 7

# -----------------------------
# Utils
# -----------------------------
def mkdir(p): os.makedirs(p, exist_ok=True)

def write_wav(path, x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    sf.write(path, x, SR, subtype="PCM_16")

def norm(x, peak=0.98):
    m = float(np.max(np.abs(x)) + 1e-12)
    return (x * (peak / m)).astype(np.float32)

def softclip(x, drive=2.0):
    return np.tanh(x * drive).astype(np.float32)

def adsr(n, a, d, s, r):
    a = max(1, int(a)); d = max(1, int(d)); r = max(1, int(r))
    if a + d + r > n:
        scale = n / (a + d + r + 1e-9)
        a = max(1, int(a * scale)); d = max(1, int(d * scale)); r = max(1, int(r * scale))
    sus = max(0, n - (a + d + r))
    A = np.linspace(0, 1, a, endpoint=False)
    D = np.linspace(1, s, d, endpoint=False)
    S = np.full(sus, s, dtype=np.float32)
    R = np.linspace(s, 0, r, endpoint=True)
    e = np.concatenate([A, D, S, R]).astype(np.float32)
    if len(e) < n:
        e = np.pad(e, (0, n - len(e)))
    return e[:n]

def noise(n):
    return np.random.uniform(-1, 1, n).astype(np.float32)

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def butter(mode, cutoff_hz, order=4):
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, SR/2 - 200))
    wn = cutoff_hz / (SR/2)
    b, a = signal.butter(order, wn, btype=("low" if mode == "lp" else "high"))
    return b, a

def filt(x, b, a):
    return signal.lfilter(b, a, x).astype(np.float32)

def detune_saw(t, f, cents=8):
    det = 2 ** (cents / 1200.0)
    s1 = 2.0 * (t * f - np.floor(0.5 + t * f))
    s2 = 2.0 * (t * (f * det) - np.floor(0.5 + t * (f * det)))
    return 0.5 * (s1 + s2)

# -----------------------------
# Drum synths (variations)
# -----------------------------
def make_kick():
    dur = random.uniform(0.45, 0.85)
    n = int(SR * dur)
    t = np.arange(n) / SR

    f0 = random.uniform(110, 170)
    f1 = random.uniform(38, 60)
    drop = random.uniform(10, 22)
    f = f1 + (f0 - f1) * np.exp(-t * drop)

    phase = 2*np.pi*np.cumsum(f) / SR
    body = np.sin(phase)

    click_amt = random.uniform(0.15, 0.55)
    b_hp, a_hp = butter("hp", random.uniform(2200, 4500), order=3)
    click = filt(noise(n), b_hp, a_hp) * adsr(n, SR*0.001, SR*0.01, 0.0, SR*0.02)

    e = adsr(n, SR*0.001, SR*random.uniform(0.03,0.09), 0.0, SR*random.uniform(0.18,0.40))
    x = body * e + click_amt * click

    drive = random.uniform(1.5, 3.5)
    x = softclip(x, drive)
    return norm(x)

def make_snare():
    dur = random.uniform(0.35, 0.75)
    n = int(SR * dur)
    t = np.arange(n) / SR

    # noisy body
    b_hp, a_hp = butter("hp", random.uniform(1100, 2400), order=3)
    nse = filt(noise(n), b_hp, a_hp)
    nse *= adsr(n, SR*0.001, SR*random.uniform(0.02,0.05), 0.0, SR*random.uniform(0.12,0.25))

    # tone
    tone_f = random.uniform(160, 260)
    tone = np.sin(2*np.pi*tone_f*t) * adsr(n, SR*0.001, SR*random.uniform(0.03,0.07), 0.0, SR*random.uniform(0.10,0.18))

    x = random.uniform(0.45,0.7)*nse + random.uniform(0.15,0.45)*tone
    x = softclip(x, random.uniform(1.6, 3.2))
    return norm(x)

def make_clap():
    dur = random.uniform(0.35, 0.65)
    n = int(SR * dur)
    x = np.zeros(n, dtype=np.float32)
    b_hp, a_hp = butter("hp", random.uniform(1400, 2600), order=3)

    # several bursts
    bursts = random.randint(3, 6)
    for i in range(bursts):
        off_ms = random.uniform(0, 80)
        off = int(SR * off_ms / 1000.0)
        if off >= n: 
            continue
        burst = filt(noise(n - off), b_hp, a_hp)
        e = adsr(n - off, SR*0.001, SR*random.uniform(0.01,0.03), 0.0, SR*random.uniform(0.06,0.12))
        x[off:] += burst * e * random.uniform(0.6, 1.0)

    x = softclip(x, random.uniform(1.4, 2.6))
    return norm(x)

def make_hat(open_hat=False):
    dur = random.uniform(0.06, 0.16) if not open_hat else random.uniform(0.18, 0.60)
    n = int(SR * dur)
    x = noise(n)

    b_hp, a_hp = butter("hp", random.uniform(5200, 8200), order=3)
    x = filt(x, b_hp, a_hp)

    rel = random.uniform(0.04, 0.10) if not open_hat else random.uniform(0.12, 0.35)
    e = adsr(n, SR*0.001, SR*random.uniform(0.006, 0.02), 0.0, SR*rel)
    x *= e

    # slight grit
    if random.random() < 0.6:
        x = softclip(x, random.uniform(1.2, 2.0))

    return norm(x)

def make_rim():
    dur = random.uniform(0.12, 0.35)
    n = int(SR * dur)
    t = np.arange(n)/SR
    f1 = random.uniform(700, 1200)
    f2 = random.uniform(1300, 2400)
    tone = np.sin(2*np.pi*f1*t) + 0.6*np.sin(2*np.pi*f2*t)
    e = adsr(n, SR*0.001, SR*random.uniform(0.008,0.02), 0.0, SR*random.uniform(0.05,0.12))
    hat = make_hat(False)
    if len(hat) < n:
        hat = np.pad(hat, (0, n - len(hat)))
    else:
        hat = hat[:n]
    x = tone*e + 0.12*hat*e
    x = softclip(x, random.uniform(1.5, 2.5))
    return norm(x)

def make_perc():
    dur = random.uniform(0.12, 0.55)
    n = int(SR * dur)
    t = np.arange(n)/SR

    mode = random.choice(["metal", "wood", "click", "tom"])
    if mode == "tom":
        f = random.uniform(90, 220)
        x = np.sin(2*np.pi*f*t)
        x *= adsr(n, SR*0.002, SR*0.05, 0.0, SR*random.uniform(0.12,0.25))
        x = softclip(x, random.uniform(1.2, 2.4))
        return norm(x)

    if mode == "click":
        x = noise(n)
        b_hp, a_hp = butter("hp", random.uniform(2500, 6500), order=3)
        x = filt(x, b_hp, a_hp)
        x *= adsr(n, SR*0.001, SR*0.01, 0.0, SR*0.05)
        return norm(x)

    # resonant-ish percussion
    f = random.uniform(240, 1200) if mode == "wood" else random.uniform(350, 1800)
    x = detune_saw(t, f, cents=random.uniform(4, 16))
    b_lp, a_lp = butter("lp", random.uniform(1800, 6500), order=3)
    x = filt(x, b_lp, a_lp)
    x *= adsr(n, SR*0.001, SR*random.uniform(0.01,0.04), 0.0, SR*random.uniform(0.10,0.30))
    x = softclip(x, random.uniform(1.4, 3.0))
    return norm(x)

def make_crash():
    dur = random.uniform(0.8, 2.4)
    n = int(SR * dur)
    x = noise(n)
    b_hp, a_hp = butter("hp", random.uniform(2500, 4500), order=3)
    x = filt(x, b_hp, a_hp)
    b_lp, a_lp = butter("lp", random.uniform(9000, 14000), order=3)
    x = filt(x, b_lp, a_lp)
    x *= adsr(n, SR*0.002, SR*0.06, random.uniform(0.1, 0.4), SR*random.uniform(0.6, 1.4))
    x = softclip(x, random.uniform(1.2, 2.0))
    return norm(x)

# -----------------------------
# Bass synths
# -----------------------------
def make_808(midi_note, long=True):
    dur = random.uniform(1.1, 2.2) if long else random.uniform(0.45, 1.0)
    n = int(SR * dur)
    t = np.arange(n)/SR

    f = midi_to_hz(midi_note)
    # slight pitch drop
    f_t = f * (1.0 + random.uniform(0.02,0.08)*np.exp(-t*random.uniform(7, 14)))
    phase = 2*np.pi*np.cumsum(f_t)/SR
    x = np.sin(phase)

    e = adsr(n, SR*0.004, SR*random.uniform(0.05,0.10), random.uniform(0.25,0.7) if long else 0.0, SR*random.uniform(0.20,0.55))
    x *= e

    # saturation + sub-control
    x = softclip(x, random.uniform(2.0, 4.0))
    b_lp, a_lp = butter("lp", random.uniform(160, 280), order=3)
    x = filt(x, b_lp, a_lp)
    return norm(x)

def make_808_slide(midi_a, midi_b):
    dur = random.uniform(0.9, 1.8)
    n = int(SR*dur)
    t = np.arange(n)/SR

    f1 = midi_to_hz(midi_a)
    f2 = midi_to_hz(midi_b)
    # glide curve
    curve = (t / dur) ** random.uniform(0.6, 1.4)
    f = f1 + (f2 - f1) * curve

    phase = 2*np.pi*np.cumsum(f)/SR
    x = np.sin(phase)

    e = adsr(n, SR*0.004, SR*random.uniform(0.05,0.10), random.uniform(0.2,0.6), SR*random.uniform(0.20,0.55))
    x *= e
    x = softclip(x, random.uniform(2.2, 4.5))
    b_lp, a_lp = butter("lp", random.uniform(160, 260), order=3)
    x = filt(x, b_lp, a_lp)
    return norm(x)

def make_wub():
    dur = random.uniform(1.0, 2.5)
    n = int(SR*dur)
    t = np.arange(n)/SR

    base = random.uniform(40, 70)
    osc = 0.55*detune_saw(t, base, cents=random.uniform(6, 18)) + 0.35*np.sin(2*np.pi*base*t)

    # LFO wobble affects filter cutoff
    lfo_rate = random.choice([1,2,3,4,6,8]) * random.choice([0.5, 1.0])
    lfo = 0.5*(1 + np.sin(2*np.pi*lfo_rate*t + random.uniform(0, 2*np.pi)))
    cutoff = random.uniform(90, 160) + lfo * random.uniform(600, 2400)

    # dynamic filtering (block)
    y = np.zeros_like(osc)
    prev = 0.0
    block = 256
    for i in range(0, n, block):
        c = float(np.mean(cutoff[i:i+block]))
        # one-pole style inside loop
        a = math.exp(-2.0*math.pi*max(30.0, c)/SR)
        for j in range(i, min(i+block, n)):
            prev = (1-a)*osc[j] + a*prev
            y[j] = prev

    # add sub + grime
    y += 0.35*np.sin(2*np.pi*base*t)
    y = softclip(y, random.uniform(2.0, 3.8))

    # optional highpass for mid-wub style
    if random.random() < 0.35:
        b_hp, a_hp = butter("hp", random.uniform(40, 110), order=2)
        y = filt(y, b_hp, a_hp)

    e = adsr(n, SR*0.01, SR*0.12, random.uniform(0.6, 0.95), SR*0.25)
    return norm(y*e)

def make_reese():
    dur = random.uniform(0.8, 2.2)
    n = int(SR*dur)
    t = np.arange(n)/SR
    f = random.uniform(45, 85)

    x = 0.7*detune_saw(t, f, cents=random.uniform(12, 30)) + 0.35*detune_saw(t, f*2, cents=random.uniform(6, 16))
    # slow phaser-ish movement using a moving lowpass cutoff
    lfo = 0.5*(1 + np.sin(2*np.pi*random.uniform(0.15, 0.6)*t))
    cutoff = 350 + lfo*random.uniform(800, 2400)
    y = np.zeros_like(x)
    prev = 0.0
    for i in range(n):
        c = float(cutoff[i])
        a = math.exp(-2.0*math.pi*c/SR)
        prev = (1-a)*x[i] + a*prev
        y[i] = prev

    y = softclip(y, random.uniform(1.8, 3.2))
    y *= adsr(n, SR*0.01, SR*0.10, random.uniform(0.6, 0.95), SR*0.20)
    return norm(y)

# -----------------------------
# Melodic / FX
# -----------------------------
def make_stab():
    dur = random.uniform(0.25, 1.0)
    n = int(SR*dur)
    t = np.arange(n)/SR
    f = midi_to_hz(random.choice([48,50,52,55,57,60,62]))
    x = detune_saw(t, f, cents=random.uniform(10, 22)) + 0.5*detune_saw(t, f*2, cents=random.uniform(6, 16))
    b_lp, a_lp = butter("lp", random.uniform(1200, 5200), order=3)
    x = filt(x, b_lp, a_lp)
    x = softclip(x, random.uniform(2.0, 4.0))
    e = adsr(n, SR*0.002, SR*0.08, 0.0, SR*random.uniform(0.10, 0.35))
    return norm(x*e)

def make_drone():
    dur = random.uniform(2.0, 6.0)
    n = int(SR*dur)
    t = np.arange(n)/SR
    f = random.uniform(30, 70)
    x = detune_saw(t, f, cents=random.uniform(4, 14)) + 0.4*detune_saw(t, f*1.5, cents=random.uniform(6, 18))
    b_lp, a_lp = butter("lp", random.uniform(600, 2600), order=3)
    x = filt(x, b_lp, a_lp)
    x = softclip(x, random.uniform(1.6, 2.6))
    e = adsr(n, SR*0.25, SR*0.5, random.uniform(0.6, 0.95), SR*0.8)
    return norm(x*e, peak=0.85)

def make_impact():
    dur = random.uniform(0.5, 1.2)
    n = int(SR*dur)
    t = np.arange(n)/SR
    thump_f = random.uniform(38, 65)
    thump = np.sin(2*np.pi*thump_f*t) * adsr(n, SR*0.002, SR*0.08, 0.0, SR*0.45)
    hit = noise(n)
    b_hp, a_hp = butter("hp", random.uniform(900, 1800), order=3)
    hit = filt(hit, b_hp, a_hp) * adsr(n, SR*0.001, SR*0.02, 0.0, SR*0.25)
    x = softclip(thump*0.9 + hit*0.7, random.uniform(2.0, 3.6))
    return norm(x)

def make_riser():
    dur = random.uniform(1.0, 3.5)
    n = int(SR*dur)
    t = np.arange(n)/SR
    x = noise(n)
    b_hp, a_hp = butter("hp", random.uniform(300, 1200), order=2)
    x = filt(x, b_hp, a_hp)

    # opening lowpass
    cutoff_start = random.uniform(400, 1200)
    cutoff_end = random.uniform(8000, 15000)
    cutoff = cutoff_start + (cutoff_end - cutoff_start) * (t / dur) ** random.uniform(0.8, 1.4)

    y = np.zeros_like(x)
    prev = 0.0
    for i in range(n):
        c = float(cutoff[i])
        a = math.exp(-2.0*math.pi*c/SR)
        prev = (1-a)*x[i] + a*prev
        y[i] = prev

    y *= adsr(n, SR*0.01, SR*0.10, 1.0, SR*0.25)
    y = softclip(y, random.uniform(1.2, 2.2))
    return norm(y)

def make_drop():
    # reverse-ish impact / downer
    x = make_impact()
    x = x[::-1].copy()
    x = softclip(x, random.uniform(1.2, 2.2))
    return norm(x)

def make_texture():
    dur = random.uniform(2.0, 6.0)
    n = int(SR*dur)
    x = noise(n) * random.uniform(0.08, 0.18)
    # vinyl-ish pops
    for _ in range(random.randint(10, 35)):
        i = random.randint(0, n-200)
        pop = noise(200)
        b_hp, a_hp = butter("hp", random.uniform(1200, 2600), order=2)
        pop = filt(pop, b_hp, a_hp)
        pop *= adsr(200, 1, 20, 0.0, 120)
        x[i:i+200] += pop * random.uniform(0.3, 0.9)
    b_lp, a_lp = butter("lp", random.uniform(6000, 12000), order=2)
    x = filt(x, b_lp, a_lp)
    return norm(x, peak=0.6)

# -----------------------------
# Build pack
# -----------------------------
def build():
    random.seed(SEED)
    np.random.seed(SEED)

    # folders
    paths = {
        "KICKS": os.path.join(OUT, "DRUMS", "KICKS"),
        "SNARES": os.path.join(OUT, "DRUMS", "SNARES"),
        "CLAPS": os.path.join(OUT, "DRUMS", "CLAPS"),
        "HATS_C": os.path.join(OUT, "DRUMS", "HATS_CLOSED"),
        "HATS_O": os.path.join(OUT, "DRUMS", "HATS_OPEN"),
        "PERCS": os.path.join(OUT, "DRUMS", "PERCS"),
        "RIMS": os.path.join(OUT, "DRUMS", "RIMS"),
        "CRASH": os.path.join(OUT, "DRUMS", "CRASH"),

        "808S": os.path.join(OUT, "BASS", "808S"),
        "808S_LONG": os.path.join(OUT, "BASS", "808S_LONG"),
        "808_SLIDES": os.path.join(OUT, "BASS", "808_SLIDES"),
        "WUBS": os.path.join(OUT, "BASS", "WUBS"),
        "REESE": os.path.join(OUT, "BASS", "REESE"),

        "STABS": os.path.join(OUT, "MELODIC", "STABS"),
        "DRONES": os.path.join(OUT, "MELODIC", "DRONES"),

        "IMPACTS": os.path.join(OUT, "FX", "IMPACTS"),
        "RISERS": os.path.join(OUT, "FX", "RISERS"),
        "DROPS": os.path.join(OUT, "FX", "DROPS"),
        "TEXTURES": os.path.join(OUT, "FX", "TEXTURES"),
    }
    for pth in paths.values():
        mkdir(pth)

    manifest = {"sample_rate": SR, "seed": SEED, "items": []}

    def emit(folder_key, name, audio, meta=None):
        path = os.path.join(paths[folder_key], f"{name}.wav")
        write_wav(path, audio)
        item = {"path": path.replace("\\", "/")}
        if meta:
            item.update(meta)
        manifest["items"].append(item)

    # Drums
    for i in range(N_KICKS):
        emit("KICKS", f"kick_{i:03d}", make_kick(), {"type":"one-shot","cat":"kick"})
    for i in range(N_SNARES):
        emit("SNARES", f"snare_{i:03d}", make_snare(), {"type":"one-shot","cat":"snare"})
    for i in range(N_CLAPS):
        emit("CLAPS", f"clap_{i:03d}", make_clap(), {"type":"one-shot","cat":"clap"})
    for i in range(N_HATS_C):
        emit("HATS_C", f"hatC_{i:03d}", make_hat(False), {"type":"one-shot","cat":"hat_closed"})
    for i in range(N_HATS_O):
        emit("HATS_O", f"hatO_{i:03d}", make_hat(True), {"type":"one-shot","cat":"hat_open"})
    for i in range(N_PERCS):
        emit("PERCS", f"perc_{i:03d}", make_perc(), {"type":"one-shot","cat":"perc"})
    for i in range(N_RIMS):
        emit("RIMS", f"rim_{i:03d}", make_rim(), {"type":"one-shot","cat":"rim"})
    for i in range(N_CRASH):
        emit("CRASH", f"crash_{i:03d}", make_crash(), {"type":"one-shot","cat":"crash"})

    # 808s (notes)
    for m in range(MIDI_MIN, MIDI_MAX + 1):
        emit("808S", f"808_{m:02d}", make_808(m, long=False), {"type":"one-shot","cat":"808_short","midi":m})
        emit("808S_LONG", f"808L_{m:02d}", make_808(m, long=True), {"type":"one-shot","cat":"808_long","midi":m})

        for s in range(SLIDES_PER_NOTE):
            m2 = int(np.clip(m + random.choice([-7,-5,-3,3,5,7,12]), MIDI_MIN, MIDI_MAX))
            emit("808_SLIDES", f"slide_{m:02d}_to_{m2:02d}_{s}", make_808_slide(m, m2),
                 {"type":"one-shot","cat":"808_slide","midi_from":m,"midi_to":m2})

    # Bass textures
    for i in range(N_WUBS):
        emit("WUBS", f"wub_{i:03d}", make_wub(), {"type":"one-shot","cat":"wub"})
    for i in range(N_REESE):
        emit("REESE", f"reese_{i:03d}", make_reese(), {"type":"one-shot","cat":"reese"})

    # Melodic
    for i in range(N_STABS):
        emit("STABS", f"stab_{i:03d}", make_stab(), {"type":"one-shot","cat":"stab"})
    for i in range(N_DRONES):
        emit("DRONES", f"drone_{i:03d}", make_drone(), {"type":"loop","cat":"drone"})

    # FX
    for i in range(N_IMPACTS):
        emit("IMPACTS", f"impact_{i:03d}", make_impact(), {"type":"fx","cat":"impact"})
    for i in range(N_RISERS):
        emit("RISERS", f"riser_{i:03d}", make_riser(), {"type":"fx","cat":"riser"})
    for i in range(N_DROPS):
        emit("DROPS", f"drop_{i:03d}", make_drop(), {"type":"fx","cat":"drop"})
    for i in range(N_TEXTURES):
        emit("TEXTURES", f"texture_{i:03d}", make_texture(), {"type":"fx","cat":"texture"})

    # Save manifest
    mkdir(OUT)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("DONE.")
    print(f"Output folder: {OUT}")
    print(f"Total WAVs: {len(manifest['items'])}")
    print("Tip: change SEED to generate a whole new pack instantly.")

if __name__ == "__main__":
    build()
