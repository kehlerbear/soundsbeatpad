import os
import sys
import time
import shutil
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from scipy import signal

# -----------------------------
# Audio settings
# -----------------------------
SR = 44100           # output sample rate
BLOCK = 256          # lower = lower latency; try 512 if crackling
MASTER = 0.90        # master volume
LIMITER = 0.98       # hard limiter (prevents clipping)
RECORD_NORMALIZE = True  # normalize recording on save
RECORD_PEAK = 0.98

# -----------------------------
# Paths + Key Mapping
# -----------------------------
def resource_root() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.normpath(sys._MEIPASS)
    return os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

def writable_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.normpath(os.path.dirname(sys.executable))
    return os.path.normpath(os.path.dirname(os.path.abspath(__file__)))

def resolve_sound_root() -> str:
    base = resource_root()
    candidates = [
        os.path.join(base, "sounds", "ABSTRUCT_Beatpack_Python"),
        os.path.join(base, "ABSTRUCT_Beatpack_Python"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.normpath(c)
    return os.path.normpath(candidates[0])

APP_ROOT = writable_root()
ROOT = resolve_sound_root()
EXTRA_ROOTS = []
RECORDINGS_DIR = os.path.join(APP_ROOT, "recordings")

def p(*parts):
    return os.path.normpath(os.path.join(*parts))

# You can change these mappings anytime:
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")

PAD_KEYS = [
    "q", "w", "e", "r",
    "a", "s", "d", "f",
    "z", "x", "c", "v",
    "1", "2", "3", "4",
]

GLOBAL_KEYS = (
    list("1234567890")
    + list("qwertyuiop")
    + list("asdfghjkl")
    + list("zxcvbnm")
)
KEY_LIMIT = len(GLOBAL_KEYS)

SOUNDS = {
    # top row
    "q": p(ROOT, "Drums", "Kick_GrimeA.wav"),
    "w": p(ROOT, "Drums", "Snare_CrackA.wav"),
    "e": p(ROOT, "Drums", "Clap_WideA.wav"),
    "r": p(ROOT, "Drums", "Rim_SnapA.wav"),

    # home row
    "a": p(ROOT, "Drums", "Hat_ClosedA.wav"),
    "s": p(ROOT, "Drums", "Hat_OpenA.wav"),
    "d": p(ROOT, "Drums", "Crash_A.wav"),
    "f": p(ROOT, "FX", "Impact_A.wav"),

    # bass row
    "z": p(ROOT, "Bass", "808_C2.wav"),
    "x": p(ROOT, "Bass", "808_A1.wav"),
    "c": p(ROOT, "Bass", "808_F1.wav"),
    "v": p(ROOT, "Wubs", "Wub_55hz_6hz.wav"),

    # extra row
    "1": p(ROOT, "FX", "Riser_A.wav"),
    "2": p(ROOT, "FX", "Downlifter_A.wav"),
    "3": p(ROOT, "Loops", "Loop_Trap_140bpm_2bar.wav"),
    "4": p(ROOT, "Textures", "Vinyl_Noise_4s.wav"),
}

# -----------------------------
# SciPy FX helpers
# -----------------------------
def resample_to(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    # resample_poly is high quality and fast
    g = np.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    y = signal.resample_poly(x, up, down).astype(np.float32)
    return y

def butter_filter(x: np.ndarray, mode: str, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    cutoff_hz = max(20.0, min(float(cutoff_hz), sr/2 - 100))
    nyq = 0.5 * sr
    wn = cutoff_hz / nyq
    b, a = signal.butter(order, wn, btype=("low" if mode == "lp" else "high"))
    return signal.lfilter(b, a, x).astype(np.float32)

def softclip(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(x * drive).astype(np.float32)

def pitch_shift_resample(x: np.ndarray, semitones: int, sr: int) -> np.ndarray:
    """
    Simple pitch shift via resampling:
    - pitch up -> shorter
    - pitch down -> longer
    Great for one-shots/808s. For loops, it will change length (which can be cool).
    """
    if semitones == 0:
        return x
    ratio = 2 ** (semitones / 12.0)
    # Resample to change pitch
    new_len = max(1, int(len(x) / ratio))
    y = signal.resample(x, new_len).astype(np.float32)
    return y

# -----------------------------
# Sample Cache
# -----------------------------
CACHE = {}  # path -> np.float32 mono at SR
CACHE_INFO = {}  # path -> dict

def load_sample(path: str) -> np.ndarray:
    if path in CACHE:
        return CACHE[path]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=1)  # to mono

    data = np.asarray(data, dtype=np.float32)
    data = resample_to(data, sr, SR)

    # tiny fade to avoid clicks at edges
    fade = min(64, len(data))
    if fade > 2:
        w = np.linspace(0, 1, fade, dtype=np.float32)
        data[:fade] *= w
        data[-fade:] *= w[::-1]

    CACHE[path] = data
    CACHE_INFO[path] = {"src_sr": sr, "len": len(data)}
    return data

def get_sample_length_sec(path: str) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    info = CACHE_INFO.get(path)
    if info and "len" in info:
        return float(info["len"]) / SR
    try:
        sf_info = sf.info(path)
        if sf_info.samplerate > 0:
            return float(sf_info.frames) / float(sf_info.samplerate)
    except Exception:
        pass
    return 0.0

# -----------------------------
# Voice Engine (polyphonic mixing)
# -----------------------------
class Voice:
    __slots__ = ("audio", "pos", "gain")
    def __init__(self, audio: np.ndarray, gain: float = 1.0):
        self.audio = audio
        self.pos = 0
        self.gain = float(gain)

class Engine:
    def __init__(self):
        self.voices = []
        self.lock = threading.Lock()

        self.recording = False
        self.rec_buf = []   # list of blocks
        self.rec_start_time = None

        # FX toggles / params
        self.fx_lp = False
        self.fx_hp = False
        self.fx_drive = False
        self.lp_cutoff = 900.0
        self.hp_cutoff = 120.0
        self.drive = 2.2

        # pitch for next trigger
        self.pitch_semitones = 0
        self.last_recording_path = None

        # playback state (for saved recordings)
        self.playback_audio = None
        self.playback_pos = 0
        self.playback_active = False
        self.playback_paused = False
        self.playback_loop = False
        self.playback_start = 0
        self.playback_end = 0

    def trigger(self, sample: np.ndarray, gain: float = 1.0):
        # apply pitch shift for this trigger (does not affect cache)
        audio = sample
        if self.pitch_semitones != 0:
            audio = pitch_shift_resample(sample, self.pitch_semitones, SR)

        v = Voice(audio=audio, gain=gain)
        with self.lock:
            self.voices.append(v)

    def toggle_record(self):
        with self.lock:
            if not self.recording:
                self.recording = True
                self.rec_buf = []
                self.rec_start_time = time.time()
                return True
            else:
                self.recording = False
                return False

    def get_recording(self) -> np.ndarray:
        with self.lock:
            if not self.rec_buf:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self.rec_buf).astype(np.float32)

    def start_playback(self, audio: np.ndarray, start: int = 0, end: int | None = None):
        with self.lock:
            self.playback_audio = audio
            self.playback_start = max(0, min(int(start), len(audio)))
            self.playback_end = max(self.playback_start, min(int(end) if end is not None else len(audio), len(audio)))
            self.playback_pos = self.playback_start
            self.playback_active = True
            self.playback_paused = False

    def pause_playback(self):
        with self.lock:
            if self.playback_active:
                self.playback_paused = True

    def resume_playback(self):
        with self.lock:
            if self.playback_active:
                self.playback_paused = False

    def stop_playback(self):
        with self.lock:
            self.playback_active = False
            self.playback_paused = False
            self.playback_audio = None
            self.playback_pos = 0

ENGINE = Engine()
LAST_PEAK = 0.0

# -----------------------------
# Audio callback
# -----------------------------
def audio_callback(outdata, frames, time_info, status):
    global LAST_PEAK
    if status:
        # Print once in a while if needed; keep callback light
        pass

    mix = np.zeros(frames, dtype=np.float32)

    with ENGINE.lock:
        alive = []
        for v in ENGINE.voices:
            remaining = len(v.audio) - v.pos
            if remaining <= 0:
                continue
            take = min(frames, remaining)
            mix[:take] += v.audio[v.pos:v.pos+take] * v.gain
            v.pos += take
            if v.pos < len(v.audio):
                alive.append(v)
        ENGINE.voices = alive

        if ENGINE.playback_active and ENGINE.playback_audio is not None and not ENGINE.playback_paused:
            end_pos = ENGINE.playback_end if ENGINE.playback_end > 0 else len(ENGINE.playback_audio)
            if ENGINE.playback_pos < ENGINE.playback_start:
                ENGINE.playback_pos = ENGINE.playback_start
            remaining = end_pos - ENGINE.playback_pos
            if remaining > 0:
                take = min(frames, remaining)
                mix[:take] += ENGINE.playback_audio[ENGINE.playback_pos:ENGINE.playback_pos+take]
                ENGINE.playback_pos += take
                if ENGINE.playback_pos >= end_pos:
                    if ENGINE.playback_loop:
                        ENGINE.playback_pos = ENGINE.playback_start
                    else:
                        ENGINE.playback_active = False
                        ENGINE.playback_audio = None
                        ENGINE.playback_pos = 0
            else:
                if ENGINE.playback_loop:
                    ENGINE.playback_pos = ENGINE.playback_start
                else:
                    ENGINE.playback_active = False
                    ENGINE.playback_audio = None
                    ENGINE.playback_pos = 0

    # FX on the master
    y = mix

    if ENGINE.fx_hp:
        y = butter_filter(y, "high", ENGINE.hp_cutoff, SR, order=3)
    if ENGINE.fx_lp:
        y = butter_filter(y, "low", ENGINE.lp_cutoff, SR, order=3)
    if ENGINE.fx_drive:
        y = softclip(y, ENGINE.drive)

    # master + limiter
    y *= MASTER
    y = np.clip(y, -LIMITER, LIMITER)

    # record exactly what we output
    with ENGINE.lock:
        if ENGINE.recording:
            ENGINE.rec_buf.append(y.copy())

    outdata[:] = y.reshape(-1, 1)
    LAST_PEAK = float(np.max(np.abs(y))) if len(y) else 0.0

# -----------------------------
# UI (Tkinter)
# -----------------------------
def scan_samples(root_dir: str) -> list:
    samples = []
    if not os.path.isdir(root_dir):
        return samples
    for base, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(AUDIO_EXTS):
                samples.append(os.path.join(base, name))
    return sorted(samples)

def scan_all_samples() -> list:
    roots = [ROOT] + EXTRA_ROOTS
    all_samples = []
    for r in roots:
        all_samples.extend(scan_samples(r))
    return sorted(set(all_samples))

def friendly_name(path: str) -> str:
    return os.path.basename(path) if path else "(empty)"

def short_name(path: str, max_len: int = 12) -> str:
    if not path:
        return "(empty)"
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace("_", " ").strip()
    if len(name) > max_len:
        return name[: max_len - 1].strip() + "…"
    return name

def neon_color_for_key(key: str) -> str:
    palette = [
        "#39ff14", "#00e5ff", "#ff2e63", "#ff9100", "#b517ff",
        "#00ffb3", "#ffea00", "#ff4dff", "#00ffa2", "#7cff00",
        "#ff6f00", "#00c8ff",
    ]
    if not key:
        return "#00e5ff"
    idx = abs(hash(key)) % len(palette)
    return palette[idx]

def contrast_text_color(hex_color: str) -> str:
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except Exception:
        return "#0b0f1a"
    # perceived brightness
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#0b0f1a" if brightness > 140 else "#e6f1ff"

def dim_color(hex_color: str, factor: float = 0.55) -> str:
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except Exception:
        return hex_color
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

def load_audio_file(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    return resample_to(data, sr, SR)

def build_ui():
    root = tk.Tk()
    root.title("SOUNDsbeatpad")
    root.geometry("1380x820")
    root.minsize(1200, 720)

    # Neon studio theme
    COL_BG = "#0b0f1a"
    COL_PANEL = "#101827"
    COL_ACCENT = "#39ff14"
    COL_ACCENT_2 = "#00e5ff"
    COL_TEXT = "#e6f1ff"
    COL_MUTED = "#7a8aa0"
    COL_WARN = "#ff2e63"

    root.configure(bg=COL_BG)
    style = ttk.Style(root)

    menubar = tk.Menu(root)
    help_menu = tk.Menu(menubar, tearoff=0)
    root.config(menu=menubar)

    status_var = tk.StringVar(value="Loading samples...")
    rec_var = tk.StringVar(value="REC: off")
    pb_var = tk.StringVar(value="Playback: stopped")
    pitch_var = tk.IntVar(value=ENGINE.pitch_semitones)
    lp_var = tk.BooleanVar(value=ENGINE.fx_lp)
    hp_var = tk.BooleanVar(value=ENGINE.fx_hp)
    drive_var = tk.BooleanVar(value=ENGINE.fx_drive)
    loop_var = tk.BooleanVar(value=ENGINE.playback_loop)
    master_var = tk.DoubleVar(value=MASTER)
    selected_pad = tk.StringVar(value=PAD_KEYS[0])

    # Top status bar
    top = tk.Frame(root, bg=COL_BG)
    top.pack(fill="x", padx=10, pady=8)

    tk.Label(top, text="The world is yours.", font=("Arial", 13), bg=COL_BG, fg=COL_TEXT).pack(side="left")
    meter_canvas = tk.Canvas(top, width=220, height=18, bg=COL_PANEL, highlightthickness=0)
    meter_canvas.pack(side="right", padx=6)
    tk.Label(top, textvariable=pb_var, font=("Arial", 11), bg=COL_BG, fg=COL_TEXT).pack(side="right", padx=12)
    tk.Label(top, textvariable=rec_var, font=("Arial", 11), bg=COL_BG, fg=COL_TEXT).pack(side="right", padx=12)

    app = tk.Frame(root, bg=COL_BG)
    app.pack(fill="both", expand=True)

    sidebar = tk.Frame(app, width=150, bg=COL_PANEL)
    sidebar.pack(side="left", fill="y", padx=(10, 6), pady=8)

    content = tk.Frame(app, bg=COL_BG)
    content.pack(side="right", fill="both", expand=True)

    main_notebook = ttk.Notebook(content)
    main_notebook.pack(fill="both", expand=True, padx=6, pady=6)

    record_tab = tk.Frame(main_notebook)
    library_tab = tk.Frame(main_notebook)

    main_notebook.add(record_tab, text="Studio")
    main_notebook.add(library_tab, text="Library")

    def go_tab(tab):
        main_notebook.select(tab)

    tk.Label(sidebar, text="Studio", font=("Arial", 13, "bold"), bg=COL_PANEL, fg=COL_TEXT).pack(fill="x", pady=(0, 8))
    tk.Button(sidebar, text="Studio", command=lambda: go_tab(record_tab), bg=COL_PANEL, fg=COL_TEXT, activebackground=COL_ACCENT, activeforeground=COL_BG).pack(fill="x", pady=2)
    tk.Button(sidebar, text="Library", command=lambda: go_tab(library_tab), bg=COL_PANEL, fg=COL_TEXT, activebackground=COL_ACCENT, activeforeground=COL_BG).pack(fill="x", pady=2)

    help_btn = tk.Button(sidebar, text="Help", bg=COL_PANEL, fg=COL_TEXT, activebackground=COL_ACCENT_2, activeforeground=COL_BG)
    help_btn.pack(fill="x", pady=(10, 2))
    save_btn = tk.Button(sidebar, text="Save Take...", bg=COL_PANEL, fg=COL_TEXT, activebackground=COL_ACCENT, activeforeground=COL_BG)
    save_btn.pack(fill="x", pady=2)
    open_btn = tk.Button(sidebar, text="Open Take...", bg=COL_PANEL, fg=COL_TEXT, activebackground=COL_ACCENT, activeforeground=COL_BG)
    open_btn.pack(fill="x", pady=2)

    closed_panel_frame = tk.LabelFrame(sidebar, text="Closed Panels", bg=COL_PANEL, fg=COL_TEXT)
    closed_panel_frame.pack(fill="both", expand=True, pady=(12, 6))
    closed_list = tk.Listbox(closed_panel_frame, height=6)
    closed_list.pack(fill="both", expand=True, padx=6, pady=6)
    closed_btns = tk.Frame(closed_panel_frame, bg=COL_PANEL)
    closed_btns.pack(fill="x", padx=6, pady=(0, 6))

    section_states = {}

    def update_closed_list():
        closed_list.delete(0, tk.END)
        for title, data in section_states.items():
            if not data["state"].get():
                closed_list.insert(tk.END, title)

    def get_closed_selection():
        sel = closed_list.curselection()
        if sel:
            return sel[0]
        active = closed_list.index(tk.ACTIVE)
        if active is not None and active >= 0 and closed_list.size() > 0:
            return active
        if closed_list.size() > 0:
            return 0
        return None

    def reopen_closed_panel(_event=None):
        idx = get_closed_selection()
        if idx is None:
            return
        title = closed_list.get(idx)
        data = section_states.get(title)
        if not data:
            return
        data["state"].set(True)
        data["update"]()
        update_closed_list()

    closed_list.bind("<Double-Button-1>", reopen_closed_panel)
    tk.Button(closed_btns, text="Open", command=reopen_closed_panel).pack(side="left", padx=2)

    def reopen_closed_with_others():
        idx = get_closed_selection()
        if idx is None:
            return
        restore_layout_containers()
        restore_sections()
        title = closed_list.get(idx)
        data = section_states.get(title)
        if not data:
            return
        data["state"].set(True)
        data["update"]()
        update_closed_list()

    tk.Button(closed_btns, text="Open With Others", command=reopen_closed_with_others).pack(side="left", padx=2)

    closed_menu = tk.Menu(closed_list, tearoff=0)
    closed_menu.add_command(label="Open", command=lambda: reopen_closed_panel())
    closed_menu.add_command(label="Open With Others", command=lambda: reopen_closed_with_others())

    def show_closed_menu(event):
        idx = closed_list.nearest(event.y)
        if idx >= 0:
            closed_list.selection_clear(0, tk.END)
            closed_list.selection_set(idx)
            closed_menu.tk_popup(event.x_root, event.y_root)

    closed_list.bind("<Button-3>", show_closed_menu)

    def make_section(parent, title, start_open=True, group_key=None, parent_kind="pack", container=None, pane_weight=None):
        frame = tk.Frame(parent)
        header = tk.Frame(frame)
        header.pack(fill="x")
        state = tk.BooleanVar(value=start_open)
        max_btn = tk.Button(header, text="Max", width=5)
        max_btn.pack(side="left", padx=(0, 4))
        min_btn = tk.Button(header, text="Min", width=5)
        min_btn.pack(side="left", padx=(0, 6))
        toggle_btn = tk.Button(header, text="", anchor="w")
        toggle_btn.pack(side="left", fill="x", expand=True)
        body = tk.Frame(frame)
        if start_open:
            body.pack(fill="both", expand=True)

        def update():
            if state.get():
                body.pack(fill="both", expand=True)
                toggle_btn.config(text=f"[-] {title}")
                if parent_kind == "pane":
                    try:
                        header.update_idletasks()
                        min_h = header.winfo_reqheight()
                        (container or parent).paneconfigure(
                            frame,
                            minsize=min_h,
                            weight=pane_weight if pane_weight is not None else 1,
                        )
                    except Exception:
                        pass
            else:
                body.pack_forget()
                toggle_btn.config(text=f"[+] {title}")
                if parent_kind == "pane":
                    try:
                        header.update_idletasks()
                        min_h = header.winfo_reqheight()
                        (container or parent).paneconfigure(frame, minsize=min_h, weight=0)
                    except Exception:
                        pass

        def on_toggle():
            state.set(not state.get())
            update()
            update_closed_list()

        def on_minimize():
            state.set(False)
            update()
            update_closed_list()

        toggle_btn.config(command=on_toggle)
        min_btn.config(command=on_minimize)
        max_btn.config(command=lambda g=group_key, f=frame: maximize_section(g, f))
        register_section(group_key, frame, parent_kind, container or parent, pane_weight)
        section_states[title] = {"state": state, "update": update}
        update()
        update_closed_list()
        return frame, body, toggle_btn, max_btn

    station_split = ttk.Panedwindow(record_tab, orient="horizontal")
    station_split.pack(fill="both", expand=True, padx=8, pady=8)

    station_record = tk.Frame(station_split)
    station_pads = tk.Frame(station_split)
    station_pattern = tk.Frame(station_split)

    station_meta = {
        station_record: {"weight": 1, "minsize": 320, "title": "Recording"},
        station_pads: {"weight": 2, "minsize": 420, "title": "Pads"},
        station_pattern: {"weight": 2, "minsize": 420, "title": "Pattern Maker"},
    }
    station_order = [station_record, station_pads, station_pattern]
    station_states = {
        station_record: tk.BooleanVar(value=True),
        station_pads: tk.BooleanVar(value=True),
        station_pattern: tk.BooleanVar(value=True),
    }
    station_max = {"frame": None}
    station_snapshot = {"order": None, "states": None, "sashes": None}

    def apply_station_order(order=None):
        for pane in station_split.panes():
            station_split.forget(pane)
        for st in order or station_order:
            if not station_states[st].get():
                continue
            meta = station_meta[st]
            station_split.add(st, weight=meta["weight"])
            try:
                station_split.pane(st, minsize=meta["minsize"])
            except Exception:
                pass

    apply_station_order()

    def make_station_header(parent, title):
        header = tk.Frame(parent, bg=COL_PANEL)
        tk.Label(header, text=title, bg=COL_PANEL, fg=COL_TEXT, font=("Arial", 12, "bold")).pack(side="left", padx=8, pady=6)
        return header

    station_record_header = make_station_header(station_record, "Recording Station")
    station_record_header.pack(fill="x")
    station_record_close_btn = tk.Button(station_record_header, text="Close", width=6)
    station_record_close_btn.pack(side="right", padx=6, pady=6)
    station_record_max_btn = tk.Button(station_record_header, text="Max", width=6)
    station_record_max_btn.pack(side="right", padx=6, pady=6)
    station_record_body = tk.Frame(station_record)
    station_record_body.pack(fill="both", expand=True)

    station_pads_header = make_station_header(station_pads, "Pads Station")
    station_pads_header.pack(fill="x")
    station_pads_close_btn = tk.Button(station_pads_header, text="Close", width=6)
    station_pads_close_btn.pack(side="right", padx=6, pady=6)
    station_pads_max_btn = tk.Button(station_pads_header, text="Max", width=6)
    station_pads_max_btn.pack(side="right", padx=6, pady=6)
    station_pads_body = tk.Frame(station_pads)
    station_pads_body.pack(fill="both", expand=True)

    station_pattern_header = make_station_header(station_pattern, "Pattern Station")
    station_pattern_header.pack(fill="x")
    station_pattern_close_btn = tk.Button(station_pattern_header, text="Close", width=6)
    station_pattern_close_btn.pack(side="right", padx=6, pady=6)
    station_pattern_max_btn = tk.Button(station_pattern_header, text="Max", width=6)
    station_pattern_max_btn.pack(side="right", padx=6, pady=6)
    station_pattern_body = tk.Frame(station_pattern)
    station_pattern_body.pack(fill="both", expand=True)

    drag_station = {"active": False, "station": None}

    def station_for_widget(w):
        while w is not None:
            if w in station_meta:
                return w
            w = getattr(w, "master", None)
        return None

    def on_station_press(event, station):
        drag_station["active"] = True
        drag_station["station"] = station

    def on_station_drag(event):
        if not drag_station["active"]:
            return
        target = station_for_widget(root.winfo_containing(event.x_root, event.y_root))
        current = drag_station["station"]
        if not target or target == current:
            return
        i = station_order.index(current)
        j = station_order.index(target)
        station_order[i], station_order[j] = station_order[j], station_order[i]
        apply_station_order()
        drag_station["station"] = target

    def on_station_release(_event):
        drag_station["active"] = False
        drag_station["station"] = None

    for header, station in (
        (station_record_header, station_record),
        (station_pads_header, station_pads),
        (station_pattern_header, station_pattern),
    ):
        header.bind("<ButtonPress-1>", lambda e, s=station: on_station_press(e, s))
        header.bind("<B1-Motion>", on_station_drag)
        header.bind("<ButtonRelease-1>", on_station_release)

    left_canvas = tk.Canvas(station_record_body, highlightthickness=0)
    left_scroll = ttk.Scrollbar(station_record_body, orient="vertical", command=left_canvas.yview)
    left_inner = tk.Frame(left_canvas)
    left_inner.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
    left_window = left_canvas.create_window((0, 0), window=left_inner, anchor="nw")
    def on_left_canvas_config(event):
        left_canvas.itemconfigure(left_window, width=event.width)
    left_canvas.bind("<Configure>", on_left_canvas_config)
    left_canvas.configure(yscrollcommand=left_scroll.set)
    left_canvas.pack(side="left", fill="both", expand=True)
    left_scroll.pack(side="right", fill="y")

    def bind_mousewheel(widget, target):
        def on_wheel(event):
            delta = -1 * int(event.delta / 120) if event.delta else (1 if event.num == 5 else -1)
            target.yview_scroll(delta, "units")
        widget.bind("<MouseWheel>", on_wheel)
        widget.bind("<Button-4>", lambda e: target.yview_scroll(-1, "units"))
        widget.bind("<Button-5>", lambda e: target.yview_scroll(1, "units"))

    for w in (station_record_body, left_canvas, left_inner):
        bind_mousewheel(w, left_canvas)

    group_left = "left"
    group_pads = "pads"
    group_seq = "seq"
    group_library = "library"

    section_groups = {}
    section_pack_info = {}
    section_pane_info = {}
    active_section = {"frame": None}
    def update_waveform_layout():
        pass

    def register_section(group_key, frame, parent_kind, container, pane_weight=None):
        section_groups.setdefault(group_key, []).append(frame)
        if parent_kind == "pane":
            section_pane_info[frame] = {"container": container, "weight": pane_weight}

    def restore_layout_containers():
        station_max["frame"] = None
        station_snapshot["order"] = None
        station_snapshot["states"] = None
        station_snapshot["sashes"] = None
        apply_station_order()

    def restore_station_snapshot():
        if station_snapshot["order"] is not None:
            station_order[:] = list(station_snapshot["order"])
        if station_snapshot["states"] is not None:
            for st, val in station_snapshot["states"].items():
                if st in station_states:
                    station_states[st].set(val)
        else:
            for st in station_states:
                station_states[st].set(True)
        apply_station_order()
        if station_snapshot["sashes"]:
            try:
                station_split.update_idletasks()
                for i, pos in enumerate(station_snapshot["sashes"]):
                    station_split.sashpos(i, pos)
            except Exception:
                pass
        station_max["frame"] = None
        station_snapshot["order"] = None
        station_snapshot["states"] = None
        station_snapshot["sashes"] = None

    def show_only_station(station):
        if station_snapshot["order"] is None:
            station_snapshot["order"] = list(station_order)
            station_snapshot["states"] = {st: station_states[st].get() for st in station_states}
            try:
                station_snapshot["sashes"] = [station_split.sashpos(i) for i in range(max(0, len(station_split.panes()) - 1))]
            except Exception:
                station_snapshot["sashes"] = None
        for pane in station_split.panes():
            station_split.forget(pane)
        meta = station_meta[station]
        station_split.add(station, weight=1)
        try:
            station_split.pane(station, minsize=meta["minsize"])
        except Exception:
            pass
        station_max["frame"] = station

    def is_station_only(station) -> bool:
        if station_max["frame"] == station:
            return True
        panes = station_split.panes()
        if len(panes) != 1:
            return False
        try:
            return root.nametowidget(panes[0]) == station
        except Exception:
            return panes[0] == str(station)

    def update_station_max_buttons():
        station_record_max_btn.config(text="Back" if is_station_only(station_record) else "Max")
        station_pads_max_btn.config(text="Back" if is_station_only(station_pads) else "Max")
        station_pattern_max_btn.config(text="Back" if is_station_only(station_pattern) else "Max")

    def toggle_station_record():
        if is_station_only(station_record):
            restore_station_snapshot()
            restore_sections()
        else:
            restore_sections()
            show_only_station(station_record)
        active_section["frame"] = None
        update_max_buttons()
        update_station_max_buttons()

    def toggle_station_pads():
        if is_station_only(station_pads):
            restore_station_snapshot()
            restore_sections()
        else:
            restore_sections()
            show_only_station(station_pads)
        active_section["frame"] = None
        update_max_buttons()
        update_station_max_buttons()

    def toggle_station_pattern():
        if is_station_only(station_pattern):
            restore_station_snapshot()
            restore_sections()
        else:
            restore_sections()
            show_only_station(station_pattern)
        active_section["frame"] = None
        update_max_buttons()
        update_station_max_buttons()

    def close_station(station):
        station_states[station].set(False)
        apply_station_order()
        if station_max["frame"] == station:
            station_max["frame"] = None
            station_snapshot["order"] = None
            station_snapshot["states"] = None
        active_section["frame"] = None
        update_max_buttons()
        update_station_max_buttons()
        update_closed_list()

    def open_station(station):
        station_states[station].set(True)
        apply_station_order()
        active_section["frame"] = None
        update_max_buttons()
        update_station_max_buttons()
        update_closed_list()

    def restore_sections():
        for frame, info in section_pack_info.items():
            if not frame.winfo_ismapped():
                try:
                    frame.pack(**info)
                except Exception:
                    pass
        for frame, info in section_pane_info.items():
            container = info["container"]
            if str(frame) not in container.panes():
                try:
                    container.add(frame, weight=info["weight"])
                except Exception:
                    container.add(frame)
        active_section["frame"] = None
        update_max_buttons()
        update_waveform_layout()

    def maximize_layout_for_group(group_key):
        restore_layout_containers()
        if group_key == group_left:
            show_only_station(station_record)
        elif group_key == group_pads:
            show_only_station(station_pads)
        elif group_key == group_seq:
            show_only_station(station_pattern)

    def maximize_section(group_key, frame):
        if active_section["frame"] == frame:
            restore_layout_containers()
            restore_sections()
            return
        maximize_layout_for_group(group_key)
        for f in section_groups.get(group_key, []):
            if f == frame:
                continue
            if f in section_pane_info:
                container = section_pane_info[f]["container"]
                if str(f) in container.panes():
                    container.forget(f)
            else:
                if f not in section_pack_info:
                    try:
                        section_pack_info[f] = f.pack_info()
                    except Exception:
                        section_pack_info[f] = {"fill": "both", "expand": True}
                f.pack_forget()
        if frame in section_pane_info:
            container = section_pane_info[frame]["container"]
            if str(frame) not in container.panes():
                try:
                    container.add(frame, weight=1)
                except Exception:
                    container.add(frame)
        else:
            info = section_pack_info.get(frame, {"fill": "both", "expand": True})
            try:
                frame.pack(**info)
            except Exception:
                frame.pack(fill="both", expand=True)
        active_section["frame"] = frame
        update_max_buttons()
        update_waveform_layout()

    pads_section, pads_body, pads_toggle_btn, pads_max_btn = make_section(station_pads_body, "Pads", True, group_key=group_pads, container=station_pads_body)
    pads_section.pack(fill="both", expand=True)
    seq_pane = ttk.Panedwindow(station_pattern_body, orient="vertical")
    seq_pane.pack(fill="both", expand=True)
    fx_section, fx_body, fx_toggle_btn, fx_max_btn = make_section(seq_pane, "FX", False, group_key=group_seq, parent_kind="pane", container=seq_pane, pane_weight=1)
    seq_section, seq_body, seq_toggle_btn, seq_max_btn = make_section(seq_pane, "Pattern Maker", True, group_key=group_seq, parent_kind="pane", container=seq_pane, pane_weight=1)
    seq_pane.add(fx_section, weight=1)
    seq_pane.add(seq_section, weight=1)

    record_section, record_body, record_toggle_btn, record_max_btn = make_section(left_inner, "Recording", True, group_key=group_left, container=left_inner)
    record_section.pack(fill="x", pady=6)
    takes_section, takes_body, takes_toggle_btn, takes_max_btn = make_section(left_inner, "Takes", True, group_key=group_left, container=left_inner)
    takes_section.pack(fill="x", pady=6)
    edit_section, edit_body, edit_toggle_btn, edit_max_btn = make_section(left_inner, "Waveform", True, group_key=group_left, container=left_inner)
    edit_section.pack(fill="x", pady=6)
    trim_section, trim_body, trim_toggle_btn, trim_max_btn = make_section(left_inner, "Trim", True, group_key=group_left, container=left_inner)
    trim_section.pack(fill="x", pady=6)


    section_max_buttons = [
        (record_max_btn, record_section),
        (takes_max_btn, takes_section),
        (edit_max_btn, edit_section),
        (trim_max_btn, trim_section),
        (pads_max_btn, pads_section),
        (fx_max_btn, fx_section),
        (seq_max_btn, seq_section),
    ]

    def update_max_buttons():
        active = active_section["frame"]
        for btn, frame in section_max_buttons:
            if btn is None:
                continue
            btn.config(text="Back" if active == frame else "Max")
        update_station_max_buttons()
        update_waveform_layout()

    def register_station_entry(title, station):
        def update():
            apply_station_order()
            update_station_max_buttons()
            update_closed_list()
        section_states[title] = {"state": station_states[station], "update": update}
        update_closed_list()

    register_station_entry("Recording Station", station_record)
    register_station_entry("Pads Station", station_pads)
    register_station_entry("Pattern Station", station_pattern)
    update_max_buttons()
    station_record_max_btn.config(command=toggle_station_record)
    station_pads_max_btn.config(command=toggle_station_pads)
    station_pattern_max_btn.config(command=toggle_station_pattern)
    station_record_close_btn.config(command=lambda: close_station(station_record))
    station_pads_close_btn.config(command=lambda: close_station(station_pads))
    station_pattern_close_btn.config(command=lambda: close_station(station_pattern))
    update_station_max_buttons()

    library_section, library_body, library_toggle_btn, library_max_btn = make_section(library_tab, "Library", True, group_key=group_library, container=library_tab)
    library_section.pack(fill="both", expand=True, padx=10, pady=6)
    section_max_buttons.append((library_max_btn, library_section))
    update_max_buttons()

    pad_buttons = {}
    pad_container = tk.Frame(pads_body, highlightthickness=1, highlightbackground=COL_ACCENT_2, bg=COL_PANEL)
    pad_container.pack(fill="both", expand=True, padx=8, pady=8)
    tab_nav = tk.Frame(pad_container, bg=COL_PANEL)
    tab_nav.pack(fill="x", padx=6, pady=(6, 2))
    pad_notebook = ttk.Notebook(pad_container)
    style.configure("SoundTabs.TNotebook.Tab", padding=(10, 4))
    pad_notebook.configure(style="SoundTabs.TNotebook")
    pad_notebook.pack(side="left", fill="both", expand=True)
    pad_scroll = tk.Scrollbar(pad_container, orient="vertical", width=16)
    pad_scroll.pack(side="right", fill="y")

    SOUND_TAB_COLORS = [
        "#39ff14", "#00e5ff", "#ff2e63", "#ff9100", "#b517ff",
        "#00ffb3", "#ffea00", "#ff4dff", "#00ffa2", "#7cff00",
        "#ff6f00", "#00c8ff",
    ]

    def tab_color_for_widget(tab_widget) -> str:
        try:
            idx = pad_notebook.index(tab_widget)
        except Exception:
            idx = 0
        return SOUND_TAB_COLORS[idx % len(SOUND_TAB_COLORS)]

    def add_tab_color_bar(tab_widget):
        color = tab_color_for_widget(tab_widget)
        bar = tk.Frame(tab_widget, bg=color, height=4)
        bar.place(x=0, y=0, relwidth=1)

    def select_tab_offset(delta: int):
        tabs = pad_notebook.tabs()
        if not tabs:
            return
        current = pad_notebook.select()
        try:
            idx = tabs.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx + delta) % len(tabs)
        pad_notebook.select(tabs[new_idx])

    tk.Button(tab_nav, text="<", width=3, command=lambda: select_tab_offset(-1)).pack(side="left", padx=2)
    tk.Button(tab_nav, text=">", width=3, command=lambda: select_tab_offset(1)).pack(side="left", padx=2)
    current_tab_var = tk.StringVar(value="")
    current_tab_color = tk.Label(tab_nav, text=" ", bg=COL_PANEL, width=2)
    current_tab_color.pack(side="left", padx=(10, 4))
    tk.Label(tab_nav, textvariable=current_tab_var, bg=COL_PANEL, fg=COL_TEXT, anchor="w").pack(side="left", padx=(0, 0))

    keys_tab = tk.Frame(pad_notebook)
    pad_notebook.add(keys_tab, text="Keys")
    add_tab_color_bar(keys_tab)

    category_tabs = {}
    category_keymap = {}
    global_keymap = {}
    path_to_key = {}

    rec_frame = record_body
    takes_frame = takes_body
    edit_frame = edit_body
    trim_frame_parent = trim_body

    def update_pad_button(key: str):
        path = SOUNDS.get(key, "")
        key_label = path_to_key.get(path, "")
        key_display = key_label.upper() if key_label else "--"
        text = f"{key_display}\n{short_name(path)}"
        btn = pad_buttons.get(key)
        if btn:
            color = neon_color_for_key(key_label or key)
            btn.config(text=text, bg=color, fg=contrast_text_color(color), activebackground=color)

    def set_selected_pad(key: str):
        selected_pad.set(key)
        for k, btn in pad_buttons.items():
            btn.config(relief=("sunken" if k == key else "raised"))

    def trigger_key(key: str):
        if key not in SOUNDS:
            return
        try:
            sample = load_sample(SOUNDS[key])
            gain = 1.0
            path = SOUNDS[key].lower()
            if "hat" in path:
                gain = 0.65
            if "wub" in path or "loop" in path:
                gain = 0.85
            ENGINE.trigger(sample, gain=gain)
        except Exception as e:
            messagebox.showerror("Play error", str(e))

    # Key pad grid
    grid_keys = [PAD_KEYS[i:i+4] for i in range(0, len(PAD_KEYS), 4)]
    for r, row in enumerate(grid_keys):
        for c, key in enumerate(row):
            color = neon_color_for_key(key)
            btn = tk.Button(
                keys_tab,
                width=16,
                height=4,
                text="",
                font=("Arial", 9, "bold"),
                bg=color,
                fg=contrast_text_color(color),
                activebackground=color,
                command=lambda k=key: (set_selected_pad(k), trigger_key(k)),
            )
            btn.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            btn.pad_key = key
            btn.bind("<Button-3>", lambda e, k=key, p=SOUNDS.get(key): show_pattern_menu(k, p, e))
            pad_buttons[key] = btn
            update_pad_button(key)
    for i in range(4):
        keys_tab.grid_columnconfigure(i, weight=1)
        keys_tab.grid_rowconfigure(i, weight=1)

    set_selected_pad(PAD_KEYS[0])

    # Recording + playback
    # Recording frame moved to top of window for visibility

    rec_light = tk.Label(rec_frame, text="REC", width=6, bg=COL_MUTED, fg=COL_BG)
    rec_light.pack(side="left", padx=6, pady=6)

    recordings = []
    recordings_source = {}  # path -> "recorded" | "external"
    rec_list_frame = tk.Frame(takes_frame)
    rec_list_frame.pack(fill="x", padx=6, pady=4)
    rec_list = tk.Listbox(rec_list_frame, height=5, width=40)
    rec_list_scroll = ttk.Scrollbar(rec_list_frame, orient="vertical", command=rec_list.yview)
    rec_list.configure(yscrollcommand=rec_list_scroll.set)
    rec_list.pack(side="left", fill="both", expand=True)
    rec_list_scroll.pack(side="right", fill="y")
    bind_mousewheel(rec_list_frame, rec_list)
    bind_mousewheel(rec_list, rec_list)
    wave_info = tk.StringVar(value="Waveform: no recording selected")
    wave_canvas = tk.Canvas(edit_frame, height=160, bg="#0b1f4b", highlightthickness=1, highlightbackground=COL_ACCENT_2)
    wave_canvas.pack(fill="x", padx=6, pady=(4, 2))
    wave_info_row = tk.Frame(edit_frame)
    wave_info_row.pack(fill="x", padx=6, pady=(0, 2))
    tk.Label(wave_info_row, textvariable=wave_info).pack(side="left")
    tk.Button(wave_info_row, text="Delete Take", command=lambda: delete_selected_recording()).pack(side="right")
    tk.Button(wave_info_row, text="Save Take...", command=lambda: save_selected_recording()).pack(side="right", padx=(0, 6))

    open_takes_frame = tk.LabelFrame(edit_frame, text="Open Takes")
    open_takes_frame.pack(fill="x", padx=6, pady=(4, 6))
    open_takes = []
    open_takes_rows = []
    mini_wave_cache = {}
    OPEN_TAKES_MAX = 4

    def update_waveform_layout():
        is_max = active_section["frame"] == edit_section
        pad = 0 if is_max else 6
        try:
            edit_section.pack_configure(fill="both" if is_max else "x", expand=is_max)
        except Exception:
            pass
        try:
            wave_canvas.pack_configure(fill="both" if is_max else "x", expand=is_max, padx=pad, pady=(4, 2))
        except Exception:
            pass
        try:
            wave_info_row.pack_configure(padx=pad, pady=(0, 2))
        except Exception:
            pass
        try:
            open_takes_frame.pack_configure(padx=pad, pady=(4, 6), fill="x", expand=False)
        except Exception:
            pass

    wave_list_frame = tk.LabelFrame(takes_frame, text="All Recordings")
    wave_list_visible = tk.BooleanVar(value=False)
    wave_list_canvas = tk.Canvas(wave_list_frame, height=160, bg="#0b1f4b", highlightthickness=0)
    wave_list_vbar = ttk.Scrollbar(wave_list_frame, orient="vertical", command=wave_list_canvas.yview)
    wave_list_inner = tk.Frame(wave_list_canvas, bg="#0b1f4b")
    wave_list_inner.bind("<Configure>", lambda e: wave_list_canvas.configure(scrollregion=wave_list_canvas.bbox("all")))
    wave_list_canvas.create_window((0, 0), window=wave_list_inner, anchor="nw")
    wave_list_canvas.configure(yscrollcommand=wave_list_vbar.set)
    wave_list_canvas.pack(side="left", fill="both", expand=True)
    wave_list_vbar.pack(side="right", fill="y")

    waveform_cache = {}
    wave_rows = []

    def add_recording(path: str):
        if path in recordings:
            return
        recordings.append(path)
        recordings_source.setdefault(path, "recorded")
        rec_list.insert(tk.END, os.path.basename(path))
        rec_list.selection_clear(0, tk.END)
        rec_list.selection_set(tk.END)
        add_wave_row(path)

    def load_existing_recordings():
        if not os.path.isdir(RECORDINGS_DIR):
            return
        for name in sorted(os.listdir(RECORDINGS_DIR)):
            if name.lower().endswith(AUDIO_EXTS):
                path = os.path.join(RECORDINGS_DIR, name)
                recordings_source[path] = "recorded"
                add_recording(path)

    def get_selected_recording():
        sel = rec_list.curselection()
        if sel:
            return recordings[sel[0]]
        return ENGINE.last_recording_path

    def delete_selected_recording():
        sel = rec_list.curselection()
        if not sel:
            return
        idx = sel[0]
        path = recordings[idx]
        if not path:
            return
        if recordings_source.get(path) == "external":
            messagebox.showinfo("Delete recording", "This take is external and won't be deleted here.")
            return
        if not messagebox.askyesno("Delete recording", f"Delete this recording?\n{os.path.basename(path)}"):
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
            return
        removed = recordings.pop(idx)
        recordings_source.pop(removed, None)
        if removed in open_takes:
            open_takes.remove(removed)
            refresh_open_takes()
        rec_list.delete(idx)
        if ENGINE.last_recording_path == path:
            ENGINE.last_recording_path = None
        remove_wave_row(removed)
        path = get_selected_recording()
        load_waveform(path)
        add_open_take(path)
        status_var.set("Recording deleted.")

    def save_selected_recording():
        path = get_selected_recording()
        if not path or not os.path.exists(path):
            messagebox.showinfo("Save take", "No recording selected.")
            return
        target = filedialog.asksaveasfilename(
            title="Save take as",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not target:
            return
        try:
            shutil.copy2(path, target)
            messagebox.showinfo("Save take", f"Saved:\n{target}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def open_take_into_waveform():
        path = filedialog.askopenfilename(
            title="Open audio file",
            filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif"), ("All files", "*.*")]
        )
        if not path:
            return
        path = os.path.abspath(path)
        recordings_source[path] = "external"
        if path in recordings:
            select_recording_by_path(path)
        else:
            add_recording(path)
        ENGINE.last_recording_path = path
        load_waveform(path)
        add_open_take(path)
        status_var.set(f"Loaded: {os.path.basename(path)}")

    def play_take_path(path: str):
        if not path or not os.path.exists(path):
            return
        try:
            audio = load_audio_file(path)
            ENGINE.trigger(audio, gain=1.0)
        except Exception as e:
            messagebox.showerror("Play error", str(e))

    def draw_mini_wave(canvas: tk.Canvas, path: str):
        canvas.delete("wave")
        w = max(200, canvas.winfo_width())
        h = max(40, canvas.winfo_height())
        if not path or not os.path.exists(path):
            canvas.create_rectangle(0, 0, w, h, fill="#0b1f4b", outline="", tags="wave")
            return
        try:
            audio = load_audio_file(path)
        except Exception:
            canvas.create_rectangle(0, 0, w, h, fill="#0b1f4b", outline="", tags="wave")
            return
        if len(audio) == 0:
            return
        step = max(1, int(len(audio) / w))
        samples = audio[::step]
        if len(samples) > w:
            samples = samples[:w]
        if len(samples) < w:
            samples = np.pad(samples, (0, w - len(samples)), mode="constant")
        mid = h // 2
        canvas.create_line(0, mid, w, mid, fill="#0f172a", tags="wave")
        amp = np.abs(samples)
        for i, v in enumerate(amp):
            y = int(v * (h * 0.45))
            canvas.create_line(i, mid - y, i, mid + y, fill=COL_ACCENT_2, tags="wave")

    def refresh_open_takes():
        for row in open_takes_rows:
            row.destroy()
        open_takes_rows.clear()
        for path in open_takes:
            row = tk.Frame(open_takes_frame, bg="#0b1f4b")
            row.pack(fill="x", padx=6, pady=4)
            tk.Label(row, text=os.path.basename(path), anchor="w", bg="#0b1f4b", fg=COL_TEXT).pack(side="left")
            tk.Button(row, text="Play", command=lambda p=path: play_take_path(p)).pack(side="right", padx=(4, 0))
            c = tk.Canvas(row, height=50, bg="#0b1f4b", highlightthickness=0)
            c.pack(fill="x", padx=8, pady=(2, 0))
            c.bind("<Configure>", lambda _e, p=path, cv=c: draw_mini_wave(cv, p))
            open_takes_rows.append(row)

    def add_open_take(path: str):
        if not path:
            return
        if path in open_takes:
            open_takes.remove(path)
        open_takes.append(path)
        while len(open_takes) > OPEN_TAKES_MAX:
            open_takes.pop(0)
        refresh_open_takes()

    rec_menu = tk.Menu(rec_frame, tearoff=0)
    rec_menu.add_command(label="Delete recording", command=delete_selected_recording)

    def toggle_wave_list():
        if wave_list_visible.get():
            wave_list_frame.pack(fill="both", padx=6, pady=(6, 4))
        else:
            wave_list_frame.pack_forget()

    ttk.Checkbutton(takes_frame, text="Show All Takes", variable=wave_list_visible, command=toggle_wave_list).pack(anchor="w", padx=8, pady=(2, 4))

    def get_preview(path: str, width: int) -> np.ndarray:
        if path in waveform_cache and len(waveform_cache[path]) == width:
            return waveform_cache[path]
        try:
            audio = load_audio_file(path)
        except Exception:
            return np.zeros(width, dtype=np.float32)
        if len(audio) == 0:
            return np.zeros(width, dtype=np.float32)
        step = max(1, int(len(audio) / width))
        samples = audio[::step]
        if len(samples) > width:
            samples = samples[:width]
        if len(samples) < width:
            samples = np.pad(samples, (0, width - len(samples)), mode="constant")
        waveform_cache[path] = samples
        return samples

    def draw_wave_row(canvas: tk.Canvas, path: str):
        canvas.delete("row")
        w = max(200, canvas.winfo_width())
        h = max(30, canvas.winfo_height())
        mid = h // 2
        canvas.create_line(0, mid, w, mid, fill="#0f172a", tags="row")
        samples = get_preview(path, w)
        amp = np.abs(samples)
        for i, v in enumerate(amp):
            y = int(v * (h * 0.45))
            canvas.create_line(i, mid - y, i, mid + y, fill="#22c55e", tags="row")

    def add_wave_row(path: str):
        row = tk.Frame(wave_list_inner, bg="#0b1f4b")
        row.pack(fill="x", padx=6, pady=4)
        label = tk.Label(row, text=os.path.basename(path), anchor="w", bg="#0b1f4b", fg="#e2e8f0")
        label.pack(fill="x")
        c = tk.Canvas(row, height=36, bg="#1e3a8a", highlightthickness=0)
        c.pack(fill="x", pady=(2, 0))
        c.bind("<Configure>", lambda _e, p=path, cv=c: draw_wave_row(cv, p))
        c.bind("<Button-1>", lambda _e, p=path: select_recording_by_path(p))
        wave_rows.append({"path": path, "frame": row, "canvas": c})

    def remove_wave_row(path: str):
        for i, row in enumerate(list(wave_rows)):
            if row["path"] == path:
                row["frame"].destroy()
                wave_rows.pop(i)
                waveform_cache.pop(path, None)
                break

    def select_recording_by_path(path: str):
        if path in recordings:
            idx = recordings.index(path)
            rec_list.selection_clear(0, tk.END)
            rec_list.selection_set(idx)
            rec_list.see(idx)
            load_waveform(path)

    load_existing_recordings()

    def save_recording():
        audio = ENGINE.get_recording()
        if len(audio) == 0:
            return None

        if RECORD_NORMALIZE:
            peak = float(np.max(np.abs(audio)) + 1e-12)
            audio = (audio * (RECORD_PEAK / peak)).astype(np.float32)

        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        outname = os.path.join(RECORDINGS_DIR, f"beatpad_recording_{ts}.wav")
        sf.write(outname, audio, SR, subtype="PCM_16")
        ENGINE.last_recording_path = os.path.abspath(outname)
        recordings_source[ENGINE.last_recording_path] = "recorded"
        add_recording(ENGINE.last_recording_path)
        return outname

    loop_recording = {"active": False}

    def toggle_record():
        now_on = ENGINE.toggle_record()
        if not now_on:
            out = save_recording()
            if out:
                messagebox.showinfo("Recording saved", f"Saved:\n{os.path.abspath(out)}")
            else:
                messagebox.showinfo("Recording", "No audio recorded.")
        update_status()

    def start_loop_record():
        if ENGINE.recording:
            return
        loop_recording["active"] = True
        ENGINE.toggle_record()
        status_var.set("Loop recording... press Stop Loop to finish.")
        update_status()

    def stop_loop_record():
        if not ENGINE.recording:
            return
        ENGINE.toggle_record()
        out = save_recording()
        loop_recording["active"] = False
        if out:
            loop_var.set(True)
            toggle_loop()
            trim_in.set(0.0)
            trim_out.set(1.0)
            load_waveform(out)
            play_selected()
        update_status()

    def play_selected():
        path = get_selected_recording()
        if not path or not os.path.exists(path):
            messagebox.showinfo("Playback", "No recording found yet.")
            return
        try:
            audio = load_audio_file(path)
            if loop_var.get():
                a = float(trim_in.get())
                b = float(trim_out.get())
                start = int(max(0.0, min(1.0, a)) * len(audio))
                end = int(max(0.0, min(1.0, b)) * len(audio))
                if end <= start:
                    start, end = 0, len(audio)
                ENGINE.start_playback(audio, start=start, end=end)
            else:
                ENGINE.start_playback(audio, start=0, end=len(audio))
            pb_var.set("Playback: playing")
        except Exception as e:
            messagebox.showerror("Playback error", str(e))

    def pause_resume():
        if ENGINE.playback_active and not ENGINE.playback_paused:
            ENGINE.pause_playback()
            pb_var.set("Playback: paused")
        elif ENGINE.playback_active and ENGINE.playback_paused:
            ENGINE.resume_playback()
            pb_var.set("Playback: playing")

    def stop_playback():
        ENGINE.stop_playback()
        pb_var.set("Playback: stopped")

    def toggle_loop():
        ENGINE.playback_loop = loop_var.get()

    def overdub():
        path = get_selected_recording()
        if not path or not os.path.exists(path):
            messagebox.showinfo("Overdub", "Select a recording to overdub.")
            return
        try:
            audio = load_audio_file(path)
            if loop_var.get():
                a = float(trim_in.get())
                b = float(trim_out.get())
                start = int(max(0.0, min(1.0, a)) * len(audio))
                end = int(max(0.0, min(1.0, b)) * len(audio))
                if end <= start:
                    start, end = 0, len(audio)
                ENGINE.start_playback(audio, start=start, end=end)
            else:
                ENGINE.start_playback(audio, start=0, end=len(audio))
            if not ENGINE.recording:
                ENGINE.toggle_record()
            pb_var.set("Playback: playing")
            update_status()
        except Exception as e:
            messagebox.showerror("Overdub error", str(e))

    def record_all_open_takes():
        if ENGINE.recording:
            toggle_record()
            return
        if not open_takes:
            messagebox.showinfo("Record All", "No open takes to record.")
            return
        ENGINE.toggle_record()
        lengths = []
        for path in open_takes:
            if not path or not os.path.exists(path):
                continue
            try:
                audio = load_audio_file(path)
            except Exception:
                continue
            lengths.append(len(audio) / SR)
            ENGINE.trigger(audio, gain=1.0)
        if lengths:
            total_ms = int((max(lengths) + 0.1) * 1000)
            root.after(total_ms, lambda: ENGINE.recording and toggle_record())
        update_status()

    btn_row = tk.Frame(rec_frame)
    btn_row.pack(fill="x", padx=6, pady=6)

    rec_buttons = [
        ("Record", toggle_record),
        ("Record All", record_all_open_takes),
        ("Loop Rec", start_loop_record),
        ("Stop Loop", stop_loop_record),
        ("Play", play_selected),
        ("Pause/Resume", pause_resume),
        ("Stop", stop_playback),
        ("Overdub", overdub),
    ]
    rec_btn_widgets = []
    for label, cmd in rec_buttons:
        rec_btn_widgets.append(tk.Button(btn_row, text=label, command=cmd))

    layout_state = {"cols": 0}

    def layout_rec_buttons():
        width = btn_row.winfo_width()
        if width <= 1:
            return
        min_btn = 90
        cols = max(1, width // min_btn)
        if cols != layout_state["cols"]:
            for i in range(max(layout_state["cols"], cols)):
                btn_row.grid_columnconfigure(i, weight=0)
            layout_state["cols"] = cols
        for btn in rec_btn_widgets:
            btn.grid_forget()
        for i, btn in enumerate(rec_btn_widgets):
            r, c = divmod(i, cols)
            btn.grid(row=r, column=c, sticky="ew", padx=2, pady=2)
        for i in range(cols):
            btn_row.grid_columnconfigure(i, weight=1)

    btn_row.bind("<Configure>", lambda _e: layout_rec_buttons())
    layout_rec_buttons()
    update_max_buttons()

    ttk.Checkbutton(rec_frame, text="Loop playback", variable=loop_var, command=toggle_loop).pack(anchor="w", padx=8, pady=(0, 6))

    trim_frame = tk.LabelFrame(trim_frame_parent, text="Trim Selection")
    trim_frame.pack(fill="x", padx=6, pady=(2, 6))
    trim_in = tk.DoubleVar(value=0.0)
    trim_out = tk.DoubleVar(value=1.0)
    trim_info = tk.StringVar(value="In: 0.00s  Out: 0.00s")

    ttk.Scale(trim_frame, from_=0.0, to=1.0, variable=trim_in, orient="horizontal").pack(fill="x", padx=8, pady=(6, 2))
    ttk.Scale(trim_frame, from_=0.0, to=1.0, variable=trim_out, orient="horizontal").pack(fill="x", padx=8, pady=(2, 6))
    tk.Label(trim_frame, textvariable=trim_info).pack(anchor="w", padx=8, pady=(0, 4))

    tk.Button(trim_frame, text="Trim + Save New Take", command=lambda: trim_selected()).pack(padx=8, pady=(0, 6))
    tk.Button(trim_frame, text="Delete Selection + Save New Take", command=lambda: cut_selection()).pack(padx=8, pady=(0, 6))

    # FX controls
    fx_frame = tk.LabelFrame(fx_body, text="FX")
    fx_frame.pack(fill="x", pady=6)

    def sync_fx():
        ENGINE.fx_lp = lp_var.get()
        ENGINE.fx_hp = hp_var.get()
        ENGINE.fx_drive = drive_var.get()

    ttk.Checkbutton(fx_frame, text="Lowpass (L)", variable=lp_var, command=sync_fx).pack(anchor="w", padx=6, pady=2)
    ttk.Checkbutton(fx_frame, text="Highpass (H)", variable=hp_var, command=sync_fx).pack(anchor="w", padx=6, pady=2)
    ttk.Checkbutton(fx_frame, text="Grime Drive (G)", variable=drive_var, command=sync_fx).pack(anchor="w", padx=6, pady=2)

    ttk.Label(fx_frame, text="LP cutoff").pack(anchor="w", padx=6, pady=(6, 0))
    lp_scale = ttk.Scale(fx_frame, from_=80.0, to=14000.0, value=ENGINE.lp_cutoff,
                         command=lambda v: setattr(ENGINE, "lp_cutoff", float(v)))
    lp_scale.pack(fill="x", padx=6)

    ttk.Label(fx_frame, text="HP cutoff").pack(anchor="w", padx=6, pady=(6, 0))
    hp_scale = ttk.Scale(fx_frame, from_=20.0, to=2000.0, value=ENGINE.hp_cutoff,
                         command=lambda v: setattr(ENGINE, "hp_cutoff", float(v)))
    hp_scale.pack(fill="x", padx=6)

    ttk.Label(fx_frame, text="Pitch (next hit)").pack(anchor="w", padx=6, pady=(6, 0))
    pitch_scale = ttk.Scale(fx_frame, from_=-12, to=12, value=ENGINE.pitch_semitones,
                            command=lambda v: pitch_var.set(int(float(v))))
    pitch_scale.pack(fill="x", padx=6)

    def on_pitch_change(*_):
        ENGINE.pitch_semitones = pitch_var.get()
    pitch_var.trace_add("write", on_pitch_change)

    ttk.Label(fx_frame, text="Master volume").pack(anchor="w", padx=6, pady=(6, 0))
    master_scale = ttk.Scale(fx_frame, from_=0.1, to=1.0, value=MASTER,
                             command=lambda v: set_master(float(v)))
    master_scale.pack(fill="x", padx=6, pady=(0, 6))

    # Sample browser + remap
    browser = tk.LabelFrame(library_body, text="Samples")
    browser.pack(fill="both", expand=True, pady=6)

    filter_row = tk.Frame(browser)
    filter_row.pack(fill="x", padx=6, pady=4)
    tk.Label(filter_row, text="Category").pack(side="left")
    category_var = tk.StringVar(value="All")
    category_combo = ttk.Combobox(filter_row, textvariable=category_var, state="readonly", width=24)
    category_combo.pack(side="left", padx=(6, 12))
    search_var = tk.StringVar(value="")
    tk.Entry(filter_row, textvariable=search_var).pack(side="left", fill="x", expand=True)
    samples_frame = tk.Frame(browser)
    samples_frame.pack(fill="both", expand=True, padx=6, pady=4)
    samples_list = tk.Listbox(samples_frame, height=8)
    samples_scroll = ttk.Scrollbar(samples_frame, orient="vertical", command=samples_list.yview)
    samples_list.configure(yscrollcommand=samples_scroll.set)
    samples_list.pack(side="left", fill="both", expand=True)
    samples_scroll.pack(side="right", fill="y")
    bind_mousewheel(samples_frame, samples_list)
    bind_mousewheel(samples_list, samples_list)

    all_samples = []
    category_samples = {}
    path_to_category = {}
    path_meta = {}
    display_to_path = {}

    def classify_beatpack_subcat(folder: str, path: str) -> str | None:
        name = os.path.splitext(os.path.basename(path))[0].lower()
        folder_u = folder.upper()
        if folder_u == "DRUMS":
            if "kick" in name:
                return "KICKS"
            if "snare" in name:
                return "SNARES"
            if "clap" in name:
                return "CLAPS"
            if "rim" in name:
                return "RIMS"
            if "crash" in name:
                return "CRASH"
            if "perc" in name:
                return "PERCS"
            if "hat" in name:
                if "open" in name:
                    return "HATS_OPEN"
                if "closed" in name:
                    return "HATS_CLOSED"
                return "HATS"
            return None
        if folder_u == "BASS":
            if "808" in name:
                if "slide" in name:
                    return "808_SLIDES"
                if "long" in name:
                    return "808S_LONG"
                return "808S"
            if "reese" in name:
                return "REESE"
            if "wub" in name:
                return "WUBS"
            return None
        if folder_u == "FX":
            if "riser" in name:
                return "RISERS"
            if "downlifter" in name or "drop" in name:
                return "DROPS"
            if "impact" in name:
                return "IMPACTS"
            if "texture" in name or "vinyl" in name:
                return "TEXTURES"
            return None
        if folder_u == "WUBS":
            return "WUBS"
        if folder_u == "TEXTURES":
            return "TEXTURES"
        if folder_u == "LOOPS":
            return "LOOPS"
        return None

    def group_by_category(paths: list) -> dict:
        cats = {}
        for path in paths:
            base_root = ROOT
            for r in [ROOT] + EXTRA_ROOTS:
                if os.path.commonpath([r, path]) == r:
                    base_root = r
                    break
            rel = os.path.relpath(path, base_root)
            parts = rel.split(os.sep)
            prefix = os.path.basename(base_root)

            if parts and parts[0].upper() == "ABSTRUCT_ALL_SAMPLES":
                prefix = "ABSTRUCT_ALL_SAMPLES"
                parts = parts[1:]
                if len(parts) >= 2:
                    cat_parts = [parts[0].upper(), parts[1].upper()]
                elif len(parts) == 1:
                    cat_parts = [parts[0].upper()]
                else:
                    cat_parts = ["MISC"]
            else:
                if parts and parts[0]:
                    top = parts[0].upper()
                    sub = classify_beatpack_subcat(parts[0], path)
                    cat_parts = [top, sub] if sub else [top]
                else:
                    cat_parts = ["MISC"]

            cat = f"{prefix}/" + "/".join(cat_parts)
            cats.setdefault(cat, []).append(path)
        for cat in cats:
            cats[cat] = sorted(cats[cat])
        return dict(sorted(cats.items()))

    def short_tab_title(cat: str) -> str:
        parts = cat.split("/")
        if parts and parts[0].upper().startswith("ABSTRUCT"):
            parts = parts[1:]
        if not parts:
            return ""
        return " / ".join(p.replace("_", " ").title() for p in parts)

    def make_scroll_tab(title: str):
        tab = tk.Frame(pad_notebook)
        pad_notebook.add(tab, text=title)
        add_tab_color_bar(tab)
        canvas = tk.Canvas(tab, highlightthickness=0)
        vbar = tk.Scrollbar(tab, orient="vertical", command=canvas.yview, width=16)
        hbar = tk.Scrollbar(tab, orient="horizontal", command=canvas.xview, width=16)
        inner = tk.Frame(canvas)

        def update_scrollregion(_event=None):
            inner.update_idletasks()
            canvas.update_idletasks()
            w = max(inner.winfo_reqwidth(), canvas.winfo_width())
            h = max(inner.winfo_reqheight(), canvas.winfo_height())
            canvas.configure(scrollregion=(0, 0, w, h))

        def on_canvas_configure(event):
            canvas.itemconfigure(inner_window, width=event.width)
            update_scrollregion()

        inner.bind("<Configure>", update_scrollregion)
        inner_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")

        def on_wheel(event):
            delta = -1 * int(event.delta / 120) if event.delta else (1 if event.num == 5 else -1)
            canvas.yview_scroll(delta, "units")

        for w in (tab, canvas, inner):
            w.bind("<MouseWheel>", on_wheel)

        tab.bind("<Visibility>", lambda _e: tab.after(0, update_scrollregion))
        return tab, inner, canvas, vbar, update_scrollregion

    def rebuild_global_keymap(all_paths: list):
        global_keymap.clear()
        path_to_key.clear()
        for i, path in enumerate(sorted(all_paths)):
            if i >= KEY_LIMIT:
                break
            key = GLOBAL_KEYS[i]
            global_keymap[key] = path
            path_to_key[path] = key

    def ensure_key_for_path(path: str) -> str:
        if path in path_to_key:
            return path_to_key[path]
        for key in GLOBAL_KEYS:
            if key not in global_keymap:
                global_keymap[key] = path
                path_to_key[path] = key
                return key
        return ""

    def build_category_tabs():
        nonlocal category_tabs
        nonlocal category_keymap
        for tab in list(category_tabs.values()):
            pad_notebook.forget(tab)
        category_tabs = {}
        category_keymap = {}

        all_paths = []
        for _, paths in category_samples.items():
            all_paths.extend(paths)
        rebuild_global_keymap(all_paths)

        for cat, paths in category_samples.items():
            tab_title = short_tab_title(cat)
            tab, inner, canvas, vbar, update_scrollregion = make_scroll_tab(tab_title)
            category_tabs[cat] = {
                "tab": tab,
                "inner": inner,
                "canvas": canvas,
                "vbar": vbar,
                "update": update_scrollregion,
            }
            category_keymap[cat] = {}
            cols = 6
            tab_color = tab_color_for_widget(tab)
            tab_fg = contrast_text_color(tab_color)
            for i, path in enumerate(paths):
                r, c = divmod(i, cols)
                key = path_to_key.get(path, "")
                label = f"{key.upper()}  {short_name(path)}" if key else short_name(path)
                if key:
                    category_keymap[cat][key] = path
                btn = tk.Button(
                    inner,
                    width=14,
                    height=2,
                    text=label,
                    font=("Arial", 9, "bold"),
                    bg=tab_color,
                    fg=tab_fg,
                    activebackground=tab_color,
                    command=lambda p=path: trigger_sample_path(p),
                )
                btn.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
                btn.bind("<Button-3>", lambda e, k=key if key else None, p=path: show_pattern_menu(k, p, e))
            for i in range(cols):
                inner.grid_columnconfigure(i, weight=1)
            tab.update_idletasks()
            update_scrollregion()

        update_pad_scrollbar()

    def trigger_sample_path(path: str):
        try:
            sample = load_sample(path)
            gain = 1.0
            lower = path.lower()
            if "hat" in lower:
                gain = 0.65
            if "wub" in lower or "loop" in lower:
                gain = 0.85
            ENGINE.trigger(sample, gain=gain)
        except Exception as e:
            messagebox.showerror("Play error", str(e))

    # -----------------------------
    # Step Sequencer (16 steps)
    # -----------------------------
    SEQ_STEPS = 48
    BPM_DEFAULT = 90
    seq_bpm = tk.IntVar(value=BPM_DEFAULT)
    seq_playing = {"on": False}
    seq_step = {"i": 0}

    seq_rows = PAD_KEYS[:]  # use key pads as rows
    seq_row_paths = {}  # row_id -> path (for non-key rows)
    seq_keymap = {}  # library key -> path for pattern rows
    pattern_assign = {"i": 0}
    seq_grid = {k: [0] * SEQ_STEPS for k in seq_rows}
    seq_buttons = {k: [] for k in seq_rows}
    seq_step_lengths = {k: [None] * SEQ_STEPS for k in seq_rows}

    bpm_row = tk.Frame(seq_body)
    bpm_row.pack(fill="x", padx=6, pady=(4, 0))
    tk.Label(bpm_row, text="Tempo (BPM)", anchor="w").pack(side="left")
    bpm_label = tk.Label(bpm_row, text=str(BPM_DEFAULT), width=4, anchor="e")
    bpm_label.pack(side="right")

    def on_bpm_change(v):
        val = int(float(v))
        seq_bpm.set(val)
        bpm_label.config(text=str(val))

    ttk.Scale(seq_body, from_=60, to=140, value=BPM_DEFAULT,
              command=on_bpm_change).pack(fill="x", padx=6, pady=(0, 6))

    seq_controls = tk.Frame(seq_body)
    seq_controls.pack(fill="x", padx=6, pady=(0, 6))
    tk.Button(seq_controls, text="Play", command=lambda: start_seq()).pack(side="left", padx=2)
    tk.Button(seq_controls, text="Stop", command=lambda: stop_seq()).pack(side="left", padx=2)
    tk.Button(seq_controls, text="Restart", command=lambda: restart_seq()).pack(side="left", padx=2)
    tk.Button(seq_controls, text="Clear", command=lambda: clear_seq()).pack(side="left", padx=2)
    seq_rec_btn = tk.Button(seq_controls, text="Rec")
    seq_rec_btn.pack(side="left", padx=2)
    seq_loop_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(seq_controls, text="Loop", variable=seq_loop_var).pack(side="left", padx=6)
    seq_loop_len_var = tk.IntVar(value=SEQ_STEPS)

    def get_loop_len() -> int:
        try:
            val = int(seq_loop_len_var.get())
        except Exception:
            val = SEQ_STEPS
        return max(1, min(SEQ_STEPS, val))

    def apply_loop_len(_event=None):
        seq_loop_len_var.set(get_loop_len())
        refresh_seq_buttons()
        update_playhead_positions()
        draw_playhead()

    tk.Label(seq_controls, text="Loop Len").pack(side="left", padx=(8, 2))
    loop_spin = tk.Spinbox(seq_controls, from_=1, to=SEQ_STEPS, width=4, textvariable=seq_loop_len_var, command=apply_loop_len)
    loop_spin.pack(side="left")
    loop_spin.bind("<FocusOut>", apply_loop_len)
    loop_spin.bind("<Return>", apply_loop_len)

    seq_fx_frame = tk.LabelFrame(seq_body, text="Pattern FX")
    seq_fx_header = tk.Frame(seq_fx_frame)
    seq_fx_header.pack(fill="x", padx=4, pady=(4, 2))
    tk.Label(seq_fx_header, text="Pattern FX").pack(side="left")
    seq_fx_visible = tk.BooleanVar(value=True)
    def toggle_pattern_fx():
        if seq_fx_visible.get():
            seq_fx_body.pack_forget()
            seq_fx_visible.set(False)
            seq_fx_close_btn.config(text="Open")
        else:
            seq_fx_body.pack(fill="x", padx=4, pady=(0, 4))
            seq_fx_visible.set(True)
            seq_fx_close_btn.config(text="Close")
    seq_fx_close_btn = tk.Button(seq_fx_header, text="Close", width=8, command=toggle_pattern_fx)
    seq_fx_close_btn.pack(side="right", padx=(4, 0))
    seq_fx_frame.pack(fill="x", padx=6, pady=(0, 6))
    seq_fx_body = tk.Frame(seq_fx_frame)
    seq_fx_body.pack(fill="x", padx=4, pady=(0, 4))

    seq_fx_lp_var = tk.BooleanVar(value=False)
    seq_fx_hp_var = tk.BooleanVar(value=False)
    seq_fx_drive_var = tk.BooleanVar(value=False)
    seq_gate_var = tk.BooleanVar(value=False)
    seq_gate_len_var = tk.DoubleVar(value=1.0)
    seq_pitch_var = tk.IntVar(value=0)
    seq_lp_cutoff_var = tk.DoubleVar(value=900.0)
    seq_hp_cutoff_var = tk.DoubleVar(value=120.0)
    seq_drive_var = tk.DoubleVar(value=2.2)

    ttk.Checkbutton(seq_fx_body, text="Lowpass", variable=seq_fx_lp_var).pack(anchor="w", padx=6, pady=2)
    ttk.Checkbutton(seq_fx_body, text="Highpass", variable=seq_fx_hp_var).pack(anchor="w", padx=6, pady=2)
    ttk.Checkbutton(seq_fx_body, text="Drive", variable=seq_fx_drive_var).pack(anchor="w", padx=6, pady=2)
    ttk.Checkbutton(seq_fx_body, text="Gate (trim)", variable=seq_gate_var).pack(anchor="w", padx=6, pady=2)

    ttk.Label(seq_fx_body, text="LP cutoff").pack(anchor="w", padx=6, pady=(6, 0))
    ttk.Scale(seq_fx_body, from_=80.0, to=14000.0, variable=seq_lp_cutoff_var).pack(fill="x", padx=6)

    ttk.Label(seq_fx_body, text="HP cutoff").pack(anchor="w", padx=6, pady=(6, 0))
    ttk.Scale(seq_fx_body, from_=20.0, to=2000.0, variable=seq_hp_cutoff_var).pack(fill="x", padx=6)

    ttk.Label(seq_fx_body, text="Pitch (pattern only)").pack(anchor="w", padx=6, pady=(6, 0))
    ttk.Scale(seq_fx_body, from_=-12, to=12, variable=seq_pitch_var).pack(fill="x", padx=6)

    ttk.Label(seq_fx_body, text="Drive amount").pack(anchor="w", padx=6, pady=(6, 0))
    ttk.Scale(seq_fx_body, from_=1.0, to=6.0, variable=seq_drive_var).pack(fill="x", padx=6, pady=(0, 6))
    ttk.Label(seq_fx_body, text="Gate length (steps)").pack(anchor="w", padx=6, pady=(6, 0))
    ttk.Scale(seq_fx_body, from_=0.25, to=4.0, variable=seq_gate_len_var).pack(fill="x", padx=6, pady=(0, 6))

    # Playhead bar above the grid
    playhead_frame = tk.Frame(seq_body, bg=COL_PANEL)
    playhead_frame.pack(fill="x", padx=6, pady=(0, 4))
    playhead_canvas = tk.Canvas(playhead_frame, height=18, bg=COL_PANEL, highlightthickness=0)
    playhead_canvas.pack(fill="x", expand=True)
    playhead_positions = []

    def update_playhead_positions():
        seq_inner.update_idletasks()
        playhead_positions.clear()
        if not seq_rows:
            return
        row_id = seq_rows[0]
        row = seq_buttons.get(row_id, [])
        if not row:
            return
        for btn in row:
            x = btn.winfo_x()
            w = btn.winfo_width()
            playhead_positions.append(x + (w / 2))
        width = max(seq_inner.winfo_reqwidth(), seq_canvas.winfo_width())
        playhead_canvas.configure(scrollregion=(0, 0, width, playhead_canvas.winfo_height()))

    def draw_playhead():
        playhead_canvas.delete("playhead")
        if not playhead_positions:
            return
        loop_len = get_loop_len()
        if loop_len <= 0:
            return
        idx = min(seq_step["i"], loop_len - 1, len(playhead_positions) - 1)
        x = playhead_positions[idx]
        h = max(8, int(playhead_canvas.winfo_height()))
        playhead_canvas.create_rectangle(x - 2, 2, x + 2, h - 2, fill=COL_ACCENT_2, outline="", tags="playhead")
    
    def set_playhead_from_x(x: int):
        if not playhead_positions:
            return
        loop_len = get_loop_len()
        if loop_len <= 0:
            return
        target = min(loop_len, len(playhead_positions))
        x = playhead_canvas.canvasx(x)
        best_idx = 0
        best_dist = float("inf")
        for i in range(target):
            dist = abs(playhead_positions[i] - x)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        seq_step["i"] = best_idx
        draw_playhead()

    playhead_canvas.bind("<Button-1>", lambda e: set_playhead_from_x(e.x))
    playhead_canvas.bind("<B1-Motion>", lambda e: set_playhead_from_x(e.x))

    seq_canvas = tk.Canvas(seq_body, highlightthickness=0)
    seq_vbar = ttk.Scrollbar(seq_body, orient="vertical", command=seq_canvas.yview)
    def on_seq_hscroll(*args):
        seq_canvas.xview(*args)
        playhead_canvas.xview(*args)
    seq_hbar = ttk.Scrollbar(seq_body, orient="horizontal", command=on_seq_hscroll)
    seq_inner = tk.Frame(seq_canvas)
    def on_seq_inner_config(_event=None):
        seq_canvas.configure(scrollregion=seq_canvas.bbox("all"))
        update_playhead_positions()
        draw_playhead()
    seq_inner.bind("<Configure>", on_seq_inner_config)
    seq_window = seq_canvas.create_window((0, 0), window=seq_inner, anchor="nw")

    def on_seq_canvas_config(event):
        seq_canvas.itemconfigure(seq_window, width=event.width)

    seq_canvas.bind("<Configure>", on_seq_canvas_config)
    def on_seq_xscroll(*args):
        seq_hbar.set(*args)
        try:
            playhead_canvas.xview_moveto(args[0])
        except Exception:
            pass
    seq_canvas.configure(yscrollcommand=seq_vbar.set, xscrollcommand=on_seq_xscroll)
    seq_canvas.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    seq_vbar.pack(side="right", fill="y")
    seq_hbar.pack(side="bottom", fill="x")
    bind_mousewheel(seq_canvas, seq_canvas)
    bind_mousewheel(seq_inner, seq_canvas)

    def row_key_label(row_id: str) -> str:
        if row_id in PAD_KEYS:
            return row_id.upper()
        path = seq_row_paths.get(row_id)
        if not path:
            return ""
        key = path_to_key.get(path)
        if not key:
            for k, p in seq_keymap.items():
                if p == path:
                    key = k
                    break
        return key.upper() if key else ""

    def draw_seq_grid():
        for child in seq_inner.winfo_children():
            child.destroy()
        for r, row_id in enumerate(seq_rows):
            key_label = row_key_label(row_id)
            if row_id in PAD_KEYS:
                name_label = short_name(SOUNDS.get(row_id, ""))
            else:
                name_label = short_name(seq_row_paths.get(row_id, row_id))
            name_cell = tk.Label(seq_inner, text=name_label, anchor="w")
            name_cell.grid(row=r, column=0, padx=2, pady=1, sticky="w")
            name_cell.bind("<Button-3>", lambda e, k=row_id: show_row_menu(k, e))
            row_buttons = []
            for c in range(SEQ_STEPS):
                btn = tk.Button(seq_inner, text="", width=1, height=1)
                btn.grid(row=r, column=c + 1, padx=0, pady=0)
                btn.bind("<Button-3>", lambda e, k=row_id, i=c: show_step_menu(k, i, e))
                row_buttons.append(btn)
            if key_label:
                key_cell = tk.Label(seq_inner, text=f"[{key_label}]", fg=COL_ACCENT_2)
                key_cell.grid(row=r, column=SEQ_STEPS + 1, padx=2, pady=1, sticky="e")
                key_cell.bind("<Button-3>", lambda e, k=row_id: show_row_menu(k, e))
            seq_buttons[row_id] = row_buttons
        refresh_seq_buttons()
        update_playhead_positions()
        draw_playhead()

    def row_sample_length_sec(row_id: str) -> float:
        path = seq_row_paths.get(row_id)
        if row_id in PAD_KEYS and row_id in SOUNDS:
            path = SOUNDS[row_id]
        return get_sample_length_sec(path)

    def get_step_length(row_id: str, start_idx: int, step_sec: float, loop_len: int) -> int:
        custom = None
        if row_id in seq_step_lengths and 0 <= start_idx < SEQ_STEPS:
            custom = seq_step_lengths[row_id][start_idx]
        if custom is not None:
            return max(1, min(loop_len, int(custom)))
        default_len = max(1, int(np.ceil(row_sample_length_sec(row_id) / step_sec)))
        return max(1, min(loop_len, default_len))

    def refresh_seq_buttons():
        loop_len = get_loop_len()

        step_sec = max(0.01, 60.0 / max(40, min(200, seq_bpm.get())) / 4.0)

        loop_len = get_loop_len()
        inactive_bg = dim_color(COL_PANEL, 0.3)
        for r, row_id in enumerate(seq_rows):
            base_color = neon_color_for_key(row_id if row_id in PAD_KEYS else (seq_row_paths.get(row_id, "") or row_id))
            dimmed = dim_color(base_color, 0.4)
            coverage = [0] * SEQ_STEPS
            for s, val in enumerate(seq_grid[row_id]):
                if val == 1 and s < loop_len:
                    length_steps = get_step_length(row_id, s, step_sec, loop_len)
                    for j in range(length_steps):
                        idx = (s + j) % loop_len
                        coverage[idx] = 2 if j == 0 else 1
            for c, btn in enumerate(seq_buttons[row_id]):
                if c >= loop_len:
                    bg = dim_color(base_color, 0.25) if seq_grid[row_id][c] == 1 else inactive_bg
                else:
                    state = coverage[c]
                    if state == 2:
                        bg = base_color
                    elif state == 1:
                        bg = dimmed
                    else:
                        bg = COL_PANEL
                fg = contrast_text_color(bg)
                btn.config(bg=bg, fg=fg, activebackground=base_color)
                btn.config(command=lambda k=row_id, i=c: toggle_seq(k, i))
        draw_playhead()

    def toggle_seq(k, i):
        seq_grid[k][i] = 0 if seq_grid[k][i] == 1 else 1
        if k in seq_step_lengths and 0 <= i < SEQ_STEPS:
            if seq_grid[k][i] == 0:
                seq_step_lengths[k][i] = None
        refresh_seq_buttons()

    def clear_seq_step(k, i):
        if k in seq_grid and 0 <= i < SEQ_STEPS:
            seq_grid[k][i] = 0
            if k in seq_step_lengths:
                seq_step_lengths[k][i] = None
            refresh_seq_buttons()

    def step_duration_ms():
        bpm = max(40, min(200, seq_bpm.get()))
        return int(60000 / bpm / 4)  # 16th note

    def apply_seq_fx(audio: np.ndarray) -> np.ndarray:
        y = audio
        if seq_pitch_var.get() != 0:
            y = pitch_shift_resample(y, seq_pitch_var.get(), SR)
        if seq_fx_hp_var.get():
            y = butter_filter(y, "high", seq_hp_cutoff_var.get(), SR, order=3)
        if seq_fx_lp_var.get():
            y = butter_filter(y, "low", seq_lp_cutoff_var.get(), SR, order=3)
        if seq_fx_drive_var.get():
            y = softclip(y, seq_drive_var.get())
        return y

    def trim_audio_to_steps(audio: np.ndarray, gate_steps: float) -> np.ndarray:
        max_len = int((step_duration_ms() / 1000.0) * SR * gate_steps)
        max_len = max(1, min(max_len, len(audio)))
        y = audio[:max_len].copy()
        fade = min(32, len(y))
        if fade > 2:
            w = np.linspace(1, 0, fade, dtype=np.float32)
            y[-fade:] *= w
        return y

    def apply_seq_gate(audio: np.ndarray) -> np.ndarray:
        if not seq_gate_var.get():
            return audio
        gate_steps = max(0.05, float(seq_gate_len_var.get()))
        return trim_audio_to_steps(audio, gate_steps)

    def trigger_seq_path(path: str, gate_steps: float | None = None):
        if not path:
            return
        try:
            sample = load_sample(path)
            gain = 1.0
            lower = path.lower()
            if "hat" in lower:
                gain = 0.65
            if "wub" in lower or "loop" in lower:
                gain = 0.85
            audio = apply_seq_fx(sample)
            if gate_steps is not None:
                audio = trim_audio_to_steps(audio, gate_steps)
            else:
                audio = apply_seq_gate(audio)
            ENGINE.trigger(audio, gain=gain)
        except Exception as e:
            messagebox.showerror("Play error", str(e))

    def trigger_seq_row(row_id: str, step_idx: int | None = None):
        gate_steps = None
        if step_idx is not None and row_id in seq_step_lengths and 0 <= step_idx < SEQ_STEPS:
            custom = seq_step_lengths[row_id][step_idx]
            if custom is not None:
                gate_steps = float(custom)
        if gate_steps is not None and seq_gate_var.get():
            gate_steps = min(gate_steps, float(seq_gate_len_var.get()))
        if row_id in seq_row_paths:
            trigger_seq_path(seq_row_paths[row_id], gate_steps=gate_steps)
            return
        if row_id in PAD_KEYS and row_id in SOUNDS:
            trigger_seq_path(SOUNDS[row_id], gate_steps=gate_steps)

    def seq_tick():
        if not seq_playing["on"]:
            return
        loop_len = get_loop_len()
        idx = seq_step["i"]
        if idx >= loop_len:
            idx = 0
            seq_step["i"] = 0
        for row_id in seq_rows:
            if seq_grid[row_id][idx] == 1:
                trigger_seq_row(row_id, idx)
        next_i = idx + 1
        if next_i >= loop_len:
            if seq_loop_var.get():
                seq_step["i"] = 0
            else:
                seq_playing["on"] = False
                return
        else:
            seq_step["i"] = next_i
        draw_playhead()
        root.after(step_duration_ms(), seq_tick)

    def start_seq():
        if seq_playing["on"]:
            return
        seq_playing["on"] = True
        draw_playhead()
        seq_tick()

    def stop_seq():
        seq_playing["on"] = False
        draw_playhead()

    def restart_seq():
        seq_step["i"] = 0
        if seq_playing["on"]:
            draw_playhead()
        else:
            start_seq()

    def clear_seq():
        for k in seq_rows:
            seq_grid[k] = [0] * SEQ_STEPS
            if k in seq_step_lengths:
                seq_step_lengths[k] = [None] * SEQ_STEPS
        refresh_seq_buttons()

    seq_recording = {"on": False, "auto_play": False}

    def toggle_seq_record():
        if not seq_recording["on"]:
            ENGINE.toggle_record()
            seq_recording["on"] = True
            if not seq_playing["on"]:
                start_seq()
                seq_recording["auto_play"] = True
            seq_rec_btn.config(text="Stop Rec")
        else:
            ENGINE.toggle_record()
            out = save_recording()
            seq_recording["on"] = False
            if seq_recording["auto_play"]:
                stop_seq()
                seq_recording["auto_play"] = False
            seq_rec_btn.config(text="Rec")
            if out:
                messagebox.showinfo("Recording saved", f"Saved:\n{os.path.abspath(out)}")
            else:
                messagebox.showinfo("Recording", "No audio recorded.")

    seq_rec_btn.config(command=toggle_seq_record)

    def add_step_for_key(key: str | None, path: str | None = None):
        row_id = key if key else None
        if row_id is None and path:
            for k in PAD_KEYS:
                if SOUNDS.get(k) == path:
                    row_id = k
                    break
        if row_id is None and path:
            row_id = PAD_KEYS[pattern_assign["i"] % len(PAD_KEYS)]
            pattern_assign["i"] = (pattern_assign["i"] + 1) % len(PAD_KEYS)
        if row_id and row_id not in seq_rows:
            seq_rows.insert(0, row_id)
            seq_grid[row_id] = [0] * SEQ_STEPS
            seq_step_lengths[row_id] = [None] * SEQ_STEPS
        if path and row_id in PAD_KEYS:
            SOUNDS[row_id] = path
            update_pad_button(row_id)
            seq_keymap[row_id] = path
            draw_seq_grid()
        if row_id and row_id not in seq_buttons:
            draw_seq_grid()
        if row_id not in seq_rows:
            messagebox.showinfo("Pattern Maker", "This pad can't be added right now.")
            return
        loop_len = get_loop_len()
        start = seq_step["i"] if seq_playing["on"] else 0
        start = start % loop_len if loop_len > 0 else 0
        idx = None
        for i in range(loop_len):
            j = (start + i) % loop_len
            if seq_grid[row_id][j] == 0:
                idx = j
                break
        if idx is None:
            idx = start
        seq_grid[row_id][idx] = 1
        refresh_seq_buttons()

    pattern_menu = tk.Menu(seq_body, tearoff=0)
    pattern_menu.add_command(label="Add to Pattern Maker", command=lambda: None)

    def show_pattern_menu(key: str | None, path: str | None, event):
        pattern_menu.entryconfigure(0, command=lambda: add_step_for_key(key, path))
        pattern_menu.tk_popup(event.x_root, event.y_root)

    step_menu = tk.Menu(seq_body, tearoff=0)
    step_menu.add_command(label="Clear step", command=lambda: None)
    step_menu.add_command(label="Shorten to here", command=lambda: None)

    row_menu = tk.Menu(seq_body, tearoff=0)
    row_menu.add_command(label="Assign selected sample", command=lambda: None)
    row_menu.add_command(label="Load file for row...", command=lambda: None)

    def assign_path_to_row(row_id: str, path: str | None):
        if not path or not os.path.exists(path):
            messagebox.showerror("Assign error", f"Missing file:\n{path}")
            return
        if row_id in PAD_KEYS:
            assign_sample_to_pad(path, row_id)
            return
        key = ensure_key_for_path(path)
        if key:
            seq_keymap[key] = path
        seq_row_paths[row_id] = path
        draw_seq_grid()

    def show_row_menu(row_id: str, event):
        row_menu.entryconfigure(0, command=lambda: assign_path_to_row(row_id, selected_sample_path()))
        row_menu.entryconfigure(1, command=lambda: assign_path_to_row(
            row_id,
            filedialog.askopenfilename(
                title="Choose sample",
                filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif"), ("All files", "*.*")]
            )
        ))
        row_menu.tk_popup(event.x_root, event.y_root)

    def show_step_menu(row_id: str, step_idx: int, event):
        loop_len = get_loop_len()
        step_menu.entryconfigure(0, command=lambda: clear_seq_step(row_id, step_idx))

        def find_hit_for_cell():
            if row_id not in seq_grid or step_idx >= loop_len:
                return None
            step_sec = max(0.01, 60.0 / max(40, min(200, seq_bpm.get())) / 4.0)
            for s in range(loop_len):
                if seq_grid[row_id][s] == 1:
                    length_steps = get_step_length(row_id, s, step_sec, loop_len)
                    for j in range(length_steps):
                        idx = (s + j) % loop_len
                        if idx == step_idx:
                            return s, length_steps
            return None

        hit = find_hit_for_cell()
        if hit is None:
            step_menu.entryconfigure(1, state="disabled", command=lambda: None)
        else:
            start_idx, _len = hit
            if step_idx >= start_idx:
                new_len = step_idx - start_idx + 1
            else:
                new_len = step_idx + loop_len - start_idx + 1
            def shorten():
                if row_id in seq_step_lengths:
                    seq_step_lengths[row_id][start_idx] = max(1, min(loop_len, new_len))
                refresh_seq_buttons()
            step_menu.entryconfigure(1, state="normal", command=shorten)
        step_menu.tk_popup(event.x_root, event.y_root)

    draw_seq_grid()

    def refresh_samples():
        nonlocal all_samples
        all_samples = scan_all_samples()
        nonlocal category_samples
        category_samples = group_by_category(all_samples)
        path_meta.clear()
        for path in all_samples:
            base_root = ROOT
            for r in [ROOT] + EXTRA_ROOTS:
                if os.path.commonpath([r, path]) == r:
                    base_root = r
                    break
            rel = os.path.relpath(path, base_root)
            path_meta[path] = {
                "root": base_root,
                "root_name": os.path.basename(base_root),
                "rel": rel,
                "rel_lower": rel.lower(),
            }
        path_to_category.clear()
        for cat, paths in category_samples.items():
            for pth in paths:
                path_to_category[pth] = cat
        categories = ["All"] + list(category_samples.keys())
        category_combo["values"] = categories
        if category_var.get() not in categories:
            category_var.set("All")
        build_category_tabs()
        for k in pad_buttons:
            update_pad_button(k)
        apply_filter()

    def update_pad_scrollbar():
        current = pad_notebook.select()
        canvas = None
        vbar = None
        updater = None
        for data in category_tabs.values():
            if str(data["tab"]) == current:
                canvas = data["canvas"]
                vbar = data.get("vbar")
                updater = data.get("update")
                break
        if canvas is None:
            pad_scroll.config(command=lambda *args: None)
            pad_scroll.set(0.0, 1.0)
            return
        pad_scroll.config(command=canvas.yview)
        if vbar is None:
            canvas.configure(yscrollcommand=pad_scroll.set)
        else:
            def yset(first, last):
                pad_scroll.set(first, last)
                vbar.set(first, last)
            canvas.configure(yscrollcommand=yset)
        if updater:
            updater()

    def update_current_tab_label():
        current = pad_notebook.select()
        if not current:
            current_tab_var.set("")
            current_tab_color.config(bg=COL_PANEL)
            return
        current_tab_var.set(pad_notebook.tab(current, "text"))
        try:
            tab_widget = pad_notebook.nametowidget(current)
            current_tab_color.config(bg=tab_color_for_widget(tab_widget))
        except Exception:
            current_tab_color.config(bg=COL_PANEL)

    pad_notebook.bind("<<NotebookTabChanged>>", lambda _e: (update_pad_scrollbar(), update_current_tab_label()))
    pad_notebook.bind("<Configure>", lambda _e: update_pad_scrollbar())
    pad_container.bind("<Configure>", lambda _e: update_pad_scrollbar())
    update_current_tab_label()

    def apply_filter(*_):
        samples_list.delete(0, tk.END)
        display_to_path.clear()
        text = search_var.get().lower().strip()
        cat_filter = category_var.get()
        for path in all_samples:
            if cat_filter != "All" and path_to_category.get(path) != cat_filter:
                continue
            meta = path_meta.get(path)
            if not meta:
                continue
            if not text or text in meta["rel_lower"]:
                key_label = path_to_key.get(path, "")
                key_display = key_label.upper() if key_label else "--"
                display = f"[{key_display}] {meta['root_name']}/{meta['rel']}"
                samples_list.insert(tk.END, display)
                display_to_path[display] = path

    search_var.trace_add("write", apply_filter)
    category_combo.bind("<<ComboboxSelected>>", apply_filter)

    def selected_sample_path():
        sel = samples_list.curselection()
        if not sel:
            return None
        display = samples_list.get(sel[0])
        if display in display_to_path:
            return display_to_path[display]
        rel = display
        if rel.startswith("["):
            rel = rel.split("] ", 1)[1] if "] " in rel else rel
        for r in [ROOT] + EXTRA_ROOTS:
            prefix = os.path.basename(r)
            if rel.startswith(prefix + os.sep) or rel.startswith(prefix + "/"):
                sub = rel[len(prefix)+1:]
                return os.path.normpath(os.path.join(r, sub))
        return os.path.normpath(os.path.join(ROOT, rel))

    def assign_sample_to_pad(path: str, key: str):
        if not path or not os.path.exists(path):
            messagebox.showerror("Assign error", f"Missing file:\n{path}")
            return
        ensure_key_for_path(path)
        SOUNDS[key] = path
        update_pad_button(key)
        draw_seq_grid()
        status_var.set(f"Assigned {os.path.basename(path)} to {key.upper()}")

    def on_sample_double_click(_):
        path = selected_sample_path()
        if path:
            assign_sample_to_pad(path, selected_pad.get())

    samples_list.bind("<Double-Button-1>", on_sample_double_click)

    drag_state = {"path": None}

    def on_sample_press(event):
        idx = samples_list.nearest(event.y)
        if idx >= 0:
            display = samples_list.get(idx)
            drag_state["path"] = display_to_path.get(display)

    def on_root_release(_):
        path = drag_state["path"]
        drag_state["path"] = None
        if not path:
            return
        x, y = root.winfo_pointerxy()
        target = root.winfo_containing(x, y)
        if hasattr(target, "pad_key"):
            assign_sample_to_pad(path, target.pad_key)

    samples_list.bind("<ButtonPress-1>", on_sample_press)
    root.bind("<ButtonRelease-1>", on_root_release)

    def load_sample_file():
        path = filedialog.askopenfilename(
            title="Choose sample",
            filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif"), ("All files", "*.*")]
        )
        if path:
            assign_sample_to_pad(path, selected_pad.get())

    btns = tk.Frame(browser)
    btns.pack(fill="x", padx=6, pady=4)
    tk.Button(btns, text="Rescan", command=refresh_samples).pack(side="left", padx=2)
    tk.Button(btns, text="Load File...", command=load_sample_file).pack(side="left", padx=2)
    tk.Button(btns, text="Assign to Pad", command=lambda: assign_sample_to_pad(selected_sample_path(), selected_pad.get())).pack(side="left", padx=2)
    tk.Button(btns, text="Audition", command=lambda: trigger_sample_path(selected_sample_path() or "")).pack(side="left", padx=2)

    # Help overlay
    def show_help():
        help_win = tk.Toplevel(root)
        help_win.title("Help / Shortcuts")
        help_win.geometry("540x420")
        txt = tk.Text(help_win, wrap="word")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert(tk.END, (
            "SOUNDsbeatpad\n"
            "\n"
            "Quick Start\n"
            "  1) Open Studio and click a pad to hear its sound.\n"
            "  2) Open Library, choose a sound, and assign it to a pad.\n"
            "  3) In Pattern Maker, click steps to build a loop, then press Play.\n"
            "\n"
            "Navigation\n"
            "  Sidebar: switch Studio / Library.\n"
            "  Station headers: Max focuses one pane; Back restores layout.\n"
            "  Station Close hides a pane; reopen via Closed Panels list.\n"
            "  Closed Panels: right-click for Open / Open With Others.\n"
            "\n"
            "Recording Station\n"
            "  Record captures the master output to a new take.\n"
            "  Play/Pause/Stop controls playback of the selected take.\n"
            "  Overdub plays the take and records on top of it.\n"
            "  Loop Rec records and auto-loops the new take.\n"
            "\n"
            "Takes + Waveform\n"
            "  Select a take to view the waveform.\n"
            "  Click and drag in the waveform to set a region.\n"
            "  Trim saves just the selected region as a new take.\n"
            "  Cut deletes the selected region and saves a new take.\n"
            "  Delete Take removes the selected file from disk.\n"
            "  Save Take exports the selected take to a new location.\n"
            "  Open Take loads an external audio file into the waveform.\n"
            "\n"
            "Pads Station\n"
            "  Click pads to trigger sounds.\n"
            "  Selected pad is the target for Library assignment.\n"
            "  Pad colors are fixed; Library tabs set the color of library pads.\n"
            "\n"
            "Library\n"
            "  Use Category + Search to find sounds.\n"
            "  Double-click a sample to assign it to the selected pad.\n"
            "  Drag a sample onto a pad to assign it.\n"
            "  Load File adds an external sound to the selected pad.\n"
            "  Audition plays the highlighted sample.\n"
            "  Use tab arrows to switch categories; full tab name shows at top.\n"
            "\n"
            "Pattern Maker\n"
            "  Click grid cells to toggle steps on/off.\n"
            "  Play/Stop controls run the pattern; Loop repeats it.\n"
            "  Loop Len changes how many steps the loop uses.\n"
            "  Restart jumps playback back to step 1.\n"
            "  Drag or click the playhead bar to set the loop start.\n"
            "  Right-click a pad or sample to add it to the Pattern Maker.\n"
            "  Right-click a row label to assign a new sample to that row.\n"
            "  Right-click a step for Clear or Shorten to here.\n"
            "  Pattern FX applies only to pattern playback.\n"
            "\n"
            "FX\n"
            "  Lowpass/Highpass filter the master output.\n"
            "  Drive adds distortion; Pitch shifts the next hit.\n"
            "  Pattern FX affects only Pattern Maker playback.\n"
            "\n"
            "Shortcuts\n"
            "  Pattern Station maximized: pad keys trigger row sounds.\n"
            "  Pattern Station maximized: library keys trigger assigned row sounds.\n"
            "  SPACE: Record toggle (Pattern Station maximized: Play/Stop)\n"
            "  L: Lowpass   H: Highpass   G: Grime Drive\n"
            "  [ / ]: Pitch down/up (next hit)\n"
            "  Arrow Up/Down: LP cutoff   Left/Right: HP cutoff\n"
            "  ESC: Quit\n"
            "\n"
            "Troubleshooting\n"
            "  If audio crackles, raise BLOCK or reduce CPU load.\n"
            "  If no sound, check your output device and volume.\n"
            "  If samples are missing, use Rescan in Library.\n"
        ))
        txt.config(state="disabled")

    help_btn.config(command=show_help)
    save_btn.config(command=save_selected_recording)
    open_btn.config(command=open_take_into_waveform)
    help_menu.add_command(label="Help / Shortcuts", command=show_help)
    help_menu.add_separator()
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
        "About",
        "SOUNDsbeatpad\nHelp menu: use Help / Shortcuts for navigation and keys."
    ))
    menubar.add_cascade(label="Help", menu=help_menu)

    # Preload samples
    if not os.path.isdir(ROOT):
        status_var.set("Sound pack not found.")
        messagebox.showerror(
            "Missing sound pack",
            "SOUNDsbeatpad could not find the bundled sounds folder.\n\n"
            f"Expected here:\n{ROOT}\n\n"
            "Keep the 'sounds' folder next to this program when moving it to another device."
        )
    errors = []
    for k, path in SOUNDS.items():
        try:
            load_sample(path)
        except Exception as e:
            errors.append(str(e))

    if errors:
        status_var.set("Some samples failed to load (see popup).")
        messagebox.showerror("Sample load errors", "\n\n".join(errors))
    else:
        status_var.set("Ready. Smash pads or keys.")

    refresh_samples()
    for k in pad_buttons:
        update_pad_button(k)

    wave_state = {"path": None, "audio": None, "sr": SR}
    drag_sel = {"active": False, "start": 0.0, "end": 1.0, "mode": "select", "offset": 0.0, "orig_start": 0.0, "orig_end": 0.0}

    def render_waveform():
        wave_canvas.delete("wave")
        path = wave_state["path"]
        audio = wave_state["audio"]
        w = max(200, wave_canvas.winfo_width())
        h = max(80, wave_canvas.winfo_height())
        if not path or audio is None or len(audio) == 0:
            wave_canvas.create_rectangle(0, 0, w, h, fill="#0b1f4b", outline="", tags="wave")
            wave_canvas.create_text(w // 2, h // 2, fill=COL_TEXT, text="No recording yet", tags="wave")
            return
        mid = h // 2
        wave_canvas.create_line(0, mid, w, mid, fill="#0f172a", tags="wave")
        step = max(1, int(len(audio) / w))
        samples = audio[::step]
        if len(samples) > w:
            samples = samples[:w]
        amp = np.abs(samples)
        for i, v in enumerate(amp):
            y = int(v * (h * 0.45))
            wave_canvas.create_line(i, mid - y, i, mid + y, fill=COL_ACCENT, tags="wave")

        # time ticks
        dur = len(audio) / SR
        tick_count = 5
        for i in range(tick_count + 1):
            x = int((i / tick_count) * w)
            t = (i / tick_count) * dur
            wave_canvas.create_line(x, 0, x, h, fill="#1e40af", tags="wave")
            wave_canvas.create_text(x + 2, 2, anchor="nw", fill=COL_TEXT, text=f"{t:.1f}s", tags="wave")

        # draw in/out markers
        x_in = int(trim_in.get() * w)
        x_out = int(trim_out.get() * w)
        wave_canvas.create_line(x_in, 0, x_in, h, fill=COL_ACCENT, tags="wave")
        wave_canvas.create_line(x_out, 0, x_out, h, fill=COL_ACCENT, tags="wave")

        # selection overlay
        x0 = min(x_in, x_out)
        x1 = max(x_in, x_out)
        wave_canvas.create_rectangle(x0, 0, x1, h, fill=COL_WARN, stipple="gray50", outline="", tags="wave")

    def update_trim_info():
        audio = wave_state["audio"]
        if audio is None:
            trim_info.set("In: 0.00s  Out: 0.00s")
            render_waveform()
            return
        dur = len(audio) / SR
        t_in = max(0.0, min(1.0, trim_in.get())) * dur
        t_out = max(0.0, min(1.0, trim_out.get())) * dur
        trim_info.set(f"In: {t_in:.2f}s  Out: {t_out:.2f}s")
        render_waveform()

    def load_waveform(path: str):
        if not path or not os.path.exists(path):
            wave_state["path"] = None
            wave_state["audio"] = None
            wave_info.set("Waveform: no selection")
            render_waveform()
            return
        try:
            data = load_audio_file(path)
            wave_state["path"] = path
            wave_state["audio"] = data
            wave_info.set(f"Waveform: {os.path.basename(path)}  ({len(data)/SR:.2f}s)")
            trim_in.set(0.0)
            trim_out.set(1.0)
            update_trim_info()
        except Exception as e:
            messagebox.showerror("Waveform error", str(e))

    def trim_selected():
        path = get_selected_recording()
        audio = wave_state["audio"]
        if not path or audio is None:
            messagebox.showinfo("Trim", "Select a recording first.")
            return
        a = float(trim_in.get())
        b = float(trim_out.get())
        if b <= a:
            messagebox.showerror("Trim", "Out must be greater than In.")
            return
        start = int(max(0.0, min(1.0, a)) * len(audio))
        end = int(max(0.0, min(1.0, b)) * len(audio))
        trimmed = audio[start:end]
        if len(trimmed) == 0:
            messagebox.showerror("Trim", "Trimmed region is empty.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(path))[0]
        outname = f"{base}_trim_{ts}.wav"
        sf.write(outname, trimmed, SR, subtype="PCM_16")
        outpath = os.path.abspath(outname)
        ENGINE.last_recording_path = outpath
        add_recording(outpath)
        load_waveform(outpath)
        add_open_take(outpath)
        status_var.set(f"Trimmed and saved: {outname}")

    def cut_selection():
        path = get_selected_recording()
        audio = wave_state["audio"]
        if not path or audio is None:
            messagebox.showinfo("Cut", "Select a recording first.")
            return
        a = float(trim_in.get())
        b = float(trim_out.get())
        if b <= a:
            messagebox.showerror("Cut", "Out must be greater than In.")
            return
        start = int(max(0.0, min(1.0, a)) * len(audio))
        end = int(max(0.0, min(1.0, b)) * len(audio))
        if start <= 0 and end >= len(audio):
            messagebox.showerror("Cut", "Selection covers the entire audio.")
            return
        keep = np.concatenate([audio[:start], audio[end:]]).astype(np.float32)
        if len(keep) == 0:
            messagebox.showerror("Cut", "Resulting audio is empty.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(path))[0]
        outname = f"{base}_cut_{ts}.wav"
        sf.write(outname, keep, SR, subtype="PCM_16")
        outpath = os.path.abspath(outname)
        ENGINE.last_recording_path = outpath
        add_recording(outpath)
        load_waveform(outpath)
        add_open_take(outpath)
        status_var.set(f"Cut and saved: {outname}")

    def move_selection(orig_start: float, orig_end: float, new_start: float):
        path = get_selected_recording()
        audio = wave_state["audio"]
        if not path or audio is None:
            return
        if orig_end <= orig_start:
            return
        total = len(audio)
        seg_len = int(max(1, (orig_end - orig_start) * total))
        o_start = int(max(0, min(total - 1, orig_start * total)))
        o_end = min(total, o_start + seg_len)
        seg = audio[o_start:o_end]
        if len(seg) == 0:
            return
        n_start = int(max(0, min(total - len(seg), new_start * total)))
        if n_start == o_start:
            return
        moved = audio.copy()
        moved[o_start:o_end] = 0.0
        moved[n_start:n_start + len(seg)] = seg
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(path))[0]
        outname = f"{base}_move_{ts}.wav"
        sf.write(outname, moved, SR, subtype="PCM_16")
        outpath = os.path.abspath(outname)
        ENGINE.last_recording_path = outpath
        add_recording(outpath)
        load_waveform(outpath)
        add_open_take(outpath)
        new_start_norm = n_start / len(moved)
        new_end_norm = (n_start + len(seg)) / len(moved)
        trim_in.set(new_start_norm)
        trim_out.set(new_end_norm)
        update_trim_info()
        status_var.set(f"Moved selection: {outname}")

    def on_rec_select(_):
        load_waveform(get_selected_recording())

    rec_list.bind("<<ListboxSelect>>", on_rec_select)
    def on_rec_right_click(event):
        idx = rec_list.nearest(event.y)
        if idx >= 0:
            rec_list.selection_clear(0, tk.END)
            rec_list.selection_set(idx)
            rec_menu.tk_popup(event.x_root, event.y_root)

    rec_list.bind("<Button-3>", on_rec_right_click)
    trim_in.trace_add("write", lambda *_: update_trim_info())
    trim_out.trace_add("write", lambda *_: update_trim_info())
    wave_canvas.bind("<Configure>", lambda _e: render_waveform())

    def coord_to_norm(x: int) -> float:
        w = max(1, wave_canvas.winfo_width())
        return max(0.0, min(1.0, x / float(w)))

    def on_wave_press(event):
        drag_sel["active"] = True
        click = coord_to_norm(event.x)
        cur_start = min(trim_in.get(), trim_out.get())
        cur_end = max(trim_in.get(), trim_out.get())
        shift_down = bool(event.state & 0x0001)
        if shift_down and wave_state["audio"] is not None and (cur_end - cur_start) > 0.001 and cur_start <= click <= cur_end:
            drag_sel["mode"] = "move"
            drag_sel["offset"] = click - cur_start
            drag_sel["orig_start"] = cur_start
            drag_sel["orig_end"] = cur_end
            new_start = max(0.0, min(1.0 - (cur_end - cur_start), click - drag_sel["offset"]))
            trim_in.set(new_start)
            trim_out.set(new_start + (cur_end - cur_start))
        else:
            drag_sel["mode"] = "select"
            drag_sel["start"] = click
            drag_sel["end"] = drag_sel["start"]
            trim_in.set(min(drag_sel["start"], drag_sel["end"]))
            trim_out.set(max(drag_sel["start"], drag_sel["end"]))

    def on_wave_drag(event):
        if not drag_sel["active"]:
            return
        pos = coord_to_norm(event.x)
        if drag_sel["mode"] == "move":
            length = max(0.0, drag_sel["orig_end"] - drag_sel["orig_start"])
            new_start = max(0.0, min(1.0 - length, pos - drag_sel["offset"]))
            trim_in.set(new_start)
            trim_out.set(new_start + length)
        else:
            drag_sel["end"] = pos
            trim_in.set(min(drag_sel["start"], drag_sel["end"]))
            trim_out.set(max(drag_sel["start"], drag_sel["end"]))

    def on_wave_release(_event):
        drag_sel["active"] = False
        if drag_sel["mode"] == "move":
            new_start = min(trim_in.get(), trim_out.get())
            move_selection(drag_sel["orig_start"], drag_sel["orig_end"], new_start)
        drag_sel["mode"] = "select"

    wave_canvas.bind("<ButtonPress-1>", on_wave_press)
    wave_canvas.bind("<B1-Motion>", on_wave_drag)
    wave_canvas.bind("<ButtonRelease-1>", on_wave_release)

    def update_status():
        rec_light.config(bg=(COL_WARN if ENGINE.recording else COL_MUTED))
        rec_var.set("REC: on" if ENGINE.recording else "REC: off")
        if not ENGINE.playback_active:
            pb_var.set("Playback: stopped")
        if len(all_samples) > KEY_LIMIT:
            status_var.set(f"Keys assigned to first {KEY_LIMIT} samples. Use search to find sounds.")

    METER_BARS = 18
    meter_levels = [0.0] * METER_BARS

    def draw_meter():
        meter_canvas.delete("bar")
        w = max(1, meter_canvas.winfo_width())
        h = max(1, meter_canvas.winfo_height())
        gap = 2
        bar_w = max(2, int((w - (METER_BARS + 1) * gap) / METER_BARS))
        x = gap
        for i, level in enumerate(meter_levels):
            bar_h = int(level * (h - 2))
            y0 = h - 1 - bar_h
            if level < 0.55:
                color = "#39ff14"
            elif level < 0.8:
                color = "#ffea00"
            else:
                color = "#ff2e63"
            meter_canvas.create_rectangle(x, y0, x + bar_w, h - 1, fill=color, outline="", tags="bar")
            x += bar_w + gap

    def update_meter():
        level = min(1.0, max(0.0, LAST_PEAK))
        t = time.time()
        for i in range(METER_BARS):
            wobble = 0.85 + 0.15 * np.sin(t * 6.0 + i)
            target = level * wobble
            if target > meter_levels[i]:
                meter_levels[i] = target
            else:
                meter_levels[i] = max(0.0, meter_levels[i] - 0.03)
        draw_meter()
        update_status()
        root.after(60, update_meter)
    update_meter()

    def set_master(val: float):
        global MASTER
        MASTER = max(0.0, min(1.0, val))
        master_var.set(MASTER)

    def on_key(event):
        k = event.keysym.lower()

        if k == "escape":
            root.destroy()
            return
        if k == "f1":
            show_help()
            return

        if k == "space":
            if is_station_only(station_pattern):
                if seq_playing["on"]:
                    stop_seq()
                else:
                    start_seq()
                return
            toggle_record()
            return

        if k == "l":
            lp_var.set(not lp_var.get())
            sync_fx()
            return
        if k == "h":
            hp_var.set(not hp_var.get())
            sync_fx()
            return
        if k == "g":
            drive_var.set(not drive_var.get())
            sync_fx()
            return

        if k == "bracketleft":
            pitch_var.set(max(-12, pitch_var.get() - 1))
            pitch_scale.set(pitch_var.get())
            return
        if k == "bracketright":
            pitch_var.set(min(12, pitch_var.get() + 1))
            pitch_scale.set(pitch_var.get())
            return

        if k == "up":
            ENGINE.lp_cutoff = min(14000.0, ENGINE.lp_cutoff + 150.0)
            lp_scale.set(ENGINE.lp_cutoff)
            return
        if k == "down":
            ENGINE.lp_cutoff = max(80.0, ENGINE.lp_cutoff - 150.0)
            lp_scale.set(ENGINE.lp_cutoff)
            return
        if k == "left":
            ENGINE.hp_cutoff = max(20.0, ENGINE.hp_cutoff - 20.0)
            hp_scale.set(ENGINE.hp_cutoff)
            return
        if k == "right":
            ENGINE.hp_cutoff = min(2000.0, ENGINE.hp_cutoff + 20.0)
            hp_scale.set(ENGINE.hp_cutoff)
            return

        if is_station_only(station_pattern):
            if k in PAD_KEYS and k in SOUNDS:
                trigger_sample_path(SOUNDS[k])
                return
            if k in seq_keymap:
                trigger_sample_path(seq_keymap[k])
                return

    root.bind("<KeyPress>", on_key)
    return root

# -----------------------------
# Main
# -----------------------------
def main():
    # start audio stream first
    stream = sd.OutputStream(
        samplerate=SR,
        channels=1,
        blocksize=BLOCK,
        callback=audio_callback,
        dtype="float32",
        latency="low",
    )

    with stream:
        root = build_ui()
        root.mainloop()

if __name__ == "__main__":
    main()
