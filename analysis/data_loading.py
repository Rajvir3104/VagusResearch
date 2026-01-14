import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

fname = "../../playing_around/Vagus_nerve_test_Oct_9_HE.txt"

# ---------- Loaders ----------
def load_data_old_4col():  # your old format
    df = pd.read_csv(fname, sep=r"\s+", header=None)
    df.columns = ["time", "breath", "ecg", "vagus"]
    return df

def load_data_msna_5col():  # header + 5 numeric cols
    rows = []
    with open(fname, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    [float(x) for x in parts]
                    rows.append(parts)
                except ValueError:
                    pass
    df = pd.DataFrame(rows, columns=["time", "ecg", "bp", "resp", "raw_arm"]).astype(float)
    return df

# ---------- Canonicalize ----------
def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a NEW df with consistent column names:
      time, ecg, resp, nerve, (optional) bp
    Works with both:
      old: time breath ecg vagus
      msna: time ecg bp resp raw_arm
    """
    cols = set(df.columns)

    # old format
    if {"time", "breath", "ecg", "vagus"}.issubset(cols):
        out = df.rename(columns={"breath": "resp", "vagus": "nerve"}).copy()
        return out

    # msna format
    if {"time", "ecg", "resp", "raw_arm"}.issubset(cols):
        out = df.rename(columns={"raw_arm": "nerve"}).copy()
        return out

    raise ValueError(f"Unknown data format. Columns found: {sorted(df.columns)}")


def load_any():
    """
    Tries old format first (fast). If it fails, falls back to MSNA-style loader.
    """
    try:
        df = load_data_old_4col()
        df = canonicalize(df)
        return df
    except Exception:
        df = load_data_msna_5col()
        df = canonicalize(df)
        return df

# ---------- Common signal utils ----------
def calculate_freq(df):
    dt = np.diff(df["time"].values)
    dt_med = np.median(dt)
    return 1.0 / dt_med

def bandpass(x, fs, low, high, order=2):
    nyq = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype="bandpass")
    return filtfilt(b, a, x)

def lowpass(x, fs, cutoff=40, order=3):
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)

# ---------- Nerve preprocessing (works for BOTH) ----------
def pre_process(df, fs, low_cut=300.0, high_cut=3000.0):
    nerve_raw = df["nerve"].to_numpy(float)
    return bandpass(nerve_raw, fs, low_cut, high_cut, order=2)

def compute_threshold(signal, k=3.25):
    sigma_n = np.median(np.abs(signal)) / 0.6745
    return k * sigma_n

def detect_spikes(thr, signal, fs, t):
    t = np.asarray(t, dtype=float)
    refrac_ms = 1.0
    refrac = int(refrac_ms * 1e-3 * fs)

    cross_neg = np.where((signal[1:] < -thr) & (signal[:-1] >= -thr))[0] + 1
    cross_pos = np.where((signal[1:] >  thr) & (signal[:-1] <=  thr))[0] + 1
    cross = np.sort(np.concatenate((cross_neg, cross_pos)))

    keep = []
    last = -np.inf
    for i in cross:
        if i - last > refrac:
            keep.append(i)
            last = i

    spike_idx = np.asarray(keep, dtype=int)
    return t[spike_idx]

def detecting_rpeaks_ecg(fs, df):
    t = df["time"].to_numpy(float)
    ecg = df["ecg"].to_numpy(float)
    ecg = ecg - np.mean(ecg)
    ecg_f = lowpass(ecg, fs=fs, cutoff=40)

    min_rr_s = 0.5
    min_distance = int(min_rr_s * fs)

    r_idx, _ = find_peaks(ecg_f, distance=min_distance, prominence=0.5)
    return t[r_idx]

def detecting_rpeaks_breathing(fs, df):
    t = df["time"].to_numpy(float)
    resp_raw = df["resp"].to_numpy(float)

    resp = lowpass(resp_raw, fs=fs, cutoff=2.0)
    resp = resp - np.median(resp)

    if np.ptp(resp[:int(fs*10)]) and np.abs(np.min(resp)) > np.max(resp):
        resp = -resp

    min_breath_s = 0.6
    pk_idx, _ = find_peaks(resp, distance=int(min_breath_s*fs),
                           prominence=np.std(resp)*0.3)
    return t[pk_idx]


