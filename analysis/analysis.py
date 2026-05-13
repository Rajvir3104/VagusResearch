from analysis import data_loading
from analysis import app_loaders
# import data_loading
import numpy as np
import matplotlib.pyplot as plt
# Ankle Pulse	ECG	Finometer	MSNA bandpass
# df = data_loading.load_parquet()
# df.columns = ["time", "resp" , "bp", "ecg", "nerve"]
# fs = data_loading.calculate_freq(df)

def time(df):
    time = df["time"].to_numpy(float)

    return time
    
def breath_data(df):
    result = df["resp"].to_numpy(float)
    return result

def ecg_data(df):
    ecg = df["ecg"].to_numpy(float)

    return ecg

def nerve_data(df, fs):
    nerve_filt = data_loading.pre_process(df, fs)
    thr = data_loading.compute_threshold(nerve_filt)
    
    spike_times = data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values)

    return spike_times

# def triple_graph_data():
#     time = df["time"].to_numpy(float)
#     time = time[time <= 500] 
#     breath = df["resp"].to_numpy(float)
#     breath = breath[:len(time)]
#     ecg = df["ecg"].to_numpy(float)
#     ecg = ecg[:len(time)]

#     nerve_filt = data_loading.pre_process(df, fs)
#     thr = data_loading.compute_threshold(nerve_filt)
    
#     spike_times = data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values)

#     return time, breath, ecg, spike_times

import numpy as np

# def get_aligned_metric_data(
#     bin_width=0.05,
#     max_time=600,
    
# ):
#     # Raw signals
#     t_raw = df["time"].to_numpy(float)
#     if t_raw.size == 0:
#         raise ValueError("Data frame contains no time values.")

#     mask = t_raw <= max_time
#     if not np.any(mask):
#         mask = np.ones_like(t_raw, dtype=bool)

#     t_raw = t_raw[mask]
#     respiration = df["resp"].to_numpy(float)[mask]
#     ecg = df["ecg"].to_numpy(float)[mask]
#     nerve = df["nerve"].to_numpy(float)[mask]

#     # Spike detection
#     nerve_filt = data_loading.pre_process(df, fs)
#     thr = data_loading.compute_threshold(nerve_filt)
#     spike_times = np.asarray(
#         data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values),
#         dtype=float,
#     )

#     # Keep spikes in plotted range
#     if spike_times.size > 0:
#         spike_times = spike_times[(spike_times >= t_raw[0]) & (spike_times <= t_raw[-1])]
#     else:
#         spike_times = np.array([], dtype=float)

#     # Bin spikes into firing rate
#     bins = np.arange(t_raw[0], t_raw[-1] + bin_width, bin_width)
#     spike_counts, _ = np.histogram(spike_times, bins=bins)
#     neuron_firing_rate = spike_counts / bin_width
#     t_rate = (bins[:-1] + bins[1:]) / 2

#     return t_raw, t_rate, neuron_firing_rate, respiration, ecg, nerve



# def compute_cmi(df, fs):
#     ecg_r_peaks = data_loading.detecting_rpeaks_ecg(fs, df)
#     resp_peaks = data_loading.detect_breathing_peaks_and_troughs(fs, df)["peak_times"]

#     rr = np.diff(ecg_r_peaks)
#     hr = 60.0 / rr
#     hr_times = ecg_r_peaks[:-1]

#     cmi_values = []

#     for i in range(len(resp_peaks) - 1):
#         start = resp_peaks[i]
#         end = resp_peaks[i + 1]

#         hr_segment = hr[(hr_times >= start) & (hr_times < end)]

#         if len(hr_segment) < 2:
#             cmi_values.append(np.nan)
#             continue

#         hr_max = np.max(hr_segment)
#         hr_min = np.min(hr_segment)
#         hr_mean = np.mean(hr_segment)

#         cmi = ((hr_max - hr_min) / hr_mean) * 100.0
#         cmi_values.append(cmi)

#     return np.array(cmi_values)

def epochs_ecg(df, fs):
    epochs = []
    timestamps = data_loading.detecting_rpeaks_ecg(fs, df)
    for i in timestamps:
        epochs.append([i, (i-2), (i+2)])
    # print(len(epochs))
    return epochs

def epochs_breath_peaks(df, fs):
    epochs = []
    result = data_loading.detect_breathing_peaks_and_troughs(fs, df)
    timestamps = result["peak_times"]
    for i in timestamps:
        epochs.append([i, (i-20), (i+20)])
    # print(len(epochs))
    return epochs

def epochs_breath_troughs(df, fs):
    epochs = []
    result = data_loading.detect_breathing_peaks_and_troughs(fs, df)
    timestamps = result["trough_times"]
    for i in timestamps:
        epochs.append([i, (i-20), (i+20)])
    # print(len(epochs))
    return epochs

def obtain_spikes(df, fs):
    nerve_filt = data_loading.pre_process(df, fs)
    thr = data_loading.compute_threshold(nerve_filt)
    
    spike_times = data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values)
    
    return spike_times

def combining_spike_and_epoch(spike_t, epochs):
    # --- make spike_t a flat 1D float array no matter what comes in ---
    if isinstance(spike_t, (list, tuple)) and len(spike_t) > 0 and isinstance(spike_t[0], (list, tuple, np.ndarray)):
        spike_t = np.concatenate([np.asarray(s, dtype=float).ravel() for s in spike_t if len(s) > 0])
    else:
        spike_t = np.asarray(spike_t, dtype=float).ravel()

    overlap = []
    for (t0, start, end) in epochs:
        mask = (spike_t >= start) & (spike_t < end)
        overlap.append(spike_t[mask].tolist())
    return overlap



def psth_from_overlap(overlap, epochs, bin_width=0.01):
    # assumes all epochs have same duration
    t0, start, end = epochs[0]
    left = start - t0    # negative
    right = end - t0     # positive

    bins = np.arange(left, right + bin_width, bin_width)
    counts_per_epoch = []

    for spikes_abs, (t0, start, end) in zip(overlap, epochs):
        spikes_rel = np.array(spikes_abs) - t0  # align around event
        counts, _ = np.histogram(spikes_rel, bins=bins)
        counts_per_epoch.append(counts)

    counts_per_epoch = np.vstack(counts_per_epoch)  

    mean_counts = counts_per_epoch.mean(axis=0)
    sem_counts = counts_per_epoch.std(axis=0, ddof=1) / np.sqrt(counts_per_epoch.shape[0])

    # Convert to rate (spikes/sec) so it’s independent of bin width
    mean_rate = mean_counts / bin_width
    sem_rate = sem_counts / bin_width

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, mean_rate, sem_rate


def average_aligned_epochs(signal, t, epochs, bin_width=0.005):
    if len(epochs) == 0:
        return np.array([]), np.array([])

    # Use the first epoch to define a common relative time grid around each event.
    t0, start, end = epochs[0]
    centers = (np.arange(start - t0, end - t0 + bin_width, bin_width)[:-1] +
               np.arange(start - t0, end - t0 + bin_width, bin_width)[1:]) / 2

    aligned = []
    for t0, start, end in epochs:
        mask = (t >= start) & (t < end)
        if not np.any(mask):
            continue

        t_rel = t[mask] - t0
        aligned_sig = np.interp(centers, t_rel, signal[mask], left=np.nan, right=np.nan)
        aligned.append(aligned_sig)

    if len(aligned) == 0:
        return centers, np.full_like(centers, np.nan)

    return centers, np.nanmean(np.vstack(aligned), axis=0)
