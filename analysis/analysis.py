from analysis import data_loading
# import data_loading
import numpy as np
import matplotlib.pyplot as plt

df = data_loading.load_any()
fs = data_loading.calculate_freq(df)

def time():
    time = df["time"].to_numpy(float)

    return time
    
def breath_data():
    result = df["resp"].to_numpy(float)
    return result

def ecg_data():
    ecg = df["ecg"].to_numpy(float)

    return ecg

def nerve_data():
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

def get_aligned_metric_data(
    bin_width=0.05,
    max_time=600
):
    # Raw signals
    t_raw = df["time"].to_numpy(float)
    mask = t_raw <= max_time
    t_raw = t_raw[mask]

    respiration = df["resp"].to_numpy(float)[mask]
    ecg = df["ecg"].to_numpy(float)[mask]
    nerve = df["nerve"].to_numpy(float)[mask]

    # Spike detection
    nerve_filt = data_loading.pre_process(df, fs)
    thr = data_loading.compute_threshold(nerve_filt)
    spike_times = np.asarray(
        data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values),
        dtype=float
    )

    # Keep spikes in plotted range
    spike_times = spike_times[(spike_times >= t_raw[0]) & (spike_times <= t_raw[-1])]

    # Bin spikes into firing rate
    bins = np.arange(t_raw[0], t_raw[-1] + bin_width, bin_width)
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    neuron_firing_rate = spike_counts / bin_width
    t_rate = (bins[:-1] + bins[1:]) / 2

    return t_raw, t_rate, neuron_firing_rate, respiration, ecg, nerve


import numpy as np

# def compute_cmi():
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

def epochs_ecg():
    epochs = []
    timestamps = data_loading.detecting_rpeaks_ecg(fs, df)
    for i in timestamps:
        epochs.append([i, (i-1), (i+1)])
    # print(len(epochs))
    return epochs

def epochs_breath_peaks():
    epochs = []
    result = data_loading.detect_breathing_peaks_and_troughs(fs, df)
    timestamps = result["peak_times"]
    for i in timestamps:
        epochs.append([i, (i-3), (i+3)])
    # print(len(epochs))
    return epochs

def epochs_breath_troughs():
    epochs = []
    result = data_loading.detect_breathing_peaks_and_troughs(fs, df)
    timestamps = result["trough_times"]
    for i in timestamps:
        epochs.append([i, (i-3), (i+3)])
    # print(len(epochs))
    return epochs

def obtain_spikes():
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

if __name__ == "__main__":
    spikes = obtain_spikes()
    ecg_epochs = epochs_ecg()
    breath_epochs = epochs_breath_peaks()
   
  

    # print(spikes[:10])

    # ecg_overlap = combining_spike_and_epoch(spikes, ecg_epochs)
    # breath_overlap = combining_spike_and_epoch(spikes, breath_epochs)

    # t_ecg, rate_ecg, sem_ecg = psth_from_overlap(ecg_overlap, ecg_epochs, bin_width=0.005)
    # t_breath, rate_breath, sem_breath = psth_from_overlap(breath_overlap, breath_epochs, bin_width=0.1)

    # plt.figure(figsize=(12,5))
    # plt.subplot(1,2,1)
    # plt.title("PSTH around ECG R-peaks")
    # plt.plot(t_ecg, rate_ecg, label="Mean firing rate")
    # plt.fill_between(t_ecg, rate_ecg - sem_ecg, rate_ecg + sem_ecg, color="blue", alpha=0.2, label="SEM")
    # plt.axvline(0, color="red", linestyle="--", label="R-peak")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Firing Rate (spikes/s)")
    # plt.legend()

    # plt.subplot(1,2,2)
    # plt.title("PSTH around Breathing Peaks")
    # plt.plot(t_breath, rate_breath, label="Mean firing rate")
    # plt.fill_between(t_breath, rate_breath - sem_breath, rate_breath + sem_breath,
    #                  color="green", alpha=0.2, label="SEM")
    # plt.axvline(0, color="orange", linestyle="--", label="Breath peak")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Firing Rate (spikes/s)")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()