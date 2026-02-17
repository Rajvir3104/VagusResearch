from analysis import data_loading
# import data_loading
import numpy as np
import matplotlib.pyplot as plt

df = data_loading.load_any()
fs = data_loading.calculate_freq(df)

def epochs_ecg():
    epochs = []
    timestamps = data_loading.detecting_rpeaks_ecg(fs, df)
    for i in timestamps:
        epochs.append([i, (i-1), (i+1)])
    print(len(epochs))
    return epochs

def epochs_breath():
    epochs = []
    timestamps = data_loading.detecting_rpeaks_breathing(fs, df)
    for i in timestamps:
        epochs.append([i, (i-3), (i+3)])
    print(len(epochs))
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

    counts_per_epoch = np.vstack(counts_per_epoch)  # shape: (n_epochs, n_bins-1)

    mean_counts = counts_per_epoch.mean(axis=0)
    sem_counts = counts_per_epoch.std(axis=0, ddof=1) / np.sqrt(counts_per_epoch.shape[0])

    # Convert to rate (spikes/sec) so it’s independent of bin width
    mean_rate = mean_counts / bin_width
    sem_rate = sem_counts / bin_width

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, mean_rate, sem_rate

def plot_breath(df):
    plt.figure(figsize=(14,5))
    plt.plot(df["time"], df["breath"])
    plt.title("Full Breathing timeline (debug)")
    plt.xlabel("Time (s)")
    plt.show()



    # -------- BREATH GRAPH --------
    plt.figure(figsize=(14,5))

    # plot breathing signal if column exists
    if "breath" in df.columns:
        plt.plot(time, df["breath"], label="Breathing signal", alpha=0.8)

    # mark peaks
    plt.scatter(breath_peaks, [0]*len(breath_peaks),
                marker="|", s=300, label="Detected breath peaks")

    plt.title("Full Breathing timeline (debug)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    spikes = obtain_spikes()
    print("Spikes:", spikes)

    e_ecg = epochs_ecg()
    print("ECG Epochs:", e_ecg)

    e_breath = epochs_breath()
    print("Breath Epochs:", e_breath)

    ov_ecg = combining_spike_and_epoch(spikes, e_ecg)
    print("ECG Overlap:", ov_ecg)

    ov_breath = combining_spike_and_epoch(spikes, e_breath)
    print("Breath Overlap:", ov_breath)

    t_ecg, mean_ecg, sem_ecg = psth_from_overlap(ov_ecg, e_ecg, bin_width=0.005)
    print("ECG PSTH:", t_ecg, mean_ecg, sem_ecg)

    t_breath, mean_breath, sem_breath = psth_from_overlap(ov_breath, e_breath, bin_width=0.1)
    print("Breath PSTH:", t_breath, mean_breath, sem_breath)

    plot_breath(df)
# import numpy as np
# import matplotlib.pyplot as plt
# from analysis import data_loading

# df = data_loading.load_any()          # <- works for both formats
# fs = data_loading.calculate_freq(df)

# def epochs_ecg(pre=0.3, post=0.3):
#     epochs = []
#     timestamps = data_loading.detecting_rpeaks_ecg(fs, df)
#     for t0 in timestamps:
#         epochs.append([t0, t0 - pre, t0 + post])
#     return epochs

# def epochs_resp(pre=3.0, post=3.0):
#     epochs = []
#     timestamps = data_loading.detecting_rpeaks_breathing(fs, df)
#     for t0 in timestamps:
#         epochs.append([t0, t0 - pre, t0 + post])
#     return epochs

# def obtain_spikes():
#     nerve_filt = data_loading.pre_process(df, fs)          # bandpassed "nerve"
#     thr = data_loading.compute_threshold(nerve_filt)       # positive threshold
#     spikes = data_loading.detect_spikes(thr, nerve_filt, fs, t=df["time"].values)
#     return spikes, nerve_filt, thr

# def combining_spike_and_epoch(spike_t, epochs):
#     spike_t = np.asarray(spike_t)
#     overlap = []
#     for (t0, start, end) in epochs:
#         mask = (spike_t >= start) & (spike_t < end)
#         overlap.append(spike_t[mask].tolist())
#     return overlap

# def psth_from_overlap(overlap, epochs, bin_width=0.01):
#     t0, start, end = epochs[0]
#     left = start - t0
#     right = end - t0

#     bins = np.arange(left, right + bin_width, bin_width)
#     counts_per_epoch = []

#     for spikes_abs, (t0, start, end) in zip(overlap, epochs):
#         spikes_rel = np.asarray(spikes_abs) - t0
#         counts, _ = np.histogram(spikes_rel, bins=bins)
#         counts_per_epoch.append(counts)

#     counts_per_epoch = np.vstack(counts_per_epoch)
#     mean_counts = counts_per_epoch.mean(axis=0)
#     sem_counts = counts_per_epoch.std(axis=0, ddof=1) / np.sqrt(counts_per_epoch.shape[0])

#     mean_rate = mean_counts / bin_width
#     sem_rate = sem_counts / bin_width
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     return bin_centers, mean_rate, sem_rate

# def plot_psth(t, mean_rate, sem_rate, title, xlabel):
#     plt.figure()
#     plt.plot(t, mean_rate)
#     plt.fill_between(t, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.2)
#     plt.axvline(0, linestyle="--")
#     plt.xlabel(xlabel)
#     plt.ylabel("Mean rate (events/s)")
#     plt.title(title)
#     plt.tight_layout()

# def plot_sanity_traces(df, fs, nerve_filt, thr, seconds=10.0, start_time=None):
#     """
#     Plots short window: ECG, Resp, raw nerve, filtered nerve w/ thresholds.
#     """
#     t = df["time"].to_numpy(float)
#     ecg = df["ecg"].to_numpy(float)
#     resp = df["resp"].to_numpy(float)
#     nerve_raw = df["nerve"].to_numpy(float)

#     if start_time is None:
#         start_time = t[0]

#     end_time = start_time + seconds
#     idx = (t >= start_time) & (t <= end_time)

#     tw = t[idx] - start_time
#     ecgw = ecg[idx]
#     respw = resp[idx]
#     nraw = nerve_raw[idx]
#     nfilt = nerve_filt[idx]

#     # light conditioning for viewing
#     ecgw = ecgw - np.mean(ecgw)
#     respw = respw - np.median(respw)

#     plt.figure()
#     plt.plot(tw, ecgw)
#     plt.title("ECG (window)")
#     plt.xlabel("Time (s)")
#     plt.tight_layout()

#     plt.figure()
#     plt.plot(tw, respw)
#     plt.title("Respiration (window)")
#     plt.xlabel("Time (s)")
#     plt.tight_layout()

#     plt.figure()
#     plt.plot(tw, nraw)
#     plt.title("Nerve channel RAW (window)")
#     plt.xlabel("Time (s)")
#     plt.tight_layout()

#     plt.figure()
#     plt.plot(tw, nfilt)
#     plt.axhline(+thr, linestyle="--")
#     plt.axhline(-thr, linestyle="--")
#     plt.title("Nerve channel bandpassed + thresholds (window)")
#     plt.xlabel("Time (s)")
#     plt.tight_layout()

# if __name__ == "__main__":
#     spikes, nerve_filt, thr = obtain_spikes()

#     # ---- sanity check plots ----
#     plot_sanity_traces(df, fs, nerve_filt, thr, seconds=10.0)

#     # ---- ECG-aligned PSTH ----
#     e_ecg = epochs_ecg(pre=0.3, post=0.3)
#     ov_ecg = combining_spike_and_epoch(spikes, e_ecg)
#     t_ecg, mean_ecg, sem_ecg = psth_from_overlap(ov_ecg, e_ecg, bin_width=0.005)
#     plot_psth(t_ecg, mean_ecg, sem_ecg,
#               title="ECG-aligned PSTH",
#               xlabel="Time relative to R-peak (s)")

#     # ---- Respiration-aligned PSTH ----
#     e_resp = epochs_resp(pre=3.0, post=3.0)
#     ov_resp = combining_spike_and_epoch(spikes, e_resp)
#     t_resp, mean_resp, sem_resp = psth_from_overlap(ov_resp, e_resp, bin_width=0.1)
#     plot_psth(t_resp, mean_resp, sem_resp,
#               title="Respiration-aligned PSTH",
#               xlabel="Time relative to respiration peak (s)")

#     plt.show()
