import os
import sys
from pathlib import Path

import dash_uploader as du
import plotly.graph_objects as go
from dash import Output, no_update

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.app_loaders import load_any
from dashboard.data import run_dashboard_analysis
from dashboard.figures import psth_figure


@du.callback(
    output=[
        Output("upload-status", "children"),
        Output("ecg-psth-graph", "figure"),
        Output("breath-peak-psth-graph", "figure"),
        Output("breath-trough-psth-graph", "figure"),
    ],
    id="upload-data",
)
def process_uploaded_file(file_paths):
    if not file_paths:
        return (
            "No file uploaded yet.",
            go.Figure(),
            go.Figure(),
            go.Figure(),
        )

    file_path = file_paths[0]
    filename = os.path.basename(file_path)

    if not filename.lower().endswith((".csv", ".txt")):
        return (
            "Please upload a .csv or .txt file.",
            no_update,
            no_update,
            no_update,
        )

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            decoded_text = f.read()

        df = load_any(decoded_text)

        print("=== SHAPE ===", flush=True)
        print(df.shape, flush=True)
        print("=== COLUMNS ===", flush=True)
        print(df.columns.tolist(), flush=True)
        print("=== HEAD ===", flush=True)
        print(df.head(), flush=True)

        if df.empty:
            return (
                "File loaded, but no rows were parsed.",
                no_update,
                no_update,
                no_update,
            )

        results = run_dashboard_analysis(df)

        ecg_fig = psth_figure(
            results["t_ecg"],
            results["rate_ecg"],
            results["sem_ecg"],
            title="ECG-aligned PSTH",
            xlabel="Time relative to R-peak (s)",
            signal_x=results["t_ecg_signal"],
            ecg=results["ecg_mean_ecg"],
        )

        breath_peak_fig = psth_figure(
            results["t_breath_peak"],
            results["rate_breath_peak"],
            results["sem_breath_peak"],
            title="Respiration-aligned PSTH",
            xlabel="Time relative to respiration peak (s)",
            signal_x=results["t_breath_peak_signal"],
            resp=results["resp_mean_breath_peak"],
        )

        breath_trough_fig = psth_figure(
            results["t_breath_trough"],
            results["rate_breath_trough"],
            results["sem_breath_trough"],
            title="Respiration-aligned PSTH",
            xlabel="Time relative to respiration trough (s)",
            signal_x=results["t_breath_trough_signal"],
            resp=results["resp_mean_breath_trough"],
        )

        return (
            f"Successfully processed {filename}. Rows: {len(df):,}. "
            f"Detected spikes: {len(results['spike_times']):,}.",
            ecg_fig,
            breath_peak_fig,
            breath_trough_fig,
        )

    except Exception as e:
        return (
            f"Error processing file: {e}",
            no_update,
            no_update,
            no_update,
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)