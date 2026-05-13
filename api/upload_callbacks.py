import base64
import os
import tempfile
import sys
from pathlib import Path
import numpy as np

# allow imports from analysis/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
from dash import Input, Output, State, callback, no_update

from analysis.app_loaders import load_any
from api.data import run_dashboard_analysis
from api.figures import psth_figure


@callback(
    Output("upload-status", "children"),
    Output("ecg-psth-graph", "figure"),
    Output("breath-peak-psth-graph", "figure"),
    Output("breath-trough-psth-graph", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def process_uploaded_file(contents, filename):
    if contents is None:
        return (
            "No file uploaded yet.",
            go.Figure(),
            go.Figure(),
            go.Figure(),
        )

    if not filename.lower().endswith((".csv", ".txt")):
        return (
            "Please upload a .csv or .txt file.",
            no_update,
            no_update,
            no_update,
        )

    temp_path = None

    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        suffix = os.path.splitext(filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(decoded)
            temp_path = temp_file.name

        with open(temp_path, "r", encoding="utf-8", errors="replace") as f:
            decoded_text = f.read()

        df = load_any(decoded_text)
        print("=== DATAFRAME INFO ===", flush=True)
        print(df, flush=True)

        print("=== SHAPE ===", flush=True)
        print(df.shape, flush=True)

        print("=== COLUMNS ===", flush=True)
        print(df.columns.tolist(), flush=True)

        print("=== DTYPES ===", flush=True)
        print(df.dtypes, flush=True)

        print("=== HEAD ===", flush=True)
        print(df.head(), flush=True)

        print("=== TAIL ===", flush=True)
        print(df.tail(), flush=True)
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
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)