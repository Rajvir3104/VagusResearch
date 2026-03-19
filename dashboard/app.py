# import sys
# from pathlib import Path
# import numpy as np

# # allow imports from analysis/
# sys.path.append(str(Path(__file__).resolve().parents[1]))

# from analysis.analysis import (
#     epochs_ecg,
#     epochs_breath_peaks,
#     epochs_breath_troughs,
#     obtain_spikes,
#     combining_spike_and_epoch,
#     psth_from_overlap,
# )

# import dash
# from dash import dcc, html
# import plotly.graph_objs as go

# #  Run analysis once 
# spikes = obtain_spikes()

# ecg_epochs = epochs_ecg()
# breath_peak_epochs = epochs_breath_peaks()
# breath_trough_epochs = epochs_breath_troughs()

# ecg_overlap = combining_spike_and_epoch(spikes, ecg_epochs)
# breath_peak_overlap = combining_spike_and_epoch(spikes, breath_peak_epochs)
# breath_trough_overlap = combining_spike_and_epoch(spikes, breath_trough_epochs)

# t_ecg, rate_ecg, sem_ecg = psth_from_overlap(ecg_overlap, ecg_epochs, bin_width=0.005)
# t_breath_peak, rate_breath_peak, sem_breath_peak = psth_from_overlap(breath_peak_overlap, breath_peak_epochs, bin_width=0.1)
# t_breath_trough, rate_breath_trough, sem_breath_trough = psth_from_overlap(breath_trough_overlap, breath_trough_epochs, bin_width=0.1)

# # Dash App
# app = dash.Dash(__name__)

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# def aligned_metrics_figure(
#     t,
#     neuron_firing_rate, respiration, ecg,
#     neuron_firing_rate_name="Neuron Firing Rate",
#     respiration_name="Respiration",
#     ecg_name="ECG",
#     x_label="Aligned time (s)"
# ):
#     fig = make_subplots(
#         rows=3,
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.03,
#         subplot_titles=(neuron_firing_rate_name, respiration_name, ecg_name)
#     )

#     fig.add_trace(
#         go.Scatter(x=t, y=neuron_firing_rate, mode="lines", name=neuron_firing_rate_name),
#         row=1, col=1
#     )
#     fig.add_trace(
#         go.Scatter(x=t, y=respiration, mode="lines", name=respiration_name),
#         row=2, col=1
#     )
#     fig.add_trace(
#         go.Scatter(x=t, y=ecg, mode="lines", name=ecg_name),
#         row=3, col=1
#     )

#     fig.update_layout(
#         height=800,
#         title="Aligned Metrics",
#         hovermode="x unified",
#         showlegend=False,
#         dragmode="pan"
#     )

#     fig.update_xaxes(title_text=x_label, row=3, col=1)

#     # Add a scroll/overview bar on the bottom shared x-axis
#     fig.update_xaxes(range=[-2, 2], row=3, col=1)

#     return fig

# def psth_figure(t, rate, sem, title, xlabel):
#     return go.Figure(
#         data=[
#             # Mean rate
#             go.Scatter(
#                 x=t,
#                 y=rate,
#                 mode="lines",
#                 name="Mean firing rate",
#                 line=dict(width=2),
#             ),
#             # SEM shading
#             go.Scatter(
#                 x=np.concatenate([t, t[::-1]]),
#                 y=np.concatenate([rate - sem, (rate + sem)[::-1]]),
#                 fill="toself",
#                 fillcolor="rgba(0,100,200,0.2)",
#                 line=dict(color="rgba(255,255,255,0)"),
#                 hoverinfo="skip",
#                 name="SEM",
#             ),
#         ],
#         layout=go.Layout(
#             title=title,
#             xaxis_title=xlabel,
#             yaxis_title="Events / second",
#             shapes=[
#                 dict(
#                     type="line",
#                     x0=0, x1=0,
#                     y0=0, y1=float(np.max(rate + sem)),
#                     line=dict(dash="dash", color="black")
#                 )
#             ],
#             margin=dict(l=60, r=20, t=60, b=50),
#         )
#     )


# app.layout = html.Div([
#     html.H1("Vagus PSTH Analysis"),

#     html.H2("ECG-Aligned PSTH"),
#     dcc.Graph(
#         figure=psth_figure(
#             t_ecg,
#             rate_ecg,
#             sem_ecg,
#             title="ECG-aligned PSTH",
#             xlabel="Time relative to R-peak (s)"
#         )
#     ),

#     html.H2("Respiration-Aligned_peaks PSTH"),
#     dcc.Graph(
#         figure=psth_figure(
#             t_breath_peak,
#             rate_breath_peak,
#             sem_breath_peak,
#             title="Respiration-aligned PSTH",
#             xlabel="Time relative to respiration peak (s)"
#         )
#     ),
#     html.H2("Respiration-Aligned_troughs PSTH"),
#     dcc.Graph(
#         figure=psth_figure(
#             t_breath_trough,
#             rate_breath_trough,
#             sem_breath_trough,
#             title="Respiration-aligned PSTH",
#             xlabel="Time relative to respiration trough (s)"
#         )
#     ),
# ])

# if __name__ == "__main__":
#     app.run(debug=True)
import dash
from dash import html, dcc, page_container

app = dash.Dash(__name__, use_pages=True)

app.layout = html.Div([
    html.H1("Vagus PSTH Analysis"),

    html.Div([
        dcc.Link("Home", href="/", style={"marginRight": "20px"}),
        dcc.Link("Triple Graph", href="/triple"),
    ]),

    html.Hr(),

    page_container,
])

if __name__ == "__main__":
    app.run(debug=True)