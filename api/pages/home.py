import dash
from dash import html, dcc
import plotly.graph_objects as go

dash.register_page(__name__, path="/")

import dash
import dash_uploader as du
from dash import html, dcc
import plotly.graph_objects as go

dash.register_page(__name__, path="/")

layout = html.Div([
    html.H2("CSV upload"),

    du.Upload(
        id="upload-data",
        text="Drag and Drop or Select File",
        max_files=1,
        filetypes=["csv", "txt"],
        upload_id="vagus_upload",
    ),

    html.Div(id="upload-status"),

    html.H2("Specific Window: ECG + Firing Rate"),
    dcc.Graph(id="ecg-psth-graph", figure=go.Figure()),

    html.H2("Respiration-Aligned Peaks PSTH"),
    dcc.Graph(id="breath-peak-psth-graph", figure=go.Figure()),

    html.H2("Respiration-Aligned Troughs PSTH"),
    dcc.Graph(id="breath-trough-psth-graph", figure=go.Figure()),
])