import dash
from dash import html, dcc
import plotly.graph_objects as go

dash.register_page(__name__, path="/")

layout = html.Div([
    html.H2("CSV upload"),

    dcc.Upload(
        id="upload-data",
        children=html.Div([
            "Drag and Drop or ",
            html.A("Select CSV File")
        ]),
        accept=".csv,.txt",
        multiple=False,
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
    ),

    html.Div(id="upload-status"),

    html.H2("Specific Window: ECG + Firing Rate"),
    dcc.Graph(id="ecg-psth-graph", figure=go.Figure()),

    html.H2("Respiration-Aligned Peaks PSTH"),
    dcc.Graph(id="breath-peak-psth-graph", figure=go.Figure()),

    html.H2("Respiration-Aligned Troughs PSTH"),
    dcc.Graph(id="breath-trough-psth-graph", figure=go.Figure()),
])