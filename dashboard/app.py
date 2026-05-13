import dash
import dash_uploader as du
from dash import html, dcc, page_container

app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

du.configure_upload(app, "/tmp/uploads")

# Import after du.configure_upload
from dashboard import upload_callbacks  # noqa: F401
import dashboard.pages.home  # noqa: F401

app.layout = html.Div([
    html.H1("Vagus PSTH Analysis"),

    html.Div([
        dcc.Link("Home", href="/", style={"marginRight": "20px"}),
    ]),

    html.Hr(),

    page_container,
])

server = app.server

if __name__ == "__main__":
    app.run(debug=True)