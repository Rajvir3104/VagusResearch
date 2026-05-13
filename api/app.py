import dash
from dash import html, dcc, page_container, callback
from api import upload_callbacks
app = dash.Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

# Import page modules so Dash page registration happens.
# import pages.home  # noqa: F401

app.layout = html.Div([
    html.H1("Vagus PSTH Analysis"),

    html.Div([
        dcc.Link("Home", href="/", style={"marginRight": "20px"}),
        # dcc.Link("Triple Graph", href="/triple"),
    ]),

    html.Hr(),

    page_container,
])



server = app.server

if __name__ == "__main__":
    app.run(debug=False)