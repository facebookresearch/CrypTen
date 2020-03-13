#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from load_data import get_available_dates, read_data
from plotly.subplots import make_subplots


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# load data using relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

available_dates = get_available_dates(DATA_PATH)
func_df, model_df = read_data(DATA_PATH, available_dates)

template = "simple_white"
green, blue = "#229954", "#1E88E5"


# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

index_page = html.Div(
    children=[
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("crypten-icon.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("CrypTen", style={"margin-bottom": "0px"}),
                                html.H4("Benchmarks", style={"margin-top": "0px"}),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Compare Dates", id="learn-more-button"),
                            href="/compare",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="select_date",
                    options=[
                        {"label": date, "value": date}
                        for date in sorted(available_dates)
                    ],
                    value=sorted(available_dates)[-1],
                ),
                html.Div(
                    [
                        html.H3("Functions"),
                        dcc.Markdown(
                            """To reproduce or view assumptions see
[benchmarks](
https://github.com/facebookresearch/CrypTen/blob/master/benchmarks/benchmark.py#L68)
                        """
                        ),
                        html.H5("Runtimes"),
                        dcc.Markdown(
                            """
* function runtimes are averaged over 10 runs using a random tensor of size (100, 100).
* `max` and `argmax` are excluded as they take considerably longer.
    * As of 02/25/2020, `max` / `argmax` take 3min 13s ± 4.73s
""",
                            className="bullets",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="func-runtime-crypten")],
                            className="six columns",
                        ),
                        html.Div(
                            [dcc.Graph(id="func-runtime-crypten-v-plain")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                html.H5("Errors"),
                dcc.Markdown(
                    """
                * function errors are over the domain (0, 100] with step size 0.01
                    * exp, sin, and cos are over the domain (0, 10) with step size 0.001
                """,
                    className="bullets",
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="func-abs-error")], className="six columns"
                        ),
                        html.Div(
                            [dcc.Graph(id="func-relative-error")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.H3("Models"),
                        dcc.Markdown(
                            """
For model details or to reproduce see
[models](https://github.com/facebookresearch/CrypTen/blob/master/benchmarks/models.py)
and
[training details](
https://github.com/facebookresearch/CrypTen/blob/master/benchmarks/benchmark.py#L293).
* trained on Gaussian clusters for binary classification
    * uses SGD with 5k samples, 20 features, over 20 epochs, and 0.1 learning rate
* feedforward has three hidden layers with intermediary RELU and
  final sigmoid activations
* note benchmarks run with world size 1 using CPython
""",
                            className="bullets",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="model-training-time")],
                            className="six columns",
                        ),
                        html.Div(
                            [dcc.Graph(id="model-accuracy")], className="six columns"
                        ),
                    ],
                    className="row",
                ),
            ]
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


comparison_layout = html.Div(
    [
        html.Div(id="compare"),
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("crypten-icon.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("CrypTen", style={"margin-bottom": "0px"}),
                                html.H4("Benchmarks", style={"margin-top": "0px"}),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Benchmarks", id="learn-more-button"), href="/"
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.H6("Previous Date"),
                dcc.Dropdown(
                    id="start_date",
                    options=[
                        {"label": date, "value": date} for date in available_dates
                    ],
                    value=sorted(available_dates)[0],
                ),
                html.H6("Current Date"),
                dcc.Dropdown(
                    id="end_date",
                    options=[
                        {"label": date, "value": date} for date in available_dates
                    ],
                    value=sorted(available_dates)[-1],
                ),
                html.Div(
                    [
                        html.H3("Functions"),
                        dcc.Dropdown(
                            options=[
                                {"label": func, "value": func}
                                for func in func_df["function"].unique()
                            ],
                            multi=True,
                            value="sigmoid",
                            id="funcs",
                        ),
                        dcc.Markdown(
                            """
* function runtimes are averaged over 10 runs using a random tensor of size (100, 100).
* `max` and `argmax` are excluded as they take considerably longer.
    * As of 02/25/2020, `max` / `argmax` take 3min 13s ± 4.73s
""",
                            className="bullets",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="runtime-diff")], className="six columns"
                        ),
                        html.Div([dcc.Graph(id="error-diff")], className="six columns"),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.H4("Historical"),
                        html.Div(
                            [dcc.Graph(id="runtime-timeseries")],
                            className="six columns",
                        ),
                        html.Div(
                            [dcc.Graph(id="error-timeseries")], className="six columns"
                        ),
                    ],
                    className="row",
                ),
            ]
        ),
    ]
)


@app.callback(Output("func-runtime-crypten", "figure"), [Input("select_date", "value")])
def update_runtime_crypten(selected_date):
    filter_df = func_df[func_df["date"] == selected_date]
    filter_df["runtime in seconds"] = filter_df["runtime crypten"]
    fig = px.bar(
        filter_df,
        x="runtime in seconds",
        y="function",
        orientation="h",
        error_x="runtime crypten error plus",
        error_x_minus="runtime crypten error minus",
        color_discrete_sequence=[green],
        template=template,
        title="Crypten",
    )
    fig.update_layout(height=500)
    return fig


@app.callback(
    Output("func-runtime-crypten-v-plain", "figure"), [Input("select_date", "value")]
)
def update_runtime_crypten_v_plain(selected_date):
    filter_df = func_df[func_df["date"] == selected_date]
    fig = px.bar(
        filter_df,
        x="runtime gap",
        y="function",
        orientation="h",
        error_x="runtime gap error plus",
        error_x_minus="runtime gap error minus",
        color_discrete_sequence=[blue],
        template=template,
        title="Crypten vs. Plaintext",
    )

    fig.update_layout(height=500)
    return fig


@app.callback(Output("func-abs-error", "figure"), [Input("select_date", "value")])
def update_abs_error(selected_date):
    filter_df = func_df[func_df["date"] == selected_date]
    fig = px.bar(
        filter_df,
        x="total abs error",
        text="total abs error",
        log_x=True,
        y="function",
        orientation="h",
        color_discrete_sequence=["#C8E6C9"],
        template=template,
        title="Crypten Absolute Error",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(height=500)
    return fig


@app.callback(Output("func-relative-error", "figure"), [Input("select_date", "value")])
def update_abs_error(selected_date):
    filter_df = func_df[func_df["date"] == selected_date]
    fig = px.bar(
        filter_df,
        x="average relative error",
        text="average relative error",
        y="function",
        orientation="h",
        color_discrete_sequence=["#C8E6C9"],
        template=template,
        title="Crypten Relative Error",
    )
    fig.update_traces(texttemplate="%{text:%}", textposition="outside")
    fig.update_layout(height=500)
    return fig


@app.callback(Output("model-training-time", "figure"), [Input("select_date", "value")])
def update_training_time(selected_date):
    filter_df = model_df[model_df["date"] == selected_date]
    filter_df["type"] = np.where(filter_df["is plain text"], "Plain Text", "CrypTen")
    fig = px.bar(
        filter_df,
        x="seconds per epoch",
        text="seconds per epoch",
        y="model",
        color="type",
        orientation="h",
        barmode="group",
        color_discrete_sequence=[blue, green],
        template=template,
        title="Model Training Time",
    )
    fig.update_layout(xaxis={"range": [0, filter_df["seconds per epoch"].max() * 1.1]})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig


@app.callback(Output("model-accuracy", "figure"), [Input("select_date", "value")])
def update_model_accuracy(selected_date):
    filter_df = model_df[model_df["date"] == selected_date]
    filter_df["type"] = np.where(filter_df["is plain text"], "Plain Text", "CrypTen")
    fig = px.bar(
        filter_df,
        x="accuracy",
        text="accuracy",
        y="model",
        color="type",
        orientation="h",
        barmode="group",
        color_discrete_sequence=[blue, green],
        template=template,
        title="Model Accuracy",
    )
    fig.update_layout(xaxis={"range": [0, 1.0]})
    fig.update_traces(texttemplate="%{text:%}", textposition="outside")
    return fig


@app.callback(
    Output("runtime-diff", "figure"),
    [Input("start_date", "value"), Input("end_date", "value"), Input("funcs", "value")],
)
def update_runtime_diff(start_date, end_date, funcs):
    if type(funcs) is str:
        funcs = [funcs]
    start_df = func_df[func_df["date"] == start_date]
    end_df = func_df[func_df["date"] == end_date]

    fig = make_subplots(
        rows=len(funcs), cols=1, specs=[[{"type": "domain"}] for _ in range(len(funcs))]
    )
    for i, func in enumerate(funcs):
        runtime = end_df[end_df["function"] == func]["runtime crypten"]
        runtime_prev = start_df[start_df["function"] == func]["runtime crypten"]
        func_text = func.capitalize()

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=float(runtime),
                title={
                    "text": f"{func_text}<br><span style='font-size:0.8em;color:gray'>"
                    + "runtime in seconds</span><br>"
                },
                delta={
                    "reference": float(runtime_prev),
                    "relative": True,
                    "increasing": {"color": "#ff4236"},
                    "decreasing": {"color": "#008000"},
                },
            ),
            row=i + 1,
            col=1,
        )
    fig.update_layout(height=300 * len(funcs))

    return fig


@app.callback(
    Output("error-diff", "figure"),
    [Input("start_date", "value"), Input("end_date", "value"), Input("funcs", "value")],
)
def update_error_diff(start_date, end_date, funcs):
    if type(funcs) is str:
        funcs = [funcs]
    start_df = func_df[func_df["date"] == start_date]
    end_df = func_df[func_df["date"] == end_date]

    fig = make_subplots(
        rows=len(funcs), cols=1, specs=[[{"type": "domain"}] for _ in range(len(funcs))]
    )
    for i, func in enumerate(funcs):
        error = end_df[end_df["function"] == func]["total abs error"]
        error_prev = start_df[start_df["function"] == func]["total abs error"]
        func_text = func.capitalize()

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=float(error),
                title={
                    "text": f"{func_text}<br><span style='font-size:0.8em;color:gray'>"
                    + "total abs error</span><br>"
                },
                delta={
                    "reference": float(error_prev),
                    "relative": True,
                    "increasing": {"color": "#ff4236"},
                    "decreasing": {"color": "#008000"},
                },
            ),
            row=i + 1,
            col=1,
        )
    fig.update_layout(height=300 * len(funcs))

    return fig


@app.callback(Output("runtime-timeseries", "figure"), [Input("funcs", "value")])
def update_runtime_timeseries(funcs):
    if type(funcs) is str:
        funcs = [funcs]

    filtered_df = func_df[func_df["function"].isin(funcs)]
    filtered_df.sort_values("date", inplace=True)

    fig = px.line(
        filtered_df, x="date", y="runtime crypten", template=template, color="function"
    )

    return fig


@app.callback(Output("error-timeseries", "figure"), [Input("funcs", "value")])
def update_error_timeseries(funcs):
    if type(funcs) is str:
        funcs = [funcs]

    filtered_df = func_df[func_df["function"].isin(funcs)]
    filtered_df.sort_values("date", inplace=True)

    fig = px.line(
        filtered_df, x="date", y="total abs error", template=template, color="function"
    )

    return fig


@app.callback(
    dash.dependencies.Output("page-content", "children"),
    [dash.dependencies.Input("url", "pathname")],
)
def display_page(pathname):
    """Routes to page based on URL"""
    if pathname == "/compare":
        return comparison_layout
    else:
        return index_page


if __name__ == "__main__":
    app.run_server(debug=True)
