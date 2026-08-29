import base64

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from analysis import analyse_customer
from config import BEST_THRESHOLD, COLORS, FONT, RISK_LABELS, _risk_level
from data import X_all, y_all


def _stat(label, value_node):
    return html.Div(
        [
            html.Div(label, style={
                "color": COLORS["text_muted"],
                "fontSize": "0.72rem",
                "fontWeight": "600",
                "textTransform": "uppercase",
                "letterSpacing": "0.06em",
                "marginBottom": "6px",
            }),
            html.Div(
                value_node,
                style={"fontSize": "1.05rem", "fontWeight": "600", "color": COLORS["text"]},
            ),
        ]
    )


def _badge(text, level):
    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": "4px 12px",
            "borderRadius": "999px",
            "fontSize": "0.85rem",
            "fontWeight": "600",
            "backgroundColor": COLORS[f"{level}_bg"],
            "color": COLORS[level],
        },
    )


def create_app():
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.FLATLY,
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
        ],
        suppress_callback_exceptions=True,
        title="Credit Scoring Dashboard",
    )

    card_style = {
        "background": COLORS["surface"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "12px",
        "padding": "24px",
        "color": COLORS["text"],
        "fontFamily": FONT,
    }

    app.layout = html.Div(
        style={"background": COLORS["bg"], "minHeight": "100vh", "fontFamily": FONT},
        children=[
            dbc.Navbar(
                dbc.Container(
                    [
                        html.Span(
                            "Credit Scoring Dashboard",
                            style={
                                "fontWeight": "600",
                                "fontSize": "1.1rem",
                                "color": COLORS["text"],
                                "letterSpacing": "0.2px",
                            },
                        ),
                        html.Span(
                            "bdaanoun | cbenlafk",
                            style={"color": COLORS["text_muted"], "fontSize": "0.82rem"},
                        ),
                    ],
                    fluid=True,
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                ),
                color=COLORS["surface"],
                dark=False,
                style={"borderBottom": f"1px solid {COLORS['border']}", "padding": "14px 0"},
            ),
            dbc.Container(
                fluid=True,
                style={"padding": "32px 40px", "maxWidth": "1280px"},
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Customer ID", style={
                                        "color": COLORS["text_muted"],
                                        "fontSize": "0.72rem",
                                        "fontWeight": "600",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.06em",
                                        "marginBottom": "6px",
                                    }),
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="customer-id-input",
                                                type="number",
                                                placeholder="Enter a customer ID, e.g. 310536",
                                                style={
                                                    "background": COLORS["surface"],
                                                    "border": f"1px solid {COLORS['border']}",
                                                    "color": COLORS["text"],
                                                    "borderRadius": "8px 0 0 8px",
                                                    "fontSize": "0.95rem",
                                                    "height": "44px",
                                                },
                                                debounce=False,
                                            ),
                                            dbc.Button(
                                                "Analyse",
                                                id="analyse-btn",
                                                n_clicks=0,
                                                style={
                                                    "background": COLORS["text"],
                                                    "border": "none",
                                                    "borderRadius": "0 8px 8px 0",
                                                    "fontWeight": "600",
                                                    "padding": "0 26px",
                                                    "height": "44px",
                                                    "fontSize": "0.95rem",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        id="error-msg",
                                        style={
                                            "color": COLORS["high"],
                                            "marginTop": "8px",
                                            "fontSize": "0.88rem",
                                            "minHeight": "20px",
                                        },
                                    ),
                                ],
                                md=6,
                            )
                        ],
                        className="mb-4",
                    ),
                    dbc.Row(
                        id="score-row",
                        style={"marginBottom": "32px"},
                        children=[
                            dbc.Col(
                                html.Div(
                                    "Enter a customer ID above to generate a risk report.",
                                    style={
                                        **card_style,
                                        "textAlign": "center",
                                        "color": COLORS["text_secondary"],
                                        "fontSize": "0.92rem",
                                    },
                                )
                            )
                        ],
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Loading(
                                        id="loading-fig",
                                        type="circle",
                                        color=COLORS["accent"],
                                        children=html.Div(id="main-chart"),
                                    )
                                ]
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Div(id="shap-section", style={"marginTop": "32px"}),
                            ])
                        ]
                    ),
                ],
            ),
            dcc.Store(id="shap-store"),
        ],
    )

    @app.callback(
        Output("score-row", "children"),
        Output("main-chart", "children"),
        Output("shap-section", "children"),
        Output("shap-store", "data"),
        Output("error-msg", "children"),
        Input("analyse-btn", "n_clicks"),
        Input("customer-id-input", "n_submit"),
        State("customer-id-input", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(_n_clicks, _n_submit, customer_id_raw):
        if not customer_id_raw:
            return dash.no_update, dash.no_update, dash.no_update, None, "Please enter a customer ID."

        try:
            customer_id = int(customer_id_raw)
        except (TypeError, ValueError):
            return dash.no_update, dash.no_update, dash.no_update, None, "Customer ID must be an integer."

        try:
            score, fig, shap_html = analyse_customer(customer_id)
        except ValueError as exc:
            return dash.no_update, dash.no_update, dash.no_update, None, str(exc)
        except Exception as exc:
            return dash.no_update, dash.no_update, dash.no_update, None, f"Unexpected error: {exc}"

        level = _risk_level(score)
        label = RISK_LABELS[level]

        actual_row = y_all[X_all["SK_ID_CURR"] == customer_id]
        actual_val = actual_row.iloc[0] if not actual_row.empty else None
        if pd.isna(actual_val):
            actual_text = "Unknown"
        else:
            actual_text = "Defaulted" if int(actual_val) == 1 else "Repaid"

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(score * 100, 1),
                number={"suffix": "%", "font": {"size": 40, "color": COLORS["text"]}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont={"color": COLORS["text_secondary"], "size": 11}),
                    bar=dict(color=COLORS[level], thickness=0.22),
                    bgcolor=COLORS["surface"],
                    bordercolor=COLORS["border"],
                    borderwidth=1,
                    steps=[
                        {"range": [0, BEST_THRESHOLD * 100], "color": COLORS["low_bg"]},
                        {"range": [BEST_THRESHOLD * 100, 30], "color": COLORS["medium_bg"]},
                        {"range": [30, 100], "color": COLORS["high_bg"]},
                    ],
                    threshold=dict(
                        line=dict(color=COLORS["text"], width=2),
                        thickness=0.75,
                        value=BEST_THRESHOLD * 100,
                    ),
                ),
                title={"text": "Default Probability", "font": {"color": COLORS["text_secondary"], "size": 13}},
            )
        )
        gauge_fig.update_layout(
            paper_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"], family=FONT),
            height=240,
            margin=dict(l=24, r=24, t=44, b=10),
        )

        prediction = int(score >= BEST_THRESHOLD)
        prediction_text = "Default" if prediction == 1 else "Repaid"
        prediction_level = "high" if prediction == 1 else "low"

        score_row = [
            dbc.Col(
                [
                    html.Div(
                        style=card_style,
                        children=[dcc.Graph(figure=gauge_fig, config={"displayModeBar": False})],
                    )
                ],
                md=5,
            ),
            dbc.Col(
                [
                    html.Div(
                        style={**card_style, "height": "100%"},
                        children=[
                            dbc.Row([
                                dbc.Col(_stat("Customer ID", str(customer_id)), width=6),
                                dbc.Col(_stat("Prediction", _badge(prediction_text, prediction_level)), width=6),
                            ], className="mb-4"),
                            dbc.Row([
                                dbc.Col(_stat("Risk level", _badge(label, level)), width=6),
                                dbc.Col(_stat("Actual outcome", actual_text), width=6),
                            ], className="mb-4"),
                            dbc.Row([
                                dbc.Col(_stat("Decision threshold", f"{BEST_THRESHOLD:.0%}"), width=6),
                                dbc.Col(_stat("Model probability", f"{score:.1%}"), width=6),
                            ]),
                        ],
                    )
                ],
                md=7,
            ),
        ]

        main_chart = dcc.Graph(
            figure=fig,
            config={"displayModeBar": True, "scrollZoom": False},
            style={"borderRadius": "12px", "overflow": "hidden"},
        )

        encoded = base64.b64encode(shap_html.encode()).decode()
        shap_section = html.Div(
            style=card_style,
            children=[
                html.H5(
                    "SHAP Force Plot",
                    style={"marginBottom": "12px", "fontWeight": "600", "color": COLORS["text"], "fontSize": "1.05rem"},
                ),
                html.P(
                    "Each arrow shows how a feature pushed the prediction away from the base value (expected value for the population).",
                    style={"color": COLORS["text_secondary"], "fontSize": "0.88rem", "marginBottom": "16px"},
                ),
                html.Iframe(
                    src=f"data:text/html;base64,{encoded}",
                    style={
                        "width": "100%",
                        "height": "220px",
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "8px",
                        "background": "#ffffff",
                    },
                ),
            ],
        )

        return score_row, main_chart, shap_section, shap_html, ""

    return app


app = create_app()
