import os
import sys
import io
import base64
import shap
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_DIR / "results" / "model" / "xgboost.pkl"

X_TRAIN_PATH = PROJECT_DIR / "data" / "X_train_processed.csv"
X_TEST_PATH  = PROJECT_DIR / "data" / "X_test_processed.csv"

Y_TRAIN_PATH = PROJECT_DIR / "data" / "y_train.csv"
Y_TEST_PATH  = PROJECT_DIR / "data" / "y_test.csv"

print("--------dsfsdfsdf", PROJECT_DIR,"------------------")
# Load model & data (once at startup)
print("Loading model...")
model_data   = joblib.load(MODEL_PATH)
preprocessor = model_data["preprocessor"]
classifier   = model_data["model"]
print("Model loaded.")

print("Loading data...")
from sklearn.model_selection import train_test_split as _tts

X_train = pd.read_csv(X_TRAIN_PATH)
X_test  = pd.read_csv(X_TEST_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test  = pd.read_csv(Y_TEST_PATH).squeeze()

# Re-derive IDs by replaying the exact same split used in preprocess.py
if "SK_ID_CURR" not in X_train.columns:
    _raw = pd.read_csv(os.path.join(PROJECT_DIR, "data", "application_train.csv"),usecols=["SK_ID_CURR", "TARGET"])
    _ids_train, _ids_test = _tts(_raw["SK_ID_CURR"],test_size=0.20,stratify=_raw["TARGET"],random_state=42,)
    X_train.insert(0, "SK_ID_CURR", _ids_train.values)
    X_test.insert(0,  "SK_ID_CURR", _ids_test.values)

# For the kaggle test set (application_test.csv) we don't have a split,
# so we won't merge it here — train+val is enough for the dashboard demo.

# Combine for unified look-up (test labels are unknown -1)
y_train_s = y_train.reset_index(drop=True)
y_test_s  = pd.Series([-1] * len(X_test), name="TARGET")

X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train_s, y_test_s], ignore_index=True)

ALL_IDS  = sorted(X_all["SK_ID_CURR"].unique().tolist())
print(f"Dataset ready — {len(ALL_IDS):,} customers.")

# Pre-transform a population sample for SHAP background
_POP_FEATURES = X_all.drop(columns=["SK_ID_CURR"])
_POP_SAMPLE   = _POP_FEATURES.sample(min(1000, len(_POP_FEATURES)), random_state=42)
_POP_TRANSFORMED = preprocessor.transform(_POP_SAMPLE)

# Feature name cleanup helper
def _clean_names(names):
    return [n.replace("remainder__", "").replace("onehotencoder__", "") for n in names]

FEATURE_NAMES = _clean_names(preprocessor.get_feature_names_out())

# Design tokens — single minimal palette used everywhere
COLORS = {
    "bg":             "#fafafa",
    "surface":        "#ffffff",
    "border":         "#e5e7eb",
    "text":           "#111827",
    "text_secondary": "#6b7280",
    "text_muted":     "#9ca3af",
    "accent":         "#2563eb",
    "low":            "#059669",
    "low_bg":         "#ecfdf5",
    "medium":         "#d97706",
    "medium_bg":      "#fffbeb",
    "high":           "#dc2626",
    "high_bg":        "#fef2f2",
}

FONT = "Inter, sans-serif"


def _risk_level(score: float) -> str:
    if score < 0.30:
        return "low"
    if score < 0.50:
        return "medium"
    return "high"


RISK_LABELS = {"low": "Low risk", "medium": "Medium risk", "high": "High risk"}


# Core inference function

def analyse_customer(customer_id: int):
    """Return (score, plotly_fig, shap_html_str) or raise ValueError."""
    row = X_all[X_all["SK_ID_CURR"] == customer_id]
    if row.empty:
        raise ValueError(f"Customer ID {customer_id} was not found in the dataset.")

    client_features = row.drop(columns=["SK_ID_CURR"]).iloc[[0]]
    client_transformed = preprocessor.transform(client_features)

    # Probability of default
    score = float(classifier.predict_proba(client_transformed)[0, 1])

    # ── SHAP ──────────────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(
        classifier,
        data=_POP_TRANSFORMED,
        feature_perturbation="interventional",
        model_output="probability",
    )
    shap_explanation = explainer(client_transformed)
    if isinstance(shap_explanation, list):
        shap_client    = shap_explanation[1][0]
        expected_value = explainer.expected_value[1]
    else:
        shap_client    = shap_explanation.values[0]
        expected_value = shap_explanation.base_values[0]

    # SHAP force plot HTML string
    force_plot = shap.force_plot(
        expected_value,
        shap_client,
        client_transformed[0],
        feature_names=FEATURE_NAMES,
        matplotlib=False,
    )
    shap_buf = io.StringIO()
    shap.save_html(shap_buf, force_plot)
    shap_html = shap_buf.getvalue()

    # ── Plotly visualisation ───────────────────────────────────────────────
    # Comparison clients (up to 4 random others)
    others = X_all[X_all["SK_ID_CURR"] != customer_id].sample(
        min(4, len(X_all) - 1), random_state=42
    )
    others_feat        = others.drop(columns=["SK_ID_CURR"])
    others_transformed = preprocessor.transform(others_feat)
    others_scores      = classifier.predict_proba(others_transformed)[:, 1]

    top10_idx = np.argsort(np.abs(shap_client))[::-1][:10]
    top5_idx  = top10_idx[:5]

    pop_transformed_full = preprocessor.transform(X_all.drop(columns=["SK_ID_CURR"]))

    def _short(name, limit=22):
        return name if len(name) <= limit else name[:19] + "…"

    pop_titles = [_short(FEATURE_NAMES[i]) for i in top5_idx]

    fig = make_subplots(
        rows=3, cols=5,
        specs=[
            [{"type": "domain", "colspan": 5}] + [None] * 4,
            [{"type": "xy",     "colspan": 5}] + [None] * 4,
            [{"type": "xy"}] * 5,
        ],
        subplot_titles=(
            "Top 10 features by SHAP importance",
            "Default probability — selected client vs peers",
            *pop_titles,
        ),
        horizontal_spacing=0.04,
        vertical_spacing=0.14,
    )

    # Row 1 — Feature table
    col_names     = [FEATURE_NAMES[i] for i in top10_idx]
    col_vals      = [client_transformed[0, i] for i in top10_idx]
    shap_contribs = [shap_client[i] for i in top10_idx]
    directions = [
        "Toward default" if c > 0 else
        "Toward repayment" if c < 0 else
        "—"
        for c in shap_contribs
    ]
    direction_bg = [
        COLORS["high_bg"] if c > 0 else COLORS["low_bg"] if c < 0 else COLORS["surface"]
        for c in shap_contribs
    ]
    direction_font = [
        COLORS["high"] if c > 0 else COLORS["low"] if c < 0 else COLORS["text_secondary"]
        for c in shap_contribs
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Feature</b>", "<b>Value (encoded)</b>",
                        "<b>SHAP</b>", "<b>Direction</b>"],
                fill_color="#f3f4f6",
                font=dict(color=COLORS["text"], size=12),
                align="left",
                height=32,
            ),
            cells=dict(
                values=[
                    col_names,
                    ["—" if pd.isna(v) else f"{v:.4f}" for v in col_vals],
                    [f"{c:+.4f}" for c in shap_contribs],
                    directions,
                ],
                fill_color=[
                    [COLORS["surface"]] * len(col_names),
                    [COLORS["surface"]] * len(col_names),
                    [COLORS["surface"]] * len(col_names),
                    direction_bg,
                ],
                font=dict(
                    color=[
                        [COLORS["text"]] * len(col_names),
                        [COLORS["text_secondary"]] * len(col_names),
                        [COLORS["text_secondary"]] * len(col_names),
                        direction_font,
                    ],
                    size=11,
                ),
                align="left",
                height=28,
            ),
        ),
        row=1, col=1,
    )

    # Row 2 — Comparison bar chart
    bar_labels = [f"Client {cid}" for cid in others["SK_ID_CURR"]]
    bar_labels.insert(0, f"Selected — {customer_id}")
    bar_scores = others_scores.tolist()
    bar_scores.insert(0, score)
    bar_colors = [
        COLORS["high"] if s >= 0.5 else COLORS["low"]
        for s in bar_scores
    ]
    bar_colors[0] = COLORS["accent"]   # highlight selected client

    fig.add_trace(
        go.Bar(
            x=bar_labels, y=bar_scores,
            marker_color=bar_colors,
            name="Default probability",
            text=[f"{s:.1%}" for s in bar_scores],
            textposition="outside",
            textfont=dict(color=COLORS["text_secondary"]),
        ),
        row=2, col=1,
    )
    fig.add_shape(
        type="line", y0=0.5, y1=0.5,
        x0=-0.5, x1=len(bar_labels) - 0.5,
        line=dict(color=COLORS["text_muted"], dash="dash", width=1.25),
        row=2, col=1,
    )

    # Row 3 — Population box plots
    for panel_i, feat_idx in enumerate(top5_idx, start=1):
        pop_vals = pop_transformed_full[:, feat_idx]
        if hasattr(pop_vals, "toarray"):
            pop_vals = pop_vals.toarray().ravel()
        pop_vals = np.asarray(pop_vals).ravel()

        client_val = client_transformed[0, feat_idx]
        if hasattr(client_val, "toarray"):
            client_val = client_val.toarray().ravel()[0]

        fig.add_trace(
            go.Box(
                y=pop_vals, name=FEATURE_NAMES[feat_idx],
                boxpoints=False,
                marker_color=COLORS["text_muted"],
                line_color=COLORS["text_secondary"],
                fillcolor="#f3f4f6",
                showlegend=False,
                hovertemplate="Population: %{y}<extra></extra>",
            ),
            row=3, col=panel_i,
        )
        fig.add_trace(
            go.Scatter(
                x=[FEATURE_NAMES[feat_idx]], y=[client_val],
                mode="markers",
                name="Selected client",
                marker=dict(color=COLORS["accent"], size=11, symbol="diamond"),
                showlegend=(panel_i == 1),
                hovertemplate="This client: %{y}<extra></extra>",
            ),
            row=3, col=panel_i,
        )
        fig.update_xaxes(showticklabels=False, row=3, col=panel_i)

    fig.update_yaxes(
        title_text="P(default)",
        tickformat=".0%",
        range=[0, max(bar_scores) * 1.25],
        row=2, col=1,
    )
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family=FONT),
        title=dict(
            text=f"<b>Local interpretation — Customer {customer_id}</b>",
            font=dict(size=17, color=COLORS["text"]),
        ),
        height=1150,
        showlegend=True,
        legend=dict(bgcolor=COLORS["surface"], bordercolor=COLORS["border"]),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        showgrid=False,
        color=COLORS["text_secondary"],
        linecolor=COLORS["border"],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["border"],
        color=COLORS["text_secondary"],
    )
    fig.update_annotations(font=dict(color=COLORS["text_secondary"], size=12))

    return score, fig, shap_html


# Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Credit Scoring Dashboard",
)

# ── Styles ────────────────────────────────────
CARD_STYLE = {
    "background": COLORS["surface"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "24px",
    "color": COLORS["text"],
    "fontFamily": FONT,
}

LABEL_STYLE = {
    "color": COLORS["text_muted"],
    "fontSize": "0.72rem",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.06em",
    "marginBottom": "6px",
}


def _stat(label, value_node):
    return html.Div([
        html.Div(label, style=LABEL_STYLE),
        html.Div(value_node, style={"fontSize": "1.05rem", "fontWeight": "600",
                                     "color": COLORS["text"]}),
    ])


def _badge(text, level):
    return html.Span(text, style={
        "display": "inline-block",
        "padding": "4px 12px",
        "borderRadius": "999px",
        "fontSize": "0.85rem",
        "fontWeight": "600",
        "backgroundColor": COLORS[f"{level}_bg"],
        "color": COLORS[level],
    })


# ── Layout ────────────────────────────────────
app.layout = html.Div(
    style={"background": COLORS["bg"], "minHeight": "100vh", "fontFamily": FONT},
    children=[

        # ── Navbar ──────────────────────────────
        dbc.Navbar(
            dbc.Container([
                html.Span("Credit Scoring Dashboard",
                          style={"fontWeight": "600", "fontSize": "1.1rem",
                                 "color": COLORS["text"], "letterSpacing": "0.2px"}),
                html.Span("bdaanoun | cbenlafk",
                          style={"color": COLORS["text_muted"], "fontSize": "0.82rem"}),
            ], fluid=True, style={"display": "flex", "justifyContent": "space-between",
                                  "alignItems": "center"}),
            color=COLORS["surface"],
            dark=False,
            style={"borderBottom": f"1px solid {COLORS['border']}", "padding": "14px 0"},
        ),

        dbc.Container(fluid=True, style={"padding": "32px 40px", "maxWidth": "1280px"}, children=[

            # ── Search bar ──────────────────────
            dbc.Row([
                dbc.Col([
                    html.Label("Customer ID", style=LABEL_STYLE),
                    dbc.InputGroup([
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
                    ]),
                    html.Div(id="error-msg",
                             style={"color": COLORS["high"], "marginTop": "8px",
                                    "fontSize": "0.88rem", "minHeight": "20px"}),
                ], md=6),
            ], className="mb-4"),

            # ── Score cards ─────────────────────
            dbc.Row(
                id="score-row",
                style={"marginBottom": "32px"},
                children=[
                    dbc.Col(
                        html.Div(
                            "Enter a customer ID above to generate a risk report.",
                            style={**CARD_STYLE, "textAlign": "center",
                                   "color": COLORS["text_secondary"],
                                   "fontSize": "0.92rem"},
                        ),
                    ),
                ],
            ),

            # ── Main Plotly figure ──────────────
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-fig",
                        type="circle",
                        color=COLORS["accent"],
                        children=html.Div(id="main-chart"),
                    )
                ])
            ]),

            # ── SHAP force plot ─────────────────
            dbc.Row([
                dbc.Col([
                    html.Div(id="shap-section", style={"marginTop": "32px"}),
                ])
            ]),

        ]),

        # Store shap html
        dcc.Store(id="shap-store"),
    ],
)


# Callbacks

@app.callback(
    Output("score-row",   "children"),
    Output("main-chart",  "children"),
    Output("shap-section","children"),
    Output("shap-store",  "data"),
    Output("error-msg",   "children"),
    Input("analyse-btn",  "n_clicks"),
    Input("customer-id-input", "n_submit"),
    State("customer-id-input", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, n_submit, customer_id_raw):
    if not customer_id_raw:
        return dash.no_update, dash.no_update, dash.no_update, None, "Please enter a customer ID."

    try:
        customer_id = int(customer_id_raw)
    except (ValueError, TypeError):
        return dash.no_update, dash.no_update, dash.no_update, None, "Customer ID must be an integer."

    try:
        score, fig, shap_html = analyse_customer(customer_id)
    except ValueError as exc:
        return dash.no_update, dash.no_update, dash.no_update, None, str(exc)
    except Exception as exc:
        return dash.no_update, dash.no_update, dash.no_update, None, f"Unexpected error: {exc}"

    level = _risk_level(score)
    label = RISK_LABELS[level]

    # Known label lookup
    actual_row = y_all[X_all["SK_ID_CURR"] == customer_id]
    actual_val = actual_row.iloc[0] if not actual_row.empty else None
    if actual_val is None or actual_val == -1:
        actual_text = "Unknown (test set)"
    else:
        actual_text = "Defaulted" if actual_val == 1 else "Repaid"

    # ── Gauge ──
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": COLORS["text"]}},
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=COLORS["text_secondary"], size=11)),
            bar=dict(color=COLORS[level], thickness=0.22),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            steps=[
                {"range": [0,  30],  "color": COLORS["low_bg"]},
                {"range": [30, 50],  "color": COLORS["medium_bg"]},
                {"range": [50, 100], "color": COLORS["high_bg"]},
            ],
            threshold=dict(
                line=dict(color=COLORS["text"], width=2),
                thickness=0.75,
                value=16,
            ),
        ),
        title={"text": "Default Probability", "font": {"color": COLORS["text_secondary"], "size": 13}},
    ))
    gauge_fig.update_layout(
        paper_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family=FONT),
        height=240,
        margin=dict(l=24, r=24, t=44, b=10),
    )

    # ── Score cards ──
    score_row = [
        dbc.Col([
            html.Div(style=CARD_STYLE, children=[
                dcc.Graph(figure=gauge_fig, config={"displayModeBar": False}),
            ])
        ], md=5),

        dbc.Col([
            html.Div(style={**CARD_STYLE, "height": "100%"}, children=[
                dbc.Row([
                    dbc.Col(_stat("Customer ID", str(customer_id)), width=6),
                    dbc.Col(_stat("Verdict", _badge(label, level)), width=6),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(_stat("Actual outcome", actual_text), width=6),
                    dbc.Col(_stat("Decision threshold", "50%"), width=6),
                ]),
            ]),
        ], md=7),
    ]

    # ── Main chart ──
    main_chart = dcc.Graph(
        figure=fig,
        config={"displayModeBar": True, "scrollZoom": False},
        style={"borderRadius": "12px", "overflow": "hidden"},
    )

    # ── SHAP section ──
    # Encode as base64 data URI to embed in an iframe without a server route
    encoded = base64.b64encode(shap_html.encode()).decode()
    shap_section = html.Div(style=CARD_STYLE, children=[
        html.H5("SHAP Force Plot",
                style={"marginBottom": "12px", "fontWeight": "600",
                       "color": COLORS["text"], "fontSize": "1.05rem"}),
        html.P("Each arrow shows how a feature pushed the prediction away from "
               "the base value (expected value for the population).",
               style={"color": COLORS["text_secondary"], "fontSize": "0.88rem",
                      "marginBottom": "16px"}),
        html.Iframe(
            src=f"data:text/html;base64,{encoded}",
            style={"width": "100%", "height": "220px",
                   "border": f"1px solid {COLORS['border']}", "borderRadius": "8px",
                   "background": "#ffffff"},
        ),
    ])

    return score_row, main_chart, shap_section, shap_html, ""


# Entry point
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)