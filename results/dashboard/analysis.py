import io

import numpy as np
import pandas as pd
import shap
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from config import BEST_THRESHOLD, COLORS
from data import X_all, _POP_TRANSFORMED, FEATURE_NAMES, classifier, preprocessor


def analyse_customer(customer_id: int):
    row = X_all[X_all["SK_ID_CURR"] == customer_id]
    if row.empty:
        raise ValueError(f"Customer ID {customer_id} was not found in the dataset.")

    client_features = row.drop(columns=["SK_ID_CURR"]).iloc[[0]]
    client_transformed = preprocessor.transform(client_features)

    score = float(classifier.predict_proba(client_transformed)[0, 1])

    explainer = shap.TreeExplainer(
        classifier,
        data=_POP_TRANSFORMED,
        feature_perturbation="interventional",
        model_output="probability",
    )
    shap_explanation = explainer(client_transformed)
    if isinstance(shap_explanation, list):
        shap_client = shap_explanation[1][0]
        expected_value = explainer.expected_value[1]
    else:
        shap_client = shap_explanation.values[0]
        expected_value = shap_explanation.base_values[0]

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

    others = X_all[X_all["SK_ID_CURR"] != customer_id].sample(
        min(4, len(X_all) - 1), random_state=42
    )
    others_feat = others.drop(columns=["SK_ID_CURR"])
    others_transformed = preprocessor.transform(others_feat)
    others_scores = classifier.predict_proba(others_transformed)[:, 1]

    top10_idx = np.argsort(np.abs(shap_client))[::-1][:10]
    top5_idx = top10_idx[:5]

    pop_transformed_full = preprocessor.transform(X_all.drop(columns=["SK_ID_CURR"]))

    def _short(name, limit=22):
        return name if len(name) <= limit else name[:19] + "…"

    pop_titles = [_short(FEATURE_NAMES[i]) for i in top5_idx]

    fig = make_subplots(
        rows=3,
        cols=5,
        specs=[
            [{"type": "domain", "colspan": 5}] + [None] * 4,
            [{"type": "xy", "colspan": 5}] + [None] * 4,
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

    col_names = [FEATURE_NAMES[i] for i in top10_idx]
    col_vals = [client_transformed[0, i] for i in top10_idx]
    shap_contribs = [shap_client[i] for i in top10_idx]
    directions = [
        "Toward default" if c > 0 else "Toward repayment" if c < 0 else "—"
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
                values=[
                    "<b>Feature</b>",
                    "<b>Value (encoded)</b>",
                    "<b>SHAP</b>",
                    "<b>Direction</b>",
                ],
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
        row=1,
        col=1,
    )

    bar_labels = [f"Client {cid}" for cid in others["SK_ID_CURR"]]
    bar_labels.insert(0, f"Selected — {customer_id}")
    bar_scores = others_scores.tolist()
    bar_scores.insert(0, score)
    bar_colors = [COLORS["high"] if s >= BEST_THRESHOLD else COLORS["low"] for s in bar_scores]
    bar_colors[0] = COLORS["accent"]

    fig.add_trace(
        go.Bar(
            x=bar_labels,
            y=bar_scores,
            marker_color=bar_colors,
            name="Default probability",
            text=[f"{s:.1%}" for s in bar_scores],
            textposition="outside",
            textfont=dict(color=COLORS["text_secondary"]),
        ),
        row=2,
        col=1,
    )
    fig.add_shape(
        type="line",
        y0=BEST_THRESHOLD,
        y1=BEST_THRESHOLD,
        x0=-0.5,
        x1=len(bar_labels) - 0.5,
        line=dict(color=COLORS["text_muted"], dash="dash", width=1.25),
        row=2,
        col=1,
    )

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
                y=pop_vals,
                name=FEATURE_NAMES[feat_idx],
                boxpoints=False,
                marker_color=COLORS["text_muted"],
                line_color=COLORS["text_secondary"],
                fillcolor="#f3f4f6",
                showlegend=False,
                hovertemplate="Population: %{y}<extra></extra>",
            ),
            row=3,
            col=panel_i,
        )
        fig.add_trace(
            go.Scatter(
                x=[FEATURE_NAMES[feat_idx]],
                y=[client_val],
                mode="markers",
                name="Selected client",
                marker=dict(color=COLORS["accent"], size=11, symbol="diamond"),
                showlegend=(panel_i == 1),
                hovertemplate="This client: %{y}<extra></extra>",
            ),
            row=3,
            col=panel_i,
        )
        fig.update_xaxes(showticklabels=False, row=3, col=panel_i)

    fig.update_yaxes(
        title_text="P(default)",
        tickformat=".0%",
        range=[0, max(0.20, max(bar_scores) * 1.25)],
        row=2,
        col=1,
    )
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"]),
        title=dict(
            text=f"<b>Local interpretation — Customer {customer_id}</b>",
            font=dict(size=17, color=COLORS["text"]),
        ),
        height=1150,
        showlegend=True,
        legend=dict(bgcolor=COLORS["surface"], bordercolor=COLORS["border"]),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=False, color=COLORS["text_secondary"], linecolor=COLORS["border"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["border"], color=COLORS["text_secondary"])
    fig.update_annotations(font=dict(color=COLORS["text_secondary"], size=12))

    return score, fig, shap_html
