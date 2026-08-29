import os

import numpy as np
import shap
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
    

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "results", "model", "xgboost.pkl")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "clients_outputs")



def explain_client( model, X, customer_id, id_column="SK_ID_CURR"):
    
    preprocessor = model["preprocessor"]
    classifier = model["model"]
    
    # 1. Get the client
    client = X[X[id_column] == customer_id]

    if client.empty:
        raise ValueError(f"Customer {customer_id} not found.")

    # Keep one row
    client = client.iloc[[0]]

    # 2. Prepare client features
    client_transformed = preprocessor.transform(client)
    
    # 3. Prediction
    score = classifier.predict_proba(client_transformed)[0, 1]


    # prepare the features for SHAP explanation
    X_features = X
    X_transformed = preprocessor.transform(X_features)
    feature_names = preprocessor.get_feature_names_out()
        


    # 5. Create SHAP explainer
    explainer = shap.TreeExplainer(
        classifier,
        data=X_transformed[:1000],
        feature_perturbation="interventional",
        model_output="probability"
    )

    shap_values = explainer(client_transformed)

    # SHAP returns an Explanation for current versions and a list for older ones.
    if isinstance(shap_values, list):
        shap_client = shap_values[1][0]
        expected_value = explainer.expected_value[1]
    else:
        shap_client = shap_values.values[0]
        expected_value = shap_values.base_values[0]


    # 6. SHAP force plot
    feature_names = [
        name.replace("remainder__", "")
            .replace("onehotencoder__", "")
        for name in preprocessor.get_feature_names_out()
    ]
    
    shap_force_plot = shap.force_plot(
        expected_value,
        shap_client,
        client_transformed[0],
        feature_names=feature_names,
        matplotlib=False
    )
    
    fig = create_client_visualization(
        X,
        score,
        customer_id,
        preprocessor,
        classifier,
        id_column,
        client_transformed,
        shap_client,
        feature_names,
        X_transformed
    )
    
    return score, shap_force_plot, fig




def create_client_visualization( X, score, customer_id, preprocessor, classifier, id_column="SK_ID_CURR", client_transformed=None, shap_client=None, feature_names=None, population_transformed=None):
    
    # Comparison clients
    other_clients = X[X[id_column] != customer_id]

    comparison = other_clients.sample(min(4, len(other_clients)), random_state=42)
    comparison_transformed = preprocessor.transform(comparison)
    comparison_scores = classifier.predict_proba(comparison_transformed)[:, 1]
    
    top_feature_indices = np.argsort(np.abs(shap_client))[::-1][:10]
    top_population_Feature_indices = top_feature_indices[:5]

    
    #shortn the namess
    population_titles = [
        feature_names[index] if len(feature_names[index]) <= 22
        else f"{feature_names[index][:19]}..."
        for index in top_population_Feature_indices
    ]

    # Plot layout
    population_columns = 5
    fig = make_subplots(
        rows=3,
        cols=population_columns,
        specs=[
            [{"type": "domain", "colspan": population_columns}] + [None] * (population_columns - 1),
            [{"type": "xy", "colspan": population_columns}] + [None] * (population_columns - 1),
            [{"type": "xy"}] * population_columns
        ],
        subplot_titles=(
            "Client information",
            "Default probability comparison",
            *population_titles
        ),
        horizontal_spacing=0.04,
        vertical_spacing=0.16
    )


    #-----------Plot One
    # Client information, ranked by transformed-feature SHAP importance.
    #--------------------------------------------------------
    colNames = [feature_names[index] for index in top_feature_indices]
    colValues = [client_transformed[0, index] for index in top_feature_indices]
    SHAPContributions = [shap_client[index] for index in top_feature_indices]
    
    directions = [ "Toward DEFAULT" if contribution > 0    else "Away from DEFAULT"  if contribution < 0   else "No contribution"
                    for contribution in SHAPContributions ]


    values = ["Missing" if pd.isna(value) else str(value) for value in colValues]
    formatted_contributions = [f"{contribution:+.6f}" for contribution in SHAPContributions]

    fig.add_trace(
        go.Table(
            header=dict(values=["Feature", "Transformed value", "SHAP contribution", "Direction"]),
            cells=dict(values=[colNames, values, formatted_contributions, directions])
        ),
        row=1,
        col=1
    )




    #---------plot 2
    # Comparison chart
    labels = [f"Client {cid}" for cid in comparison[id_column]]
    labels.insert(0, f"Client {customer_id}")

    scores = comparison_scores.tolist()
    scores.insert(0, score)

    fig.add_trace(
        go.Bar(
            x=labels,
            y=scores,
            name="Default probability"
        ),
        row=2,
        col=1
    )



    #----------- plot 3
    # Compare the client's most important transformed features with the population.
    if population_transformed is None:
        population_transformed = preprocessor.transform(X)
    for panel_index, feature_index in enumerate(top_population_Feature_indices, start=1):
        population_values = population_transformed[:, feature_index]
        if hasattr(population_values, "toarray"):
            population_values = population_values.toarray().ravel()
        else:
            population_values = np.asarray(population_values).ravel()

        client_value = client_transformed[0, feature_index]
        if hasattr(client_value, "toarray"):
            client_value = client_value.toarray().ravel()[0]

        feature_name = feature_names[feature_index]
        fig.add_trace(
            go.Box(
                y=population_values,
                name=feature_name,
                boxpoints=False,
                marker_color="#9aa7b8",
                line_color="#536273",
                showlegend=False,
                hovertemplate="Population: %{y}<extra></extra>"
            ),
            row=3,
            col=panel_index
        )
        fig.add_trace(
            go.Scatter(
                x=[feature_name],
                y=[client_value],
                mode="markers",
                name="Selected client",
                marker=dict(color="#d1495b", size=10, symbol="diamond"),
                showlegend=panel_index == 1,
                hovertemplate="Selected client: %{y}<extra></extra>"
            ),
            row=3,
            col=panel_index
        )
        fig.update_xaxes(showticklabels=False, row=3, col=panel_index)

    fig.update_yaxes(
        title_text="Probability of default",
        tickformat=".0%",
        row=2,
        col=1
    )

    fig.update_layout(
        title=f"Local interpretation — Customer {customer_id}",
        height=1100,
        showlegend=True
    )

    return fig




def save_client_outputs(client, X, y, model_data):
    
    
    customer_id = client["id"]
    split = client["split"]
    description = client["description"]
    
    client_output_dir = os.path.join(OUTPUT_DIR, f"client_{customer_id}")
    os.makedirs(client_output_dir, exist_ok=True)

    score, shap_force_plot, plotly_fig = explain_client(
        model=model_data,
        X=X,
        customer_id=customer_id,
    )

    actual_target_raw = y.loc[X["SK_ID_CURR"] == customer_id]
    if isinstance(actual_target_raw, pd.DataFrame):
        actual_target = actual_target_raw.squeeze()
        if isinstance(actual_target, pd.Series):
            actual_target = actual_target.iloc[0]
    elif isinstance(actual_target_raw, pd.Series):
        actual_target = actual_target_raw.iloc[0]
    else:
        actual_target = actual_target_raw

    actual_target = int(actual_target) if pd.notna(actual_target) else None
    predicted_target = int(score >= 0.5)

    result = "unknown" if actual_target is None else (
        "correct" if predicted_target == actual_target else "wrong"
    )

    shap_path = os.path.join(client_output_dir, "shap.html")
    plotly_path = os.path.join(client_output_dir, "visualization.html")
    summary_path = os.path.join(client_output_dir, "summary.txt")

    shap.save_html(shap_path, shap_force_plot)
    plotly_fig.write_html(plotly_path)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Client ID: {customer_id}\n")
        summary_file.write(f"Split: {split}\n")
        summary_file.write(f"Description: {description}\n")
        summary_file.write(f"Default probability: {score:.2%}\n")
        summary_file.write(f"Predicted target (threshold 0.5): {predicted_target}\n")
        summary_file.write(f"Actual target: {actual_target}\n")
        summary_file.write(f"Prediction: {result}\n")

    print(f"\nClient {customer_id} ({split}, {description})")
    print(f"Default probability: {score:.2%}")
    print(f"Predicted target: {predicted_target}")
    print(f"Actual target: {actual_target}")
    print(f"Prediction: {result}")
    print(f"Saved to: {client_output_dir}")
