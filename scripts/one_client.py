import os

import joblib
import shap
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
    

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "results", "model", "xgboost.pkl")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "clients_outputs")



def explain_client( model, X, customer_id, id_column="SK_ID_CURR", n_comparison_clients=5):
    
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
    print(f"Customer ID : {customer_id}")
    print(f"Default probability : {score:.2%}")


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
            .replace("onehot__", "")
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
        client,
        X,
        score,
        customer_id,
        preprocessor,
        classifier,
        id_column,
        n_comparison_clients
    )
    
    return score, shap_force_plot, fig




def create_client_visualization(client, X, score, customer_id, preprocessor, classifier, id_column="SK_ID_CURR", n_comparison_clients=5):
    
    # Comparison clients
    other_clients = X[X[id_column] != customer_id]

    comparison = other_clients.sample(min(n_comparison_clients, len(other_clients)), random_state=42)
    comparison_transformed = preprocessor.transform(comparison)
    comparison_scores = classifier.predict_proba(comparison_transformed)[:, 1]

    # Plot layout
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "domain"}], [{"type": "xy"}]],
        subplot_titles=(
            "Client information",
            "Default probability comparison"
        ),
        vertical_spacing=0.20
    )

    # Client information
    info_columns = [
        col for col in client.columns
        if col != id_column
    ][:10]

    values = [
        "Missing" if pd.isna(client.iloc[0][col])
        else str(client.iloc[0][col])
        for col in info_columns
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=["Variable", "Value"]),
            cells=dict(values=[info_columns, values])
        ),
        row=1,
        col=1
    )

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

    fig.update_yaxes(
        title_text="Probability of default",
        tickformat=".0%",
        row=2,
        col=1
    )

    fig.update_layout(
        title=f"Local interpretation — Customer {customer_id}",
        height=800,
        showlegend=False
    )

    return fig




def save_client_outputs(client, X, y, model_data):
    customer_id = client["id"]
    split = client["split"]
    description = client["description"]
    client_output_dir = os.path.join(OUTPUT_DIR, f"client_{customer_id}")
    os.makedirs(client_output_dir, exist_ok=True)

    if customer_id not in set(X["SK_ID_CURR"]):
        raise ValueError(f"Customer {customer_id} was not found in the {split} set.")

    score, shap_force_plot, plotly_fig = explain_client(
        model=model_data,
        X=X,
        customer_id=customer_id,
    )

    actual_target = None if y is None else int(y.loc[X["SK_ID_CURR"] == customer_id].iloc[0])
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
