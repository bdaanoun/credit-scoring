import os
import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt

def plot_feature_importance(model, preprocessor, X,output_path,top_n=20):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    feature_names = preprocessor.get_feature_names_out()

    importance = np.abs(shap_values).mean(axis=0)

    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 8))

    plt.barh(
        range(top_n),
        importance[indices]
    )

    plt.yticks(
        range(top_n),
        feature_names[indices]
    )

    plt.xlabel("Mean |SHAP value|")
    plt.title("Top Feature Importance")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()