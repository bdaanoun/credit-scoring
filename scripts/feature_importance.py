import os
import pandas as pd

from matplotlib import pyplot as plt


def plot_feature_importance(model, preprocessor, output_path, top_n=10):
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({"feature": feature_names,"importance": importances})
    importance_df = importance_df.sort_values("importance",ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importance")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

    plt.close()