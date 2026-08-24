import os
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def generate_learning_curve(model, X, y, output_path):

    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    train_scores = []
    validation_scores = []

    print("\nGenerating learning curve...")
    print(f"Training sizes: {train_sizes}")
    print(f"CV folds: {cv.n_splits}\n")

    for size_index, size in enumerate(train_sizes, start=1):

        n_samples = int(len(X) * size)

        print(
            f"[{size_index}/{len(train_sizes)}] "
            f"Training size: {n_samples:,} ({size:.0%})"
        )

        X_subset = X.iloc[:n_samples]
        y_subset = y.iloc[:n_samples]

        fold_train_scores = []
        fold_validation_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_subset, y_subset),
            start=1
        ):

            print(
                f"    Fold {fold}/{cv.n_splits}...",
                flush=True
            )

            model_clone = clone(model)

            model_clone.fit(
                X_subset.iloc[train_idx],
                y_subset.iloc[train_idx]
            )

            train_proba = model_clone.predict_proba(
                X_subset.iloc[train_idx]
            )[:, 1]

            val_proba = model_clone.predict_proba(
                X_subset.iloc[val_idx]
            )[:, 1]

            train_auc = roc_auc_score(y_subset.iloc[train_idx],train_proba)

            val_auc = roc_auc_score(y_subset.iloc[val_idx],val_proba)

            fold_train_scores.append(train_auc)
            fold_validation_scores.append(val_auc)

            print(
                f"        Train AUC: {train_auc:.4f} | "
                f"Validation AUC: {val_auc:.4f}"
            )

        train_mean = sum(fold_train_scores) / len(fold_train_scores)
        val_mean = sum(fold_validation_scores) / len(fold_validation_scores)

        train_scores.append(train_mean)
        validation_scores.append(val_mean)

        print(
            f"    → Mean Train AUC: {train_mean:.4f} | "
            f"Mean Validation AUC: {val_mean:.4f}\n"
        )

    # Plot

    plt.figure(figsize=(10, 6))

    plt.plot(
        train_sizes,
        train_scores,
        marker="o",
        label="Training AUC"
    )

    plt.plot(
        train_sizes,
        validation_scores,
        marker="o",
        label="Validation AUC"
    )

    plt.xlabel("Training Set Size")
    plt.ylabel("ROC AUC")
    plt.title("Learning Curve - Random Forest")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"Learning curve saved to: {output_path}")