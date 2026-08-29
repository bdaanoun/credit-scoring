from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_DIR / "results" / "model" / "xgboost.pkl"

X_TRAIN_PATH = PROJECT_DIR / "data" / "X_train_processed.csv"
X_TEST_PATH = PROJECT_DIR / "data" / "X_test_processed.csv"

Y_TRAIN_PATH = PROJECT_DIR / "data" / "y_train.csv"
Y_TEST_PATH = PROJECT_DIR / "data" / "y_test.csv"

BEST_THRESHOLD = 0.16

COLORS = {
    "bg": "#fafafa",
    "surface": "#ffffff",
    "border": "#e5e7eb",
    "text": "#111827",
    "text_secondary": "#6b7280",
    "text_muted": "#9ca3af",
    "accent": "#2563eb",
    "low": "#059669",
    "low_bg": "#ecfdf5",
    "medium": "#d97706",
    "medium_bg": "#fffbeb",
    "high": "#dc2626",
    "high_bg": "#fef2f2",
}

FONT = "Inter, sans-serif"


def _risk_level(score: float) -> str:
    if score < 0.10:
        return "low"
    if score < 0.20:
        return "medium"
    return "high"


RISK_LABELS = {
    "low": "Low risk",
    "medium": "Medium risk",
    "high": "High risk",
}
