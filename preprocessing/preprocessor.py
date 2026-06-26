"""
Heart Disease Prediction — Preprocessing Pipeline
Handles feature engineering, encoding, and scaling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Column metadata ────────────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age":      "Age (years)",
    "sex":      "Sex (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)",
    "restecg":  "Resting ECG Results (0-2)",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise Induced Angina (1=Yes, 0=No)",
    "oldpeak":  "ST Depression Induced by Exercise",
    "slope":    "Slope of Peak Exercise ST Segment (0-2)",
    "ca":       "Number of Major Vessels Colored by Fluoroscopy (0-4)",
    "thal":     "Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)",
}

TARGET_COL = "target"
FEATURE_COLS = list(FEATURE_DESCRIPTIONS.keys())

CP_MAP    = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
THAL_MAP  = {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Reversible Defect"}
SLOPE_MAP = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
SEX_MAP   = {1: "Male", 0: "Female"}


# ── Core functions ─────────────────────────────────────────────────────────────

def load_data(path: str | Path) -> pd.DataFrame:
    """Load and do lightweight sanity checks on the CSV."""
    df = pd.read_csv(path)
    missing = df.isnull().sum().sum()
    if missing:
        logger.warning("Dataset has %d missing values — will impute.", missing)
    dups = df.duplicated().sum()
    if dups:
        logger.info("Dropping %d duplicate rows.", dups)
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(estimator) -> Pipeline:
    """Wrap an sklearn estimator with StandardScaler."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  estimator),
    ])


def prepare_single_input(data: dict) -> pd.DataFrame:
    """Convert a chatbot input dict into a DataFrame suitable for prediction."""
    row = {col: [data.get(col, 0)] for col in FEATURE_COLS}
    return pd.DataFrame(row)


def decode_features(data: dict) -> dict[str, str]:
    """Return human-readable labels for raw feature values."""
    return {
        "Age":               f"{data.get('age')} years",
        "Sex":               SEX_MAP.get(data.get("sex", 0), "Unknown"),
        "Chest Pain Type":   CP_MAP.get(data.get("cp", 0), "Unknown"),
        "Blood Pressure":    f"{data.get('trestbps')} mm Hg",
        "Cholesterol":       f"{data.get('chol')} mg/dl",
        "Fasting Blood Sugar>120": "Yes" if data.get("fbs") else "No",
        "Resting ECG":       str(data.get("restecg")),
        "Max Heart Rate":    str(data.get("thalach")),
        "Exercise Angina":   "Yes" if data.get("exang") else "No",
        "ST Depression":     str(data.get("oldpeak")),
        "Slope":             SLOPE_MAP.get(data.get("slope", 0), "Unknown"),
        "Major Vessels":     str(data.get("ca")),
        "Thalassemia":       THAL_MAP.get(data.get("thal", 0), "Unknown"),
    }
