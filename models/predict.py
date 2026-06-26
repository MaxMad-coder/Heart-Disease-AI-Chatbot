"""
Heart Disease Prediction — Inference Module
Loads the saved model and performs single-patient prediction with explanations.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"


def load_model():
    """Load the trained pipeline from disk."""
    model_path = MODELS / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Run `python models/train_model.py` first."
        )
    return joblib.load(model_path)


def load_meta() -> dict:
    meta_path = MODELS / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def load_shap_importance() -> dict:
    shap_path = ASSETS / "shap_values.json"
    if shap_path.exists():
        with open(shap_path) as f:
            return json.load(f)
    return {}


def predict(model, input_df: pd.DataFrame) -> dict:
    """
    Returns:
        prediction  : 0 or 1
        probability : float (0-1), probability of heart disease
        confidence  : "High" / "Moderate" / "Low"
        risk_label  : str
    """
    proba = model.predict_proba(input_df)[0]
    pred  = int(model.predict(input_df)[0])
    prob  = float(proba[1])

    if prob >= 0.75:
        confidence = "High"
    elif prob >= 0.50:
        confidence = "Moderate"
    else:
        confidence = "Low"

    risk_label = "High Risk of Heart Disease" if pred == 1 else "Low Risk of Heart Disease"

    return {
        "prediction":  pred,
        "probability": round(prob, 4),
        "confidence":  confidence,
        "risk_label":  risk_label,
    }


def generate_shap_explanation(input_df: pd.DataFrame, model) -> dict[str, float]:
    """Return per-feature SHAP values for a single patient row."""
    try:
        import shap
        model_obj = model.named_steps["model"]
        scaler    = model.named_steps["scaler"]
        X_scaled  = pd.DataFrame(
            scaler.transform(input_df), columns=input_df.columns
        )
        model_name = type(model_obj).__name__
        if "Forest" in model_name or "Tree" in model_name or "XGB" in model_name:
            explainer = shap.TreeExplainer(model_obj)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                sv = sv[1]
        else:
            background = shap.maskers.Independent(X_scaled, max_samples=50)
            explainer  = shap.Explainer(model_obj.predict_proba, background)
            sv = explainer(X_scaled).values[:, :, 1]
        return dict(zip(input_df.columns.tolist(), sv[0].tolist()))
    except Exception as e:
        logger.warning("SHAP single-sample failed: %s", e)
        # Fall back to stored mean SHAP importances
        return load_shap_importance()


def build_patient_explanation(shap_vals: dict[str, float], top_n: int = 5) -> str:
    """Build a natural-language explanation from SHAP values."""
    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    increasing = [(f, v) for f, v in sorted_feats if v > 0][:top_n]
    decreasing = [(f, v) for f, v in sorted_feats if v < 0][:top_n]

    friendly = {
        "age":      "Age",
        "sex":      "Gender",
        "cp":       "Chest Pain Type",
        "trestbps": "Resting Blood Pressure",
        "chol":     "Cholesterol Level",
        "fbs":      "Fasting Blood Sugar",
        "restecg":  "Resting ECG Result",
        "thalach":  "Maximum Heart Rate",
        "exang":    "Exercise-induced Angina",
        "oldpeak":  "ST Depression",
        "slope":    "ST Slope",
        "ca":       "Number of Major Vessels",
        "thal":     "Thalassemia Type",
    }

    lines = []
    if increasing:
        feats = ", ".join(friendly.get(f, f) for f, _ in increasing)
        lines.append(f"**Risk-increasing factors:** {feats}")
    if decreasing:
        feats = ", ".join(friendly.get(f, f) for f, _ in decreasing)
        lines.append(f"**Risk-decreasing factors:** {feats}")
    return "\n".join(lines) if lines else "Feature contributions are balanced."
