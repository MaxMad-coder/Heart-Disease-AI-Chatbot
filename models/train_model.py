"""
Heart Disease Prediction — Model Training Pipeline
Trains multiple classifiers, selects the best, tunes it, and saves it.
Run: python models/train_model.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data" / "heart.csv"
ASSETS  = ROOT / "assets"
MODELS  = ROOT / "models"
ASSETS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from preprocessing.preprocessor import (
    FEATURE_COLS, TARGET_COL, build_pipeline, load_data, split_data,
)

# ── 1. Load data ───────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    logger.info("Running EDA …")
    logger.info("Shape: %s", df.shape)
    logger.info("Dtypes:\n%s", df.dtypes)
    logger.info("Stats:\n%s", df.describe())
    logger.info("Missing:\n%s", df.isnull().sum())
    logger.info("Target dist:\n%s", df[TARGET_COL].value_counts())

    # ── target distribution ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#2196F3", "#F44336"]
    df[TARGET_COL].value_counts().plot(
        kind="bar", ax=axes[0], color=colors, edgecolor="white"
    )
    axes[0].set_title("Target Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xticklabels(["No Disease (0)", "Disease (1)"], rotation=0)
    axes[0].set_ylabel("Count")
    pct = df[TARGET_COL].value_counts(normalize=True) * 100
    axes[1].pie(pct, labels=["No Disease", "Disease"], colors=colors,
                autopct="%1.1f%%", startangle=140)
    axes[1].set_title("Class Balance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── correlation heatmap ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── feature distributions ──────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        df[col].hist(ax=axes[i], bins=20, color="#42A5F5", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
    for j in range(len(FEATURE_COLS), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── outlier boxplots ───────────────────────────────────────────────────────
    cont_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(cont_cols), figsize=(16, 5))
    for i, col in enumerate(cont_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#66BB6A")
        axes[i].set_title(col, fontsize=10)
    fig.suptitle("Outlier Detection (Box Plots)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "outliers.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("EDA plots saved to assets/")


# ── 2. Train models ────────────────────────────────────────────────────────────

MODELS_DEF: dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "XGBoost":             XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                          random_state=42, verbosity=0),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(),
}


def evaluate_model(pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),   4),
        "precision": round(precision_score(y_test, y_pred),  4),
        "recall":    round(recall_score(y_test, y_pred),      4),
        "f1":        round(f1_score(y_test, y_pred),          4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),    4),
        "cv_auc":    round(cv.mean(), 4),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "pipeline":  pipeline,
        "y_proba":   y_proba,
    }


def train_all_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results, pipelines, probas = {}, {}, {}
    for name, est in MODELS_DEF.items():
        logger.info("Training %s …", name)
        pipe = build_pipeline(est)
        res  = evaluate_model(pipe, X_train, X_test, y_train, y_test)
        results[name]   = {k: v for k, v in res.items()
                           if k not in ("pipeline", "y_proba", "conf_matrix")}
        pipelines[name] = res["pipeline"]
        probas[name]    = res["y_proba"]

    df_results = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    logger.info("\nModel Comparison:\n%s", df_results.to_string())
    return df_results, pipelines, probas


def plot_model_comparison(df_results: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    palette = sns.color_palette("Set2", len(df_results))
    for i, m in enumerate(metrics):
        bars = axes[i].barh(df_results.index, df_results[m], color=palette)
        axes[i].set_xlim(0.7, 1.01)
        axes[i].set_title(m.upper(), fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Score")
        for bar, val in zip(bars, df_results[m]):
            axes[i].text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=7)
    plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(pipelines, probas, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = plt.cm.get_cmap("tab10")
    for i, (name, proba) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=palette(i))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ASSETS / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 3. Hyperparameter tuning ───────────────────────────────────────────────────

def tune_best_model(best_name, pipelines, X_train, y_train):
    logger.info("Tuning %s …", best_name)
    param_grids = {
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth":    [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "XGBoost": {
            "model__n_estimators":  [100, 200],
            "model__max_depth":     [3, 5, 7],
            "model__learning_rate": [0.05, 0.1, 0.2],
        },
        "Logistic Regression": {
            "model__C":      [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"],
        },
        "SVM": {
            "model__C":      [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights":     ["uniform", "distance"],
        },
        "Decision Tree": {
            "model__max_depth":         [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
        },
    }
    grid = param_grids.get(best_name, {})
    if not grid:
        logger.warning("No param grid for %s; skipping tuning.", best_name)
        return pipelines[best_name]

    search = GridSearchCV(
        pipelines[best_name], grid,
        cv=5, scoring="roc_auc", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    logger.info("Best params: %s", search.best_params_)
    logger.info("Best CV AUC: %.4f", search.best_score_)
    return search.best_estimator_


# ── 4. SHAP explainability ─────────────────────────────────────────────────────

def compute_shap(pipeline, X_test: pd.DataFrame, best_name: str) -> None:
    logger.info("Computing SHAP values …")
    model  = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    try:
        if best_name in ("Random Forest", "Decision Tree", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                sv = sv[1]
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba, shap.sample(X_scaled, 50)
            )
            sv = explainer.shap_values(X_scaled, nsamples=100)[:, :, 1] \
                if hasattr(explainer.shap_values(shap.sample(X_scaled, 5), nsamples=50), '__len__') \
                else explainer.shap_values(X_scaled, nsamples=100)

        # Summary plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X_scaled, show=False, plot_type="bar")
        plt.title("SHAP Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(ASSETS / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Bee-swarm
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X_scaled, show=False)
        plt.title("SHAP Beeswarm Plot", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(ASSETS / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save mean |SHAP| for later use
        mean_shap = np.abs(sv).mean(axis=0)
        shap_dict = dict(zip(X_test.columns.tolist(), mean_shap.tolist()))
        with open(ASSETS / "shap_values.json", "w") as f:
            json.dump(shap_dict, f, indent=2)
        logger.info("SHAP values saved.")
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)


# ── 5. Main entry ──────────────────────────────────────────────────────────────

def main() -> None:
    df = load_data(DATA)
    run_eda(df)

    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    df_results, pipelines, probas = train_all_models(X_train, X_test, y_train, y_test)
    plot_model_comparison(df_results)
    plot_roc_curves(pipelines, probas, y_test)

    best_name = df_results.index[0]
    logger.info("Best model: %s", best_name)

    final_model = tune_best_model(best_name, pipelines, X_train, y_train)

    # Final evaluation
    y_pred  = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    logger.info("\nFinal Model Report:\n%s", classification_report(y_test, y_pred))
    logger.info("Final ROC-AUC: %.4f", roc_auc_score(y_test, y_proba))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(ASSETS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance (for tree-based)
    model_obj = final_model.named_steps["model"]
    if hasattr(model_obj, "feature_importances_"):
        fi = pd.Series(model_obj.feature_importances_, index=FEATURE_COLS)
        fig, ax = plt.subplots(figsize=(8, 6))
        fi.sort_values().plot(kind="barh", ax=ax, color="#42A5F5")
        ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(ASSETS / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    compute_shap(final_model, X_test, best_name)

    # Save model + metadata
    joblib.dump(final_model, MODELS / "best_model.pkl")
    meta = {
        "best_model":  best_name,
        "roc_auc":     float(roc_auc_score(y_test, y_proba)),
        "f1":          float(f1_score(y_test, y_pred)),
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "features":    FEATURE_COLS,
        "comparison":  df_results.drop(columns=["conf_matrix"], errors="ignore").to_dict(),
    }
    with open(MODELS / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model saved to models/best_model.pkl")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
