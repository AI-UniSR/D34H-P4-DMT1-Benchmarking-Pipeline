"""
Explainability functions for plotting.py
Add these imports and functions to your existing plotting.py file.

Strategy per model type:
  - Tree-based (RF, ExtraTrees, XGB, HGB)  → shap.TreeExplainer  (fast, exact)
  - Linear (LogisticRegression)             → shap.LinearExplainer (fast, exact)
  - Opaque (SVC, KNN, NB, BNB, MLP)        → permutation importance (model-agnostic)

KernelExplainer is intentionally avoided — it is too slow for a benchmarking loop
with 10 models × 3 targets.

Author: Muhammad Salaar Arslan
Date: 14/05/2026
"""

# ── Additional imports needed at the top of plotting.py ──────────────────────
import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Tree-based models that have a fast native SHAP explainer
try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from xgboost import XGBClassifier

    _TREE_MODELS = (
        RandomForestClassifier,
        ExtraTreesClassifier,
        XGBClassifier,
        HistGradientBoostingClassifier,
    )
except ImportError:
    _TREE_MODELS = ()

_LINEAR_MODELS = (LogisticRegression,)

# ── Public entry point ────────────────────────────────────────────────────────


def plot_explainability(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    target_name: str,
    output_dir: str,
    top_n: int = 20,
) -> None:
    """Compute and save feature-importance plots for a fitted pipeline.

    Uses SHAP for tree/linear models; falls back to permutation importance for
    opaque models (SVC, KNN, NaiveBayes, MLP …).

    Args:
        pipeline:    Fitted sklearn Pipeline with named steps 'preprocessing'
                     and 'model'.
        X_train:     Training features — used as the SHAP background distribution.
        X_val:       Hold-out features used for explanation / permutation scoring.
        y_val:       Hold-out labels (needed for permutation importance scoring).
        model_name:  Short identifier used in plot titles and filenames.
        target_name: Name of the prediction target (used in titles / filenames).
        output_dir:  Directory where PNG files are saved.
        top_n:       How many features to show in each plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = pipeline.named_steps["model"]
    preprocessing = pipeline.named_steps["preprocessing"]

    # Transform features (feature_engine transformers preserve column names)
    X_train_t = _safe_transform(preprocessing, X_train)
    X_val_t = _safe_transform(preprocessing, X_val)

    if isinstance(model, _TREE_MODELS):
        _explain_with_shap_tree(
            model, X_train_t, X_val_t, model_name, target_name, output_dir, top_n
        )
    elif isinstance(model, _LINEAR_MODELS):
        _explain_with_shap_linear(
            model, X_train_t, X_val_t, model_name, target_name, output_dir, top_n
        )
    else:
        logging.info(
            f"[explainability] {model_name}: no fast SHAP explainer available, "
            "falling back to permutation importance."
        )
        _explain_with_permutation(
            pipeline, X_val, y_val, model_name, target_name, output_dir, top_n
        )


# ── SHAP — tree models ────────────────────────────────────────────────────────


def _explain_with_shap_tree(
    model,
    X_train_t: pd.DataFrame,
    X_val_t: pd.DataFrame,
    model_name: str,
    target_name: str,
    output_dir: str,
    top_n: int,
) -> None:
    """SHAP TreeExplainer path (RF, ExtraTrees, XGB, HGB)."""
    try:
        explainer = shap.TreeExplainer(model, data=X_train_t)
        shap_values = explainer(X_val_t)

        # shap.Explanation object — pick the positive-class slice for binary problems
        if shap_values.values.ndim == 3:
            # shape (n_samples, n_features, n_classes) → class index 1
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values  # already (n_samples, n_features)

        _save_shap_plots(sv, X_val_t, model_name, target_name, output_dir, top_n)

    except Exception as exc:
        logging.warning(f"[explainability] TreeExplainer failed for {model_name}: {exc}")
        logging.warning("[explainability] Skipping SHAP for this model.")


# ── SHAP — linear models ──────────────────────────────────────────────────────


def _explain_with_shap_linear(
    model,
    X_train_t: pd.DataFrame,
    X_val_t: pd.DataFrame,
    model_name: str,
    target_name: str,
    output_dir: str,
    top_n: int,
) -> None:
    """SHAP LinearExplainer path (LogisticRegression …)."""
    try:
        # Use a masker built from the training distribution
        masker = shap.maskers.Independent(X_train_t, max_samples=200)
        explainer = shap.LinearExplainer(model, masker=masker)
        shap_values = explainer(X_val_t)

        # Binary LogReg: shap_values is (n_samples, n_features) for class 1
        _save_shap_plots(shap_values, X_val_t, model_name, target_name, output_dir, top_n)

    except Exception as exc:
        logging.warning(f"[explainability] LinearExplainer failed for {model_name}: {exc}")
        logging.warning("[explainability] Skipping SHAP for this model.")


# ── Permutation importance (model-agnostic fallback) ─────────────────────────


def _explain_with_permutation(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    target_name: str,
    output_dir: str,
    top_n: int,
    n_repeats: int = 10,
) -> None:
    """Permutation importance on the full pipeline (pre-processed features)."""
    try:
        result = permutation_importance(
            pipeline,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=42,
            scoring="roc_auc",
        )

        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_val.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .head(top_n)
            .iloc[::-1]  # reverse so highest bar is at the top
        )

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
        ax.barh(
            importance_df["feature"],
            importance_df["importance_mean"],
            xerr=importance_df["importance_std"],
            color="#4C72B0",
            ecolor="grey",
            capsize=3,
            alpha=0.85,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean decrease in ROC AUC (± 1 std)")
        ax.set_title(f"Permutation Importance — {model_name} ({target_name})")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(
            output_dir, f"perm_importance_{model_name}_{target_name}.png"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        mlflow.log_artifact(save_path)
        plt.close()
        logging.info(f"[explainability] Saved permutation importance → {save_path}")

    except Exception as exc:
        logging.warning(
            f"[explainability] Permutation importance failed for {model_name}: {exc}"
        )


# ── Shared SHAP plot helpers ──────────────────────────────────────────────────


def _save_shap_plots(
    shap_explanation,       # shap.Explanation object
    X_val_t: pd.DataFrame,
    model_name: str,
    target_name: str,
    output_dir: str,
    top_n: int,
) -> None:
    """Save a SHAP beeswarm plot and a SHAP bar plot for the top-N features."""

    # ── 1. Beeswarm (shows direction + magnitude per sample) ──────────────────
    # Restrict to top_n features by mean |shap|
    mean_abs = np.abs(shap_explanation.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]

    shap_top = shap_explanation[:, top_idx]

    fig, _ax = plt.subplots(figsize=(10, max(6, top_n * 0.42)))
    shap.plots.beeswarm(shap_top, show=False, max_display=top_n)
    plt.title(f"SHAP Beeswarm — {model_name} ({target_name})", pad=10)
    plt.tight_layout()

    beeswarm_path = os.path.join(
        output_dir, f"shap_beeswarm_{model_name}_{target_name}.png"
    )
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=150)
    mlflow.log_artifact(beeswarm_path)
    plt.close()
    logging.info(f"[explainability] Saved SHAP beeswarm → {beeswarm_path}")

    # ── 2. Bar plot (mean |SHAP| — cleaner for reports) ───────────────────────
    top_features = [X_val_t.columns[i] for i in top_idx]
    top_means = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(top_features[::-1], top_means[::-1], color="#DD8452", alpha=0.9)
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title(f"SHAP Feature Importance — {model_name} ({target_name})")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    bar_path = os.path.join(
        output_dir, f"shap_bar_{model_name}_{target_name}.png"
    )
    plt.savefig(bar_path, bbox_inches="tight", dpi=150)
    mlflow.log_artifact(bar_path)
    plt.close()
    logging.info(f"[explainability] Saved SHAP bar → {bar_path}")


# ── Utility ───────────────────────────────────────────────────────────────────


def _safe_transform(preprocessing, X: pd.DataFrame) -> pd.DataFrame:
    """Transform X and guarantee the output is a DataFrame with column names."""
    out = preprocessing.transform(X)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out, columns=X.columns, index=X.index)
    
    out = out.astype(float)
    return out

from translations import VARIABLE_TRANSLATION

def _safe_transform(preprocessing, X: pd.DataFrame) -> pd.DataFrame:
    """Transform X, guarantee DataFrame output, cast to float64, and translate column names."""
    out = preprocessing.transform(X)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out, columns=X.columns, index=X.index)

    # Cast to float64 for SHAP compatibility
    out = out.astype(float)

    # Translate column names to English for plot labels
    out = out.rename(columns=VARIABLE_TRANSLATION)

    return out