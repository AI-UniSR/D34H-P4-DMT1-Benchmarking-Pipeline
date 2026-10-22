"""
Explanation of File:
File that is used to benchmark different models against each other on the Diabetes T1 Dataset

Author: Muhammad Salaar Arslan
Date: 10/06/2026
Email: arslan.muhammad@hsr.it
"""

import argparse
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from plotting import (
    apply_completeness_filters,
    plot_completeness,
    plot_drop_order_comparison,
    plot_feature_counts_by_target,
    plot_feature_selection_stability,
)
from explainability_plotting import plot_explainability
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from typing import Optional


def main(args: argparse.Namespace):
    """
    Runs all of the functions and creates the graphs for model evaluation
    """
    data_dir = args.data_dir
    output_dir = args.output_dir
    train_size = args.train_size
    min_age = args.min_age
    max_age = args.max_age
    patient_wise_threshold = args.patient_wise_threshold
    exam_wise_threshold = args.exam_wise_threshold
 
    np.random.seed(42)
    random.seed(42)
 
    target_names = ["IDAA1c_12m", "IDAA1c_12_24m", "IDAA1c_12_36m"]
 
    with mlflow.start_run() as parent_run:
        dataframes = load_data(data_dir, target_names=target_names)
 
        # Drop columns that should not be used as features
        cols_to_drop = [
            "Year of Diagnosis",
            "IDAA1c_min_12m",
            "IDAA1c_min_12_24m",
            "IDAA1c_min_12_36m",
            "T0"
        ]
        for tn in target_names:
            existing = [col for col in cols_to_drop if col in dataframes[tn].columns]
            dataframes[tn] = dataframes[tn].drop(columns=existing)
 
        # Accumulate scores across targets for combined plots
        # target → model → scores
        all_model_scores_full: dict[str, dict[str, list[float]]] = {}
        all_model_scores_sig:  dict[str, dict[str, list[float]]] = {}

        # target → {"full": n, "sig": n} -- how many features each target is
        # actually modelled with, for the cross-target summary plot below
        feature_counts_by_target: dict[str, dict[str, int]] = {}
 
        for i, target_name in enumerate(target_names):
            logging.info(f"Processing target {i + 1}/{len(target_names)}: {target_name}")
 
            with mlflow.start_run(nested=True, run_name=f"target_{target_name}"):
                mlflow.log_param("target_name", target_name)
 
                df = dataframes[target_name]
                mlflow.log_param("total_patients_raw", len(df))

                # Diagnostic only — compares both drop orders side by side and does
                # not affect the dataframe used below (see plotting.py).
                plot_drop_order_comparison(
                    df=df,
                    passthrough_cols=["Sesso", "Age at Diagnosis", target_name],
                    patient_wise_threshold=patient_wise_threshold,
                    exam_wise_threshold=exam_wise_threshold,
                    output_dir=output_dir,
                    prefix=target_name,
                )

                X_train, X_test, y_train, y_test, preprocessing_stats = preprocess(
                    df=df,
                    min_age=min_age,
                    max_age=max_age,
                    target_name=target_name,
                    train_size=train_size,
                    patient_wise_threshold=patient_wise_threshold,
                    exam_wise_threshold=exam_wise_threshold,
                    output_dir=output_dir,
                    drop_order=args.drop_order,
                )
 
                mlflow.log_param(f"{target_name}_drop_order", preprocessing_stats["drop_order"])
                mlflow.log_metrics({
                    f"{target_name}_patients_before_age_filter": preprocessing_stats["patients_before_age_filter"],
                    f"{target_name}_patients_after_age_filter":  preprocessing_stats["patients_after_age_filter"],
                    f"{target_name}_patients_train_final":       preprocessing_stats["patients_train_final"],
                    f"{target_name}_patients_test_final":        preprocessing_stats["patients_test_final"],
                    f"{target_name}_total_patients_final":       preprocessing_stats["total_patients_final"],
                    f"{target_name}_train_class_True_count":     preprocessing_stats.get("train_class_True_count", 0),
                    f"{target_name}_train_class_False_count":    preprocessing_stats.get("train_class_False_count", 0),
                    f"{target_name}_test_class_True_count":      preprocessing_stats.get("test_class_True_count", 0),
                    f"{target_name}_test_class_False_count":     preprocessing_stats.get("test_class_False_count", 0),
                    f"{target_name}_n_patients_after_drop_order_step1": preprocessing_stats["n_patients_after_drop_order_step1"],
                    f"{target_name}_n_features_after_drop_order_step1": preprocessing_stats["n_features_after_drop_order_step1"],
                    f"{target_name}_n_features_final":                  preprocessing_stats["n_features_final_after_completeness_filter"],
                })
 
                # Mirror the bool cast that evaluate_models will apply to X_train
                object_cols_test = X_test.select_dtypes(include="object").columns.tolist()
                X_test[object_cols_test] = X_test[object_cols_test].astype(bool)

                # ── Log CV fold sizes (consistent across folds for stratified CV) ─────────
                kf_diagnostic = RepeatedStratifiedKFold(
                    n_splits=args.n_folds, n_repeats=args.n_repeats, random_state=42
                )
                _fold_train_idx, _fold_val_idx = next(kf_diagnostic.split(X_train, y_train))
                mlflow.log_metrics({
                    f"{target_name}_cv_fold_train_size": len(_fold_train_idx),
                    f"{target_name}_cv_fold_val_size":   len(_fold_val_idx),
                })
                logging.info(
                    f"[{target_name}] CV fold sizes — "
                    f"train: {len(_fold_train_idx)}, val: {len(_fold_val_idx)} "
                    f"(across {args.n_folds} folds × {args.n_repeats} repeats)"
                )
 
                # ── RUN 1: Full feature set ───────────────────────────────────
                logging.info(f"[full features] Running benchmark for {target_name}")
 
                model_scores_full = evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    n_folds=args.n_folds,
                    n_repeats=args.n_repeats,
                )
                all_model_scores_full[target_name] = model_scores_full
 
                for model_name, scores in model_scores_full.items():
                    mlflow.log_metric(f"{model_name}_full_mean_roc_auc", np.mean(scores))
                    mlflow.log_metric(f"{model_name}_full_std_roc_auc",  np.std(scores))
 
                output_path = os.path.join(output_dir, f"model_benchmark_scores_{target_name}.csv")
                pd.DataFrame(model_scores_full).to_csv(output_path, index=False)
                mlflow.log_artifact(output_path)
 
                plot_results(model_scores_full, args.output_dir, target_name)
 
                compute_model_explainability(
                    X_train=X_train,
                    y_train=y_train,
                    output_dir=args.output_dir,
                    target_name=target_name,
                    top_n=20,
                )
 
                test_scores_full = evaluate_on_test_set(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
 
                for model_name, metrics in test_scores_full.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_name}_full_test_{metric_name}", value)
 
                output_path = os.path.join(output_dir, f"model_test_scores_{target_name}.csv")
                pd.DataFrame(test_scores_full).T.to_csv(output_path)
                mlflow.log_artifact(output_path)
 
                create_summary_table(
                    model_scores=model_scores_full,
                    test_scores=test_scores_full,
                    output_dir=args.output_dir,
                    target_name=target_name,
                    suffix="full",
                )
 
                # ── RUN 2: Significant features, selected INSIDE the pipeline ──
                # Selection is nested: refit per CV fold for model_scores_sig, and
                # fit once on the full X_train (never on X_test) for test_scores_sig.
                # See SignificantFeatureSelector for why this replaces the old
                # approach of pre-filtering columns from a full-cohort CSV.
                logging.info(f"[sig features] Running nested significant-feature benchmark for {target_name}")

                selection_counts, total_selection_folds = compute_feature_selection_stability(
                    X_train=X_train,
                    y_train=y_train,
                    n_folds=args.n_folds,
                    n_repeats=args.n_repeats,
                    p_threshold=args.sig_p_threshold,
                )
                plot_feature_selection_stability(
                    selection_counts=selection_counts,
                    total_folds=total_selection_folds,
                    output_dir=args.output_dir,
                    prefix=target_name,
                    p_threshold=args.sig_p_threshold,
                )

                # Also fit the selector once on the full X_train, matching what
                # evaluate_on_test_set will do internally — gives a single
                # reportable "final" significant-feature list for the paper,
                # derived from train data only.
                numeric_cols_sig = X_train.select_dtypes(include="number").columns.tolist()
                object_cols_sig  = X_train.select_dtypes(include="object").columns.tolist()
                final_selection_pipeline = Pipeline([
                    ("preprocessing", make_preprocessing_pipeline(
                        numeric_cols=numeric_cols_sig, object_cols=object_cols_sig
                    )),
                    ("feature_selection", SignificantFeatureSelector(p_threshold=args.sig_p_threshold)),
                ])
                final_selection_pipeline.fit(X_train, y_train)
                final_selected_features = final_selection_pipeline.named_steps["feature_selection"].selected_features_

                logging.info(
                    f"[sig features] {target_name}: {len(final_selected_features)} features "
                    f"significant (p<{args.sig_p_threshold}) on the full training set: {final_selected_features}"
                )
                mlflow.log_metric(f"{target_name}_n_significant_features_final", len(final_selected_features))

                output_path = os.path.join(output_dir, f"{target_name}_significant_features_train_only.csv")
                pd.DataFrame({"Variable": final_selected_features}).to_csv(output_path, index=False)
                mlflow.log_artifact(output_path)

                model_scores_sig = evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    n_folds=args.n_folds,
                    n_repeats=args.n_repeats,
                    use_significant_features=True,
                    p_threshold=args.sig_p_threshold,
                )
                all_model_scores_sig[target_name] = model_scores_sig

                for model_name, scores in model_scores_sig.items():
                    mlflow.log_metric(f"{model_name}_sig_mean_roc_auc", np.mean(scores))
                    mlflow.log_metric(f"{model_name}_sig_std_roc_auc",  np.std(scores))

                output_path = os.path.join(
                    output_dir, f"model_benchmark_scores_sig_{target_name}.csv"
                )
                pd.DataFrame(model_scores_sig).to_csv(output_path, index=False)
                mlflow.log_artifact(output_path)

                test_scores_sig = evaluate_on_test_set(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    use_significant_features=True,
                    p_threshold=args.sig_p_threshold,
                )

                for model_name, metrics in test_scores_sig.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_name}_sig_test_{metric_name}", value)

                output_path = os.path.join(
                    output_dir, f"model_test_scores_sig_{target_name}.csv"
                )
                pd.DataFrame(test_scores_sig).T.to_csv(output_path)
                mlflow.log_artifact(output_path)

                create_summary_table(
                    model_scores=model_scores_sig,
                    test_scores=test_scores_sig,
                    output_dir=args.output_dir,
                    target_name=target_name,
                    suffix="sig",
                )

                # Per-target comparison plot (full vs sig) -- annotated with how
                # many features each side actually uses
                plot_feature_set_comparison(
                    model_scores_full=model_scores_full,
                    model_scores_sig=model_scores_sig,
                    target_name=target_name,
                    output_dir=args.output_dir,
                    n_features_full=preprocessing_stats["n_features_final_after_completeness_filter"],
                    n_features_sig=len(final_selected_features),
                )

                logging.info(f"Successfully processed target: {target_name}")
 
            # ── child run closes here; active run reverts to parent_run ──────
            feature_counts_by_target[target_name] = {
                "full": preprocessing_stats["n_features_final_after_completeness_filter"],
                "sig":  len(final_selected_features),
            }
            mlflow.log_metrics({
                f"{target_name}_train_remitters":    preprocessing_stats.get("train_class_True_count", 0),
                f"{target_name}_train_nonremitters": preprocessing_stats.get("train_class_False_count", 0),
                # Feature counts logged at the PARENT run too (not just the nested
                # target run), so all three targets are visible together without
                # drilling into each nested run.
                f"{target_name}_n_features_full":        feature_counts_by_target[target_name]["full"],
                f"{target_name}_n_features_significant": feature_counts_by_target[target_name]["sig"],
            })
 
        # ── After all targets: combined boxplot for full features ─────────────
        plot_combined_results(
            all_model_scores=all_model_scores_full,
            target_names=target_names,
            output_dir=args.output_dir,
            filename="roc_auc_combined_boxplot_full.png",
            title="CV ROC AUC by Model and Remission Target (Full Features)",
        )
 
        # ── If sig features ran for all targets: combined boxplot for sig ─────
        if all_model_scores_sig:
            plot_combined_results(
                all_model_scores=all_model_scores_sig,
                target_names=[t for t in target_names if t in all_model_scores_sig],
                output_dir=args.output_dir,
                filename="roc_auc_combined_boxplot_sig.png",
                title="CV ROC AUC by Model and Remission Target (Significant Features)",
            )

        # ── Cross-target summary: how many features is each target modelled with ──
        plot_feature_counts_by_target(
            feature_counts=feature_counts_by_target,
            output_dir=args.output_dir,
        )


def load_data(data_dir: str, target_names: list[str]) -> dict[str, pd.DataFrame]:
    """
    Loading the data from Azure Machine Learning

    Args:
        data_dir (str): input path for the dataset

    Returns:
        dict[str, pd.DataFrame]: Dict of loaded dataframes.
    """
    dataframes = {}
    for target in target_names:
        path = os.path.join(data_dir, f"{target}_data.csv")
        dataframes[target] = pd.read_csv(path)

    return dataframes


def preprocess(
    df: pd.DataFrame,
    min_age: int,
    max_age: int,
    target_name: str,
    train_size: float,
    patient_wise_threshold: float,
    exam_wise_threshold: float,
    output_dir: str,
    drop_order: str = "features_first",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Function that aims to do a quick preprocessing before splitting the dataset into Train and Test set.

    Args:
        df (pd.DataFrame): The Dataframe containing 200+ features
        min_age (int): Minimum Age to stratify the data on
        max_age (int): Maximum Age to stratify the data on
        patient_wise_threshold (float): Threshold to enforce for patient-wise completeness.
        exam_wise_threshold (float): Threshold to enforce for exam-wise completeness.
        target_name (str): Name of the target variable
        train_size (float): Size of the training set
        drop_order (str): "features_first" (default — drop sparse exams before sparse
            patients) or "patients_first" (drop sparse patients before sparse exams).
            See apply_completeness_filters in plotting.py for why this matters.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
            X_train, X_test, y_train, y_test, preprocessing_stats
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # "Year of Diagnosis" is dropped in main before this function is called
    passthrough_cols = ["Sesso", "Age at Diagnosis", target_name]

    # Apply patient-wise / exam-wise completeness filters in the requested order.
    # Defaults to "features_first": dropping sparse exams before sparse patients
    # means a patient isn't penalised for missingness in columns that were going
    # to be dropped anyway, which tends to retain more patients.
    filter_result = apply_completeness_filters(
        df=df,
        passthrough_cols=passthrough_cols,
        patient_wise_threshold=patient_wise_threshold,
        exam_wise_threshold=exam_wise_threshold,
        drop_order=drop_order,
    )
    df = filter_result["df"]

    # Plot completeness report — the two series reflect whichever stage they were
    # actually used to decide at (see apply_completeness_filters docstring); under
    # "features_first" that means exam completeness is judged over all patients and
    # patient completeness is judged over the surviving exams, and vice versa.
    plot_completeness(
        filter_result["patient_completeness"] * 100,
        filter_result["exam_completeness"] * 100,
        patient_wise_threshold=patient_wise_threshold,
        exam_wise_threshold=exam_wise_threshold,
        output_dir=output_dir,
        prefix=target_name,
    )

    # Defensive re-check of the age range -- clean.py already enforces this
    # upstream, so this should be a no-op (previously this was the ONLY place
    # the upper bound (max_age) got enforced at all; the lower bound alone was
    # checked in target.py, letting patients >= max_age through as far as here).
    patients_before_age_filter = len(df)

    df = df[(df["Age at Diagnosis"] >= min_age) & (df["Age at Diagnosis"] < max_age)]

    patients_after_age_filter = len(df)
    n_filtered_by_age = patients_before_age_filter - patients_after_age_filter
    if n_filtered_by_age > 0:
        logging.warning(
            f"[preprocess] {n_filtered_by_age} patient(s) outside age range "
            f"[{min_age}, {max_age}) reached benchmark.py for {target_name} -- "
            f"clean.py's upstream age filter should have already removed these."
        )

    # Train-test split
    df = df.set_index("CPI")

    X = df.drop(columns=target_name)
    y = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=42
    )

    patients_train_final = len(y_train)
    patients_test_final = len(y_test)

    train_class_counts = y_train.value_counts().to_dict()
    test_class_counts = y_test.value_counts().to_dict()

    preprocessing_stats = {
        "patients_before_age_filter": patients_before_age_filter,
        "patients_after_age_filter":  patients_after_age_filter,
        "patients_filtered_by_age":   patients_before_age_filter - patients_after_age_filter,
        "patients_train_final":       patients_train_final,
        "patients_test_final":        patients_test_final,
        "total_patients_final":       patients_train_final + patients_test_final,
        "train_test_ratio":           patients_train_final / (patients_train_final + patients_test_final),
        "drop_order":                              filter_result["drop_order"],
        "n_patients_after_drop_order_step1":       filter_result["n_patients_after_step1"],
        "n_features_after_drop_order_step1":       filter_result["n_features_after_step1"],
        "n_features_final_after_completeness_filter": filter_result["n_features_final"],
    }

    for class_label, count in train_class_counts.items():
        preprocessing_stats[f"train_class_{class_label}_count"] = count
    for class_label, count in test_class_counts.items():
        preprocessing_stats[f"test_class_{class_label}_count"] = count

    for class_label, count in train_class_counts.items():
        preprocessing_stats[f"train_class_{class_label}_proportion"] = count / patients_train_final
    for class_label, count in test_class_counts.items():
        preprocessing_stats[f"test_class_{class_label}_proportion"] = count / patients_test_final

    return X_train, X_test, y_train, y_test, preprocessing_stats


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int,
    n_repeats: int,
    use_significant_features: bool = False,
    p_threshold: float = 0.05,
) -> dict[str, list[float]]:
    """
    Evaluate models using repeated stratified K-fold cross-validation.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_folds (int): Number of CV folds.
        n_repeats (int): Number of CV repeats.
        use_significant_features (bool): If True, insert a SignificantFeatureSelector
            step between preprocessing and the model, so univariate feature
            selection is refit on each fold's training portion only (see
            SignificantFeatureSelector docstring for why this matters).
        p_threshold (float): Significance threshold passed to SignificantFeatureSelector.

    Returns:
        dict[str, list[float]]: Model name → list of ROC AUC scores per fold.
    """
    base_models = [
        ("logreg", LogisticRegression(max_iter=10000, random_state=42, class_weight="balanced")),
        ("rf",     RandomForestClassifier(random_state=42, class_weight="balanced")),
        ("erf",    ExtraTreesClassifier(random_state=42, class_weight="balanced")),
        ("xgb",    XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=(
                len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                if len(y_train[y_train == 1]) > 0 else 1
            ),
        )),
        ("svc", SVC(probability=True, random_state=42, class_weight="balanced", kernel="rbf")),
        ("knn", KNeighborsClassifier()),
        ("nb",  GaussianNB()),
        ("bnb", BernoulliNB()),
        ("mlp", MLPClassifier(random_state=42, max_iter=1000)),
        ("hgb", HistGradientBoostingClassifier(random_state=42)),
    ]

    # models = base_models + ensemble_models
    models = base_models

    # Define the preprocessing pipeline
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    object_cols  = X_train.select_dtypes(include="object").columns.tolist()
    X_train[object_cols] = X_train[object_cols].astype(bool)

    preprocessing_pipeline = make_preprocessing_pipeline(
        numeric_cols=numeric_cols, object_cols=object_cols
    )

    model_scores = {}
    for model_name, model in models:
        logging.info(f"Evaluating the model name: {model_name}")

        kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
        steps = [("preprocessing", preprocessing_pipeline)]
        if use_significant_features:
            steps.append(("feature_selection", SignificantFeatureSelector(p_threshold=p_threshold)))
        steps.append(("model", model))
        pipeline = Pipeline(steps)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="roc_auc")
        model_scores[model_name] = cv_scores

    return model_scores


def make_preprocessing_pipeline(numeric_cols: list, object_cols: list) -> Pipeline:
    """Make the preprocessing pipeline.

    Args:
        numeric_cols (list): Numeric columns.
        object_cols (list): Object columns.

    Returns:
        Pipeline: Preprocessing pipeline.
    """
    steps = []

    if numeric_cols:
        steps.extend([
            ("median_imputer", MeanMedianImputer(imputation_method="median", variables=numeric_cols)),
            ("scaler", SklearnTransformerWrapper(RobustScaler(), variables=numeric_cols)),
        ])

    if object_cols:
        steps.append((
            "cat_imputer",
            CategoricalImputerWithTieBreaker(
                variables=object_cols,
                tie_breaker="first",
            ),
        ))

    if not steps:
        from sklearn.preprocessing import FunctionTransformer
        steps.append(("passthrough", FunctionTransformer()))

    return Pipeline(steps)


class CategoricalImputerWithTieBreaker(CategoricalImputer):
    """A version of the categorical imputer that breaks ties."""

    def __init__(self, tie_breaker: str = "first", **kwargs):
        super().__init__(**kwargs)
        self.tie_breaker = tie_breaker

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # Cast variables to object so feature_engine validates them correctly —
        # necessary because columns may be bool after preprocessing
        X_fit = X.copy()
        vars_to_cast = self.variables if self.variables is not None else X_fit.columns.tolist()
        for var in vars_to_cast:
            if var in X_fit.columns:
                X_fit[var] = X_fit[var].astype(object)

        # Always call super() to set all required attributes:
        # variables_, n_features_in_, feature_names_in_, imputer_dict_
        super().fit(X_fit, y)

        # Overwrite imputer_dict_ with tie-breaking logic for the frequent case
        if self.imputation_method == "frequent":
            for var in self.variables_:
                value_counts = X[var].value_counts()
                max_count = value_counts.max()
                tied_values = value_counts[value_counts == max_count].index.tolist()

                if len(tied_values) > 1:
                    if self.tie_breaker == "first":
                        self.imputer_dict_[var] = value_counts.index[0]
                    elif self.tie_breaker == "alphabetical":
                        self.imputer_dict_[var] = min(tied_values)
                # else: super() already set the correct single-mode value

        return self


class SignificantFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep only features whose univariate association with the target has p < threshold.

    Meant to sit INSIDE a CV pipeline, after preprocessing/imputation and before
    the model, so the univariate logistic regression that decides which features
    are "significant" is refit on each fold's training portion only — it never
    sees the fold's validation rows (or, for the final test-set fit, the test
    set). This replaces the old approach of pre-filtering columns using a
    significant-features CSV computed once on the full cohort, which leaked
    test-set information into feature selection.

    Fits a separate univariate sm.Logit(y ~ const + x) per feature (standardized
    via StandardScaler for numerical stability — this does not change the
    resulting p-value for a continuous predictor, only the fit's convergence
    behaviour). Note: scaling does not affect *which* features clear the
    threshold, only makes the optimizer better-behaved across differently-scaled
    lab values.

    Attributes set by fit():
        selected_features_: list of column names with p < p_threshold (or, if
            fewer than min_features clear the threshold, the min_features
            columns with the lowest p-values — a defensive fallback so a single
            unlucky fold can't produce an empty feature set and break the model
            step downstream).
        p_values_: dict of column name -> p-value, for every column that could
            be fit (useful for diagnostics; not used by transform()).
    """

    def __init__(self, p_threshold: float = 0.05, min_features: int = 1):
        self.p_threshold = p_threshold
        self.min_features = min_features

    def fit(self, X, y):
        X = pd.DataFrame(X)
        # Cast the same way explainability_plotting._safe_transform does, so
        # bool/object columns coming out of the categorical imputer are handled
        # uniformly regardless of their exact dtype after transform.
        X_numeric = X.astype(float)
        y_arr = np.asarray(y)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric), columns=X_numeric.columns, index=X_numeric.index
        )

        selected: list = []
        p_values: dict = {}

        for col in X_scaled.columns:
            temp_df = pd.DataFrame({"const": 1.0, col: X_scaled[col].to_numpy(), "y": y_arr})
            temp_df = temp_df.dropna()
            temp_df = temp_df[np.isfinite(temp_df[col])]

            if len(temp_df) < 3 or temp_df["y"].nunique() < 2:
                continue

            try:
                fitted = sm.Logit(temp_df["y"], temp_df[["const", col]]).fit(disp=0)
                p = fitted.pvalues[col]
            except Exception as exc:
                logging.debug(f"[SignificantFeatureSelector] Could not fit {col}: {exc}")
                continue

            p_values[col] = p
            if p < self.p_threshold:
                selected.append(col)

        if len(selected) < self.min_features and p_values:
            ranked = sorted(p_values, key=p_values.get)
            selected = ranked[: self.min_features]
            logging.info(
                f"[SignificantFeatureSelector] Only "
                f"{sum(1 for p in p_values.values() if p < self.p_threshold)} feature(s) "
                f"cleared p<{self.p_threshold} this fold; falling back to top "
                f"{len(selected)} by p-value so the model step isn't left with zero features."
            )

        self.selected_features_ = selected
        self.p_values_ = p_values
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X[self.selected_features_]


def compute_feature_selection_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int,
    n_repeats: int,
    p_threshold: float = 0.05,
) -> tuple[dict, int]:
    """Tally how often each feature clears p<threshold across CV folds.

    Runs the exact same preprocessing + SignificantFeatureSelector combination
    used inside the "sig features" model pipelines, but as a single dedicated
    pass rather than once per model — selection depends only on X_train/y_train
    and the fold split, not on which of the 10 downstream models is used, so
    repeating it 10x would be redundant. This is purely a stability diagnostic;
    it does not feed back into the benchmarking pipeline itself.

    Args:
        X_train: Training features (already bool-cast, as in evaluate_models).
        y_train: Training labels.
        n_folds: Number of CV folds.
        n_repeats: Number of CV repeats.
        p_threshold: Significance threshold used by SignificantFeatureSelector.

    Returns:
        (selection_counts, total_folds):
            selection_counts: feature name -> number of folds it was selected in.
            total_folds: n_folds * n_repeats (denominator for proportions).
    """
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    object_cols  = X_train.select_dtypes(include="object").columns.tolist()

    kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
    pipeline = Pipeline([
        ("preprocessing", make_preprocessing_pipeline(numeric_cols=numeric_cols, object_cols=object_cols)),
        ("feature_selection", SignificantFeatureSelector(p_threshold=p_threshold)),
    ])

    selection_counts: dict = dict.fromkeys(X_train.columns, 0)
    total_folds = 0

    for train_idx, _ in kf.split(X_train, y_train):
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        try:
            pipeline.fit(X_fold, y_fold)
        except Exception as exc:
            logging.warning(f"[feature selection stability] Fold fit failed, skipping: {exc}")
            continue

        for feat in pipeline.named_steps["feature_selection"].selected_features_:
            selection_counts[feat] = selection_counts.get(feat, 0) + 1
        total_folds += 1

    return selection_counts, total_folds


def plot_results(model_scores: dict[str, list[float]], output_dir: str, target_name: str) -> None:
    """
    Create boxplot and pointplot of CV ROC AUC scores.

    Args:
        model_scores (dict[str, list[float]]): Model name → list of ROC AUC scores.
        output_dir (str): Directory to save plots.
        target_name (str): Target variable name (used in titles/filenames).
    """
    df_scores = pd.DataFrame([
        {"Model": model_name, "ROC AUC": score}
        for model_name, scores in model_scores.items()
        for score in scores
    ])

    model_order = (
        df_scores.groupby("Model")["ROC AUC"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    best_model = model_order[0]
    best_score = df_scores[df_scores["Model"] == best_model]["ROC AUC"].mean()
    mlflow.log_param("best_model", best_model)
    mlflow.log_metric("best_model_mean_roc_auc", best_score)

    # Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Model", y="ROC AUC", data=df_scores, order=model_order)
    plt.ylim(0.0, 1)
    plt.title(f"ROC AUC Scores Sorted by Mean Performance ({target_name})")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"roc_auc_boxplot_{target_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()

    # Pointplot
    plt.figure(figsize=(10, 6))
    sns.pointplot(
        y="Model", x="ROC AUC", data=df_scores, order=model_order,
        errorbar=("ci", 95), linestyle="none", hue="Model",
    )
    plt.xticks(rotation=45)
    plt.xlim(0.0, 1.0)
    save_path = os.path.join(output_dir, f"roc_auc_pointplot_{target_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()

def plot_combined_results(
    all_model_scores: dict[str, dict[str, list[float]]],
    target_names: list[str],
    output_dir: str,
    filename: str = "roc_auc_combined_boxplot.png",
    title: str = "CV ROC AUC by Model and Remission Target",
) -> None:
    """Boxplot of CV ROC AUC with one group per model, three boxes per group.

    Each box represents one remission target (12m, 12-24m, 12-36m),
    coloured consistently across models.

    Args:
        all_model_scores: target_name → model_name → list of CV ROC AUC scores.
        target_names:     Ordered list of target names.
        output_dir:       Directory to save the plot.
    """
    # Build a long-form DataFrame across all targets and models
    rows = []
    for target_name, model_scores in all_model_scores.items():
        for model_name, scores in model_scores.items():
            for score in scores:
                rows.append({
                    "Model":  model_name,
                    "Target": target_name,
                    "ROC AUC": score,
                })
    df = pd.DataFrame(rows)

    # Order models by mean ROC AUC averaged across all targets
    model_order = (
        df.groupby("Model")["ROC AUC"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Readable target labels and consistent palette
    label_map = {
        "IDAA1c_12m":     "12m",
        "IDAA1c_12_24m":  "12–24m",
        "IDAA1c_12_36m":  "12–36m",
    }
    df["Target Label"] = df["Target"].map(label_map).fillna(df["Target"])

    target_labels = [label_map.get(t, t) for t in target_names]
    palette = dict(zip(target_labels, ["#4C72B0", "#DD8452", "#55A868"]))

    fig, ax = plt.subplots(figsize=(16, 7))

    sns.boxplot(
        data=df,
        x="Model",
        y="ROC AUC",
        hue="Target Label",
        hue_order=target_labels,
        order=model_order,
        palette=palette,
        width=0.6,
        ax=ax,
    )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance (0.5)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("ROC AUC (CV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Remission Target", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()
    logging.info(f"[combined plot] Saved → {save_path}")

def compute_model_explainability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: str,
    target_name: str,
    explain_val_size: float = 0.2,
    top_n: int = 20,
) -> None:
    """Fit each base model and generate explainability plots.

    Expects X_train already cast to its final dtypes (as mutated by
    evaluate_models), so column type detection is consistent.

    Args:
        X_train: Training features (already bool-cast).
        y_train: Training labels.
        output_dir: Directory to save plots.
        target_name: Target variable name (used in filenames).
        explain_val_size: Fraction of X_train held out for explanation.
        top_n: Number of top features to display per plot.
    """
    base_models = [
        ("logreg", LogisticRegression(max_iter=10000, random_state=42, class_weight="balanced")),
        ("rf",     RandomForestClassifier(random_state=42, class_weight="balanced")),
        ("erf",    ExtraTreesClassifier(random_state=42, class_weight="balanced")),
        ("xgb",    XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=(
                len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                if len(y_train[y_train == 1]) > 0 else 1
            ),
        )),
        ("svc", SVC(probability=True, random_state=42, class_weight="balanced", kernel="rbf")),
        ("knn", KNeighborsClassifier()),
        ("nb",  GaussianNB()),
        ("bnb", BernoulliNB()),
        ("mlp", MLPClassifier(random_state=42, max_iter=1000)),
        ("hgb", HistGradientBoostingClassifier(random_state=42)),
    ]

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    object_cols  = X_train.select_dtypes(include="object").columns.tolist()

    X_fit, X_explain, y_fit, y_explain = train_test_split(
        X_train, y_train, test_size=explain_val_size, stratify=y_train, random_state=42,
    )

    for model_name, model in base_models:
        logging.info(f"[explainability] Fitting and explaining: {model_name}")

        preprocessing_pipeline = make_preprocessing_pipeline(
            numeric_cols=numeric_cols, object_cols=object_cols
        )
        pipeline = Pipeline([("preprocessing", preprocessing_pipeline), ("model", model)])

        try:
            pipeline.fit(X_fit, y_fit)
        except Exception as exc:
            logging.warning(f"[explainability] Could not fit {model_name}: {exc}")
            continue

        plot_explainability(
            pipeline=pipeline,
            X_train=X_fit,
            X_val=X_explain,
            y_val=y_explain,
            model_name=model_name,
            target_name=target_name,
            output_dir=output_dir,
            top_n=top_n,
        )


def evaluate_on_test_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_significant_features: bool = False,
    p_threshold: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Fit each model on the full training set and evaluate on the held-out test set.

    Args:
        X_train: Full training features (already bool-cast by evaluate_models).
        y_train: Full training labels.
        X_test:  Held-out test features.
        y_test:  Held-out test labels.
        use_significant_features (bool): If True, insert a SignificantFeatureSelector
            step between preprocessing and the model. Because it's fit once on the
            full X_train (never on X_test), this is leakage-free — it's the same
            mechanism as the CV-nested selection, just with one "fold" that happens
            to be the entire training set.
        p_threshold (float): Significance threshold passed to SignificantFeatureSelector.

    Returns:
        dict mapping model name → dict of metric name → value.
    """
    base_models = [
        ("logreg", LogisticRegression(max_iter=10000, random_state=42, class_weight="balanced")),
        ("rf",     RandomForestClassifier(random_state=42, class_weight="balanced")),
        ("erf",    ExtraTreesClassifier(random_state=42, class_weight="balanced")),
        ("xgb",    XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=(
                len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                if len(y_train[y_train == 1]) > 0 else 1
            ),
        )),
        ("svc", SVC(probability=True, random_state=42, class_weight="balanced", kernel="rbf")),
        ("knn", KNeighborsClassifier()),
        ("nb",  GaussianNB()),
        ("bnb", BernoulliNB()),
        ("mlp", MLPClassifier(random_state=42, max_iter=1000)),
        ("hgb", HistGradientBoostingClassifier(random_state=42)),
    ]

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    object_cols  = X_train.select_dtypes(include="object").columns.tolist()

    test_scores = {}
    for model_name, model in base_models:
        logging.info(f"[test set] Fitting and evaluating: {model_name}")

        steps = [("preprocessing", make_preprocessing_pipeline(
            numeric_cols=numeric_cols, object_cols=object_cols
        ))]
        if use_significant_features:
            steps.append(("feature_selection", SignificantFeatureSelector(p_threshold=p_threshold)))
        steps.append(("model", model))
        pipeline = Pipeline(steps)

        try:
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred       = pipeline.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            test_scores[model_name] = {
                "roc_auc":     round(roc_auc_score(y_test, y_pred_proba), 4),
                "recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
                "specificity": round(specificity, 4),
            }

        except Exception as exc:
            logging.warning(f"[test set] Could not evaluate {model_name}: {exc}")
            test_scores[model_name] = {
                "roc_auc":     np.nan,
                "recall":      np.nan,
                "specificity": np.nan,
            }

    return test_scores

def create_summary_table(
    model_scores: dict[str, list[float]],
    test_scores: dict[str, dict[str, float]],
    output_dir: str,
    target_name: str,
    suffix: str = "full",
) -> None:
    """Create and save a summary table combining CV and test set results.

    Args:
        model_scores: CV ROC AUC scores per model from evaluate_models.
        test_scores:  Test set metrics per model from evaluate_on_test_set.
        output_dir:   Directory to save the table.
        target_name:  Target variable name (used in filename).
    """
    rows = []
    for model_name, cv_scores in model_scores.items():
        test_metrics = test_scores.get(model_name, {})
        rows.append({
            "Model":                    model_name,
            "Train ROC AUC (mean)":     round(np.mean(cv_scores), 4),
            "Train ROC AUC (std)":      round(np.std(cv_scores), 4),
            "Train ROC AUC (mean±std)": f"{round(np.mean(cv_scores), 3)} ± {round(np.std(cv_scores), 3)}",
            "Test ROC AUC":             test_metrics.get("roc_auc", np.nan),
            "Recall (Sensitivity)":     test_metrics.get("recall", np.nan),
            "Specificity":              test_metrics.get("specificity", np.nan),
        })

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("Train ROC AUC (mean)", ascending=False)
        .reset_index(drop=True)
    )

    output_path = os.path.join(output_dir, f"model_summary_{suffix}_{target_name}.csv")
    summary_df.to_csv(output_path, index=False)
    mlflow.log_artifact(output_path)
    logging.info(f"[summary] Saved summary table → {output_path}")

def plot_feature_set_comparison(
    model_scores_full: dict[str, list[float]],
    model_scores_sig:  dict[str, list[float]],
    target_name: str,
    output_dir: str,
    n_features_full: int,
    n_features_sig: int,
) -> None:
    """Side-by-side boxplot comparing full vs significant feature set per model.

    Args:
        model_scores_full: model_name → CV ROC AUC scores using all features.
        model_scores_sig:  model_name → CV ROC AUC scores using significant features only.
        target_name:       Target variable name (used in title and filename).
        output_dir:        Directory to save the plot.
        n_features_full:   How many features the "Full" run actually used (post
            completeness filter) -- baked into the legend/title so the plot is
            self-explanatory without cross-referencing mlflow metrics.
        n_features_sig:    How many features the "Significant" run actually used
            (the nested-selection count from the full-X_train fit).
    """
    full_label = f"Full (n={n_features_full})"
    sig_label  = f"Significant (n={n_features_sig})"

    rows = []
    for model_name, scores in model_scores_full.items():
        for score in scores:
            rows.append({"Model": model_name, "Feature Set": full_label, "ROC AUC": score})

    for model_name, scores in model_scores_sig.items():
        for score in scores:
            rows.append({"Model": model_name, "Feature Set": sig_label, "ROC AUC": score})

    df = pd.DataFrame(rows)

    # Order by full-feature mean ROC AUC
    model_order = (
        df[df["Feature Set"] == full_label]
        .groupby("Model")["ROC AUC"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    palette = {full_label: "#4C72B0", sig_label: "#DD8452"}

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.boxplot(
        data=df,
        x="Model",
        y="ROC AUC",
        hue="Feature Set",
        hue_order=[full_label, sig_label],
        order=model_order,
        palette=palette,
        width=0.6,
        ax=ax,
    )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("ROC AUC (CV)", fontsize=12)
    ax.set_title(
        f"Full vs Significant Features — {target_name}",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Feature Set", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"roc_auc_comparison_{target_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    mlflow.log_artifact(save_path)
    plt.close()
    logging.info(f"[comparison plot] Saved → {save_path}")

def parse_args() -> argparse.Namespace:
    """
    Set up argument parser.

    Returns:
        argparse.Namespace: parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Module for benchmarking data against different models"
    )
    parser.add_argument("--data_dir",                type=str,   help="Path to dataset")
    parser.add_argument("--dict_path",               type=str,   help="Path to data dictionary")
    parser.add_argument("--output_dir",              type=str,   help="Path to output the results")
    parser.add_argument("--min_age",                 type=int,   default=2,   help="Minimum age of study")
    parser.add_argument("--max_age",                 type=int,   default=18,  help="Maximum age of study")
    parser.add_argument("--n_folds",                 type=int,   default=10,  help="Number of folds")
    parser.add_argument("--n_repeats",               type=int,   default=10,  help="Number of repeats")
    parser.add_argument("--train_size",              type=float, default=0.7, help="Size of training dataset")
    parser.add_argument("--patient_wise_threshold",  type=float, help="Patient-wise completeness threshold.")
    parser.add_argument("--exam_wise_threshold",     type=float, help="Exam-wise completeness threshold.")
    parser.add_argument(
        "--sig_p_threshold",
        type=float,
        default=0.05,
        help=(
            "Significance threshold for the nested SignificantFeatureSelector used in "
            "the 'sig features' benchmarking run. Selection is refit per CV fold (and "
            "once on the full training set for the test-set numbers) so it never sees "
            "held-out rows — see SignificantFeatureSelector docstring."
        ),
    )
    parser.add_argument(
        "--drop_order",
        type=str,
        default="features_first",
        choices=["features_first", "patients_first"],
        help=(
            "Order to apply completeness filters: drop sparse exams before sparse "
            "patients ('features_first', default — tends to retain more patients) "
            "or drop sparse patients before sparse exams ('patients_first')."
        ),
    )
    return parser.parse_args()


def setup_logging():
    """Configure logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("development.log", mode="w"),
        ],
    )


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)