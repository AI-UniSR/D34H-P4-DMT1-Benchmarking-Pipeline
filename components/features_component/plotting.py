"""
Plotting functions for features step.
Author: Edoardo Draetta
Date: 2025/06/12
"""

import logging
import os

import matplotlib.pyplot as plt
import missingno as msno
import mlflow
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_completeness(
    patient_completeness: pd.Series,
    exam_completeness: pd.Series,
    patient_wise_threshold: float,
    exam_wise_threshold: float,
    output_dir: str | None = None,
    prefix: str | None = None,
):
    """Create and log feature that defines completeness.

    Args:
        patient_completeness (pd.Series): Series of patient-wise completeness.
        exam_completeness (pd.Series): Series of exam-wise completeness.
        patient_wise_threshold (float): Threshold to draw for patient-wise completeness.
        exam_wise_threshold (float): Threshold to draw for exam-wise completeness.
        output_dir (str, optional): Save path for the image.
        prefix (str, optional): Prefix for the save path.
    """
    # Set up modern plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("Set2")  # Professional, colorblind-friendly palette

    # Create single figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Feature Completeness Analysis", fontsize=20, fontweight="bold", y=0.95)

    # Define consistent colors
    primary_color = "#2E86AB"  # Professional blue
    secondary_color = "#A23B72"  # Complementary magenta
    threshold_color = "#F18F01"  # Warning orange
    accent_color = "#C73E1D"  # Deep red

    # Sort data
    patient_completeness_sorted = patient_completeness.sort_values(ascending=False)
    exam_completeness_sorted = exam_completeness.sort_values(ascending=False)

    # Top left: Patient-wise completeness line plot
    axes[0, 0].plot(patient_completeness_sorted.values, linewidth=2.5, color=primary_color, alpha=0.8)
    axes[0, 0].fill_between(range(len(patient_completeness_sorted)), patient_completeness_sorted.values, alpha=0.3, color=primary_color)
    axes[0, 0].axhline(
        y=patient_wise_threshold * 100,
        color=threshold_color,
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {patient_wise_threshold * 100}%",
    )
    axes[0, 0].set_xlabel("Patient Index (Ranked)", fontsize=12, fontweight="semibold")
    axes[0, 0].set_ylabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[0, 0].set_title("Patient-wise Completeness Profile", fontsize=14, fontweight="bold", pad=20)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    # Top right: Patient completeness histogram
    n, bins, patches = axes[0, 1].hist(patient_completeness, bins=40, color=primary_color, alpha=0.7, edgecolor="white", linewidth=0.5)
    # Color bars based on threshold
    for _, (patch, bin_val) in enumerate(zip(patches, bins[:-1], strict=False)):
        if bin_val < patient_wise_threshold * 100:
            patch.set_facecolor(accent_color)
            patch.set_alpha(0.8)

    axes[0, 1].axvline(
        x=patient_wise_threshold * 100,
        color=threshold_color,
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {patient_wise_threshold * 100}%",
    )
    axes[0, 1].set_xlabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[0, 1].set_ylabel("Number of Patients", fontsize=12, fontweight="semibold")
    axes[0, 1].set_title("Patient Completeness Distribution", fontsize=14, fontweight="bold", pad=20)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    # Bottom left: Exam-wise completeness line plot
    axes[1, 0].plot(exam_completeness_sorted.values, linewidth=2.5, color=secondary_color, alpha=0.8)
    axes[1, 0].fill_between(range(len(exam_completeness_sorted)), exam_completeness_sorted.values, alpha=0.3, color=secondary_color)
    axes[1, 0].axhline(
        y=exam_wise_threshold * 100, color=threshold_color, linestyle="--", linewidth=2, label=f"Threshold: {exam_wise_threshold * 100}%"
    )
    axes[1, 0].set_xlabel("Lab Exam Index (Ranked)", fontsize=12, fontweight="semibold")
    axes[1, 0].set_ylabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[1, 0].set_title("Exam-wise Completeness Profile", fontsize=14, fontweight="bold", pad=20)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    # Bottom right: Exam completeness histogram
    n, bins, patches = axes[1, 1].hist(exam_completeness, bins=40, color=secondary_color, alpha=0.7, edgecolor="white", linewidth=0.5)
    # Color bars based on threshold
    for _, (patch, bin_val) in enumerate(zip(patches, bins[:-1], strict=False)):
        if bin_val < exam_wise_threshold * 100:
            patch.set_facecolor(accent_color)
            patch.set_alpha(0.8)

    axes[1, 1].axvline(
        x=exam_wise_threshold * 100, color=threshold_color, linestyle="--", linewidth=2, label=f"Threshold: {exam_wise_threshold * 100}%"
    )
    axes[1, 1].set_xlabel("Completeness (%)", fontsize=12, fontweight="semibold")
    axes[1, 1].set_ylabel("Number of Lab Exams", fontsize=12, fontweight="semibold")
    axes[1, 1].set_title("Exam Completeness Distribution", fontsize=14, fontweight="bold", pad=20)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    if output_dir:
        path = os.path.join(output_dir, f"{prefix}_completeness.png")
    else:
        path = f"{prefix}_completeness.png"

    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.show()


# def plot_features(feature_dataframe: pd.DataFrame, output_dir: str):

#     df = feature_dataframe

#     # ALL FEATURES
#     msno.bar(df, sort="ascending")
#     plot_path = os.path.join(output_dir, "features_completeness_bar.png")
#     plt.savefig(plot_path, bbox_inches="tight")
#     mlflow.log_artifact(plot_path)

#     # For each category, create:
#     # 1. msno bar plot
#     # 2. msno matrix
#     # 3. msno corr
#     # 4. row of histogram - average over time - scatter
#     # or: countplot - prop over time - jitter

#     date_column = "Year of Diagnosis"
#     categories = list(data_dict_df["category"].unique())
#     categories.remove("patient_index")

#     for category in categories:

#         logging.info(f"Plotting {category}...")

#         columns = data_dict_df[data_dict_df["category"] == category]["name"]
#         n = len(columns)

#         msno.matrix(df[columns])
#         plot_path = os.path.join(output_dir, f"{category}_features_completeness_matrix.png")
#         plt.savefig(plot_path, bbox_inches="tight")
#         mlflow.log_artifact(plot_path)
#         plt.close()

#         msno.bar(df[columns], sort="ascending")
#         plot_path = os.path.join(output_dir, f"{category}_features_completeness_bar.png")
#         plt.savefig(plot_path, bbox_inches="tight")
#         mlflow.log_artifact(plot_path)
#         plt.close()

#         msno.heatmap(df[columns], sort="ascending")
#         plot_path = os.path.join(output_dir, f"{category}_features_completeness_heatmap.png")
#         plt.savefig(plot_path, bbox_inches="tight")
#         mlflow.log_artifact(plot_path)
#         plt.close()

#         fig = plt.figure(constrained_layout=True, figsize=(18, 3 * n))
#         gs = GridSpec(n, 3, figure=fig)  # maximum columns = 3

#         for i, col in enumerate(columns):

#             if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):  # Numeric columns

#                 ax_hist = fig.add_subplot(gs[i, 0])
#                 ax_ts = fig.add_subplot(gs[i, 1])
#                 ax_scatter = fig.add_subplot(gs[i, 2])

#                 # Histogram
#                 sns.histplot(df[col], ax=ax_hist)  # type: ignore
#                 ax_hist.set_title(f"Histogram of {col}", fontsize=10)
#                 ax_hist.set_ylabel("Count")

#                 # Average over time (time series)
#                 # Group by the date column to compute mean and std for each date.
#                 date_stats = df.groupby(date_column)[col].agg(["mean", "std"])
#                 date_stats = date_stats.dropna(subset=["mean", "std"])  # drop rows where mean or std is nan
#                 date_mean = date_stats["mean"]
#                 date_std = date_stats["std"]
#                 ax_ts.plot(date_mean.index, date_mean.values, color="darkgreen", label="Average")  # type: ignore
#                 ax_ts.fill_between(
#                     date_mean.index.astype(float),
#                     (date_mean - date_std).astype(float),
#                     (date_mean + date_std).astype(float),
#                     color="lightgreen",
#                     alpha=0.3,
#                     label="Std Dev",
#                 )
#                 global_mean = df[col].mean()
#                 ax_ts.axhline(global_mean, color="red", linestyle="--", label="Global Average")
#                 ax_ts.set_title(f"{col} over time", fontsize=10)
#                 ax_ts.set_xlabel(date_column)
#                 ax_ts.set_ylabel("Value")
#                 ax_ts.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

#                 # Scatter plot
#                 sns.scatterplot(x=df[date_column], y=df[col], ax=ax_scatter, alpha=0.5)
#                 ax_scatter.set_title(f"Scatter: {col}", fontsize=10)
#                 ax_scatter.set_xlabel(date_column)
#                 ax_scatter.set_ylabel(col)

#             elif not pd.api.types.is_datetime64_any_dtype(df[col]):  # non numeric

#                 # Create subplots
#                 ax_pie = fig.add_subplot(gs[i, 0])
#                 ax_ts = fig.add_subplot(gs[i, 1])
#                 ax_stacked = fig.add_subplot(gs[i, 2])

#                 # --- Pie Plot ---
#                 value_counts = df[col].value_counts(dropna=False)

#                 if pd.api.types.is_bool_dtype(df[col]):
#                     if len(value_counts) == 3:
#                         labels = ["Pos", "Neg", "Missing"]
#                         colors = ["#1f77b4", "#ff7f0e", "#d62728"]
#                         colors = ["#006400", "#90EE90", "#000000"]
#                     else:
#                         labels = [
#                             "Pos",
#                             "Neg",
#                         ]
#                         colors = [
#                             "#1f77b4",
#                             "#ff7f0e",
#                         ]
#                     # Create Pie Chart
#                     ax_pie.pie(
#                         value_counts,
#                         labels=labels,
#                         autopct="%1.1f%%",
#                         colors=colors,
#                         startangle=90,
#                         wedgeprops={"edgecolor": "black"},
#                     )
#                     ax_pie.set_title(col)
#                 else:
#                     # Labels for the pie chart
#                     labels = value_counts.index
#                     # Create Pie Chart
#                     ax_pie.pie(
#                         value_counts,
#                         labels=labels,
#                         autopct="%1.1f%%",
#                         startangle=90,
#                         wedgeprops={"edgecolor": "black"},
#                     )
#                     ax_pie.set_title(col)

#                 # --- Time Series Plot ---
#                 # For computing the proportion of True values over time,
#                 # drop rows where the column is NaN.
#                 df_non_nan = df[df[col].notnull()].copy()
#                 date_prop = df_non_nan.groupby(date_column)[col].mean()
#                 ax_ts.plot(date_prop.index, date_prop.values, marker="o", color="darkgreen", label="Proportion True")

#                 # Compute the global proportion from non-NaN values
#                 global_prop = df_non_nan[col].mean()
#                 ax_ts.axhline(global_prop, color="red", linestyle="--", label="Global Proportion")
#                 ax_ts.set_title(f"{col} over time", fontsize=10)
#                 ax_ts.set_xlabel(date_column)
#                 ax_ts.set_ylabel("Proportion")
#                 ax_ts.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

#                 # --- Stacked Bar Plot over Time ---
#                 df_stacked = df.copy()
#                 df_stacked[col] = df_stacked[col].astype(object).fillna("Missing")
#                 # Group data by the date column and the categorical column, then count occurrences.
#                 date_category_counts = df_stacked.groupby([date_column, col]).size().unstack(fill_value=0)
#                 # Sort the DataFrame by date if needed.
#                 date_category_counts = date_category_counts.sort_index()
#                 # Define colors for True, False, and Missing values
#                 color_mapping = {True: "#1f77b4", False: "#ff7f0e", "Missing": "#d62728"}  # Blue  # Orange  # Red
#                 # Ensure the colors match available categories
#                 colors = [
#                     color_mapping.get(cat, "#2ca02c") for cat in date_category_counts.columns
#                 ]  # Default to green if undefined
#                 # Create a stacked bar plot.
#                 date_category_counts.plot(kind="bar", stacked=True, ax=ax_stacked, width=0.8, color=colors)
#                 ax_stacked.set_title(f"Stacked Bar Plot of {col} over time", fontsize=10)
#                 ax_stacked.set_xlabel(date_column)
#                 ax_stacked.set_ylabel("Count")
#                 ax_stacked.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left")
#                 plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

#         plt.tight_layout()
#         plot_path = os.path.join(output_dir, f"{category}_feature_histograms.png")
#         plt.savefig(plot_path, bbox_inches="tight")
#         plt.close()
#         mlflow.log_artifact(plot_path)
