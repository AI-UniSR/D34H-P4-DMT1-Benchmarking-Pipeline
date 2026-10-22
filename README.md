# Benchmarking Pipeline

This repository contains an Azure ML pipeline for benchmarking predictive models on diabetes-related outcomes using curated clinical metadata and laboratory observations. The pipeline evaluates several classifiers across multiple target windows and produces benchmark summaries, feature-level diagnostics, and cohort attrition analyses.

## Overview

The pipeline is defined in `benchmarking_pipeline.yml` and consists of sequential and parallel Azure ML command jobs:

1. `clean_step`
   - Cleans raw `meteda` and lab data.
   - Applies age and year filters.
   - Produces a standardized clean dataset.

2. `target_step`
   - Builds the target variables used for prediction.
   - Current targets include:
     - `IDAA1c_12m`
     - `IDAA1c_12_24m`
     - `IDAA1c_12_36m`

3. `features_step`
   - Creates feature tables based on the cleaned dataset and target definitions.

4. `benchmark_step`
   - Benchmarks multiple model families using repeated stratified cross-validation.
   - Evaluates model performance using ROC-AUC and related metrics.
   - Produces benchmark plots and model explainability outputs.

5. `statistics_step`
   - Summarizes model performance and feature-level statistics.

6. `diagnose_attrition_step`
   - Identifies why certain patients are excluded from IDAA1c calculations.

7. `idaa1c_availability_step`
   - Compares baseline characteristics between patients with computable and non-computable IDAA1c values.

## Repository structure

- `benchmarking_pipeline.yml` — Azure ML pipeline definition
- `pipeline.sh` — convenience script for launching the Azure ML job
- `components/` — reusable Azure ML command components and Python logic
  - `clean_component/`
  - `target_component/`
  - `features_component/`
  - `benchmarking_component/`
  - `statistics_component/`
  - `diagnose_attrition_component/`
  - `idaa1c_availability_component/`

## Pipeline inputs

The main pipeline configuration in `benchmarking_pipeline.yml` includes the following key parameters:

- `meteda_path`: raw metadata folder
- `lab_path`: laboratory report file
- `tolerance`: target calculation tolerance
- `min_age`: minimum patient age
- `max_age`: maximum patient age
- `min_year`: earliest year to include in analysis
- `patient_wise_threshold`: completeness threshold for patient-wise filtering
- `exam_wise_threshold`: completeness threshold for exam-wise filtering
- `drop_order`: order of completeness filtering (`patients_first` or `features_first`)
- `n_folds`: number of CV folds
- `n_repeats`: number of repeats for repeated stratified CV
- `train_size`: train/test split ratio
- `sig_p_threshold`: significance threshold used in feature selection/plots

## Prerequisites

Before running the pipeline, ensure the following are available:

- Azure CLI installed and authenticated
- Access to the target Azure ML workspace
- Correct Azure ML subscription, resource group, and workspace information in `pipeline.sh`
- Python dependencies available in the Azure ML environment used by the components

## How to run

### 1) Authenticate to Azure

```bash
az login
az account set --subscription "<your-subscription-id>"
```

### 2) Update workspace configuration

Edit `pipeline.sh` and ensure these values match your Azure ML environment:

```bash
SUBSCRIPTION_ID="..."
RESOURCE_GROUP="..."
WORKSPACE_NAME="..."
```

### 3) Launch the pipeline

```bash
bash pipeline.sh
```

This invokes:

```bash
az ml job create \
  --subscription "$SUBSCRIPTION_ID" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME" \
  --file "$YAML_FILE" \
  --stream
```

## Expected outputs

The pipeline writes outputs to the Azure ML job outputs defined in `benchmarking_pipeline.yml`:

- `benchmarking_results`
- `statistics`
- `attrition_diagnostics`
- `idaa1c_availability_comparison`

These outputs generally include:

- benchmark score files
- model comparison plots
- feature selection summaries
- statistical summaries
- attrition diagnostics
- IDAA1c availability comparisons

## Notes

- The project uses Azure ML job definitions and command components rather than a standard local Python script pipeline.
- Some files in the repo are generated or environment-specific and are intentionally ignored by `.gitignore`.
- For production use, validate the input data paths and Azure workspace configuration before running a full benchmark.

## License

This project is intended for internal research and benchmarking workflows. Please check your organization’s data and publication policies before sharing or publishing outputs.
