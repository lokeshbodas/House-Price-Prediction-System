# Prices Predictor System

A production-grade, end-to-end machine learning pipeline for predicting house prices using the Ames Housing dataset. This project leverages ZenML for pipeline orchestration, MLflow for experiment tracking, and scikit-learn for model building.

## Project Structure

```
.
├── analysis/                # Exploratory Data Analysis (EDA) notebooks and scripts
├── data/                    # Raw data files (e.g., AmesHousing.csv)
├── mlruns/                  # MLflow experiment tracking artifacts
├── pipelines/               # ZenML pipeline definitions
├── src/                     # Core source code (data ingestion, feature engineering, etc.)
├── steps/                   # ZenML pipeline steps
├── tests/                   # Unit and integration tests
├── config.yaml              # Project configuration
├── requirements.txt         # Python dependencies
├── run_deployment.py        # Script to run deployment pipeline
├── run_pipeline.py          # Script to run training pipeline
├── sample_predict.py        # Example prediction script
```

## Features

- **Data Ingestion:** Modular ingestion from ZIP files ([`src/ingest_data.py`](src/ingest_data.py))
- **Data Preprocessing:** Handling missing values, outlier detection, and feature engineering ([`src/handle_missing_values.py`](src/handle_missing_values.py), [`src/feature_engineering.py`](src/feature_engineering.py))
- **Model Building:** Linear Regression pipeline with preprocessing ([`steps/model_building_step.py`](steps/model_building_step.py))
- **Experiment Tracking:** Integrated with MLflow
- **Deployment:** Continuous deployment pipeline ([`pipelines/deployment_pipeline.py`](pipelines/deployment_pipeline.py))
- **EDA:** Comprehensive analysis in Jupyter ([`analysis/EDA.ipynb`](analysis/EDA.ipynb))

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/prices-predictor-system.git
cd prices-predictor-system
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the Training Pipeline

```sh
python run_pipeline.py
```

After running, follow the instructions in the output to launch the MLflow UI:

```sh
mlflow ui --backend-store-uri '<tracking-uri>'
```

### 4. Run the Deployment Pipeline

```sh
python run_deployment.py
```

### 5. Explore EDA

Open [`analysis/EDA.ipynb`](analysis/EDA.ipynb) in Jupyter Notebook to explore the data and insights.

## Customization

- **Feature Engineering:** Modify strategies in [`steps/feature_engineering_step.py`](steps/feature_engineering_step.py)
- **Modeling:** Adjust model or pipeline in [`steps/model_building_step.py`](steps/model_building_step.py)
- **Pipelines:** Update or add steps in [`pipelines/training_pipeline.py`](pipelines/training_pipeline.py)
