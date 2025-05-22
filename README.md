# MLOps Demo – Wine Quality Prediction

This demo project implements a fully serverless MLOps pipeline using **AWS Lambda**, **Amazon SageMaker**, **Amazon S3**, and **AWS Step Functions** to automate the complete lifecycle of a machine learning model built with **XGBoost**. The model predicts wine quality based on physicochemical characteristics of red wine samples.

Although this project wasn't deployed to production, it was designed to demonstrate real-world MLOps best practices through a modular, scalable, and event-driven architecture.

## 🍷 About the Dataset

The dataset used is the public **Wine Quality** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality), which contains records of red wine samples with the following characteristics:

- Input features: `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, etc.
- Target: `quality` (score from 0 to 10)

The goal is to train a and deploy a model that can accurately predict wine quality based on these features.


## 🧱 Architecture & S3 Organization

All input data, output artifacts, metadata, and configuration files are organized under an S3 bucket with the following structure:

/wine-quality-project/
- wine_quality.csv # Original dataset
- config_file/ # Configs for training, processing, HPO
- preprocessed_data/ # Cleaned and split data
- pipeline-metadata/ # Metadata for training, processing, batch, HPO
- predictions/ # Batch inference outputs
- hpo_jobs/, training-jobs/ # Raw job artifacts
- lambda_functions/ # (Optional) Lambda ZIP packages or code
- requirements/, code/ # Dependencies and processing scripts


## 🧠 AWS Lambda Functions 

### 🚀 Job Launchers
- `launch-processing-job.py`: prepares data for training and HPO by running a processing job.
- `launch-hpo.py`: launches a hyperparameter tuning job using SageMaker HPO.
- `launch-training-job.py`: trains a model using either tuned or default hyperparameters.
- `launch-batch-job.py`: performs batch inference on the most recent dataset using the best available model.

### 🔍 Job Status Checkers
- `get-processing-job-status.py`
- `get-hpo-job-status.py`
- `get-training-job-status.py`
- `get-batch-job-status.py`

### 📦 Metadata Management
- `save-processing-job-metadata.py`: Stores info about datasets and processing job runtime.
- `save-hpo-metadata.py`: Saves metadata for the best HPO trial.
- `save-training-metadata.py`: Captures training metrics and model details.
- `save-batch-metadata.py`: Logs metadata for batch inference jobs.

### 📊 Dataset Analysis
- `dataset-metadata.py`: Reads the raw dataset from S3 and extracts statistics, schema, missing values, and data quality insights.

### 🏆 Model Versioning
- `update-best-model-package.py`: Compares the latest trained model with the current production model and updates the approval status in SageMaker Model Registry based on evaluation metrics.

## 🔄 Workflow Orchestration (Step Functions)

To automate and coordinate the different stages of the pipeline, this project uses **AWS Step Functions**.

Each stage of the MLOps process — from data preparation to model inference — is represented as an individual **state machine**. These workflows manage the execution of the Lambda functions in the correct order, handling errors, retries, and state tracking automatically.

The orchestrated pipelines include:

- **Preprocessing pipeline**: Loads the raw dataset, splits it for training and tuning, and saves metadata to S3.
- **HPO pipeline**: Runs a hyperparameter tuning job in SageMaker, tracks its progress, and saves the best configuration.
- **Training pipeline**: Trains the model using the best (or default) hyperparameters, logs metrics, and registers the model.
- **Batch inference pipeline**: Runs predictions on new data using the latest approved model and stores the results.

These state machines make the pipeline easy to manage, scalable, and fully serverless, allowing each stage to run automatically and reliably. All of them were coded and deployed using **python scripts inside a Jupyter Notebook**, using the AWS SDK (`boto3`) to build, configure, and trigger Step Functions directly. This approach allows flexible, code-driven orchestration within a familiar notebook environment.
