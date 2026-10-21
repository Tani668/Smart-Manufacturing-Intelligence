# Smart Manufacturing Intelligence System

An end-to-end machine learning application for predictive maintenance and manufacturing failure analysis, built using Python, Scikit-learn, and Streamlit.

## Overview

Unexpected machine failures can cause production downtime, increased maintenance costs, and operational inefficiencies. This project uses machine operating conditions to predict potential failures and presents model results through an interactive dashboard.

The system compares machine learning models, provides failure predictions from user-entered measurements, and visualizes dataset characteristics and model performance.

## Key Features

- Predictive maintenance using machine operating conditions
- Machine learning model comparison
- Interactive Streamlit dashboard
- Dataset exploration and visualization
- Model evaluation using multiple performance metrics
- Saved machine learning pipeline for predictions
- Optional image-based anomaly inspection baseline

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset, containing 10,000 records of simulated industrial machine operating conditions.

The predictive features include:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The target variable is machine failure.

Identifier columns and failure-mode labels are excluded from the predictive features to reduce target leakage.

## Machine Learning Methodology

1. Load and explore the dataset.
2. Separate predictive features and the target variable.
3. Preprocess numerical and categorical features.
4. Split the data into training and testing sets.
5. Train Logistic Regression and Random Forest classifiers.
6. Evaluate both models using multiple metrics.
7. Save the trained pipeline and evaluation results.
8. Integrate the trained model into the Streamlit dashboard.

## Model Performance

The following results were obtained on the current held-out test split.

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| Accuracy | 97.40% | 82.45% |
| Precision | 58.89% | 14.18% |
| Recall | 77.94% | 82.35% |
| F1-score | 67.09% | 24.19% |
| ROC-AUC | 96.70% | 90.70% |

Random Forest achieved the strongest overall balance of precision, recall, and F1-score in this experiment.

These results are specific to the dataset and test split and do not guarantee performance on real industrial equipment.

## Project Structure

```text
Smart-Manufacturing-Intelligence/
├── app/
│   └── streamlit_app.py
├── src/
│   └── train_failure_model.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── failure_pipeline.joblib
├── reports/
│   ├── failure_model_metrics.json
│   └── model_comparison.csv
├── notebooks/
├── screenshots/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Smart-Manufacturing-Intelligence.git
cd Smart-Manufacturing-Intelligence
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

For macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the models

```bash
python -m src.train_failure_model
```

### 6. Run the dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal to access the dashboard.

## Limitations

- The primary dataset is synthetic and may not represent all real industrial conditions.
- Machine failures are relatively rare, so accuracy alone is insufficient for evaluating model performance.
- False negatives may result in missed failures, while false positives may lead to unnecessary inspections.
- The optional image inspection module is a baseline and is not a validated industrial defect-detection system.
- Further validation using real-world industrial data is necessary before operational deployment.

## Future Improvements

- Implement cross-validation and additional testing.
- Add confusion matrices and feature-importance visualizations.
- Improve failure detection through threshold tuning.
- Integrate real-time sensor data and maintenance history.
- Validate image anomaly detection using an industrial inspection dataset.

## Academic Project

An applied machine learning project focused on predictive maintenance, model evaluation, and interactive data visualization.

**Disclaimer:** This project is an educational prototype and is not intended to replace professional industrial safety or maintenance systems.
