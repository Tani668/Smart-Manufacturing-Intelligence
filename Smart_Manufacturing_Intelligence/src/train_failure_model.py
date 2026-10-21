"""Train and evaluate machine-failure prediction baselines on UCI AI4I 2020."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
CSV_NAME = "ai4i2020.csv"
TARGET = "Machine failure"
DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


def locate_or_download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    expected = RAW_DIR / CSV_NAME
    if expected.exists():
        return expected

    # UCI provides a ZIP archive. Download and extract its CSV without relying on
    # a particular internal filename.
    zip_path = RAW_DIR / "ai4i2020.zip"
    try:
        urlretrieve(DATA_URL, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path) as archive:
            csv_members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not csv_members:
                raise RuntimeError("The downloaded archive did not contain a CSV file.")
            with archive.open(csv_members[0]) as src, open(expected, "wb") as dst:
                dst.write(src.read())
        return expected
    except Exception as exc:
        raise RuntimeError(
            "Could not automatically download AI4I. Download the dataset from "
            "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset "
            f"and save its CSV as {expected}. Original error: {exc}"
        ) from exc


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize whitespace in column names.
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        # Some mirrors use slightly different capitalization.
        matches = [c for c in df.columns if c.strip().lower() == TARGET.lower()]
        if not matches:
            raise ValueError(f"Target '{TARGET}' not found. Columns: {list(df.columns)}")
        df = df.rename(columns={matches[0]: TARGET})
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")


def evaluate_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    row = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    try:
        proba = model.predict_proba(X_test)[:, 1]
        row["roc_auc"] = float(roc_auc_score(y_test, proba))
    except Exception:
        row["roc_auc"] = None
    return row, pred


def main():
    for folder in (PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    csv_path = locate_or_download_dataset()
    df = load_data(csv_path)
    df.to_csv(PROCESSED_DIR / "ai4i_cleaned.csv", index=False)

    # Avoid using identifier columns as predictive features.
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    # Drop identifiers and failure-mode labels: these can reveal the overall target
    # and cause target leakage during evaluation.
    X = df.drop(columns=[TARGET] + drop_cols)
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("The target contains fewer than two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1, min_samples_leaf=2
            )),
        ]),
    }

    results = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        row, pred = evaluate_model(name, model, X_test, y_test)
        results.append(row)
        predictions[name] = pred

    results_df = pd.DataFrame(results).sort_values(
        ["roc_auc", "f1"], ascending=[False, False], na_position="last"
    )
    best_name = str(results_df.iloc[0]["model"])
    best_model = models[best_name]
    best_pred = predictions[best_name]

    joblib.dump(best_model, MODEL_DIR / "failure_pipeline.joblib")
    metadata = {
        "target": TARGET,
        "feature_columns": list(X.columns),
        "dropped_identifier_and_leakage_columns": drop_cols,
        "best_model": best_name,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "results": results,
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "classification_report": classification_report(
            y_test, best_pred, output_dict=True, zero_division=0
        ),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
        "dataset_note": "AI4I 2020 is a synthetic predictive-maintenance dataset.",
    }
    with open(REPORT_DIR / "failure_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    results_df.to_csv(REPORT_DIR / "failure_model_comparison.csv", index=False)

    print("\nTraining completed.")
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(results_df.to_string(index=False))
    print(f"Best model saved to: {MODEL_DIR / 'failure_pipeline.joblib'}")
    print(f"Metrics saved to: {REPORT_DIR / 'failure_model_metrics.json'}")
    print("\nNote: these metrics describe this dataset and split, not guaranteed real-factory performance.")


if __name__ == "__main__":
    main()
