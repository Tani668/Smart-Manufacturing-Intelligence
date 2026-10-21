"""Basic post-training checks for the Smart Manufacturing Intelligence project."""
from pathlib import Path
import json
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / "data" / "processed" / "ai4i_cleaned.csv",
    ROOT / "models" / "failure_pipeline.joblib",
    ROOT / "reports" / "failure_model_metrics.json",
    ROOT / "reports" / "failure_model_comparison.csv",
]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    print("Missing generated artifacts:")
    print("\n".join(f" - {p}" for p in missing))
    print("\nRun: python -m src.train_failure_model")
    sys.exit(1)

df = pd.read_csv(required[0])
model = joblib.load(required[1])
metrics = json.loads(required[2].read_text(encoding="utf-8"))

features = metrics["feature_columns"]
sample = df[features].head(5)
predictions = model.predict(sample)

print("PASS: processed dataset exists:", df.shape)
print("PASS: saved model loaded:", type(model).__name__)
print("PASS: prediction smoke test:", predictions.tolist())
print("PASS: comparison results:", metrics.get("results", []))
print("Excluded columns:", metrics.get("dropped_identifier_and_leakage_columns", []))
print("Reminder: inspect metrics and test the UI manually before submission.")
