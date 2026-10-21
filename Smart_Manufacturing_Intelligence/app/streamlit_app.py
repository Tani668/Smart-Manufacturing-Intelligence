from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "ai4i_cleaned.csv"
MODEL_PATH = ROOT / "models" / "failure_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "failure_model_metrics.json"
MVTEC_ROOT = ROOT / "data" / "raw" / "mvtec_anomaly_detection"

st.set_page_config(
    page_title="Smart Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏭 Smart Manufacturing Intelligence System")
st.caption("Academic prototype | Predictive maintenance + image-based quality inspection")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page",
        ["Overview", "Dataset Explorer", "Predictive Maintenance", "Model Evaluation", "Image Inspection", "About & Limitations"],
    )
    st.divider()
    st.caption("Dataset: UCI AI4I 2020 (synthetic). Image inspection is optional and requires MVTec AD locally.")


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


df = load_dataset()
model = load_model()
metrics = load_metrics()


def require_training():
    if df is None or model is None or metrics is None:
        st.warning("The failure model has not been trained yet.")
        st.code("python -m src.train_failure_model", language="bash")
        st.info("Run this command from the project root, then refresh the app.")
        return False
    return True


if page == "Overview":
    st.subheader("Manufacturing overview")
    if df is None:
        st.warning("No processed dataset found. Train the model first.")
        st.code("python -m src.train_failure_model", language="bash")
    else:
        total = len(df)
        failures = int(df["Machine failure"].sum()) if "Machine failure" in df else 0
        failure_rate = failures / total * 100 if total else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Records analyzed", f"{total:,}")
        col2.metric("Failure-labelled records", f"{failures:,}")
        col3.metric("Labelled failure rate", f"{failure_rate:.2f}%")
        col4.metric("Model status", "Trained" if model is not None else "Not trained")

        st.markdown("### What this means")
        st.write(
            "The figures above summarize labelled records in the selected public dataset. "
            "They are not live factory telemetry or proof of current machine condition."
        )
        if "Type" in df:
            counts = df["Type"].value_counts().rename_axis("Product type").reset_index(name="Records")
            st.plotly_chart(px.bar(counts, x="Product type", y="Records", title="Records by product type"), use_container_width=True)
        if "Machine failure" in df:
            fail_counts = df["Machine failure"].value_counts().rename(index={0: "No failure label", 1: "Failure label"}).rename_axis("Label").reset_index(name="Records")
            st.plotly_chart(px.bar(fail_counts, x="Label", y="Records", title="Target class distribution"), use_container_width=True)

        if metrics:
            st.markdown("### Held-out model comparison")
            st.dataframe(pd.DataFrame(metrics.get("results", [])), use_container_width=True, hide_index=True)


elif page == "Dataset Explorer":
    st.subheader("Dataset understanding and preprocessing")
    if df is None:
        st.warning("Train the model to download and prepare the dataset.")
        st.code("python -m src.train_failure_model", language="bash")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        search = st.text_input("Filter columns by name")
        cols = [c for c in df.columns if search.lower() in c.lower()]
        st.dataframe(df[cols].head(100), use_container_width=True)
        st.markdown("### Data types")
        types_df = pd.DataFrame({"Column": df.columns, "Data type": [str(x) for x in df.dtypes], "Missing values": [int(x) for x in df.isna().sum()]})
        st.dataframe(types_df, use_container_width=True, hide_index=True)
        st.markdown("### Numeric summary")
        st.dataframe(df.select_dtypes(include=np.number).describe().T, use_container_width=True)
        st.download_button("Download cleaned CSV", df.to_csv(index=False).encode("utf-8"), "ai4i_cleaned.csv", "text/csv")


elif page == "Predictive Maintenance":
    st.subheader("Machine failure-risk prediction")
    st.write("Enter operating measurements to obtain a model prediction. The result is an estimate based on the training data, not a diagnosis.")
    if not require_training():
        st.stop()

    feature_cols = metrics["feature_columns"]
    row = {}
    numeric_defaults = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
    for feature in feature_cols:
        if feature == "Type":
            row[feature] = st.selectbox("Product type", ["L", "M", "H"], index=1)
        elif feature in numeric_defaults:
            row[feature] = st.number_input(feature, value=float(numeric_defaults[feature]))
        else:
            if df is not None and feature in df:
                series = pd.to_numeric(df[feature], errors="coerce").dropna()
                default = float(series.median()) if not series.empty else 0.0
            else:
                default = 0.0
            row[feature] = st.number_input(feature, value=default)

    if st.button("Estimate failure risk", type="primary"):
        input_df = pd.DataFrame([row], columns=feature_cols)
        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])
        if prediction == 1:
            st.error("The model predicts the failure class for these inputs.")
        else:
            st.success("The model predicts the no-failure class for these inputs.")
        if probability is not None:
            st.metric("Estimated failure probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
        st.caption("This is a model estimate, not a calibrated guarantee or a substitute for maintenance procedures.")


elif page == "Model Evaluation":
    st.subheader("Model evaluation and comparison")
    if not require_training():
        st.stop()
    results = pd.DataFrame(metrics.get("results", []))
    st.markdown("### Comparison on the held-out test set")
    st.dataframe(results, use_container_width=True, hide_index=True)
    if not results.empty:
        metric_options = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in results.columns]
        selected_metric = st.selectbox("Metric to compare", metric_options, index=metric_options.index("f1") if "f1" in metric_options else 0)
        st.plotly_chart(px.bar(results, x="model", y=selected_metric, title=f"Model comparison: {selected_metric}", range_y=[0, 1]), use_container_width=True)

    cm = metrics.get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        st.markdown(f"### Confusion matrix — {metrics.get('best_model', 'best model')}")
        st.dataframe(cm_df, use_container_width=True)
        st.write("Rows are actual classes; columns are predicted classes. Class 1 denotes a machine-failure label.")
    st.info(
        "Evaluation note: failure-mode columns (TWF, HDF, PWF, OSF, RNF) and identifier columns "
        "are excluded from model inputs to reduce target leakage. AI4I 2020 is synthetic; results "
        "do not establish performance on live factory equipment."
    )
    report = metrics.get("classification_report")
    if report:
        st.markdown("### Classification report")
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    st.download_button("Download metrics JSON", json.dumps(metrics, indent=2), "failure_model_metrics.json", "application/json")


elif page == "Image Inspection":
    st.subheader("Computer vision: product image inspection")
    st.write(
        "Optional educational baseline for MVTec AD. It compares image features with features from normal training images. "
        "This is not a production-grade defect detector."
    )
    if not MVTEC_ROOT.exists():
        st.info("MVTec AD was not found. Download it from the official MVTec AD page and extract it to:")
        st.code(str(MVTEC_ROOT))
        st.markdown("[Official MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)")
        st.stop()

    categories = sorted([p.name for p in MVTEC_ROOT.iterdir() if p.is_dir()])
    if not categories:
        st.warning("No category folders were found. Check the extracted dataset path.")
        st.stop()
    category = st.selectbox("Dataset category", categories)
    train_good = MVTEC_ROOT / category / "train" / "good"
    image_files = []
    if train_good.exists():
        image_files = [p for p in train_good.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    if len(image_files) < 5:
        st.warning(f"Expected normal training images in {train_good}; not enough images were found.")
        st.stop()

    def feature_vector(pil_img):
        # Simple, reproducible baseline feature vector: resized RGB pixels + grayscale summary.
        image = pil_img.convert("RGB").resize((32, 32))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return np.concatenate([arr.reshape(-1), [gray.mean(), gray.std()]])

    with st.spinner("Building normal reference from up to 300 training images..."):
        vectors = []
        for path in image_files[:300]:
            try:
                vectors.append(feature_vector(Image.open(path)))
            except Exception:
                continue
        if len(vectors) < 5:
            st.error("Could not read enough normal training images.")
            st.stop()
        normal_vectors = np.vstack(vectors)
        center = normal_vectors.mean(axis=0)
        scale = normal_vectors.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normal_distances = np.sqrt((((normal_vectors - center) / scale) ** 2).mean(axis=1))
        threshold = float(np.quantile(normal_distances, 0.99))

    uploaded = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded inspection image", use_container_width=True)
        vector = feature_vector(image)
        distance = float(np.sqrt((((vector - center) / scale) ** 2).mean()))
        st.metric("Anomaly score (relative)", f"{distance:.3f}")
        if distance > threshold:
            st.warning("This image differs from the normal training-image reference and should be reviewed.")
        else:
            st.success("This image is within the baseline normal-reference threshold.")
        st.caption(f"Threshold estimated from normal training images: {threshold:.3f}. This simple pixel baseline can be sensitive to lighting, alignment, and product variation.")


else:
    st.subheader("About this project")
    st.markdown("""
### Project goals
- Monitor manufacturing records and summarize production-related indicators.
- Estimate machine-failure risk from operating measurements.
- Provide an optional image anomaly-inspection demonstration.
- Present metrics and insights through an interactive dashboard.

### SDG alignment
- **SDG 9 — Industry, Innovation and Infrastructure:** demonstrates data-driven monitoring and predictive maintenance concepts.
- **SDG 12 — Responsible Consumption and Production:** earlier identification of defects may support waste reduction, subject to real-world validation.

### Limitations
- AI4I 2020 is synthetic and does not establish real-world factory performance.
- The app does not connect to live industrial sensors or PLC/SCADA systems.
- MVTec image inspection is an educational baseline and may not generalize across products, lighting, cameras, or factory conditions.
- No metric should be claimed until the training pipeline has been run successfully.
- Predictions should be reviewed by qualified personnel before operational decisions.
""")
