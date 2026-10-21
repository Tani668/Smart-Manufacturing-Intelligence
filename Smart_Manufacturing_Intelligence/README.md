# Smart Manufacturing Intelligence System

An academic AI/ML prototype for:
1. Predictive maintenance using the UCI AI4I 2020 Predictive Maintenance dataset.
2. Image-based product anomaly inspection using the MVTec AD dataset (optional; dataset must be downloaded separately and used according to its license).
3. An interactive Streamlit dashboard for production KPIs, model evaluation, machine-risk predictions, and image inspection.

## Important limitations
- AI4I 2020 is a synthetic dataset, not live factory telemetry.
- The dashboard's production KPIs are calculated from the selected dataset records; they are not live plant measurements.
- The MVTec AD dataset is not bundled. Download it from its official source and check its license before use.
- The image inspection module uses a simple baseline feature model trained on the chosen MVTec category. It is an educational prototype, not a certified industrial inspection system.
- Model outputs are decision support only and must not be treated as proof of an actual machine fault.

## Setup (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train_failure_model
streamlit run app/streamlit_app.py
```

## Dataset
The training script downloads the UCI AI4I 2020 CSV from the UCI repository. Internet access is needed for the first run. If automatic download fails, download the dataset from:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
and place the CSV at `data/raw/ai4i2020.csv`.

## Optional computer-vision dataset
Download MVTec AD from:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad
Unzip it to `data/raw/mvtec_anomaly_detection/`. Choose a category (for example `bottle`) in the app. The expected structure includes:
`train/good`, `test/good`, `test/<defect_type>`, and `ground_truth/<defect_type>`.

## What is implemented
- Reproducible train/test split and stratified validation strategy for machine failure prediction
- Logistic Regression and Random Forest comparison
- Recall, precision, F1, ROC-AUC (when defined), confusion matrix, classification report
- Saved model, preprocessing pipeline, and evaluation metadata
- Streamlit pages for overview, dataset exploration, maintenance predictions, model evaluation, and optional image inspection
- Image inspection uses a transparent baseline: normal-image feature distribution from MVTec train/good and anomaly score by distance from the normal feature center. It requires enough normal images and is not a state-of-the-art detector.

## Report guidance
Use actual outputs from your run. Do not report example or placeholder metrics as real results. Document the synthetic nature of AI4I and the limitations of the image baseline.


## Verify after training
After the training command completes, run:
```bash
python -m src.verify_project
```
This checks that generated artifacts exist, the model loads, and a small prediction smoke test works.

## Important modeling note
The overall `Machine failure` target is not used as an input feature. The UCI AI4I failure-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifier columns are also excluded to avoid target leakage. Retrain after this code update; old model files and metrics are not valid for the corrected pipeline.
