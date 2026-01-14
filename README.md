# Fraud Detection System (Streamlit)

Detect fraudulent financial transactions with a production-style machine learning pipeline and an interactive Streamlit app.

## Business Problem
Financial institutions need to flag suspicious transactions quickly to prevent losses while minimizing false alarms. This project trains and compares multiple models to prioritize investigations with interpretable metrics and deployable code.

## Dataset
- Expected format: tabular CSV where each row is a transaction.
- Target column: `is_fraud` (1 = fraudulent, 0 = legitimate).
- Feature columns: numeric or categorical transaction attributes (amount, device, geography, time, etc.).
- Place raw data in `data/`. Example: `data/transactions.csv`.

## ML Pipeline
- Data loading and cleaning (duplicate removal, missing handling).
- Train/validation split with stratification.
- Feature scaling (`StandardScaler`) applied within pipelines.
- Class imbalance handling via `SMOTE` on the training set.
- Models trained and compared:
  - Logistic Regression
  - Random Forest
  - Isolation Forest (anomaly detection, trained on majority class)
- Metrics: F1-score and ROC-AUC.
- Best model persisted with `joblib` to `models/best_model.joblib`.

## Repository Structure
```
data/
notebooks/
src/
  preprocessing.py
  model.py
  evaluate.py
app.py
requirements.txt
```

## How to Run (Local)
1) Install deps (Python 3.10+):
```
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
2) Train models (expects `data/transactions.csv` with `is_fraud`):
```
python -m src.model --data_path data/transactions.csv --model_dir models
```
3) Evaluate (uses saved splits/metrics where applicable):
```
python -m src.evaluate --data_path data/transactions.csv --model_path models/best_model.joblib
``` 
4) Launch Streamlit app:
```
streamlit run app.py
```

## Deployment Notes
- Package the repo, install dependencies, and run `streamlit run app.py` on the target host.
- Ensure the saved model file (`models/best_model.joblib`) is present; retrain as data drifts.
- For containerization, copy the repo and run the same commands in the image.

## Model Comparison
The training script reports F1 and ROC-AUC for each model on the validation set and saves the top performer. Isolation Forest uses fraud scores inverted to align with fraud probability interpretation.

## Evaluation Outputs
- Confusion matrix and ROC curve plots produced during evaluation and within the Streamlit app when ground-truth labels are available in the uploaded CSV.
- Fraud probability scores and top high-risk transactions are highlighted in the UI.

## Production Considerations
- Retrain periodically to handle concept drift.
- Calibrate thresholds based on investigation capacity.
- Log predictions and decisions for auditing.
- Secure data handling (PII), and monitor model performance in production.
# Fraud Detection System (Streamlit)

Detect fraudulent financial transactions with a deployable, resume-ready machine learning pipeline and Streamlit app.

## Business problem
Financial platforms face losses and customer churn from fraudulent transactions. The goal is to flag risky transactions in (near) real time with high recall while keeping false positives low to avoid disrupting legitimate users.

## Dataset
- Expected format: tabular CSV with a binary target column `is_fraud` (1 = fraud, 0 = legit).
- Example feature columns: `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `transaction_type`.
- Place your raw data in `data/`. Update the `--target` flag if your label column differs.

## Project structure
```
data/
notebooks/
src/
  preprocessing.py
  model.py
  evaluate.py
app.py
requirements.txt
```

## ML pipeline
1. Load + clean: remove duplicates, drop rows missing the target.
2. Split: stratified train/validation split.
3. Scale: `StandardScaler` for numeric stability.
4. Balance: SMOTE applied on the training fold only.
5. Models compared:
   - Logistic Regression (L2, balanced class weights)
   - Random Forest
   - Isolation Forest (unsupervised anomaly detection)
6. Metrics: F1-score and ROC AUC; pick the best by F1 then AUC.
7. Persist: save best model, scaler, and feature columns with `joblib`.

## Quickstart
1) Install dependencies (Python 3.10+):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2) Train and save a model (replace with your data path):
```
python -m src.model --data data/transactions.csv --target is_fraud --model-out data/best_model.joblib --metrics-out data/metrics.json
```

3) Run the Streamlit app:
```
streamlit run app.py
```
If `data/best_model.joblib` is missing, the app will prompt you to train first.

## Streamlit app
- Upload CSV with the same feature schema used in training.
- The app loads the persisted model and scaler, outputs per-transaction fraud probabilities, highlights high-risk rows, and plots confusion matrix + ROC curve using stored evaluation artifacts.

## Model comparison & evaluation
- Metrics are printed during training and stored in `metrics_out` (default `data/metrics.json`).
- Isolation Forest is evaluated by mapping anomaly scores to fraud probabilities.
- Confusion matrix and ROC data are available for visualization in the app.

## Deployment notes
- Package the project as a Streamlit app (suitable for Streamlit Cloud, Azure App Service, or containerized).
- Persist artifacts (`.joblib`, `.json`) alongside the app or in object storage.
- Monitor drift by periodically re-training with fresh data and re-validating metrics.

## Notebooks
Use `notebooks/` for EDA or experimentation. The production code lives in `src/`.

## License
MIT

