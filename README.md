# Fraud Detection (Tabular ML) — End-to-End Pipeline

This repository implements a **production-oriented fraud detection pipeline** for tabular data. It covers **data ingestion, leakage-safe feature engineering, model training (LightGBM/XGBoost/Logistic Regression baselines), evaluation, calibration, explainability (SHAP),** and **threshold selection aligned to business costs**.

> ⚠️ Note: Replace file paths and environment variables with your own. Metrics below are placeholders until you run on your data.

---

## Contents
- [Project Goals](#project-goals)
- [Data Schema](#data-schema)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Reproducibility](#reproducibility)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Results (Placeholders)](#results-placeholders)
- [Production Notes](#production-notes)
- [License](#license)

---

## Project Goals
1. Build **leakage-safe** features using **time-aware rolling windows** (per user/device/merchant/IP).
2. Train performant but interpretable models (LogReg → LightGBM/XGBoost) with **class-imbalance** handling.
3. Optimize for **recall at low FPR** and **PR-AUC**, and **calibrate** probabilities for risk scoring.
4. Provide **explainability** (global & local) and a **threshold playbook** tied to business cost/benefit.

---

## Data Schema
Assumes a transactions table with the following columns (adapt as needed):
```
transaction_id: str
event_time: datetime
user_id: str
card_id: str (optional)
merchant_id: str
amount: float
currency: str
country: str
device_id: str
ip_address: str
lat, lon: float (optional)
mcc: str/int (merchant category code; optional)
label: int  (1 = fraud, 0 = legit)  [for supervised training]
```
**Important:** If you don’t have geo/MCC fields, the feature builder will skip those blocks.

---

## Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel

pip install -r requirements.txt
# Minimal requirements (if you don’t have a file):
pip install numpy pandas scikit-learn lightgbm xgboost pyarrow polars==0.20.16 \
            matplotlib shap optuna pydantic pyyaml tqdm joblib
```
Optional:
```bash
# for notebooks
pip install jupyterlab ipykernel
python -m ipykernel install --user --name fraud-venv
```

---

## Project Structure
```
.
├─ data/
│  ├─ raw/                 # input CSV/Parquet
│  ├─ interim/             # intermediate features
│  └─ processed/           # model-ready datasets
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_feature_checks.ipynb
│  └─ 03_threshold_analysis.ipynb
├─ src/
│  ├─ config.py            # load YAML/ENV configs
│  ├─ schema.py            # data schema & validators
│  ├─ io_utils.py          # readers/writers
│  ├─ time_splits.py       # temporal split helpers
│  ├─ features/
│  │  ├─ base.py
│  │  ├─ user_features.py
│  │  ├─ merchant_features.py
│  │  ├─ device_ip_features.py
│  │  ├─ geo_features.py
│  │  └─ categorical_encoders.py
│  ├─ models/
│  │  ├─ baselines.py      # logistic regression, random forest
│  │  ├─ xgb_model.py
│  │  └─ lgbm_model.py
│  ├─ eval/
│  │  ├─ metrics.py        # PR-AUC, recall@FPR, cost curves
│  │  ├─ calibration.py    # isotonic / platt scaling
│  │  └─ explanations.py   # SHAP
│  ├─ train.py             # end-to-end training entrypoint
│  ├─ predict.py           # batch inference
│  └─ thresholding.py      # operating point selection
├─ configs/
│  └─ default.yaml
├─ requirements.txt
└─ README.md
```

---

## Feature Engineering
**Leakage guardrails**: All features are computed with **time-aware rolling windows** so each row only “sees” past data up to `event_time`. Target-encoding uses **out-of-fold** logic.

### Feature Groups & Rationale
1. **Behavioral Velocity (bursts & spikes)**
   - Rolling counts/sums/means per `user_id`, `device_id`, `merchant_id`, `ip_address` over windows (1h, 24h, 7d, 30d).
   - Rationale: Fraud often manifests as sudden bursts; velocity captures abnormal frequency/amount patterns.

2. **Profile Baselines & Ratios**
   - Per-user and per-merchant `amount_p50/p75/p95/p99`, std, iqr; ratios like `amount / user_p95`.
   - Rationale: Outlier detection relative to an entity’s own history is stronger than global thresholds.

3. **Recency & Periodicity**
   - Time since last transaction for user/device/merchant; hour-of-day, day-of-week, weekend/holiday flags.
   - Rationale: Abnormal recency, odd hours, and atypical periodicity are predictive signals.

4. **Geo Signals**
   - Haversine distance to last known user or merchant coordinates; cross-border or country changes.
   - Rationale: Sudden geo jumps and location mismatches are classic risk signals.

5. **Network/Graph Proxies**
   - Degrees and uniqueness: `unique_devices_per_user`, `unique_users_per_device`, `unique_ips_per_user`.
   - Rationale: Shared devices/IPs and high-degree nodes often indicate mule networks.

6. **Categorical Encodings (Leakage-Safe)**
   - K-fold **target encoding** with smoothing for MCC, email domain, BIN, country, merchant_id (if sufficient support).
   - Frequency encodings + rare-bucket grouping.
   - Rationale: Encodes historical risk while preventing lookahead leakage.

7. **Risk Lists / Heuristics (optional)**
   - Disposable email domains, datacenter IP ranges, prepaid BIN flags.
   - Rationale: Domain knowledge features complement learned signals.

**Implementation notes**
- Heavy aggregations use **Polars**/**Pandas** with `groupby_rolling`/cumulative windows + checkpointing to Parquet.
- Feature schemas are versioned; online/offline parity documented in `/src/features/base.py`.

---

## Modeling
- **Baselines**: Logistic Regression (L2), Random Forest.
- **Gradient Boosting**: LightGBM / XGBoost with `scale_pos_weight` for imbalance.
- **Hyperparameters**: `RandomizedSearchCV` or `Optuna` + early stopping.
- **Class Imbalance**: Prefer weighting over oversampling; oversampling only in **training folds** if needed.
- **Temporal CV**: Rolling **time-based** folds; optional **GroupKFold(user_id)** to reduce identity leakage.

**Objective & Constraints**
- Primary: **PR-AUC**, **Recall@FPR<=X%** (configure `fpr_budget`).
- Secondary: **ROC-AUC**, **Brier score** post-calibration.
- Optional monotonic constraints (where domain knowledge applies).

---

## Evaluation
- Curves: **PR** and **ROC**.
- Operating points: pick thresholds by **business cost curves** (false positive cost vs. fraud loss prevented).
- Stability: report metrics by **time-slice** and by **segment** (country, MCC, device type).
- Robustness: KS drift tests for top features between train and recent production windows.

---

## Explainability
- **Global importance**: gain-based + SHAP summary.
- **Local explanations**: SHAP force/decision plots for sampled predictions.
- Export artifacts to `artifacts/`:
```
artifacts/
  shap_summary.png
  shap_values.parquet
  pr_curve.png
  roc_curve.png
  threshold_report.csv
  model.pkl
  encoders.pkl
```

---

## Reproducibility
- **Config-driven**: all paths, features, windows, and model params live in `configs/default.yaml`.
- **Seeds**: set per library (numpy, lightgbm, xgboost).
- **Deterministic**: avoid time-of-run nondeterminism in grouping/joins by sorting + stable keys.

---

## Running the Pipeline
**1) Prepare data**
```bash
python -m src.train \
  --config configs/default.yaml \
  --stage prepare
```

**2) Build features**
```bash
python -m src.train \
  --config configs/default.yaml \
  --stage features
```

**3) Train & evaluate**
```bash
python -m src.train \
  --config configs/default.yaml \
  --stage train,evaluate
```

**4) Explainability & thresholding**
```bash
python -m src.train \
  --config configs/default.yaml \
  --stage explain,threshold
```

**5) Batch inference**
```bash
python -m src.predict \
  --config configs/default.yaml \
  --input data/processed/scoring.parquet \
  --output data/processed/scores.parquet
```

---

## Configuration
Example `configs/default.yaml`:
```yaml
data:
  raw_path: data/raw/transactions.parquet
  workdir: data
splits:
  strategy: temporal
  n_folds: 5
  group_key: user_id  # optional
features:
  windows: [ '1h', '24h', '7d', '30d' ]
  enable_geo: true
  enable_network: true
  target_encoding:
    k_folds: 5
    min_samples: 200
model:
  type: lightgbm
  params:
    num_leaves: 64
    learning_rate: 0.05
    n_estimators: 2000
    subsample: 0.8
    colsample_bytree: 0.8
    reg_alpha: 1.0
    reg_lambda: 2.0
    scale_pos_weight: 20   # adjust to class imbalance
evaluation:
  fpr_budget: 0.005      # 0.5%
  metrics: [pr_auc, roc_auc, recall_at_fpr]
explain:
  shap_sample_size: 20000
```

---

## Results (Placeholders)
- **PR-AUC**: _TBD after training on your data_
- **Recall @ FPR 0.5%**: _TBD_
- **Calibrated Brier score**: _TBD_

Add a screenshot of PR/ROC curves in `artifacts/` after your first run.

---

## Production Notes
- Promote features to an **online feature store** (e.g., Redis/Feast) with identical definitions.
- Serve model via **FastAPI**; enforce **schema validation** on inputs.
- Monitor: **prediction drift**, **label lag**, and **segment metrics**; alert on threshold breaches.
- Retraining: time-based retrains (weekly/monthly) with drift-aware triggers.
- Security: PII hashing/salting; access controls on artifacts and logs.

---

## License
MIT (or your license of choice).
