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



# Service Fees Guide for All Visas

This guide explains how to access service fees for all visa types through the API endpoints available to the frontend.

## API Endpoints

### 1. Get All Services (with service fees)
**Endpoint:** `GET /api/v1/services`

**Response:** Returns an array of all services with their basic information including service fees.

**Example Response:**
```json
{
  "success": true,
  "data": [
    {
      "_id": "...",
      "slug": "adjustment-of-status",
      "title": "Adjustment of Status",
      "subtitle": "...",
      "description": "...",
      "price": {
        "serviceFee": 100000,
        "other": "USCIS Filing Fee"
      },
      "forms": ["I-130", "I-130A", "I-485", "I-864", "I-765", "I-131"],
      "icon": "repeat",
      "category": "Family"
    },
    ...
  ]
}
```

### 2. Get Single Service Details
**Endpoint:** `GET /api/v1/services/:slug`

**Response:** Returns detailed information about a specific service, including steps and prerequisites.

## Service Fees by Visa Type

### Static Service Fees (Fixed Price)

These visas have a fixed service fee that doesn't change based on user responses:

| Visa Type | Slug | Service Fee | Notes |
|-----------|------|-------------|-------|
| **Adjustment of Status (AOS)** | `adjustment-of-status` | $1,000.00 (100000 cents) | Fixed fee |
| **Citizenship / Naturalization** | `citizenship-naturalization` | $500.00 (50000 cents) | Fixed fee |
| **Fiancé Visa (K-1)** | `fiance-visa` | $500.00 (50000 cents) | Fixed fee |
| **Family Petition (I-130)** | `family-petition` | $500.00 (50000 cents) | Fixed fee |
| **Green Card Renewal / Replacement** | `green-card-renewal-replacement` | $500.00 (50000 cents) | Fixed fee |
| **Removal of Conditions (I-751)** | `removal-of-conditions` | $500.00 (50000 cents) | Fixed fee |
| **TN Visa** | `tn` | $500.00 (50000 cents) | Fixed fee |

### Dynamic Service Fees (Price Varies Based on Responses)

These visas have base fees that can increase based on dependents or additional services:

#### E-1 Visa
- **Slug:** `e1-visa`
- **Base Fee:** $2,000.00 (200000 cents) for principal applicant
- **Dynamic Pricing:** 
  - Dependent fees are calculated dynamically
  - Use pricing calculator: `src/utilities/pricing/e1VisaInitialPricing.ts`
- **Additional Fees:** Varies by filing route and dependents

#### E-1 Visa Renewal
- **Slug:** `e1-visa-renewal`
- **Base Fee:** $2,000.00 (200000 cents) for principal applicant
- **Dynamic Pricing:**
  - Dependent fees: $1,000 per dependent
  - Use pricing calculator: `src/utilities/pricing/e1VisaRenewalPricing.ts`

#### E-2 Visa
- **Slug:** `e2-visa`
- **Base Fee:** $2,000.00 (200000 cents) for principal applicant
- **Dynamic Pricing:**
  - Spouse fee: $1,000.00 (if spouse is applying for E-2)
  - Child fee: $1,000.00 per child
  - Spouse EAD fee: $250.00 (if spouse wants work authorization)
  - Use pricing calculator: `src/utilities/pricing/e2VisaPricing.ts`
  - Function: `calculateE2ServiceFee(input)`

**E-2 Pricing Example:**
- Principal only: $2,000
- Principal + Spouse: $3,000
- Principal + Spouse + EAD: $3,250
- Principal + Spouse + 2 Children: $5,000

#### E-2 Visa Renewal
- **Slug:** `e2-visa-renewal`
- **Base Fee:** $2,000.00 (200000 cents) for principal renewal
- **Dynamic Pricing:** Similar structure to E-2 Initial
  - Use pricing calculator: `src/utilities/pricing/e2VisaRenewalPricing.ts`

#### E-3 Visa
- **Slug:** `e3-visa`
- **Base Fee:** $600.00 (60000 cents) for principal applicant
- **Dynamic Pricing:**
  - Dependent fee: $600.00 per dependent (spouse + children)
  - EAD fee: $250.00 (if spouse wants work authorization)
  - Use pricing calculator: `src/utilities/pricing/e3VisaPricing.ts`
  - Function: `calculateE3ServiceFee(input)`

**E-3 Pricing Example:**
- Principal only: $600
- Principal + 1 dependent: $1,200
- Principal + Spouse + 2 Children: $2,400

#### E-3 Visa Renewal
- **Slug:** `e3-visa-renewal`
- **Base Fee:** $600.00 (60000 cents) for principal renewal
- **Dynamic Pricing:**
  - Each dependent renewal: $600 per dependent
  - Total = ($600 × number of people)
  - Use pricing calculator: `src/utilities/pricing/e3VisaRenewalPricing.ts`

**E-3 Renewal Pricing Example:**
- Principal only: $600
- Principal + Spouse: $1,200
- Principal + Spouse + 2 Children: $2,400

#### F-1 Student Visa
- **Slug:** `f1-student-visa`
- **Base Fee:** $600.00 (60000 cents) for principal applicant
- **Dynamic Pricing:**
  - F-2 Dependent fee: $600.00 per dependent (spouse + children under 21)
  - Use pricing calculator: `src/utilities/pricing/f1VisaPricing.ts`
  - Function: `calculateF1ServiceFee(responses)` or `getF1TotalServiceFee(responses)`

**F-1 Pricing Example:**
- Principal only: $600
- Principal + 1 dependent: $1,200
- Principal + 2 dependents: $1,800
- Principal + 3 dependents: $2,400

#### J-1 Exchange Visitor Program
- **Slug:** `j1-exchange-visitor`
- **Base Fee:** $600.00 (60000 cents) for principal applicant
- **Dynamic Pricing:**
  - J-2 Dependent fee: $600.00 per dependent (spouse + children under 21)
  - Use pricing calculator: `src/utilities/pricing/j1VisaPricing.ts`
  - Function: `calculateJ1ServiceFee(responses)` or `getJ1TotalServiceFee(responses)`

**J-1 Pricing Example:**
- Principal only: $600
- Principal + 1 dependent: $1,200
- Principal + 2 dependents: $1,800

#### M-1 Vocational Student Visa
- **Slug:** `m1-vocational-visa`
- **Base Fee:** $600.00 (60000 cents) for principal applicant
- **Dynamic Pricing:**
  - M-2 Dependent fee: $600.00 per dependent (spouse + children under 21)
  - Use pricing calculator: `src/utilities/pricing/m1VisaPricing.ts`
  - Function: `calculateM1ServiceFee(responses)` or `getM1TotalServiceFee(responses)`

**M-1 Pricing Example:**
- Principal only: $600
- Principal + 1 dependent: $1,200
- Principal + 2 dependents: $1,800

## Complete Service Fee Summary

### Family Visas
| Service | Slug | Base Fee | Dynamic? |
|---------|------|----------|----------|
| Adjustment of Status | `adjustment-of-status` | $1,000.00 | No |
| Citizenship / Naturalization | `citizenship-naturalization` | $500.00 | No |
| Fiancé Visa | `fiance-visa` | $500.00 | No |
| Family Petition | `family-petition` | $500.00 | No |
| Green Card Renewal/Replacement | `green-card-renewal-replacement` | $500.00 | No |
| Removal of Conditions | `removal-of-conditions` | $500.00 | No |

### Work Visas
| Service | Slug | Base Fee | Dynamic? |
|---------|------|----------|----------|
| E-1 Visa | `e1-visa` | $2,000.00 | Yes |
| E-1 Visa Renewal | `e1-visa-renewal` | $2,000.00 | Yes |
| E-2 Visa | `e2-visa` | $2,000.00 | Yes |
| E-2 Visa Renewal | `e2-visa-renewal` | $2,000.00 | Yes |
| E-3 Visa | `e3-visa` | $600.00 | Yes |
| E-3 Visa Renewal | `e3-visa-renewal` | $600.00 | Yes |
| TN Visa | `tn` | $500.00 | No |

### Student/Exchange Visas
| Service | Slug | Base Fee | Dynamic? |
|---------|------|----------|----------|
| F-1 Student Visa | `f1-student-visa` | $600.00 | Yes |
| J-1 Exchange Visitor | `j1-exchange-visitor` | $600.00 | Yes |
| M-1 Vocational Visa | `m1-vocational-visa` | $600.00 | Yes |

## How Service Fees Are Stored

Service fees are stored in **cents** in the database:
- $500.00 = 50000 cents
- $600.00 = 60000 cents
- $1,000.00 = 100000 cents
- $2,000.00 = 200000 cents

**Important:** When displaying to users, convert from cents to dollars by dividing by 100.

## Implementation Details for Frontend

### 1. Fetching All Service Fees

```javascript
// Fetch all services with their fees
const response = await fetch('/api/v1/services');
const { data: services } = await response.json();

// Extract service fees
const serviceFees = services.map(service => ({
  slug: service.slug,
  title: service.title,
  baseFee: service.price.serviceFee / 100, // Convert cents to dollars
  baseFeeCents: service.price.serviceFee,
  hasDynamicPricing: isDynamicPricingService(service.slug),
  otherFees: service.price.other
}));
```

### 2. Identifying Dynamic Pricing Services

```javascript
const dynamicPricingServices = [
  'e1-visa',
  'e1-visa-renewal',
  'e2-visa',
  'e2-visa-renewal',
  'e3-visa',
  'e3-visa-renewal',
  'f1-student-visa',
  'j1-exchange-visitor',
  'm1-vocational-visa'
];

function isDynamicPricingService(slug) {
  return dynamicPricingServices.includes(slug);
}
```

### 3. Calculating Dynamic Fees (Frontend)

For dynamic pricing services, you'll need to:
1. Collect questionnaire responses from the user
2. Call the appropriate pricing calculator function (if available as an API endpoint)
3. Or implement the pricing logic on the frontend based on the pricing calculators

**Note:** Currently, the pricing calculators exist only in the backend. You may need to:
- Create API endpoints that accept questionnaire responses and return calculated fees
- Or replicate the pricing logic on the frontend

### 4. Example: Displaying Service Fees

```javascript
function formatServiceFee(service) {
  const baseFee = service.price.serviceFee / 100;
  const currency = 'USD';
  
  return {
    base: `$${baseFee.toFixed(2)}`,
    baseCents: service.price.serviceFee,
    other: service.price.other,
    note: service.slug in dynamicPricingServices 
      ? 'Price may vary based on dependents or additional services'
      : 'Fixed price'
  };
}
```

## Service Fee Payment Process

1. **Service Selection:** User selects a service
2. **Prerequisite Check:** User completes prerequisite form (if required)
3. **Questionnaire:** User completes service-specific questionnaire
4. **Service Fee Calculation:** 
   - Static services: Use base fee from service object
   - Dynamic services: Calculate based on questionnaire responses
5. **Payment:** User pays the calculated service fee
6. **Processing:** Continue with document upload and filing fee steps

## API Response Structure

The service fee information is part of the `price` object in each service:

```typescript
{
  price: {
    serviceFee: number,  // in cents
    other: string        // description of other fees (e.g., "USCIS Filing Fee")
  }
}
```

## Notes

1. **All fees are in cents** - Remember to divide by 100 when displaying to users
2. **Dynamic pricing** - Some services require questionnaire responses to calculate final fees
3. **Additional fees** - The `other` field describes government filing fees that are separate from service fees
4. **Service fee vs. Filing fee** - Service fees are different from USCIS/government filing fees

## Quick Reference: All Visa Service Fees

```
Family Visas:
- AOS: $1,000 (fixed)
- Citizenship: $500 (fixed)
- Fiancé: $500 (fixed)
- Family Petition: $500 (fixed)
- Green Card Renewal: $500 (fixed)
- Removal of Conditions: $500 (fixed)

Work Visas:
- E-1: $2,000 base (dynamic)
- E-1 Renewal: $2,000 base (dynamic)
- E-2: $2,000 base (dynamic)
- E-2 Renewal: $2,000 base (dynamic)
- E-3: $600 base (dynamic)
- E-3 Renewal: $600 base (dynamic)
- TN: $500 (fixed)

Student Visas:
- F-1: $600 base (dynamic)
- J-1: $600 base (dynamic)
- M-1: $600 base (dynamic)
```


