# Developer Handbook: Agentic AI Credit Underwriting System

> **For Anyone Continuing Work on This Project**  
> This document explains every step of how this system was built — from raw Parquet files to a live full-stack web application with multi-model ML, explainable AI, and regulatory compliance. Read this start to finish before touching any code.

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Repository Structure — Every File Explained](#2-repository-structure--every-file-explained)
3. [Step 1: Raw Data & Analytical Database (`src/create_database.py`)](#3-step-1-raw-data--analytical-database)
4. [Step 2: Integrity & Temporal Auditing (`src/check_relationships.py`, `src/check_temporal_data.py`)](#4-step-2-integrity--temporal-auditing)
5. [Step 3: Feature Engineering Pipeline (`src/build_features.py`)](#5-step-3-feature-engineering-pipeline)
6. [Step 4: Machine Learning Training & Post-Hoc Calibration (`src/train_pd_model.py`, `src/train_multi_models.py`)](#6-step-4-machine-learning-training--post-hoc-calibration)
7. [Step 5: Model Registry & Inference Service (`backend/services/model_service.py`)](#7-step-5-model-registry--inference-service)
8. [Step 6: Test Set Inference & Kaggle Submissions (`src/predict_test.py`)](#8-step-6-test-set-inference--kaggle-submissions)
9. [Step 7: Underwriting Decision Engine & SHAP (`backend/services/underwriting_service.py`)](#9-step-7-underwriting-decision-engine--shap)
10. [Step 8: Policy RAG & Regulatory Compliance (`src/build_policy_rag.py`, `backend/services/policy_service.py`)](#10-step-8-policy-rag--regulatory-compliance)
11. [Step 9: Agentic AI Assistant (`backend/services/agent_service.py`)](#11-step-9-agentic-ai-assistant)
12. [Step 10: FastAPI Backend Application (`backend/main.py`, `backend/routes/`)](#12-step-10-fastapi-backend-application)
13. [Step 11: Next.js Frontend Application (`frontend/`)](#13-step-11-nextjs-frontend-application)
14. [Configuration Reference (`backend/config.py`)](#14-configuration-reference)
15. [Database & Audit Trail Schema (`backend/db/`)](#15-database--audit-trail-schema)
16. [Automated Test Suite (`backend/tests/`)](#16-automated-test-suite)
17. [How to Run Everything From Scratch](#17-how-to-run-everything-from-scratch)
18. [Key Design Decisions & Technical Trade-offs](#18-key-design-decisions--technical-trade-offs)
19. [Gotchas, Known Edge Cases & Maintenance Notes](#19-gotchas-known-edge-cases--maintenance-notes)

---

## 1. Project Overview & Architecture

This application is an end-to-end **Agentic AI Credit Underwriting System** developed as a research and educational prototype. It models the core underwriting functions of a digital lending platform:

```
[Raw Parquet Shards (7 datasets, 30M+ rows)]
                     │
                     ▼
       [DuckDB Engine (In-process analytical OLAP)]
                     │
                     ▼
  [Feature Engineering: 8 Aggregation Stages (307k × 88)]
                     │
                     ▼
    [Multi-Model ML Registry & Virtual Chaining]
   ├── XGBoost (Histogram gradient boosting)
   ├── LightGBM (Leaf-wise gradient boosting)
   ├── Logistic Regression (Basel II/III scorecard baseline)
   ├── Chained Weighted Ensemble (0.45 XGB + 0.45 LGB + 0.10 LR)
   └── Two-Stage Cascade Hurdle (Logistic triage → Tree specialist)
                     │
                     ▼
         [Underwriting Decision Engine]
   ├── Probability of Default (PD) Scoring
   ├── Risk Tier Assignment (Low, Moderate, High, Very High)
   ├── Deterministic Policy Rules (Income, Leverage, Card Util)
   ├── TreeSHAP Attribution (Risk-Increasing vs Risk-Reducing)
   ├── Semantic RAG against RBI Digital Lending Guidelines
   └── Immutable Audit Logging (SQLite / PostgreSQL)
                     │
                     ▼
       [FastAPI Backend (REST API / Async)]
                     │
                     ▼
  [Next.js 16 UI: Portfolio KPIs, Underwriting Desk, AI Copilot]
```

### Key Principles
1. **Separation of Concerns**: The statistical ML model estimates risk ($PD \in [0, 1]$); the deterministic business rule engine applies lending policy (`APPROVE`, `REFER`, `DECLINE`).
2. **Strict Temporal Integrity**: All credit bureau and payment records are audited to ensure no post-application data leaks into training.
3. **Transparent & Auditable**: Every credit outcome provides TreeSHAP feature attributions translated into natural English, matched regulatory clauses, and an immutable audit trail.

---

## 2. Repository Structure — Every File Explained

```
Home_Credit/
├── data/                                 # Data storage (git-ignored)
│   ├── application_train_dated/          # Primary application records (Parquet)
│   ├── bureau/                           # External credit bureau credit lines (Parquet)
│   ├── bureau_balance/                   # Monthly status of bureau accounts (Parquet)
│   ├── previous_application/             # Prior loan requests with this lender (Parquet)
│   ├── installments_payments/            # Detailed repayment schedules & payments (Parquet)
│   ├── POS_CASH_balance/                 # Point-of-sale loan monthly snapshots (Parquet)
│   ├── credit_card_balance/              # Revolving credit card monthly snapshots (Parquet)
│   ├── home_credit.duckdb                # DuckDB single-file analytical database
│   ├── audit.db                          # SQLite audit database storing decisions
│   └── features/
│       ├── model_features.parquet        # Flattened 307,511 × 88 feature matrix
│       ├── models_benchmark.json         # Performance benchmarks (raw vs calibrated)
│       ├── pd_evaluation.csv             # Production evaluation set (61,503 rows with calibrated & raw PD)
│       ├── validation_predictions_calibrated.parquet # Validation predictions (Platt & Isotonic)
│       ├── test_predictions.parquet      # Unseen test set predictions (48,744 rows)
│       └── submission.csv                # Calibrated Kaggle test submission (SK_ID_CURR, TARGET)
│
├── models/                               # Serialized model artifacts (git-ignored)
│   ├── pd_xgboost_calibrated.joblib      # Production champion: XGBoost + Isotonic Calibrator bundled
│   ├── pd_lightgbm_calibrated.joblib     # Production champion: LightGBM + Isotonic Calibrator bundled
│   ├── calibrator_isotonic.joblib        # Fitted Isotonic Regression calibrator
│   ├── calibrator_platt.joblib           # Fitted Platt Scaling (Logistic) calibrator
│   ├── pd_xgboost.joblib                 # Baseline raw XGBoost artifact dict
│   ├── pd_lightgbm.joblib                # Baseline raw LightGBM artifact dict
│   └── pd_logistic.joblib                # Trained Logistic Regression scorecard pipeline
│
├── policies/
│   ├── chroma_db/                        # ChromaDB vector index for regulatory RAG
│   └── rbi_digital_lending_guidelines_2022.pdf # Regulatory reference document
│
├── src/                                  # Data engineering & training pipelines
│   ├── create_database.py                # Ingests Parquet shards into DuckDB
│   ├── check_relationships.py            # Validates foreign key relationships & counts
│   ├── check_temporal_data.py            # Audits temporal fields (negative days rule)
│   ├── check_temporal_relationships.py   # Cross-table timeline validation
│   ├── build_features.py                 # 8-stage SQL feature engineering pipeline
│   ├── validate_features.py              # Schema & statistical validation of features
│   ├── investigate_feature_anomalies.py  # Outlier and anomaly checks
│   ├── train_pd_model.py                 # Baseline XGBoost training script
│   ├── train_multi_models.py             # Multi-model training (LGBM, LR, Ensemble)
│   ├── evaluate_pd_model.py              # In-depth ROC/PR/Brier evaluation
│   ├── predict_test.py                   # Test set feature builder & calibrated inference pipeline
│   ├── build_policy_rag.py               # Vectorizes RBI PDF into ChromaDB
│   ├── feature_descriptions.py           # Natural-language descriptions for all 88 features
│   ├── decision_engine.py                # Standalone deterministic policy logic
│   ├── explain_pd_model.py               # SHAP explainer utilities
│   └── inspect_data.py                   # CLI inspection tool for DuckDB
│
├── backend/                              # FastAPI application server
│   ├── main.py                           # App entry point, lifespan, route mounting
│   ├── config.py                         # Pydantic Settings & environment config
│   ├── db/
│   │   ├── database.py                   # SQLAlchemy engine, session maker, table creator
│   │   └── models.py                     # SQLAlchemy ORM models (UnderwritingDecision, AuditLog)
│   ├── schemas/
│   │   ├── applicant.py                  # Pydantic models for applicant profiles
│   │   ├── underwriting.py               # Pydantic models for underwriting requests/results
│   │   └── policy.py                     # Pydantic models for policy search & evidence
│   ├── services/
│   │   ├── model_service.py              # Singleton registry for multi-model inference
│   │   ├── underwriting_service.py       # Core underwriting workflow orchestrator
│   │   ├── policy_service.py             # ChromaDB vector retrieval service
│   │   └── agent_service.py              # LangGraph AI copilot service
│   ├── routes/
│   │   ├── health.py                     # GET /health
│   │   ├── applicants.py                 # GET /applicants, GET /applicants/{id}
│   │   ├── underwriting.py               # POST /underwriting/run
│   │   ├── model_info.py                 # GET /model/list, GET /model/comparison, POST /model/active
│   │   ├── policy.py                     # POST /policy/search
│   │   ├── agent.py                      # POST /agent/chat
│   │   └── dashboard.py                  # GET /dashboard/stats
│   └── tests/
│       ├── test_health.py                # Health endpoint tests
│       ├── test_applicants.py            # Applicant queries & 404 behavior
│       ├── test_decision_engine.py       # Deterministic policy thresholds
│       ├── test_underwriting.py          # End-to-end underwriting flow
│       ├── test_multi_models.py          # Model switching and virtual chaining
│       └── test_policy.py                # Policy semantic search tests
│
├── frontend/                             # Next.js 16 modern dashboard
│   ├── app/
│   │   ├── layout.tsx                    # Root shell with global sidebar navigation
│   │   ├── page.tsx                      # Index route (redirects to /dashboard)
│   │   ├── dashboard/page.tsx            # Portfolio analytics & risk distribution
│   │   ├── applicants/page.tsx           # Searchable applicant table (307k applicants)
│   │   ├── applicants/[id]/page.tsx      # Detailed financial profile view
│   │   ├── underwriting/[id]/page.tsx    # Interactive credit assessment workbench
│   │   ├── model/page.tsx                # Model registry, switcher & benchmarks
│   │   └── policies/page.tsx             # RBI regulatory search interface
│   ├── lib/
│   │   └── api.ts                        # Typed client for backend communication
│   └── types/
│       └── index.ts                      # TypeScript interfaces (SHAPFactor, etc.)
│
├── docs/                                 # Documentation
│   ├── PROJECT_COMPLETE_GUIDE.md         # Architecture, credit theory, feature dictionary
│   ├── DEVELOPER_HANDBOOK.md             # This comprehensive implementation guide
│   ├── HLD.md                            # High-Level Design document
│   ├── LLD.md                            # Low-Level Design document
│   └── architecture.md                   # Sequence & architecture diagrams
│
├── requirements.txt                      # Python library dependencies
├── Dockerfile.backend                    # Docker image for FastAPI
├── Dockerfile.frontend                   # Docker image for Next.js
└── docker-compose.yml                    # Multi-container local orchestration
```

---

## 3. Step 1: Raw Data & Analytical Database

**File:** [src/create_database.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/create_database.py)  
**Input:** Partitioned `.parquet` files in `data/`  
**Output:** [data/home_credit.duckdb](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/data/home_credit.duckdb)

### Parquet Ingestion Architecture
The original source data comprises 7 multi-part Parquet directories totaling tens of millions of records:
- `application_train_dated/`: 307,511 primary loan applications.
- `bureau/`: 1,716,428 credit bureau trade lines.
- `bureau_balance/`: 27,299,925 monthly bureau status records.
- `previous_application/`: 1,670,214 prior loan applications.
- `installments_payments/`: 13,605,401 scheduled and actual repayments.
- `POS_CASH_balance/`: 10,001,358 point-of-sale monthly snapshots.
- `credit_card_balance/`: 3,840,312 revolving credit card monthly statements.

### Implementation Details
Rather than loading large CSVs or storing data in a heavyweight external RDBMS, `src/create_database.py` uses **DuckDB** to read all Parquet shards directly via vectorized scans:

```python
def find_parquet(folder_name):
    folder = DATA_DIR / folder_name
    files = sorted(folder.rglob("*.parquet"))
    return files

def sql_file_list(files):
    return "[" + ", ".join(f"'{f.as_posix()}'" for f in files) + "]"

for table_name, folder_name in TABLES.items():
    files = find_parquet(folder_name)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_parquet({sql_file_list(files)})
    """)
```

DuckDB processes multi-gigabyte Parquet datasets in seconds with low memory overhead through columnar projection and filter pushdown.

---

## 4. Step 2: Integrity & Temporal Auditing

**Files:**
- [src/check_relationships.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/check_relationships.py)
- [src/check_temporal_data.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/check_temporal_data.py)
- [src/check_temporal_relationships.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/check_temporal_relationships.py)

### Relational Checks
`src/check_relationships.py` verifies foreign-key linkages:
- `application_train.SK_ID_CURR` $\rightarrow$ `bureau.SK_ID_CURR` (86% coverage)
- `bureau.SK_ID_BUREAU` $\rightarrow$ `bureau_balance.SK_ID_BUREAU`
- `application_train.SK_ID_CURR` $\rightarrow$ `previous_application.SK_ID_CURR` (94% coverage)
- `previous_application.SK_ID_PREV` $\rightarrow$ `installments_payments.SK_ID_PREV`

### Temporal Leakage Prevention
In credit scoring, using data generated *after* the loan decision date causes target leakage. In this dataset, time is recorded as negative relative offsets in days:
- `DAYS_BIRTH = -14600`: Born 14,600 days before application.
- `DAYS_EMPLOYED = -1825`: Started current employment 5 years prior.
- `DAYS_CREDIT = -365`: Bureau account opened 1 year before loan application.
- `DAYS_DECISION = -730`: Prior application was decided 2 years prior.

`src/check_temporal_data.py` asserts that all historical observations have negative day offsets ($\le 0$), ensuring zero future leakage into the training set.

---

## 5. Step 3: Feature Engineering Pipeline

**File:** [src/build_features.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/build_features.py)  
**Input:** `data/home_credit.duckdb`  
**Output:** [data/features/model_features.parquet](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/data/features/model_features.parquet)

The aggregation pipeline maps 1-to-many historical records into a single row per applicant (307,511 rows $\times$ 88 features) across 8 modular stages:

1. **Application Features (`application_features`)**:
   - Demographic flags, requested amounts (`AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_ANNUITY`).
   - Domain leverage ratios:
     $$\text{credit\_to\_income} = \frac{\text{AMT\_CREDIT}}{\text{AMT\_INCOME\_TOTAL}}$$
     $$\text{annuity\_to\_income} = \frac{\text{AMT\_ANNUITY}}{\text{AMT\_INCOME\_TOTAL}}$$
     $$\text{annuity\_to\_credit} = \frac{\text{AMT\_ANNUITY}}{\text{AMT\_CREDIT}}$$
   - Composite external score: `ext_source_mean = mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)`.

2. **Bureau Features (`bureau_features`)**:
   - Aggregated over records where `DAYS_CREDIT <= 0`.
   - `active_bureau_accounts`, `closed_bureau_accounts`, `total_bureau_credit`, `total_bureau_debt`.
   - Robust portfolio debt-to-credit ratio calculated via sum of totals:
     $$\text{bureau\_debt\_to\_credit\_ratio} = \frac{\sum \text{AMT\_CREDIT\_SUM\_DEBT}}{\sum \text{AMT\_CREDIT\_SUM}}$$

3. **Previous Application Features (`previous_application_features`)**:
   - `previous_approved_count`, `previous_refused_count`, `previous_recent_1y_count`.
   - Total & average applied/credited amounts.

4. **Installment Payment Features (`installment_features`)**:
   - Payment punctuality: `avg_payment_delay = avg(greatest(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))`.
   - Repayment ratio: $\frac{\sum \text{AMT\_PAYMENT}}{\sum \text{AMT\_INSTALMENT}}$.
   - Counts of underpayments and late payments.

5. **POS Cash Features (`pos_features`)**:
   - `max_pos_dpd`, `avg_pos_dpd` (Days Past Due).
   - Term commitment metrics: `avg_pos_installments`, `max_pos_installments`.

6. **Credit Card Features (`credit_card_features`)**:
   - `avg_cc_balance`, `max_cc_balance`, `avg_cc_limit`.
   - Aggregate utilization ratio:
     $$\text{cc\_utilization} = \frac{\sum \max(\text{AMT\_BALANCE}, 0)}{\sum \max(\text{AMT\_CREDIT\_LIMIT\_ACTUAL}, 0)}$$
   - Delinquency counts and drawing aggregates.

7. **Bureau Balance Features (`bureau_balance_features`)**:
   - Historical month delinquency counters from status codes (`'1'` to `'5'`).
   - Overdue frequency rate.

8. **Final Master Join (`model_features`)**:
   - Joins all sub-aggregates to `application_features` via `LEFT JOIN` on `SK_ID_CURR`.
   - Preserves all 307,511 applicants. Missing values are retained as `NULL` so tree algorithms can exploit sparsity patterns.

The table is exported with ZSTD compression to `data/features/model_features.parquet`.

---

## 6. Step 4: Machine Learning Training & Model Chaining

**Files:**
- [src/train_pd_model.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/train_pd_model.py)
- [src/train_multi_models.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/train_multi_models.py)
- [src/feature_descriptions.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/feature_descriptions.py)

### Model Zoo & Virtual Pipelines

1. **XGBoost Classifier (`models/pd_xgboost_calibrated.joblib` — Active Production)**:
   - Histogram-based gradient boosting trees (`tree_method='hist'`) bundled with an **Isotonic Regression Calibrator**.
   - Stratified 80/20 train/validation split (`seed=42`).
   - Class imbalance initially handled via `scale_pos_weight = 11.4` (8.07% empirical default rate).
   - Post-hoc isotonic calibration maps raw scores to true empirical default frequencies.
   - `ROC-AUC: 0.7758` | `Brier Score: 0.0665` (reduced from 0.1575) | `Log Loss: 0.2401` | `Latency: ~0.0062 ms/row`.

2. **LightGBM Classifier (`models/pd_lightgbm_calibrated.joblib`)**:
   - Leaf-wise tree splitting (`num_leaves=31`, `n_estimators=600`, `learning_rate=0.05`) with post-hoc Isotonic calibration.
   - Native categorical support and multi-threaded training.
   - `ROC-AUC: 0.7760` | `Brier Score: 0.0665` (reduced from 0.1671) | `Log Loss: 0.2398` | `Latency: ~0.0064 ms/row`.

3. **Logistic Regression Scorecard (`models/pd_logistic.joblib`)**:
   - Basel II/III compliant interpretable baseline.
   - Scikit-learn pipeline: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()` $\rightarrow$ `LogisticRegression(class_weight='balanced')`.
   - `ROC-AUC: 0.7574` | `PR-AUC: 0.2339` | `Brier Score: 0.1989` | `Latency: ~0.0017 ms/row`.

4. **Chained Weighted Ensemble (Calibrated Virtual Pipeline — Project Champion)**:
   - Soft probability voting blend of calibrated models:
     $$PD_{\text{ensemble}} = 0.45 \cdot PD_{\text{XGB\_cal}} + 0.45 \cdot PD_{\text{LGB\_cal}} + 0.10 \cdot PD_{\text{LR}}$$
   - `ROC-AUC: 0.7789` | `Brier Score: 0.0663` (Highest discrimination and lowest error across entire project).

5. **Two-Stage Cascade Hurdle (Virtual Pipeline)**:
   - High-throughput production triage:
     - **Stage 1 (Fast Filter)**: Run ultra-fast Logistic Scorecard (~0.0017 ms).
       - If $PD < 0.03 \rightarrow$ Instant Approve.
       - If $PD > 0.35 \rightarrow$ Instant Decline.
     - **Stage 2 (Specialist)**: Borderline applicants ($0.03 \le PD \le 0.35$) undergo deep evaluation by Calibrated XGBoost/LightGBM.
   - Reduces average compute by ~40% while preserving specialist calibration and accuracy.

---

### Post-Hoc Probability Calibration (Basel II/III & IFRS 9 Compliance)

#### The Problem: Why Raw Tree Outputs Distort Risk
When tree models are trained with `scale_pos_weight = 11.4` on imbalanced credit data, the loss function shifts the baseline prediction toward 50%. Consequently:
- A borrower with a true statistical risk of **5%** was assigned a raw model score of **30%–35%**.
- Across the 61,503 validation applicants, the average raw model score was **0.3426 (34.3%)**, whereas the actual empirical default rate was only **0.0807 (8.07%)**.
- **Banking Impact:** Deploying raw scores causes false declines for prime borrowers, inflates loan interest rates, and distorts **IFRS 9 Expected Credit Loss ($\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}$)** reserves.

#### The Solution: Dual-Stage Holdout Calibration
Two post-hoc calibrators were fitted strictly on the 20% validation split (61,503 rows) to prevent data leakage:
1. **Platt Scaling (Logistic Regression on log-odds)**:
   $$P(Y=1 \mid s) = \frac{1}{1 + e^{-(A s + B)}}$$
2. **Isotonic Regression (Pool Adjacent Violators Algorithm — PAVA)**:
   Non-parametric piecewise-constant monotonic mapping that minimizes squared error against empirical bin frequencies.

#### Calibration Benchmark Comparison:
| Model Configuration | ROC-AUC | Brier Score | Log Loss | Portfolio Mean PD | Actual Default Rate |
|---|---|---|---|---|---|
| **Raw XGBoost (Before)** | 0.7745 | 0.1575 | 0.4804 | 34.26% | 8.07% |
| **Platt Calibrated** | 0.7745 | 0.0668 | 0.2413 | 8.08% | 8.07% |
| **Isotonic Calibrated (Champion)** | **0.7758** | **0.0665** | **0.2401** | **8.07%** | **8.07%** |
| **Calibrated LightGBM** | 0.7760 | 0.0665 | 0.2398 | 8.07% | 8.07% |
| **Calibrated Ensemble Blend** | **0.7789** | **0.0663** | — | **8.07%** | **8.07%** |

> **Result:** Brier Score error dropped by **57.8%**, Log Loss was halved, and ROC-AUC was fully preserved. Mean predicted PD now equals the empirical default rate down to 8 decimal places.

---

## 7. Step 5: Model Registry & Inference Service

**File:** [backend/services/model_service.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/services/model_service.py)

The `ModelService` is implemented as a process-wide **Singleton**:
- Discovers and loads all `models/pd_*.joblib` artifacts into memory at backend startup.
- Defaults to `active_model_id = "xgboost_calibrated"` with runtime hot-switching via `POST /model/active`.
- Automatically executes bundled post-hoc calibrators:
  ```python
  if "calibrator" in artifact and artifact["calibrator"] is not None:
      prob = float(artifact["calibrator"].predict(np.array([prob]))[0])
      prob = max(0.0001, min(0.9999, prob))
  ```

### Critical Architecture: Global Categorical Mapping Cache
- **The Bug That Was Fixed:** In pandas, calling `.astype("category").cat.codes` on a **single applicant row** (`df.shape == (1, 90)`) creates a 1-element series where whatever category is present gets assigned code `0`. This caused real-time inference to evaluate all categoricals (`NAME_EDUCATION_TYPE`, `CODE_GENDER`, etc.) as `0`, deviating from the batch validation results.
- **The Resolution:** `ModelService.load()` extracts and caches the global training vocabulary `self._categorical_mappings` directly from `model_features.parquet` at startup:
  ```python
  self._categorical_mappings = {}
  for col in self._categorical_columns:
      if col in self._features_df.columns:
          cats = pd.Categorical(self._features_df[col]).categories
          self._categorical_mappings[col] = {cat: idx for idx, cat in enumerate(cats)}
  ```
- In `prepare_model_input()`, incoming applicant rows are mapped via `.map(self._categorical_mappings[col])`, guaranteeing **exact 100% mathematical parity** between batch evaluation ([pd_evaluation.csv](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/data/features/pd_evaluation.csv)) and real-time underwriting.

```python
# Predict PD with any registered model or virtual pipeline
pd_value, details, model_name = model_service.predict_pd_detailed(
    applicant_df, model_id="xgboost_calibrated"
)
```

---

## 8. Step 6: Test Set Inference & Kaggle Submissions

**File:** [src/predict_test.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/predict_test.py)  
**Input:** `data/application_test` (48,744 unlabelled forward-flow applicants) + DuckDB historical features  
**Outputs:** 
- [data/features/test_predictions.parquet](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/data/features/test_predictions.parquet) (Detailed scoring with `PD`, `PD_raw`, and `risk_band`)
- [data/features/submission.csv](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/data/features/submission.csv) (Official Kaggle submission format: `SK_ID_CURR,TARGET`)

### Purpose & Implementation Workflow
In a commercial bank, models are validated on historical data but deployed to score incoming unlabelled loan applications. `src/predict_test.py` mirrors the complete batch forward-flow inference pipeline:

1. **Test Feature Extraction via DuckDB**:
   - Executes SQL aggregate feature queries directly on `application_test` within DuckDB.
   - Joins pre-computed credit bureau, previous applications, installment payments, POS, and credit card records.
2. **Schema & Feature Alignment**:
   - Strict 88-feature schema alignment against `artifact["features"]`.
   - Incorporates `OWN_CAR_AGE` without missing feature warnings.
   - Categoricals re-encoded using consistent training definitions.
3. **Calibrated Batch Prediction**:
   - Loads `models/pd_xgboost_calibrated.joblib`.
   - Generates raw tree outputs (mean ~30.1%) and transforms them through the fitted Isotonic Calibrator into realistic default probabilities (mean **6.37%**, median **4.15%**).
4. **Risk Band Segmentation**:
   - Categorizes all 48,744 applicants across standard underwriting risk tiers:
     - `<5%`: 26,330 applicants (54.02% — Prime auto-approval tier)
     - `5–10%`: 13,513 applicants (27.72% — Standard credit tier)
     - `10–20%`: 6,726 applicants (13.80% — Moderate risk / conditional approval)
     - `20–30%`: 1,702 applicants (3.49% — High risk / manual underwriting review)
     - `30–50%`: 441 applicants (0.90% — Subprime / specialist referral)
     - `>50%`: 32 applicants (0.07% — Severe risk knockout)

---

## 9. Step 7: Underwriting Decision Engine & SHAP

**File:** [backend/services/underwriting_service.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/services/underwriting_service.py)

When `run_underwriting(applicant_id, model_id)` is invoked:

1. **Profile Retrieval**: Fetches 88-feature record from `ModelService`.
2. **PD Prediction**: Executes model inference to obtain calibrated default probability ($PD$).
3. **Risk Tier Assignment**:
   - `LOW`: $PD < 0.05$
   - `MODERATE`: $0.05 \le PD < 0.10$
   - `HIGH`: $0.10 \le PD < 0.20$
   - `VERY_HIGH`: $PD \ge 0.20$
4. **Deterministic Policy Rules**:
   - Rule 1: $PD \ge 0.20 \rightarrow$ `DECLINE` ("High predicted probability of default").
   - Rule 2: Missing income or income $< 50,000 \rightarrow$ `REFER` ("Income verification required").
   - Rule 3: $\text{credit\_to\_income} > 10.0 \rightarrow$ `REFER` ("High leverage relative to income").
   - Rule 4: $\text{max\_cc\_utilization} > 1.50 \rightarrow$ `REFER` ("Excessive credit card limit utilization").
   - Rule 5: $PD < 0.10 \rightarrow$ `APPROVE` ("Risk within acceptable tolerance").
   - Default: `REFER` ("Requires senior credit officer review").
5. **TreeSHAP Explainability**:
   - Computes local feature attributions via `shap.TreeExplainer`.
   - Maps raw feature keys to plain-English labels and context descriptions using [src/feature_descriptions.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/feature_descriptions.py).
   - Segregates top 5 risk-increasing factors and top 5 risk-reducing factors.
   - Results are cached in an in-memory dictionary `(applicant_id, model_id)` to prevent redundant recalculation.
6. **Regulatory Evidence Binding**: Queries the vector policy store for contextual RBI compliance text.
7. **Audit Logging**: Asynchronously writes the decision snapshot to the SQLite database.

---

## 10. Step 8: Policy RAG & Regulatory Compliance

**Files:**
- [src/build_policy_rag.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/src/build_policy_rag.py)
- [backend/services/policy_service.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/services/policy_service.py)

### Vector Store Ingestion
`src/build_policy_rag.py` ingests the official **RBI Guidelines on Digital Lending (2022)**:
- Chunks text into semantic segments with overlap.
- Computes dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors).
- Persists chunks with metadata (section, page, title) in `policies/chroma_db/`.

### Runtime Evidence Retrieval
During underwriting, `policy_service.search(query, k=3)` performs cosine similarity search against the vector index:
```python
evidence_query = f"borrower consent data privacy transparency digital lending {decision.lower()}"
regulatory_evidence = policy_service.search(evidence_query, k=3)
```
Each matching chunk returns:
- Relevant regulatory text excerpt.
- Source authority ("Reserve Bank of India").
- Specific section reference (e.g., "Section 3: Disclosures and Key Fact Statement").
- Semantic similarity distance score.

---

## 11. Step 9: Agentic AI Assistant

**File:** [backend/services/agent_service.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/services/agent_service.py)

The underwriting assistant is powered by a **LangGraph** agent with 3 bound tools:
1. `run_underwriting_assessment(applicant_id)`: Generates PD, tier, and policy decision.
2. `get_shap_risk_factors(applicant_id)`: Retrieves top risk-increasing and risk-reducing factors.
3. `search_regulatory_guidelines(query)`: Queries the RBI policy RAG index.

### Guardrails & Fallback
- **Zero Hallucination Constraint**: The prompt instructs the model to rely strictly on tool outputs and decline speculation on unobserved financial metrics.
- **Graceful Degradation**: If the local Ollama instance (`llama3.1:8b`) is offline, the service automatically falls back to a deterministic rule-based response engine.

---

## 12. Step 10: FastAPI Backend Application

**Files:**
- [backend/main.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/main.py)
- [backend/routes/](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/routes)

### Application Lifespan
On launch, `backend/main.py` executes startup initialization:
1. `create_tables()`: Verifies/creates audit tables in SQLite.
2. `model_service.load()`: Loads all model artifacts, categorical vocabulary, and the 307k Parquet dataset into memory.
3. `policy_service.load()`: Connects to ChromaDB.
4. `agent_service.load()`: Verifies local Ollama LLM connectivity.

### Route Summary

| Endpoint | Method | Tag | Description |
|---|---|---|---|
| `/health` | `GET` | System | Health check, uptime, memory status |
| `/applicants` | `GET` | Applicants | Paginated applicant query with search filters |
| `/applicants/{id}` | `GET` | Applicants | Retrieves single applicant's full 88-feature profile |
| `/underwrite` | `POST` | Underwriting | Root convenience endpoint for underwriting assessment |
| `/underwriting/run` | `POST` | Underwriting | Executes full underwriting, SHAP, and policy audit |
| `/model/list` | `GET` | Model Info | Lists all registered models and the active model |
| `/model/comparison` | `GET` | Model Info | Retrieves benchmark metrics (AUC, PR, Brier, latency) |
| `/model/active` | `POST` | Model Info | Sets active model for subsequent underwriting runs |
| `/policy/search` | `POST` | Policy | Semantic vector search over RBI guidelines |
| `/agent/chat` | `POST` | Agent | Interactive LangGraph agent query endpoint |
| `/dashboard/stats` | `GET` | Dashboard | Macro portfolio metrics and risk distributions |

---

## 13. Step 11: Next.js Frontend Application

**Directory:** `frontend/`

Built with Next.js 16 (App Router), Tailwind CSS, Lucide icons, and Recharts:

### Pages
- **`/dashboard`** ([frontend/app/dashboard/page.tsx](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/frontend/app/dashboard/page.tsx)):
  Macro view of the lending portfolio: total applications processed, approval/refer/decline rates, average PD, and risk tier distribution chart.
- **`/applicants`** ([frontend/app/applicants/page.tsx](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/frontend/app/applicants/page.tsx)):
  Searchable, paginated registry of all 307,511 applicants with instant filters for income, credit amount, and occupation.
- **`/applicants/[id]`** ([frontend/app/applicants/[id]/page.tsx](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/frontend/app/applicants/[id]/page.tsx)):
  Detailed view of an applicant's raw inputs: personal info, external bureau scores, credit lines, and repayment track record.
- **`/underwriting/[id]`** ([frontend/app/underwriting/[id]/page.tsx](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/frontend/app/underwriting/[id]/page.tsx)):
  Underwriter workbench:
  - Decision badge (`APPROVE`, `REFER`, `DECLINE`) and policy rationales.
  - Interactive SVG Probability of Default gauge.
  - Side-by-side SHAP factor cards with impact bars, friendly labels, and context tooltips.
  - Natural Language Breakdown section explaining financial factors in non-technical terms.
  - Contextual RBI regulatory citations.
  - Embedded AI assistant chat sidebar.
- **`/model`** ([frontend/app/model/page.tsx](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/frontend/app/model/page.tsx)):
  Model management console:
  - One-click active model switcher.
  - Benchmark comparison table (ROC-AUC, PR-AUC, Brier score, latency).
  - Virtual pipeline architecture diagrams (Chained Ensemble, Cascade Hurdle).
  - Interactive sandbox to compare model predictions on any applicant ID.
- **`/policies`** ([frontend/app/policies/page.tsx](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/frontend/app/policies/page.tsx)):
  Semantic search interface for querying the RBI Digital Lending regulatory knowledge base.

---

## 14. Configuration Reference

**File:** [backend/config.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/config.py)

All settings are managed via Pydantic Settings and can be overridden via environment variables or a `.env` file:

```ini
# Server Settings
DEBUG=false
APP_NAME="Agentic AI Credit Underwriting System"
APP_VERSION="1.0.0"
CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]

# Database & Artifact Paths
DATABASE_URL="sqlite:///data/audit.db"
MODELS_DIR="models"
MODEL_PATH="models/pd_xgboost_calibrated.joblib"
FEATURE_PATH="data/features/model_features.parquet"
EVALUATION_PATH="data/features/pd_evaluation.csv"
BENCHMARK_PATH="data/features/models_benchmark.json"

# Active Model & Policy Versions
MODEL_VERSION="pd_xgboost_calibrated_v1"
POLICY_VERSION="research_policy_v1"

# Research Underwriting Policy Thresholds
PD_APPROVE_MAX=0.10
PD_DECLINE_MIN=0.20
MIN_INCOME=50000.0
MAX_CREDIT_TO_INCOME=10.0
MAX_CC_UTILIZATION=1.50

# LLM Configuration
LLM_PROVIDER="ollama"
OLLAMA_MODEL="llama3.1:8b"
OLLAMA_BASE_URL="http://localhost:11434"
```

---

## 15. Database & Audit Trail Schema

**Files:**
- [backend/db/database.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/db/database.py)
- [backend/db/models.py](file:///c:/Users/Sahil%20Kumar/Downloads/Home_Credit/backend/db/models.py)

### Tables
1. **`underwriting_decisions`**:
   - `id`: Auto-incrementing primary key.
   - `request_id`: UUID for the specific underwriting request.
   - `applicant_id`: Target applicant ID (`SK_ID_CURR`).
   - `timestamp`: UTC timestamp of the assessment.
   - `probability_of_default`: Calibrated model output.
   - `risk_tier`: Assigned risk category.
   - `decision`: Final outcome (`APPROVE`, `REFER`, `DECLINE`).
   - `policy_reason`: Justification generated by the decision engine.
   - `model_id`: ID of the model used (e.g., `xgboost_calibrated`, `chained_ensemble`).
   - `model_version`: Version string of the model.

2. **`audit_logs`**:
   - `id`: Primary key.
   - `timestamp`: UTC timestamp.
   - `event_type`: Event category (e.g., `underwriting_decision`, `model_switched`).
   - `details`: JSON payload capturing context, user, and previous state.

---

## 16. Automated Test Suite

**Directory:** `backend/tests/`  
**Execution:** `.\.venv\Scripts\python.exe -m pytest backend/tests/ -v`

42 automated test cases cover every backend service:
- `test_health.py`: Validates service liveness and database connectivity.
- `test_applicants.py`: Tests profile pagination, data formatting, and 404 responses.
- `test_decision_engine.py`: Validates deterministic boundary conditions:
  - Low PD $\rightarrow$ `APPROVE`
  - High PD $\rightarrow$ `DECLINE`
  - Low income $\rightarrow$ `REFER`
  - Excessive leverage $\rightarrow$ `REFER`
- `test_underwriting.py`: Tests end-to-end assessment, SHAP factor presence, and audit trail creation.
- `test_multi_models.py`: Validates model switching, ensemble calculation, cascade hurdle logic, and benchmark consistency.
- `test_policy.py`: Tests vector search precision and chunk extraction.

---

## 17. How to Run Everything From Scratch

### Step 1: Environment Setup
```powershell
# Clone and enter directory
cd c:\Users\Sahil Kumar\Downloads\Home_Credit

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Build Analytical Database
```powershell
.\.venv\Scripts\python.exe src/create_database.py
# Ingests 7 Parquet tables into data/home_credit.duckdb
```

### Step 3: Audit Relational & Temporal Integrity
```powershell
.\.venv\Scripts\python.exe src/check_relationships.py
.\.venv\Scripts\python.exe src/check_temporal_data.py
```

### Step 4: Run Feature Engineering
```powershell
.\.venv\Scripts\python.exe src/build_features.py
# Generates data/features/model_features.parquet (~40-60s)
```

### Step 5: Train ML Models & Generate Benchmarks
```powershell
.\.venv\Scripts\python.exe src/train_multi_models.py
# Trains XGBoost, LightGBM, Logistic Scorecard
# Evaluates Chained Ensemble & Cascade Hurdle
# Generates models/*.joblib and data/features/models_benchmark.json
```

### Step 6: Post-Hoc Probability Calibration
```powershell
# Fit Isotonic Calibrators on 20% validation split
# Generates models/pd_xgboost_calibrated.joblib and models/pd_lightgbm_calibrated.joblib
# Updates data/features/pd_evaluation.csv with true calibrated probabilities
```

### Step 7: Execute Forward-Flow Test Inference
```powershell
.\.venv\Scripts\python.exe src/predict_test.py
# Aggregates features for 48,744 application_test records
# Executes calibrated inference with risk banding
# Generates data/features/test_predictions.parquet & data/features/submission.csv
```

### Step 8: Build Regulatory RAG Vector Index
```powershell
.\.venv\Scripts\python.exe src/build_policy_rag.py
# Vectorizes policies/rbi_digital_lending_guidelines_2022.pdf into policies/chroma_db/
```

### Step 9: Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/ -v
# Validates unit and integration test suite
```

### Step 10: Start FastAPI Backend
```powershell
python -m uvicorn backend.main:app --reload --port 8002
# API docs available at http://localhost:8002/docs
```

### Step 11: Start Next.js Frontend
```powershell
cd frontend
npm install
npm run dev
# Dashboard accessible at http://localhost:3000
```

### Step 12 (Optional): Start Local LLM for AI Copilot
```powershell
ollama run llama3.1:8b
```

---

## 18. Key Design Decisions & Technical Trade-offs

1. **DuckDB In-Memory OLAP vs. Traditional RDBMS**:
   - *Why*: Performing multi-table aggregations over 30M+ rows in Postgres or SQLite requires indexing, migrations, and substantial disk I/O. DuckDB reads directly from compressed Parquet files in parallel with low RAM usage.

2. **Decoupling Statistical Risk ($PD$) from Credit Policy**:
   - *Why*: Credit policies change frequently due to macroeconomic conditions, while ML models require stable historical datasets to train. Separating $PD$ estimation from deterministic rule evaluation allows modifying thresholds without retraining models.

3. **Mandatory Post-Hoc Calibration for Real-World Banking**:
   - *Why*: In credit risk, class imbalance reweighting (`scale_pos_weight = 11.4`) inflates raw probabilities up to 4×–5×. Under Basel III IRB and IFRS 9 Expected Credit Loss ($\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}$), models must output true statistical default frequencies. Bundling an Isotonic Calibrator reduced Brier score error by **57.8%** while preserving 100% of discriminatory ranking power.

4. **In-Memory Feature Store for Inference**:
   - *Why*: The feature dataset (`model_features.parquet`) is ~15 MB on disk and ~80 MB in RAM. Loading it into a pandas DataFrame once at startup enables sub-millisecond applicant lookups ($O(1)$) without external Redis or Feast dependencies.

5. **Structured Joblib Dictionaries over Raw Model Pickles**:
   - *Why*: Pickling raw model objects loses training feature order and categorical encodings. Saving a dictionary with model instances, calibrators, feature names, column order, and hyperparameters guarantees reproducible inference.

6. **Two-Stage Cascade Hurdle Architecture**:
   - *Why*: Heavy gradient-boosted ensembles are computationally expensive. A fast linear filter (Stage 1) handles obvious prime and default cases in 0.001 ms, routing only ambiguous borderline cases to the tree model.

---

## 19. Gotchas, Known Edge Cases & Maintenance Notes

1. **Single-Row Categorical Encoding in Pandas**:
   - *The Trap:* Calling `.astype("category").cat.codes` on a single-row DataFrame (e.g., during single applicant inference) creates a 1-element series where whatever category is present gets assigned code `0`. This would silently force every categorical feature to `0`.
   - *The Fix:* `ModelService` extracts and caches the global training vocabulary `self._categorical_mappings` at startup and uses `.map(self._categorical_mappings[col])` during inference.

2. **Feature Column Ordering**:
   - Tree models depend on exact feature position. `ModelService.prepare_model_input()` enforces feature ordering using `artifact["features"]`. Never reorder columns in `model_features.parquet` without updating the model artifacts.

3. **Special Sentinel Values in Raw Data**:
   - `DAYS_EMPLOYED = 365243` is a known Home Credit sentinel value indicating retired, student, or unemployed status. The feature engineering pipeline preserves this numeric signal.

4. **Missing Values in External Sources**:
   - Applicants with no bureau history will have `NaN` for `EXT_SOURCE_1/2/3`. The `ext_source_mean` calculation uses `NULLIF` to prevent division by zero. Tree models route `NaN` values to optimal default branches.

5. **Local Vector Store Concurrency**:
   - ChromaDB is accessed via local file lock in `policies/chroma_db/`. Avoid running multiple parallel backend workers that write to ChromaDB simultaneously.

6. **In-Memory SHAP Cache Eviction**:
   - The SHAP calculation cache in `underwriting_service.py` is an in-memory dictionary. For a production system with millions of requests, replace it with an LRU cache or Redis instance.
