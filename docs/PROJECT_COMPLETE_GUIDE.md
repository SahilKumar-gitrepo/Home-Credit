# The Complete Guide to the Agentic Credit Underwriting System

Welcome to the definitive documentation for the **Agentic AI-Based Credit Underwriting and Loan Risk Assessment System**.

This document covers everything you need to understand, explain, present, and build upon this project: from high-level banking fundamentals to machine learning architectures, vector search (RAG), agentic workflows, codebase organization, and step-by-step execution.

---

## Table of Contents

1. [Executive Summary & Core Philosophy](#1-executive-summary--core-philosophy)
2. [Credit Risk & Underwriting Fundamentals](#2-credit-risk--underwriting-fundamentals)
3. [The Home Credit Dataset & Feature Engineering](#3-the-home-credit-dataset--feature-engineering)
4. [Machine Learning Architecture & Model Registry](#4-machine-learning-architecture--model-registry)
5. [Explainable AI (XAI) with SHAP](#5-explainable-ai-xai-with-shap)
6. [Regulatory Compliance & Policy RAG](#6-regulatory-compliance--policy-rag)
7. [Agentic AI Orchestration (LangGraph & Ollama)](#7-agentic-ai-orchestration-langgraph--ollama)
8. [End-to-End System Architecture](#8-end-to-end-system-architecture)
9. [Project Directory & Codebase Walkthrough](#9-project-directory--codebase-walkthrough)
10. [API Reference & Schema Specifications](#10-api-reference--schema-specifications)
11. [How to Run, Train, Test, and Deploy](#11-how-to-run-train-test-and-deploy)
12. [Presentation & Interview Cheat Sheet](#12-presentation--interview-cheat-sheet)
13. [Natural-Language Feature & Column Dictionary: What Every Column Means](#13-natural-language-feature--column-dictionary-what-every-column-means)

---

## 1. Executive Summary & Core Philosophy

### What is this project?
This system is an **Agentic Credit Underwriting and Risk Assessment Platform** designed for retail and digital lending. It combines:
1. **Statistical Machine Learning** (XGBoost, LightGBM, Logistic Scorecards, and Chained Ensembles) to predict **Probability of Default (PD)**.
2. **Deterministic Governance** (Hard policy rules, debt-to-income caps, credit card utilization thresholds) for non-negotiable credit policy enforcement.
3. **Explainable AI (XAI)** (TreeSHAP & linear attributions) to explain why a decision was reached for transparency and borrower rights.
4. **Regulatory Retrieval-Augmented Generation (RAG)** (ChromaDB + SentenceTransformers) to ground underwriting decisions in the **Reserve Bank of India (RBI) Digital Lending Guidelines (2022)** and Basel II/III principles.
5. **Agentic LLM Assistant** (LangGraph + local Ollama models) allowing credit officers to query applicant risk via an interactive conversational interface that is strictly grounded against hallucinations.

### The 4 Core Architectural Tenets
| Tenet | Principle | Implementation |
| :--- | :--- | :--- |
| **1. Separation of Concerns** | *The ML model does not decide; it estimates risk. The policy decides.* | ML outputs a continuous PD (0.0 to 1.0). The deterministic policy engine applies rules to render `APPROVE`, `REFER`, or `DECLINE`. |
| **2. Zero Hallucination** | *An LLM must never invent financial figures or credit scores.* | The LLM acts solely as a synthesis agent consuming deterministic outputs via LangGraph tools. It has fallback logic if offline. |
| **3. Explainability is Mandatory** | *Every rejection or referral requires clear evidence.* | Every assessment extracts top 5 risk-increasing and top 5 risk-reducing SHAP factors. |
| **4. Regulatory Alignment** | *Lending models must adhere to Central Bank guidelines.* | Real-time semantic vector retrieval injects official RBI digital lending clauses directly into the audit record. |

---

## 2. Credit Risk & Underwriting Fundamentals

### What is Credit Underwriting?
Credit underwriting is the process by which a lender evaluates the risk of lending money to an applicant. In retail lending, the central metric is the **Expected Loss (EL)** formula:

$$\text{Expected Loss (EL)} = \text{PD} \times \text{EAD} \times \text{LGD}$$

- **PD (Probability of Default)**: The likelihood (0% to 100%) that the borrower fails to meet debt obligations within a specified time horizon (typically 12 months). **This system focuses on predicting PD.**
- **EAD (Exposure at Default)**: The total outstanding debt balance when default occurs.
- **LGD (Loss Given Default)**: The percentage of the loan amount lost after collateral liquidation or recovery efforts.

### Risk Tier Categorization
The continuous PD score is mapped into discrete risk tiers:

```
0.0% ─────────── 5.0% ─────────── 10.0% ─────────── 20.0% ─────────── 100%
      LOW                MODERATE             HIGH             VERY HIGH
  (Prime Credit)      (Near-Prime)        (Subprime)         (Deep Subprime)
```

- **LOW (< 5.0%)**: High creditworthiness, stable financial history, low external bureau risk.
- **MODERATE (5.0% – 10.0%)**: Acceptable risk profile, may require standard monitoring.
- **HIGH (10.0% – 20.0%)**: Elevated risk; triggered for manual review or referral.
- **VERY HIGH (> 20.0%)**: Severe risk of non-repayment; candidate for automated decline.

### Deterministic Decision Rules
Even with a low PD, automated lending requires guardrails to comply with responsible lending rules:

1. **Automated Decline (`DECLINE`)**:
   - If $\text{PD} > 0.20$ (Very High Risk), OR
   - If $\text{Annual Income} < \text{₹1,00,000}$ (Below minimum statutory threshold), OR
   - If $\text{Max Credit Card Utilization} > 95\%$ (Severe credit distress).
2. **Manual Referral (`REFER`)**:
   - If $0.10 \le \text{PD} \le 0.20$, OR
   - If $\text{Credit-to-Income Ratio} > 4.0$ (Debt burden is 4× annual earnings), OR
   - If External Bureau Score is missing/insufficient.
3. **Automated Approval (`APPROVE`)**:
   - If $\text{PD} < 0.10$, AND
   - All income, leverage, and utilization checks pass.

---

## 3. The Home Credit Dataset & Feature Engineering

### The Dataset
The model is trained on the **Home Credit Default Risk** dataset (a real-world financial benchmark from Kaggle).
- **Total Applicants**: 307,511 loans.
- **Class Imbalance**: ~8.07% default rate (target `1`), ~91.93% repaid (target `0`).
- **Challenge**: The unbanked population often has little or no formal credit history.

### The 7 Relational Data Sources
The dataset originates from 7 relational tables that our pipeline engineers into a single consolidated feature space:
1. `application_train`: Demographic info, loan amount, annuity, income, education, external scores.
2. `bureau`: Historical credit lines reported by other financial institutions to credit bureaus.
3. `bureau_balance`: Monthly status of previous credits in the bureau.
4. `previous_application`: Past loan applications within Home Credit (approved, rejected, canceled).
5. `POS_CASH_balance`: Monthly balance snapshots of point-of-sale and cash loans.
6. `installments_payments`: Repayment history (days past due, late payment amounts).
7. `credit_card_balance`: Monthly credit card utilization and balance snapshots.

### Engineered Feature Space (88 Features)
Persisted in high-speed columnar Apache Parquet: `data/features/features_engineered.parquet`.
Key features include:
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`: Normalized credit bureau risk scores.
- `ext_source_mean`, `ext_source_std`, `ext_source_min`, `ext_source_max`: Statistical aggregations across bureau providers.
- `credit_to_income`: Total loan principal divided by annual income ($\text{AMT\_CREDIT} / \text{AMT\_INCOME\_TOTAL}$).
- `annuity_to_credit`: Payment burden per unit of credit ($\text{AMT\_ANNUITY} / \text{AMT\_CREDIT}$).
- `bureau_debt_to_credit_ratio`: Existing outstanding debt compared to available credit lines.
- `max_cc_utilization`: Highest credit card credit line utilization rate.
- `days_employed_to_days_birth`: Employment tenure as a fraction of applicant age.
- `prev_app_count`: Total number of past loan applications.

---

## 4. Machine Learning Architecture & Model Registry

The system employs a **Multi-Model Registry** pattern (`backend/services/model_service.py`), moving beyond a single hardcoded model into an extensible multi-model framework.

```
                     ┌────────────────────────────────┐
                     │     Model Service Registry     │
                     └───────┬──────────────┬─────────┘
                             │              │
      ┌──────────────────────┴──────┐       └─────────────────────┐
      ▼                             ▼                             ▼
┌──────────────┐             ┌──────────────┐              ┌──────────────┐
│   XGBoost    │             │   LightGBM   │              │   Logistic   │
│  Classifier  │             │  Classifier  │              │  Scorecard   │
│ (Depth-wise) │             │ (Leaf-wise)  │              │ (Linear/Std) │
└──────┬───────┘             └──────┬───────┘              └──────┬───────┘
       │                            │                             │
       └────────────────────┬───────┴─────────────────────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Virtual Chaining Pipelines│
              ├───────────────────────────┤
              │ 1. Weighted Ensemble      │
              │ 2. Two-Stage Cascade      │
              └───────────────────────────┘
```

### 1. Standalone Classifiers
1. **XGBoost Classifier (`xgboost_calibrated` — Production Champion)**:
   - Depth-wise histogram tree growth with bundled **Isotonic Regression Calibrator**.
   - Handles missing values natively through default split directions.
   - Class imbalance handled via `scale_pos_weight = 11.4` and post-hoc calibrated to empirical default frequency.
   - Artifact: `models/pd_xgboost_calibrated.joblib`.
2. **LightGBM Classifier (`lightgbm_calibrated`)**:
   - Leaf-wise tree growth with 255-bin histogram discretization and post-hoc Isotonic calibration.
   - Extremely low memory footprint and high inference throughput (0.0064 ms/row).
   - Artifact: `models/pd_lightgbm_calibrated.joblib`.
3. **Logistic Regression Scorecard (`logistic`)**:
   - Linear baseline using median imputation and standard scaling.
   - Meets traditional banking transparency and Basel compliance requirements.
   - Artifact: `models/pd_logistic.joblib`.

### 2. Chained Decision Pipelines
1. **Chained Weighted Ensemble (`chained_ensemble`)**:
   - Combines calibrated models using an optimized weighted soft-voting consensus:
     $$\text{PD}_{\text{ensemble}} = 0.45 \times \text{PD}_{\text{XGB\_cal}} + 0.45 \times \text{PD}_{\text{LightGBM\_cal}} + 0.10 \times \text{PD}_{\text{Logistic}}$$
   - Cancels individual model variance and tree discretization errors.
   - **Achieves the highest ROC-AUC (0.7789) and lowest Brier score (0.0663) across the entire project.**
2. **Two-Stage Cascade Hurdle (`cascade_hurdle`)**:
   - **Stage 1 (Triage Filter)**: Fast evaluation with Logistic Scorecard (0.0017 ms). If $\text{PD} < 3\%$ (unquestioned prime) or $\text{PD} > 35\%$ (clear reject), returns immediately.
   - **Stage 2 (Specialist Tree Assessment)**: For borderline applicants ($3\% \le \text{PD} \le 35\%$), routes to Calibrated XGBoost/LightGBM for deep non-linear interaction modeling.
   - Optimizes compute latency by ~40% while preserving high-precision decisioning.

### 3. Post-Hoc Probability Calibration (Basel II/III & IFRS 9)
In real-world credit underwriting, a model must not merely rank applicants; its outputs must represent **true statistical probabilities**:
- **The Imbalance Issue:** Training with `scale_pos_weight = 11.4` shifts raw tree outputs to an inflated average of **34.3%**, while the true population default rate is **8.07%**.
- **Isotonic Calibration:** Fitting an isotonic regression calibrator on a 20% holdout validation set reduced the Brier score from **0.1575 to 0.0665 (a 57.8% improvement)** and Log Loss from **0.4804 to 0.2401**, matching true empirical probabilities while fully preserving ranking power.

### 4. Empirical Validation Benchmarks
Evaluated on a stratified holdout validation set of **61,503 applicants**:

| Model / Pipeline | Type | ROC-AUC | PR-AUC | Brier Loss | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Chained Weighted Ensemble** | Multi-Model Blend | **0.7789** | **0.2699** | **0.0663** | Production Blend |
| **LightGBM (Calibrated)** | Gradient Boosted Trees | **0.7760** | 0.2665 | **0.0665** | Production Calibrated |
| **XGBoost (Calibrated)** | Gradient Boosted Trees | **0.7758** | 0.2660 | **0.0665** | Production Active Default |
| **LightGBM Classifier (Raw)** | Gradient Boosted Trees | 0.7754 | 0.2665 | 0.1671 | Baseline Tree |
| **XGBoost Classifier (Raw)** | Gradient Boosted Trees | 0.7745 | 0.2660 | 0.1575 | Baseline Tree |
| **Two-Stage Cascade Hurdle** | Sequential Gatekeeper | 0.7760 | 0.2665 | 0.0665 | Virtual Pipeline |
| **Logistic Scorecard** | Standardized Linear | 0.7574 | 0.2339 | 0.1989 | Basel Baseline |

---

## 5. Explainable AI (XAI) with SHAP

### Why Explainability is Legally Required
Under fair lending regulations, borrowers have a legal right to know why a loan was denied (Adverse Action Notices). Traditional tree models are "black boxes" without explainability tools.

### TreeSHAP Implementation
The system calculates exact Shapley values using Lundberg & Lee's **TreeSHAP** algorithm (`shap.TreeExplainer`):
- Measures the marginal contribution of each feature to pushing the log-odds of default above or below the base rate.
- For each applicant, the service extracts:
  - **Top 5 Risk-Increasing Factors**: Features that made the applicant look more likely to default (e.g., low external credit score, high credit card utilization, high annuity burden).
  - **Top 5 Risk-Reducing Factors**: Features that improved creditworthiness (e.g., high employment tenure, clean prior payment record, low debt ratio).
- For linear models (Logistic Scorecard), exact standardized coefficient products ($w_i \cdot x_i$) are calculated.

> [!IMPORTANT]
> **Governance Disclaimer**: SHAP values explain *model behavior and statistical associations*. They do **NOT** establish physical causality. The UI and API explicitly display this governance notice.

---

## 6. Regulatory Compliance & Policy RAG

### The Problem with Standalone LLMs in Compliance
Generic LLMs do not know internal credit policies or recent local regulations, and they frequently hallucinate legal sections.

### Policy Vector Knowledge Base
We embedded the complete **Reserve Bank of India (RBI) Guidelines on Digital Lending (September 2022)** into an embedded vector database:
- **Vector Database**: ChromaDB (`policies/chroma_db`).
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, running locally via SentenceTransformers).
- **Document Chunking**: Structured into semantic chunks with full metadata (Authority: RBI, Section, Page, Version Date, Title).

### Retrieval Process
When an underwriting decision is rendered:
1. A semantic query is dynamically formulated based on the decision context (e.g., borrower consent, credit evaluation, privacy, disclosure of APR).
2. ChromaDB retrieves top-$k$ relevant clauses.
3. The retrieved clauses are bundled into the audit record and presented in the UI under **Regulatory Evidence**.

---

## 7. Agentic AI Orchestration (LangGraph & Ollama)

### Role of the Agent
The agent acts as an **intelligent underwriter's copilot**, enabling human underwriters to ask natural-language questions like:
- *"Why was applicant 370920 approved?"*
- *"What are the biggest risk drivers for this borrower?"*
- *"Does this loan satisfy RBI digital lending guidelines on key fact statements?"*

### Architecture & Grounding
Built with **LangGraph** (`backend/services/agent_service.py`):
```
User Question ──► [ LangGraph Orchestrator ]
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
[ Tool 1: Underwrite ] [ Tool 2: SHAP ] [ Tool 3: Policy RAG ]
        │                │                │
        └────────────────┼────────────────┘
                         ▼
             [ Ollama / LLM Synthesis ]
                         ▼
               Grounded Response
```

- **Bound Tools**:
  1. `get_underwriting_result(applicant_id)`: Fetches PD, tier, and deterministic decision.
  2. `get_shap_explanation(applicant_id)`: Fetches top increasing/reducing factors.
  3. `search_rbi_policy(query)`: Fetches exact regulatory articles.
- **Zero-Hallucination Prompting**: The system prompt strictly prohibits inferring or inventing financial data not returned by the tools.
- **Deterministic Offline Fallback**: If Ollama is not running on `localhost:11434`, the system gracefully falls back to a structured rule-based summary without crashing.

---

## 8. End-to-End System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Next.js 16 UI Frontend                         │
│   ├── /dashboard         (Portfolio analytics & distributions)         │
│   ├── /applicants        (Searchable directory of 300k+ applicants)    │
│   ├── /underwriting/[id] (Full credit assessment & agentic chat)       │
│   ├── /model             (Registry, switching, benchmarks & sandbox)   │
│   └── /policies          (Interactive RBI regulatory RAG search)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST (JSON)
┌───────────────────────────────────▼────────────────────────────────────┐
│                         FastAPI Backend Engine                         │
│                                                                        │
│   ┌─────────────────────┐  ┌────────────────────┐  ┌─────────────────┐ │
│   │   Model Registry    │  │   Decision Engine  │  │   ChromaDB RAG  │ │
│   │ (XGB, LGB, Log, Ens)│  │ (Rules, Tiers, PD) │  │  (RBI Policies) │ │
│   └──────────┬──────────┘  └─────────┬──────────┘  └────────┬────────┘ │
│              │                       │                      │          │
│              └───────────────────────┼──────────────────────┘          │
│                                      ▼                                 │
│                     ┌─────────────────────────────────┐                │
│                     │  LangGraph / Ollama Agent Copilot│                │
│                     └─────────────────────────────────┘                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│   Data & Artifact Stores     │        │    Auditing & Persistence    │
│  - Parquet (307k applicants) │        │  - SQLite / Postgres DB      │
│  - Trained .joblib models    │        │  - Immutable audit logs      │
│  - Precomputed benchmarks    │        │  - Underwriting history      │
└──────────────────────────────┘        └──────────────────────────────┘
```

---

## 9. Project Directory & Codebase Walkthrough

```
Home_Credit/
├── backend/                       # FastAPI backend application
│   ├── main.py                    # Lifespan startup, router mounting, CORS
│   ├── config.py                  # Pydantic Settings & environment config
│   ├── db/
│   │   ├── database.py            # SQLAlchemy engine & session factory
│   │   └── models.py              # UnderwritingDecision & AuditLog ORM tables
│   ├── routes/
│   │   ├── health.py              # Health check endpoint
│   │   ├── applicants.py          # Paginated applicant profile retrieval
│   │   ├── underwriting.py        # POST /underwrite execution endpoint
│   │   ├── model_info.py          # Model registry, switching & comparison
│   │   ├── policy.py              # Vector search against RBI regulations
│   │   ├── agent.py               # Conversational AI agent endpoint
│   │   └── dashboard.py           # Aggregate portfolio risk statistics
│   ├── schemas/
│   │   ├── applicant.py           # Pydantic models for applicant profiles
│   │   ├── underwriting.py        # Request/response schemas for decisions
│   │   └── policy.py              # RAG search query & response models
│   ├── services/
│   │   ├── model_service.py       # Model registry, inference, ensembling
│   │   ├── underwriting_service.py# Workflow coordinator: PD + Rules + SHAP
│   │   ├── policy_service.py      # ChromaDB vector store interface
│   │   └── agent_service.py       # LangGraph agent with tool bindings
│   └── tests/                     # Comprehensive test suite (42 tests)
│       ├── test_applicants.py
│       ├── test_decision_engine.py
│       ├── test_health.py
│       ├── test_multi_models.py   # Tests model switching & chaining
│       ├── test_policy.py
│       └── test_underwriting.py
│
├── frontend/                      # Next.js 16 App Router interface
│   ├── app/
│   │   ├── layout.tsx             # Root navigation shell & styling
│   │   ├── page.tsx               # Landing redirect
│   │   ├── dashboard/page.tsx     # Portfolio KPIs and risk tier distribution
│   │   ├── applicants/page.tsx    # Applicant explorer with instant search
│   │   ├── applicants/[id]/page.tsx# Applicant details and profile view
│   │   ├── underwriting/[id]/page.tsx# Detailed assessment, SHAP & AI chat
│   │   ├── model/page.tsx         # Model registry, comparisons & sandbox
│   │   └── policies/page.tsx      # RBI digital lending policy search UI
│   ├── lib/
│   │   └── api.ts                 # Typed API client for FastAPI
│   └── types/
│       └── index.ts               # TypeScript data interfaces
│
├── data/
│   ├── raw/                       # Original Home Credit CSV files
│   └── features/
│       ├── features_engineered.parquet  # 307k applicants x 88 features
│       └── models_benchmark.json        # Precomputed benchmark report
│
├── models/
│   ├── pd_model.joblib            # Trained XGBoost classifier
│   ├── lightgbm_pd_model.joblib   # Trained LightGBM classifier
│   ├── logistic_pd_model.joblib   # Trained Logistic Scorecard
│   └── metadata.json              # Model hyperparameters & feature list
│
├── policies/
│   ├── chroma_db/                 # Embedded ChromaDB vector index
│   └── rbi_digital_lending_guidelines_2022.pdf # Regulatory source document
│
├── src/
│   ├── feature_engineering.py     # Aggregates raw tables into Parquet
│   ├── train_pd_model.py          # Standalone XGBoost training
│   ├── train_multi_models.py      # Multi-model training & benchmarking suite
│   └── index_policies.py          # PDF chunking & vector indexing into Chroma
│
├── requirements.txt               # Python package dependencies
├── Dockerfile.backend             # Backend container spec
├── Dockerfile.frontend            # Frontend container spec
└── docker-compose.yml             # Full-stack container orchestration
```

---

## 10. API Reference & Schema Specifications

### Underwriting Endpoints

#### `POST /underwrite`
Runs the complete underwriting workflow for a specific applicant.
- **Request Body**:
  ```json
  {
    "applicant_id": 370920,
    "model_id": "chained_ensemble"
  }
  ```
  *(Note: `model_id` is optional; if omitted, the current active model is used).*
- **Response**:
  ```json
  {
    "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "applicant_id": 370920,
    "timestamp": "2026-09-03T15:46:58Z",
    "probability_of_default": 0.0349,
    "risk_tier": "LOW",
    "decision": "APPROVE",
    "policy_reason": "Low probability of default (3.49%) with satisfactory debt-to-income.",
    "risk_increasing_factors": [
      { "feature": "AMT_ANNUITY", "value": "24500", "shap": 0.041, "effect": "increases_risk", "description": "High loan repayment burden." }
    ],
    "risk_reducing_factors": [
      { "feature": "EXT_SOURCE_2", "value": "0.684", "shap": -0.112, "effect": "reduces_risk", "description": "Strong external credit bureau rating." }
    ],
    "regulatory_evidence": [
      { "authority": "Reserve Bank of India", "title": "Digital Lending Guidelines", "content": "..." }
    ],
    "model_information": {
      "model_id": "chained_ensemble",
      "model_type": "Chained Weighted Ensemble",
      "model_version": "chained_ensemble_v1",
      "feature_count": 88
    },
    "policy_version": "v1.0"
  }
  ```

---

### Model Registry Endpoints

#### `GET /model/list`
Lists all registered models, their training status, and the current active model.
- **Response**:
  ```json
  {
    "active_model_id": "xgboost",
    "models": [
      { "id": "xgboost", "name": "XGBoost Classifier", "type": "Gradient Boosted Trees", "is_active": true, "roc_auc": 0.7745 },
      { "id": "lightgbm", "name": "LightGBM Classifier", "type": "Gradient Boosted Trees (Leaf-wise)", "is_active": false, "roc_auc": 0.7754 },
      { "id": "logistic", "name": "Logistic Scorecard", "type": "Standardized Linear Model", "is_active": false, "roc_auc": 0.7574 },
      { "id": "chained_ensemble", "name": "Chained Weighted Ensemble", "type": "Multi-Model Blend", "is_active": false, "roc_auc": 0.7777 },
      { "id": "cascade_hurdle", "name": "Two-Stage Cascade Hurdle", "type": "Sequential Gatekeeper", "is_active": false, "roc_auc": 0.7754 }
    ]
  }
  ```

#### `GET /model/comparison`
Returns comparative benchmark evaluation across all models.

#### `POST /model/active`
Switches the production active underwriting model system-wide.
- **Request Body**:
  ```json
  { "model_id": "chained_ensemble" }
  ```

---

## 11. How to Run, Train, Test, and Deploy

### 1. Prerequisites
- **Python 3.10+** (with virtual environment in `.venv`)
- **Node.js 18+** and npm
- *(Optional)* **Ollama** installed and running locally with `ollama run llama3` or `ollama run mistral`.

---

### 2. Starting the Backend Server
```powershell
# In the project root:
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --port 8002
```
- Interactive Swagger API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Starting the Frontend UI
```powershell
# In a second terminal:
cd frontend
npm install
npm run dev
```
- Web Application: [http://localhost:3000](http://localhost:3000)

---

### 4. Retraining Models & Generating Benchmarks
If you ever want to retrain the multi-model suite from scratch:
```powershell
.\.venv\Scripts\python.exe src/train_multi_models.py
```
This runs in ~40 seconds, trains LightGBM and Logistic Regression, evaluates all models on the 61k test set, and updates `data/features/models_benchmark.json`.

---

### 5. Running Automated Tests
```powershell
# Run the complete test suite (42 unit tests)
.\.venv\Scripts\python.exe -m pytest backend/tests/ -v
```

---

### 6. Running with Docker Compose
```powershell
docker-compose up --build
```
Spins up both the FastAPI backend and Next.js frontend in isolated containers.

---

## 12. Presentation & Interview Cheat Sheet

When presenting this project to professors, evaluators, or engineering interviewers, highlight these key design points:

### 1. "Why not just let an LLM decide the loan?"
> **Answer**: LLMs are non-deterministic, prone to hallucinations, cannot reliably compute probability distributions, and lack mathematical auditability. In our architecture, the **LLM is strictly an explainer and copilot**. The calculation of risk is done by gradient-boosted trees and calibrated regression, and the final decision is governed by deterministic credit policy rules.

### 2. "Why use an ensemble or cascade hurdle?"
> **Answer**: 
> - Standalone XGBoost and LightGBM models learn slightly different decision boundaries due to depth-wise vs. leaf-wise splitting. Our **Chained Weighted Ensemble** blends both with a linear scorecard, boosting the ROC-AUC to **0.7777** and PR-AUC to **0.2699** by canceling out individual model variance.
> - The **Two-Stage Cascade Hurdle** addresses operational cost: in a high-volume lending environment processing millions of requests, using a fast linear model for obvious approvals and rejections reduces server load by 40%, reserving computationally intensive tree evaluations for borderline applicants.

### 3. "How does the system prevent regulatory non-compliance?"
> **Answer**: Every assessment runs an automated similarity search against our embedded ChromaDB vector index containing official Reserve Bank of India digital lending guidelines. The system automatically retrieves and attaches relevant clauses (such as borrower consent, disclosure of APR, and fair debt collection practices) to the immutable audit record stored in SQLite/Postgres.

### 4. "What is the meaning of SHAP in this context?"
> **Answer**: SHAP (SHapley Additive exPlanations) provides local feature attribution based on cooperative game theory. It breaks down the exact difference between an applicant's predicted log-odds and the population base rate. We explicitly separate risk-increasing and risk-reducing factors to generate Adverse Action Notices while clearly warning users that statistical attribution does not imply physical causation.

---

## 13. Natural-Language Feature & Column Dictionary: What Every Column Means

In machine learning pipelines, features often have abbreviated, technical column names (like `max_pos_installments` or `ext_source_mean`). This section translates all primary features into plain, everyday English, explaining what each column measures, what the values represent, and how underwriters interpret them.

---

### How to Interpret the Two Factor Columns

In credit risk modeling, we separate feature influences into two distinct camps:

1. **Risk-Increasing Factors (Positive SHAP, Red)**:
   - **What it means**: When a feature has a positive SHAP value, it means this specific borrower's attribute made them appear **more risky** to the model than the average applicant.
   - **Examples**: High loan-to-income, missed installment due dates, high credit card utilization, or a high number of past loan rejections.
   - **Underwriter view**: *"These are the red flags or vulnerabilities that threaten loan recovery."*

2. **Risk-Reducing Factors (Negative SHAP, Green)**:
   - **What it means**: When a feature has a negative SHAP value, it means this attribute helped the applicant, pulling their predicted default probability **downwards** toward zero.
   - **Examples**: Strong credit bureau ratings, long employment stability, high historical installment counts without default, or low debt burdens.
   - **Underwriter view**: *"These are the compensating strengths that support loan approval."*

---

### Key Features Explained in Plain English

#### 1. Factors Shown in Underwriting Assessments (From the Screenshot)

- **`max_pos_installments` (Maximum POS Loan Term in Months)**
  - *Plain English*: The longest term (number of monthly payments) on previous point-of-sale or electronics store loans.
  - *Why it matters*: If an applicant had previous loans stretched out over 60 months (5 years), they committed to long-term monthly outflows. Long tenures can signal extended debt servitude, slightly increasing default risk (+SHAP).
  
- **`AMT_GOODS_PRICE` (Retail Price of Goods Financed)**
  - *Plain English*: The actual retail cost of the product (car, appliance, phone, furniture) being purchased with the loan.
  - *Why it matters*: Higher ticket goods require larger monthly commitments. If the goods price is high relative to income, it puts upward pressure on the risk score (+SHAP).

- **`total_previous_credit_amount` (Cumulative Credit Borrowed in Past)**
  - *Plain English*: Total cumulative money borrowed across all prior loans with the lender (e.g. ₹24,30,972).
  - *Why it matters*: Having borrowed large cumulative sums in the past shows high borrowing velocity. If past debt was substantial, the model checks whether the applicant is becoming overleveraged (+SHAP).

- **`NAME_FAMILY_STATUS` (Marital and Family Status)**
  - *Plain English*: Whether the applicant is Married, Single, in a Civil Marriage, or Divorced.
  - *Why it matters*: Household structure dictates financial responsibilities. Married applicants often share household expenses, but also have family dependency commitments.

- **`avg_payment_delay` (Average Payment Delay in Days)**
  - *Plain English*: The average number of days between an installment's official due date and the day the borrower actually made the payment.
  - *Why it matters*: Even minor positive delays (e.g. 0.41 days past due) indicate operational or cash flow friction. Habitual late payment, even by a few days, is a statistical warning sign for banks (+SHAP).

- **`NAME_EDUCATION_TYPE` (Highest Education Level)**
  - *Plain English*: The highest formal education completed (e.g., Secondary / secondary special, Higher education, Academic degree).
  - *Why it matters*: Higher educational credentials strongly correlate with employment resilience and income growth, significantly lowering default risk (-SHAP).

- **`ext_source_mean` (Average External Credit Bureau Score)**
  - *Plain English*: The average credit score from external credit rating agencies (similar to CIBIL, Experian, or Equifax).
  - *Why it matters*: A score of 0.67 out of 1.0 indicates a prime, well-vetted borrower with a clean history at other banks. This is the single strongest factor pulling risk down (-SHAP).

- **`installment_count` (Total Past Installments Paid)**
  - *Plain English*: The total number of monthly loan installments the borrower has successfully paid over their lifetime (e.g. 156 payments).
  - *Why it matters*: A borrower who has successfully paid 156 installments has a proven, multi-year track record of paying on time. This is concrete empirical proof of repayment reliability (-SHAP).

- **`AMT_ANNUITY` (Monthly Loan Installment / EMI)**
  - *Plain English*: The required monthly payment for the requested loan (e.g. ₹11,826 per month).
  - *Why it matters*: When the monthly payment is modest and easily covered by verified income, it acts as a strong protective factor against default (-SHAP).

- **`OWN_CAR_AGE` (Vehicle Age in Years)**
  - *Plain English*: The age of the applicant's automobile (e.g. 10 years).
  - *Why it matters*: Owning a vehicle (even an older one) confirms asset ownership and personal mobility. Car owners have tangible assets that buffer against extreme poverty or liquidity crises (-SHAP).

---

#### 2. Credit Bureau & Financial Capacity Columns

- **`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`**:
  - *Plain English*: Normalized external credit scores from three independent credit bureaus. Scores are between 0.0 (worst) and 1.0 (best). Scores above 0.50 indicate safe credit; scores below 0.30 indicate adverse credit history.
- **`credit_to_income`**:
  - *Plain English*: Total loan amount divided by annual income. If an applicant earning ₹5,00,000 requests a ₹20,00,000 loan, their ratio is 4.0x. Ratios above 3.5x to 4.0x are considered high leverage.
- **`annuity_to_income`**:
  - *Plain English*: Monthly debt payment divided by monthly income (Debt-to-Income / DTI ratio). Indicates what percentage of the borrower's paycheck is immediately eaten up by loan payments.
- **`bureau_debt_to_credit_ratio`**:
  - *Plain English*: Total outstanding debt across other banks divided by total credit limits. A ratio of 0.80 means the borrower has used 80% of their available credit capacity across the banking system.
- **`max_cc_utilization`**:
  - *Plain English*: The peak percentage of credit card limit ever used. If a credit card limit is ₹1,00,000 and the borrower had a ₹95,000 balance, utilization is 95%. Max utilization above 90% is a critical distress signal.
- **`DAYS_BIRTH`**:
  - *Plain English*: Applicant's age recorded as negative days. For example, `-14600` days corresponds to $14600 / 365 = 40$ years old.
- **`DAYS_EMPLOYED`**:
  - *Plain English*: Days at current employer. Negative numbers represent active employment (e.g. `-1825` days = 5 years at the job). A value of `365243` is a special dataset code denoting pensioners or unemployed individuals.

