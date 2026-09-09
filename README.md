# Agentic AI-Based Credit Underwriting and Loan Risk Assessment System

> **Final-Year B.Tech / Academic Research Project**  
> **DISCLAIMER:** Research and educational prototype only. This application is designed to demonstrate hybrid machine learning (XGBoost, LightGBM, Logistic Scorecards & Ensembles), explainable AI (TreeSHAP), agentic orchestration (LangGraph), and regulatory knowledge retrieval (RAG). It does **not** constitute official credit advice, legal advice, or an actual lending decision engine.

📖 **Documentation:**
- [docs/PROJECT_COMPLETE_GUIDE.md](docs/PROJECT_COMPLETE_GUIDE.md) — Credit risk theory, column dictionary, and interview talking points.
- [docs/DEVELOPER_HANDBOOK.md](docs/DEVELOPER_HANDBOOK.md) — Comprehensive developer walkthrough detailing every file, pipeline step, and gotchas.

---

## System Architecture Overview

This project implements a multi-tiered architecture for modern credit risk assessment:

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js 16 Frontend                      │
│      (Dashboard, Applicants, Underwriting, Policy, Model)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ REST / JSON (port 3000 -> 8000)
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Service                  │
│   ├── /underwrite    (Inference + SHAP + RAG + Decisions)   │
│   ├── /applicants    (Profile Retrieval from Parquet)       │
│   ├── /policy/search (Chroma Vector DB / MiniLM RAG)        │
│   ├── /model/info    (Metadata, Top Features, Metrics)      │
│   └── /agent/chat    (LangGraph + Ollama Agent Assistant)   │
└───────────────┬───────────────────────────────┬─────────────┘
                │                               │
        ┌───────▼───────┐               ┌───────▼────────┐
        │  Postgres/    │               │ Home Credit    │
        │  SQLite Audit │               │ Parquet +      │
        │  Trail DB     │               │ XGBoost Model  │
        └───────────────┘               └────────────────┘
```

---

## Key Features

1. **Deterministic Underwriting as Ground Truth**
   - Calculates Probability of Default (PD) using a trained XGBoost classifier on 88 features.
   - Categorizes risk into discrete tiers (`LOW`, `MODERATE`, `HIGH`, `VERY_HIGH`).
   - Evaluates rule-based research policy criteria (income threshold, credit-to-income, credit card utilization).
   - Generates decisions: `APPROVE`, `REFER`, `DECLINE`.

2. **Strict Evidence Segregation**
   - **Model Evidence:** Feature contributions computed via `TreeExplainer` (TreeSHAP) and labeled clearly as statistical association rather than causality.
   - **Regulatory Evidence:** Retrieved from the Reserve Bank of India (RBI) Guidelines on Digital Lending (2022) with explicit authority and source attribution.

3. **LangGraph Agentic Assistant**
   - Bound with custom tools (`get_underwriting_result`, `get_shap_explanation`, `search_rbi_policy`).
   - Grounded strictly in system outputs—cannot hallucinate applicant statistics or financial facts.
   - Graceful fallback to deterministic summaries when local LLMs (Ollama) are offline.

4. **Modern UI & Visualizations**
   - Built with Next.js (App Router), Tailwind CSS, and Recharts.
   - Risk gauges, SHAP impact bar graphs, decision analytics, and policy explorer.

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (active virtual environment in `.venv`)
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Start the FastAPI server
python -m uvicorn backend.main:app --reload --port 8002
```
- API Docs: `http://localhost:8002/docs`
- Health check: `http://localhost:8002/health`

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
- Frontend UI: `http://localhost:3001`

### 4. Running with Docker Compose
```bash
docker-compose up --build
```

---

## Testing & Validation

Run the automated test suite:
```bash
python -m pytest backend/tests/ -v
```

Included test suites:
- `test_health.py`: Verifies service status and lifecycle.
- `test_decision_engine.py`: Unit tests for rule branches and risk tiers.
- `test_applicants.py`: Tests profile parsing and validation.
- `test_underwriting.py`: End-to-end inference and regression checks on applicant `370920` (Expected PD ≈ 3.52%, `APPROVE`).
- `test_policy.py`: Vector retrieval and authority metadata validation.

---

## Regulatory and Academic Notice
- **Dataset:** Home Credit Default Risk.
- **Regulatory Guidance:** Reserve Bank of India (RBI) Guidelines on Digital Lending (2022).
- Neither this software nor its authors are affiliated with or endorsed by Home Credit or the Reserve Bank of India.