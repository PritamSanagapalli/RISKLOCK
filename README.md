# 🔒 RISKLOCK

> A Conservative Loan-Risk ML System — Self-Validating CI/CD with Explainability

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-Automated-blue)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**RISKLOCK** is a production-oriented ML system for loan default risk that enforces strict data governance, blocks unsafe training/deployment, and provides auditable, explainable predictions. Built for finance teams that cannot tolerate silent failures or model drift.

---

## 📋 Table of Contents

- [Why RISKLOCK](#why-risklock)
- [Design Principles](#design-principles)
- [What Makes It Unique](#what-makes-it-unique)
- [Architecture](#architecture)
- [Dataset & Target Policy](#dataset--target-policy)
- [Data Quality & Safety Gates](#data-quality--safety-gates)
- [Pipeline Stages](#pipeline-stages)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Artifacts & Auditability](#artifacts--auditability)
- [Testing & CI Outcomes](#testing--ci-outcomes)
- [Presenting This Project](#presenting-this-project)
- [Next Steps](#next-steps)

---

## 🎯 Why RISKLOCK

Financial ML projects must be engineered differently from research prototypes. A model that performs well offline but is trained on low-quality or leaked data can cause catastrophic business and regulatory outcomes. 

**RISKLOCK's primary objective is safety**: to ensure that training and deployment are policy-driven operations that only occur when data and model quality meet explicit, auditable standards.

---

## 🛡️ Design Principles

- **Zero-trust data**: Treat all incoming data as untrusted until validated
- **Conservative policy**: Thresholds err on the side of rejecting suspicious inputs
- **Fail-fast, fail-loud**: A single critical violation aborts training (CI fails)
- **Explainability first**: Predictions must be interpretable and carry provenance metadata
- **CI/CD controls ML outcomes**: Model artifacts and deployment are gated by ML tests, not by human whim

---

## ✨ What Makes It Unique

| Feature | Description |
|---------|-------------|
| 🚦 **Guarded Training** | Not every commit triggers a model build — only validated data does |
| 🏛️ **Governance Baked into CI** | Data quality and model metrics determine pipeline success |
| 📊 **Audit Artifacts** | Data hash, quality report, and metrics stored for every run |
| 🔍 **Built-in Explainability** | Every prediction includes model version, data quality score, and top contributing factors |

---

## 🏗️ Architecture

```
commit → GitHub Actions → (lint, tests, data_quality)
    ├─ if data_quality PASS → train → evaluate
    │    ├─ if evaluate PASS → build docker → publish artifacts
    │    └─ else → abort (CI fail)
    └─ else → abort (CI fail)

deployed container → FastAPI → inference + explanations + metadata
```

---

## 📊 Dataset & Target Policy

### Dataset
**Lending Club loan CSV** — Place raw file at `data/raw/loan.csv`  
⚠️ **DO NOT commit raw data to Git**

### Target Policy (Conservative & Leak-Safe)

| Loan Status | Label | Action |
|-------------|-------|--------|
| Fully Paid | 0 | Non-default |
| Charged Off | 1 | Default |
| Default | 1 | Default |
| Current, Late, In Grace Period, etc. | — | **Excluded from training** |

**Rationale**: Exclude in-flight loans to prevent label leakage and future outcome contamination.

---

## 🚨 Data Quality & Safety Gates

These checks run **before any training step** and are implemented in `core/data_quality.py`. Any **FAIL** causes immediate CI abort.

| Check | Threshold | Action |
|-------|-----------|--------|
| Missing values (any feature) | > 10% | ❌ Abort |
| Class imbalance (dominant class) | > 85% | ❌ Abort |
| Duplicate rows | > 2% | ❌ Abort |
| Zero-variance feature | any | ❌ Abort |
| Outlier rate (IQR-based) | > 12% | ⚠️ Warn |
| Financial sanity | any violation | ❌ Abort |

**Financial sanity checks**:
- `annual_inc ≤ 0`
- `loan_amnt ≤ 0`
- `loan_amnt > annual_inc × 10`

> ⚠️ **Important**: Abort is non-negotiable. Warnings are recorded but allow training to proceed.

### Data Quality Score

A single scalar to summarize data health:

```
QualityScore = 100 
  - 2 × (missing_rate_pct) 
  - 3 × max(0, (dominant_class_pct - 50)) 
  - 1 × (outlier_rate_pct)
```

Stored in `artifacts/data_quality.json` with status, rows used, and rule violations.

---

## 🔄 Pipeline Stages

1. **Lint & Format** — Static checks (ruff/flake8)
2. **Unit Tests** — Fast validation of core logic
3. **Data Quality Checks** (`core/data_quality.py`)
   - Schema validation, cleansing, gating
   - If FAIL → CI stops (no training)
4. **Training** (`core/train.py`)
   - Deterministic (seeded)
   - Uses saved sklearn Pipeline for features + model
5. **Evaluation** (`core/evaluate.py`)
   - Computes ROC-AUC, KS, calibration
   - Compares to baseline
   - If thresholds not met → CI fails; model saved but deployment blocked
6. **Package** — Build Docker image only if evaluation passes
7. **Artifacts** — Upload model, metrics, and data quality JSON for audit

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment tool
- Lending Club dataset

### Installation

1. **Create and activate virtual environment**:

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS / Linux
source venv/bin/activate

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Place dataset**:

Put the Lending Club CSV at `data/raw/loan.csv`  
⚠️ **DO NOT commit this file**

### Running the Pipeline

**Run data quality checks**:
```bash
python core/data_quality.py
# On success: artifacts/data_quality.json created and printed
# On failure: non-zero exit and JSON contains violations
```

**Run training** (after data passes):
```bash
python core/train.py
```

**Run evaluation** (after training):
```bash
python core/evaluate.py
# artifacts/metrics.json created
```

**Run FastAPI locally**:
```bash
uvicorn app.main:app --reload --port 8000
# GET /health
# POST /predict with validated payload
```

---

## 📁 Repository Structure

```
RISKLOCK/
├── app/
│   ├── main.py              # FastAPI entry (loads model + exposes endpoints)
│   └── predict.py           # Inference + explainability utilities
├── core/
│   ├── data_quality.py      # Data validation, gating (critical)
│   ├── ingestion.py         # Safe data loaders
│   ├── features.py          # sklearn Pipeline: transforms (persisted)
│   ├── train.py             # Deterministic training
│   ├── evaluate.py          # Metric computation and gating
│   └── registry.py          # Model provenance management
├── artifacts/               # model.joblib, data_quality.json, metrics.json
├── tests/                   # Unit & integration tests
├── .github/
│   └── workflows/
│       └── ci.yml           # CI pipeline with ML gates
├── Dockerfile               # Production container
├── requirements.txt
└── README.md
```

---

## 📦 Artifacts & Auditability

All artifacts are uploaded by CI for traceability and rollback:

- **`artifacts/data_quality.json`** — Full quality report (status, score, violations)
- **`artifacts/metrics.json`** — Evaluation metrics and baseline comparison
- **`artifacts/model.joblib`** — Serialized pipeline (transformers + model)

---

## 🧪 Testing & CI Outcomes

- **Unit tests** validate parsing, numeric conversions, and gate logic
- **Data quality** intentionally fails with clear messages on real problems:
  - Schema mismatch
  - Unacceptable missing rate
  - Invalid finance values
- **Evaluation step** must fail the pipeline (non-zero exit) when metrics underperform thresholds — this prevents unsafe builds

---

## 💼 Presenting This Project

### Resume Bullets

- Built **RISKLOCK**: a conservative loan-risk ML system that automatically aborts training on unsafe data and blocks deployment when model metrics degrade
- Implemented **CI/CD ML gating**: data validation and evaluation steps determine CI success, ensuring only auditable, safe models are containerized and published
- Delivered **explainable inference**: per-loan risk drivers, model versioning, and data-quality provenance included in API responses

### Interview Talking Points

- **Why exclude Current loans?** → Future leakage / label certainty
- **Trade-offs**: Logistic regression for interpretability vs boosting for raw accuracy
- **How CI gating prevents silent production incidents**
- **How to extend the system** to monitoring and automated retraining with drift detection

---

## 🔮 Next Steps & Extension Ideas

- [ ] Add drift detection (monthly or streaming) and automatic alerts
- [ ] Champion–challenger framework to validate new models in production
- [ ] Add calibration monitoring and re-calibration jobs
- [ ] Replace local artifacts storage with a secure model registry
- [ ] Add role-based access control and audit logs for governance

---

## 📝 Final Notes

> **Do not commit raw data.** Add `data/` to `.gitignore`.

> Keep thresholds and policy constants under a single config (future config file) so business owners can adjust policy without code changes.

> This project is intentionally conservative — that is the point. It demonstrates production discipline and governance, which is what financial teams value.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/RISKLOCK/issues).

---

**Made with ❤️ for production-grade ML systems**
