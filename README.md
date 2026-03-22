# 🏥 Sentinel Health AI
### Clinical Risk Prediction Platform · MIMIC-III · XGBoost · LLM · Docker

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC_0.88-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

A full-stack clinical AI platform that predicts patient risk using real MIMIC-III ICU data, explains decisions via SHAP, generates AI clinical explanations using a local LLM, and exports professional PDF reports — all running in a single Docker command.

> **ROC-AUC: 0.8847** — above the typical published benchmark for this task on MIMIC-III (0.75–0.85)

---

## 📸 Screenshots

| Dashboard | Patient Profile |
|---|---|
| 46,520 real patients, live risk stats | ICU assignment, risk bar, SHAP features |

---

## ✨ Features

- **Risk Prediction** — XGBoost model with 11 clinical features including Charlson Comorbidity Index
- **SHAP Explainability** — every prediction explained feature by feature
- **Privacy-First LLM** — Llama 3.2 via Ollama, no data leaves the machine (HIPAA-principled)
- **RAG Chatbot** — LangChain + ChromaDB querying real clinical notes
- **Ward Assignment** — ML-based ICU / MICU / Private / General prediction
- **PDF Reports** — 2-page professional clinical reports with discharge planning
- **Live Dashboard** — real-time stats, clickable risk tables, Chart.js distribution
- **Docker** — one command deployment

---

## 🏗 Architecture

```
MIMIC-III Data (PhysioNet)
        ↓
PostgreSQL 15
        ↓
Feature Engineering → XGBoost + SHAP → Risk Scores
        ↓
FastAPI Backend → Web Dashboard (HTML/JS/Chart.js)
        ↓
Ollama (llama3.2) → AI Explanations + Discharge Plans
        ↓
LangChain + ChromaDB → RAG Chatbot on Clinical Notes
        ↓
ReportLab → PDF Clinical Reports
        ↓
Docker Compose → Everything Containerized
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | **0.8847** |
| Cross-Val Mean | 0.887 ± 0.02 |
| Training Data | 46,520 MIMIC-III patients |
| Features | 11 (including Charlson CCI) |

### Top Features by Importance
```
total_icu_hours        31.7%  ← strongest predictor
has_previous_admission 21.4%
num_diagnoses          11.1%
admission_count         9.9%
age                     6.7%
num_icu_stays           6.3%
```

### Feature Engineering Journey
```
Before: 4 basic features  → ROC-AUC 0.59 (barely above random)
After:  11 features + CCI → ROC-AUC 0.88 (above published benchmarks)
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 15 |
| ML Model | XGBoost + SHAP |
| Backend | FastAPI + SQLAlchemy |
| Frontend | HTML + CSS + Chart.js |
| LLM | Ollama + Llama 3.2 (local) |
| RAG | LangChain + ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| PDF | ReportLab |
| Container | Docker Compose |
| Data | MIMIC-III v1.4 (PhysioNet) |

---

## 📁 Project Structure

```
sentinel-health-ai/
├── api/
│   ├── main.py                  ← Web dashboard UI
│   ├── routers/
│   │   ├── patients.py          ← Patient endpoints
│   │   └── chat.py              ← RAG chatbot
│   └── services/
│       ├── ml_service.py        ← DB queries + risk data
│       ├── genai_service.py     ← LLM + RAG + timeline
│       └── pdf_service.py       ← PDF generation
├── ml/
│   ├── train.py                 ← XGBoost training
│   ├── train_enhanced.py        ← Enhanced model (CCI)
│   ├── predict.py               ← SHAP + predictions
│   └── ward_model.py            ← Ward classifier
├── data_ingestion/
│   ├── ingest.py                ← Synthetic data (testing)
│   ├── mimic_ingest.py          ← MIMIC-III loader
│   ├── feature_engineering.py  ← Charlson CCI + age + ICU
│   └── migrate_schema.py        ← Schema migrations
├── genai/
│   ├── risk_explainer.py        ← Terminal LLM explainer
│   └── rag_chatbot.py           ← Terminal RAG chatbot
├── models/                      ← Trained model files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose
- Ollama installed locally
- MIMIC-III access (see below)

### 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/sentinel-health-ai.git
cd sentinel-health-ai
```

### 2 — Start Ollama Locally
```bash
ollama pull llama3.2
ollama serve
```

### 3 — Start Docker
```bash
docker compose up -d
```

### 4 — Get MIMIC-III Data
You need PhysioNet credentials to access MIMIC-III.
See the [MIMIC-III Access Guide](#mimic-iii-access) below.

Once you have access, download and place files:
```
data/mimic/PATIENTS.csv.gz
data/mimic/ADMISSIONS.csv.gz
data/mimic/NOTEEVENTS.csv.gz
data/mimic/DIAGNOSES_ICD.csv.gz
data/mimic/ICUSTAYS.csv.gz
```

### 5 — Load Data + Train Models
```bash
conda activate healthcare

# Load MIMIC-III data
python data_ingestion/migrate_schema.py
python data_ingestion/mimic_ingest.py

# Feature engineering (Charlson CCI + age + ICU)
python data_ingestion/feature_engineering.py

# Train enhanced model
python ml/train_enhanced.py
python ml/predict.py
python ml/ward_model.py
```

### 6 — Rebuild Docker with Real Data
```bash
pg_dump --no-owner --no-privileges healthcare_db > mimic_clean_backup.sql
docker compose down
docker compose build
docker compose up -d
sleep 15
docker exec -i healthcare_postgres psql -U postgres -d healthcare_db < mimic_clean_backup.sql
```

### 7 — Open Dashboard
```
http://localhost:8000
```

---

## 🔑 MIMIC-III Access

MIMIC-III is a restricted dataset. To access it:

1. Create an account at [PhysioNet](https://physionet.org)
2. Complete the [CITI Data or Specimens Only Research](https://physionet.org/about/citi-course/) training
3. Sign the [MIMIC-III Data Use Agreement](https://physionet.org/content/mimiciii/1.4/)
4. Download files from the [MIMIC-III v1.4 page](https://physionet.org/content/mimiciii/1.4/)

> ⚠️ MIMIC-III data is not included in this repository. You must obtain your own access.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard |
| GET | `/health` | Health check |
| GET | `/patients/` | List patients (tier filter) |
| GET | `/patients/{id}/risk` | Risk score + features |
| GET | `/patients/{id}/explain` | LLM clinical explanation |
| GET | `/patients/{id}/ward` | Ward + discharge timeline |
| GET | `/patients/{id}/report` | PDF download |
| POST | `/chat/` | RAG chatbot |

---

## 💊 Charlson Comorbidity Index

The CCI is a clinical scoring system developed in 1987 that assigns weights to 17 conditions based on ICD-9 diagnosis codes:

| Condition | Weight |
|---|---|
| Myocardial Infarction | 1 |
| Congestive Heart Failure | 1 |
| Diabetes (uncomplicated) | 1 |
| Diabetes (with complications) | 2 |
| Cancer | 2 |
| Renal Disease | 2 |
| Metastatic Cancer | 6 |
| AIDS | 6 |
| ... 9 more conditions | 1–3 |

Higher score = higher comorbidity burden = higher risk.

---

## 🔒 Privacy & Ethics

- **Local LLM only** — Llama 3.2 runs via Ollama, no data sent to external APIs
- **HIPAA-principled** — no patient data leaves the machine
- **CITI Certified** — Human Subjects Research (95% score)
- **PhysioNet Credentialed** — authorized researcher
- **SHAP explainability** — no black box decisions

---

## 📋 Run Commands Reference

```bash
# Start everything
docker compose up -d

# Restore data after restart
docker exec -i healthcare_postgres psql \
  -U postgres -d healthcare_db < mimic_clean_backup.sql

# Stop
docker compose down

# Force rebuild
docker compose down
docker rmi sentinel-health-ai-api
docker compose build --no-cache
docker compose up -d

# Check logs
docker compose logs api --tail=30

# Database access
docker exec -it healthcare_postgres psql -U postgres -d healthcare_db
```

---

## 🗺 Roadmap

- [ ] Authentication (JWT)
- [ ] Live ward census (bed occupancy)
- [ ] Real-time admission tracking
- [ ] Lab values as features (ROC-AUC target: 0.92+)
- [ ] AWS/Azure deployment
- [ ] More MIMIC tables (procedures, prescriptions)

---

## 🎓 Credentials

- **CITI Certified** — Human Subjects Research Investigators (95%)
- **PhysioNet Credentialed Researcher**
- **MS Data Science** — University of Massachusetts Dartmouth

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

> ⚠️ This project is for educational and research purposes only. Not for clinical use. Always consult a qualified healthcare professional.

---

## 🤝 Author

**Nitish Bhattad**
MS Data Science, University of Massachusetts Dartmouth

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/yourusername)
