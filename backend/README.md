# MediCare AI - Backend REST API

FastAPI backend service powering the MediCare AI Machine Learning Disease Prediction & Clinical Triage system.

## Features
- **Symptom Catalog API (`GET /api/symptoms`)**: Returns categorized symptoms for patient selection.
- **Disease Prediction API (`POST /api/predict`)**: Infers probable diseases using a Random Forest ensemble model with differential confidence percentages, clinical triage levels, precautions, and specialist recommendations.
- **Health Check API (`GET /api/health`)**: Reports ML model status and loaded database records.
- **CORS Enabled**: Configured to work seamlessly with separate frontend clients (Live Server, React, Vue, HTML static files, etc.).

## Setup & Running

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. (Optional) Train / Re-train ML Model
```bash
python model/train_model.py
```

### 3. Start API Server
```bash
python -m uvicorn app:app --reload --port 8000
```

The API will be available at:
- **API Root**: `http://127.0.0.1:8000/`
- **Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`
