# MediCare AI - Intelligent Disease Prediction & Clinical Triage System

MediCare AI is a machine learning powered healthcare triage platform. The repository is completely decoupled into independent **Backend** and **Frontend** services.

---

## 📁 Repository Architecture

```text
medicare/
├── backend/                  # FastAPI REST API & Machine Learning Engine
│   ├── app.py                # FastAPI entrypoint (Port 8000)
│   ├── requirements.txt      # Python dependencies
│   ├── README.md             # Backend setup & API docs
│   └── model/
│       ├── train_model.py    # Random Forest ML model training pipeline
│       ├── model.joblib      # Calibrated Random Forest model artifact
│       ├── symptom_encoder.joblib
│       ├── disease_info.json # Clinical knowledgebase
│       └── disease_dataset.csv
├── frontend/                 # Decoupled Web Application (HTML5/CSS3/Vanilla JS)
│   ├── index.html            # Main UI Dashboard
│   ├── README.md             # Frontend server options & instructions
│   ├── css/
│   │   └── style.css         # Modern styling system (Dark/Light mode)
│   └── js/
│       └── main.js           # Async API Controller
└── README.md                 # Main workspace documentation
```

---

## 🚀 Quick Start Guide

### Step 1: Start the Backend REST API
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8000
```
- API Health Status: `http://127.0.0.1:8000/api/health`
- Swagger Documentation: `http://127.0.0.1:8000/docs`

### Step 2: Launch the Frontend Web Interface
Open a second terminal window or use VS Code Live Server:

```bash
cd frontend
python -m http.server 5500
```
Open `http://127.0.0.1:5500` in your web browser.

---

## 🔬 Machine Learning Pipeline

- **Algorithm**: Random Forest Classifier with Calibrated Probabilities (`CalibratedClassifierCV`)
- **Accuracy**: 91.7% on test evaluation with 5-fold cross-validation
- **Re-training**: To update or retrain the model, run `python backend/model/train_model.py`.
