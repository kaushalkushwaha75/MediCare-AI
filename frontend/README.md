# MediCare AI - Frontend UI

Standalone modern web interface for the MediCare AI Machine Learning Disease Prediction & Clinical Triage System.

## Features
- **Interactive Symptom Selector**: Searchable, category-filtered grid with chip tray tags.
- **Demographics & Severity Control**: Custom range sliders and patient demographic inputs.
- **Inference Dashboard**: Visual gauge SVG match confidence indicator, triage badges, recommended precautions, emergency warnings, and differential diagnosis percentage bar charts.
- **Theme Toggle**: Built-in Dark / Light mode UI.
- **Clinical Report Export**: One-click printable health summary.

## Running the Frontend

Since the frontend is decoupled static HTML/CSS/JS, you can run it using any static file server:

### Option 1: VS Code Live Server (Recommended)
1. Open the `frontend/` folder in VS Code.
2. Right-click `index.html` and click **"Open with Live Server"**.
3. Browser opens automatically (usually `http://127.0.0.1:5500` or `http://127.0.0.1:5501`).

### Option 2: Python HTTP Server
```bash
cd frontend
python -m http.server 5500
```
Open `http://127.0.0.1:5500` in your web browser.

### Option 3: Direct Browser File Access
Simply double click `frontend/index.html` to open it directly in Google Chrome, Microsoft Edge, or Firefox.

> **Note:** Ensure the backend API is running on port 8000 (`cd backend && python -m uvicorn app:app --port 8000`).
