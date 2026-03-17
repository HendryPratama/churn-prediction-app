# 📊 Telco Customer Churn Prediction System

An end-to-end Machine Learning application featuring an **XGBoost** model, a **FastAPI** backend, and a **Next.js** frontend.

## 🛠️ Tech Stack
- **ML:** Python, XGBoost, Scikit-Learn, Pandas
- **API:** FastAPI, Pydantic, Poetry
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS

## 🚀 Local Installation & Setup

Follow these steps to run the project on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/HendryPratama/churn-prediction-app.git](https://github.com/HendryPratama/churn-prediction-app.git)
cd churn-prediction-app
```

### 2. Backend Setup (Python)
Ensure you have Poetry installed.
# Install dependencies
poetry install

# Run the FastAPI server
poetry run uvicorn api.main:app --reload --port 8000

The API will be live at http://127.0.0.1:8000. You can view the docs at /docs.

3. Frontend Setup (Next.js)
Open a new terminal:
```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```
The UI will be live at http://localhost:3000.

### 📈 Model Information
Algorithm: XGBoost Classifier

Features: Tenure, Contract Type, Monthly Charges, Internet Service, etc.

Accuracy: ~80% (ROC-AUC 0.84)

### 📁 Project Structure
/api: FastAPI source code and schemas.

/models: Serialized XGBoost pipeline (.joblib).

/frontend: Next.js web application.

/notebooks: Jupyter notebooks for EDA and Model Training.

### Developed by Hendry Pratama
