# 📊 Telco Customer Churn Prediction System

An end-to-end Machine Learning application that predicts the likelihood of customer attrition using **XGBoost**. This project features a high-performance **FastAPI** backend and a modern **Next.js** frontend, fully deployed as a monorepo on **Vercel**.

## 🚀 Live Demo
[Insert your Vercel URL here]

## 🛠️ Tech Stack
- **Modeling:** Python, XGBoost, Scikit-Learn, Pandas
- **API:** FastAPI, Pydantic, Joblib
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Deployment:** Vercel (Serverless Functions)
- **Environment:** Poetry (Dependency Management)

## 📈 Project Overview
This project addresses the classic "Churn" problem in the telecommunications industry. By analyzing customer behavior—such as contract type, tenure, and internet service—the system provides a real-time risk score (0-100%) to help businesses take proactive retention measures.

### Key Insights from EDA:
* **Contract Type:** Customers with "Month-to-month" contracts are the highest risk segment.
* **Fiber Optic:** Surprisingly, Fiber Optic users churned at a higher rate than DSL users, suggesting a potential service quality issue.
* **The Safety Zone:** Churn probability drops significantly after a customer reaches **24 months** of tenure.

## 🏗️ Architecture
The system is built with an MLOps-first mindset:
1. **Research Phase:** Exploratory Data Analysis and Model Training performed in VS Code Interactive Notebooks.
2. **Persistence:** The trained XGBoost pipeline (including scalers and encoders) is serialized using `joblib`.
3. **Serving:** A FastAPI server wraps the model, providing a `/predict` POST endpoint.
4. **Interface:** A responsive React dashboard allows users to input data and visualize the risk level (Low, Medium, High).



## 💻 Local Setup

### 1. Backend (FastAPI)
```bash
# Install dependencies
poetry install

# Run the server
poetry run uvicorn api.main:app --reload
```

### 2. Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

📊 Model Performance
Model: XGBoost Classifier

Metric: ROC-AUC Score: ~0.84

Handling Imbalance: Utilized scale_pos_weight to improve recall for the churned class (minority).

📄 License
Distributed under the MIT License.
