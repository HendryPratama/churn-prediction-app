import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from src.schemas import CustomerData, PredictionResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Telco Churn Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the app starts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline_xgb.joblib")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Did you run the training notebook?")
    model = None

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Convert Pydantic model to Dictionary, then to DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # 2. Get Probability and Prediction
    # pipeline.predict_proba returns [[prob_0, prob_1]]
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(model.predict(input_df)[0])

    # 3. Logic for Risk Level
    if probability > 0.7:
        risk = "High"
    elif probability > 0.3:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
        "risk_level": risk
    }