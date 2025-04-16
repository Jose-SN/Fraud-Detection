from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import uvicorn

# Load your model
with open("best_fraud_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI instance
app = FastAPI(title="Fraud Detection API")

# Define request schema
class Transaction(BaseModel):
    TransactionAmt: float
    card1: int
    card2: int
    dist1: float = 0
    C1: int
    C2: int
    D1: float = 0
    V1: float
    V2: float

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is live"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = [[
        transaction.TransactionAmt,
        transaction.card1,
        transaction.card2,
        transaction.dist1,
        transaction.C1,
        transaction.C2,
        transaction.D1,
        transaction.V1,
        transaction.V2
    ]]
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(probability, 4)
    }

# Run API
# uvicorn main:app --reload  ← Use this in terminal to run
