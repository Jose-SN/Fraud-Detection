from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import logging
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import pandas as pd

# Load environment variables
load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load model and scaler
try:
    model = joblib.load("best_fraud_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model or scaler: {e}")
    raise

# MongoDB setup
try:
    mongo_client = MongoClient(MONGODB_URL)
    db = mongo_client["fraud_detection"]
    prediction_collection = db["predictions"]
    logging.info("Connected to MongoDB successfully.")
except Exception as e:
    logging.error(f"MongoDB connection failed: {e}")
    raise

# FastAPI app setup
app = FastAPI(title="Fraud Detection API")

# Enable CORS (useful for PowerBI / frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
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

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API is healthy"}

@app.get("/")
def root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Format data
        # input_data = np.array([[  # You may update fields based on full model
        #     transaction.TransactionAmt,
        #     transaction.card1,
        #     transaction.card2,
        #     transaction.dist1,
        #     transaction.C1,
        #     transaction.C2,
        #     transaction.D1,
        #     transaction.V1,
        #     transaction.V2
        # ]])
        # input_data = np.array([[getattr(transaction, f) for f in features]])
        input_data = pd.DataFrame([[getattr(transaction, f) for f in features]], columns=features)
        # Scale input
        scaled_input = scaler.transform(input_data)

        # Predict
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]
        timestamp = datetime.now(timezone.utc).isoformat()


        # Store to MongoDB
        record = {
            "timestamp": timestamp,
            "input": transaction.model_dump(),
            "is_fraud": int(pred),
            "fraud_probability": round(prob, 4)
        }
        prediction_collection.insert_one(record)

        logging.info(f"Prediction stored in MongoDB - Fraud: {bool(pred)}, Prob: {round(prob, 4)}")

        return {
            "is_fraud": bool(pred),
            "fraud_probability": round(prob, 4),
            "timestamp": timestamp
        }

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")
