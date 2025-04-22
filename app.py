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
import traceback


# Load environment variables
load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")

# Set up logging
# logging.basicConfig(
#     filename="app.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,
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
        # Prepare input data from features list
        input_data = np.array([[getattr(transaction, f) for f in features]])

        # Scale input
        scaled_input = scaler.transform(input_data)

        # Predict
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        # 🧠 Adjust threshold
        fraud_threshold = 0.3  # You can tune this
        is_fraud = prob > fraud_threshold

        timestamp = datetime.now().astimezone().isoformat()

        # Store to MongoDB
        record = {
            "timestamp": timestamp,
            "input": transaction.model_dump(),  # for Pydantic v2+
            "is_fraud": int(is_fraud),
            "fraud_probability": round(prob, 4),
            "threshold_used": fraud_threshold
        }
        prediction_collection.insert_one(record)

        # Log the output for debugging
        logging.info(f"Prediction: {is_fraud} (Prob: {round(prob, 4)}, Threshold: {fraud_threshold})")

        return {
            "is_fraud": int(is_fraud),
            "fraud_probability": round(prob, 4),
            "threshold_used": fraud_threshold,
            "timestamp": timestamp
        }

    except Exception as e:
        logging.error(f"Prediction failed: {e}", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Prediction error")
