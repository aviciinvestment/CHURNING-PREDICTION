from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import pandas as pd
import logging
import json
from datetime import datetime
import time
import asyncio
from typing import List
import os

import psycopg2
import joblib
import tensorflow as tf

# ================= APP =================
app = FastAPI()

# ================= RATE LIMIT =================
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ================= ENV =================
API_KEY = os.getenv("API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL not set")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ================= GLOBALS =================
model = None
preprocessor = None
cache = {}

# ================= DB =================
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            input TEXT,
            prediction FLOAT,
            latency FLOAT,
            model_version TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ================= MODEL LOADING =================
def load_model():
    global model, preprocessor

    if model is None:
        model = tf.keras.models.load_model("tf_model.h5")

    if preprocessor is None:
        preprocessor = joblib.load("preprocessing.pkl")

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)

def log_event(event, data):
    logging.info(json.dumps({
        "event": event,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }))

# ================= AUTH =================
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ================= SAVE TO DB =================
def save_prediction(data, result, latency):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predictions (input, prediction, latency, model_version, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        str(data),
        float(result),
        float(latency),
        "v1",
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

# ================= INPUT SCHEMA =================
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    CreditScore: float = Field(..., ge=300, le=900)
    Age: int = Field(..., ge=18, le=100)
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography: str
    Gender: str

# ================= ROOT =================
@app.get("/")
def home():
    return {"message": "API is running"}

# ================= ERROR HANDLER =================
@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests"}
    )

# ================= SINGLE PREDICTION =================
@app.post("/v1/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: UserInput, api_key: str = Depends(verify_api_key)):

    load_model()

    key = str(data.dict())

    if key in cache:
        return {"prediction": cache[key], "cached": True}

    df = pd.DataFrame([data.dict()])

    processed = preprocessor.transform(df)
    processed = processed.toarray() if hasattr(processed, "toarray") else processed

    start = time.time()
    prediction = await asyncio.to_thread(model.predict, processed)
    result = float(prediction.flatten()[0] >= 0.5)
    end = time.time()

    cache[key] = result

    log_event("prediction", data.dict())

    save_prediction(data.dict(), result, end - start)

    return {
        "prediction": result,
        "latency": round(end - start, 4)
    }

# ================= BATCH PREDICTION =================
@app.post("/v2/predict")
@limiter.limit("10/minute")
async def predict_batch(request: Request, data: List[UserInput], api_key: str = Depends(verify_api_key)):

    load_model()

    df = pd.DataFrame([d.dict() for d in data])

    processed = preprocessor.transform(df)
    processed = processed.toarray() if hasattr(processed, "toarray") else processed

    prediction = await asyncio.to_thread(model.predict, processed)

    results = [float(p >= 0.5) for p in prediction.flatten()]

    for i, item in enumerate(data):
        save_prediction(item.dict(), results[i], 0)

    log_event("batch_prediction", {"count": len(results)})

    return {"predictions": results}

# ================= ANALYTICS =================
@app.get("/analytics")
def analytics():

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]

    cur.execute("SELECT AVG(latency) FROM predictions")
    avg_latency = cur.fetchone()[0]

    cur.execute("SELECT AVG(prediction) FROM predictions")
    avg_prediction = cur.fetchone()[0]

    conn.close()

    return {
        "total_requests": total,
        "average_latency": avg_latency,
        "average_prediction": avg_prediction
    }

# ================= DRIFT MONITORING (DAY 26 CORE) =================
@app.get("/drift")
def drift_detection():

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT AVG(latency) FROM predictions")
    avg_latency = cur.fetchone()[0]

    cur.execute("SELECT AVG(prediction) FROM predictions")
    avg_prediction = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]

    conn.close()

    drift_flag = False

    # SIMPLE DRIFT RULES
    if avg_latency and avg_latency > 1.0:
        drift_flag = True

    if avg_prediction and (avg_prediction < 0.3 or avg_prediction > 0.8):
        drift_flag = True

    return {
        "total_requests": total,
        "average_latency": avg_latency,
        "average_prediction": avg_prediction,
        "drift_detected": drift_flag
    }