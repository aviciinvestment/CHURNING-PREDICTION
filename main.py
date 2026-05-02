from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import pandas as pd
from pydantic import BaseModel, Field
import logging
import json
from datetime import datetime
import time
import asyncio
from typing import List
import psycopg2
import os

# ================= PORT (RENDER FIX) =================
PORT = int(os.getenv("PORT", 8000))

# ================= APP =================
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ================= ENV =================
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")

if not DATABASE_URL:
    raise Exception("DATABASE_URL not set")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ================= DB CONNECT FUNCTION (SAFE) =================
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# ================= INIT DB =================
def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        input TEXT,
        prediction FLOAT,
        latency FLOAT DEFAULT 0,
        model_version TEXT DEFAULT 'v1',
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ================= MODEL =================
model = None
preprocessor = None

def load_model():
    global model, preprocessor

    import tensorflow as tf
    import joblib

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
        "time": datetime.utcnow().isoformat()
    }))

# ================= CACHE =================
cache = {}

# ================= AUTH =================
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ================= SAVE TO DB =================
def save_prediction(data, result, latency, model_version="v1"):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO predictions (input, prediction, latency, model_version, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        json.dumps(data),
        float(result),
        float(latency),
        model_version,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

# ================= INPUT =================
class UserInput(BaseModel):
    CreditScore: float = Field(..., ge=300, le=900)
    Age: int = Field(..., ge=18, le=100)
    Tenure: float = Field(..., ge=0, le=50)
    Balance: float = Field(..., ge=0)
    NumOfProducts: float = Field(..., ge=1, le=10)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)
    Geography: str
    Gender: str

# ================= ROOT =================
@app.get("/")
def home():
    return {"message": "API running 🚀"}

# ================= RATE LIMIT =================
@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc):
    return JSONResponse(status_code=429, content={"error": "Too many requests"})

# ================= SINGLE PREDICTION =================
@app.post("/v1/predict")
@limiter.limit("10/minute")
async def predict(data: UserInput, request: Request, api_key: str = Depends(verify_api_key)):

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
    latency = time.time() - start

    cache[key] = result

    save_prediction(data.dict(), result, latency)

    log_event("prediction", data.dict())

    return {
        "prediction": result,
        "latency": latency
    }

# ================= BATCH =================
@app.post("/v2/predict")
@limiter.limit("10/minute")
async def predict_batch(data: List[UserInput], request: Request, api_key: str = Depends(verify_api_key)):

    load_model()

    df = pd.DataFrame([d.dict() for d in data])
    processed = preprocessor.transform(df)
    processed = processed.toarray() if hasattr(processed, "toarray") else processed

    start = time.time()
    prediction = await asyncio.to_thread(model.predict, processed)
    latency = time.time() - start

    results = [float(p >= 0.5) for p in prediction.flatten()]

    for i, item in enumerate(data):
        save_prediction(item.dict(), results[i], latency)

    return {
        "predictions": results,
        "latency": latency
    }

# ================= ANALYTICS (DAY 28 UPGRADE) =================
@app.get("/analytics")
def analytics(api_key: str = Depends(verify_api_key)):

    conn = get_db()
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
        "avg_latency": avg_latency,
        "avg_prediction_rate": avg_prediction
    }

# ================= DRIFT DETECTION (UPGRADED) =================
@app.get("/drift")
def drift(api_key: str = Depends(verify_api_key)):

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT AVG(prediction) FROM predictions")
    avg_pred = cur.fetchone()[0] or 0.5

    conn.close()

    drift_score = abs(avg_pred - 0.5)

    return {
        "drift_score": drift_score,
        "status": "drift detected" if drift_score > 0.2 else "stable"
    }

@app.get("/logs")
def get_logs(api_key: str = Depends(verify_api_key)):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, input, prediction, latency, timestamp
        FROM predictions
        ORDER BY id DESC
        LIMIT 50
    """)
    
    rows = cur.fetchall()

    logs = []
    for r in rows:
        logs.append({
            "id": r[0],
            "input": r[1],
            "prediction": r[2],
            "latency": r[3],
            "timestamp": r[4]
        })

    return logs

# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)