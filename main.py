from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import pandas as pd
import io
from pydantic import BaseModel, Field
import logging
import json
from datetime import datetime
import os
import time
import asyncio
from typing import List
import psycopg2

# ================= DATABASE =================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL not set")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    input TEXT,
    prediction FLOAT,
    timestamp TEXT
)
""")
conn.commit()

def load_model():
    global model, preprocessor
    import tensorflow as tf   # 👈 IMPORTANT: lazy import

    if model is None:
        model = tf.keras.models.load_model("tf_model.h5")

    if preprocessor is None:
        import joblib
        preprocessor = joblib.load("preprocessing.pkl")

# ================= APP =================
app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

API_KEY = os.getenv("API_KEY")

# ================= MODEL (LAZY LOAD FIX) =================
model = None
preprocessor = None


def load_model():
    global model, preprocessor
    import tensorflow as tf   # 👈 IMPORTANT: lazy import

    if model is None:
        model = tf.keras.models.load_model("tf_model.h5")

    if preprocessor is None:
        import joblib
        preprocessor = joblib.load("preprocessing.pkl")

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)

def log_event(event_type, data):
    logging.info(json.dumps({
        "event": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }))

# ================= CACHE =================
cache = {}

# ================= AUTH =================
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ================= SAVE =================
def save_prediction(data, result):
    cursor.execute(
        "INSERT INTO predictions (input, prediction, timestamp) VALUES (%s, %s, %s)",
        (str(data), float(result), datetime.utcnow().isoformat())
    )
    conn.commit()

# ================= MODEL INPUT =================
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
    return {"message": "API is working"}

# ================= RATE LIMIT =================
@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc):
    return JSONResponse(status_code=429, content={"error": "Too many requests"})

# ================= SINGLE PREDICTION =================
@app.post("/v1/predict")
@limiter.limit("10/minute")
async def predict_score(request: Request, data: UserInput, api_key: str = Depends(verify_api_key)):
    try:
        load_model()

        key = str(data.dict())

        if key in cache:
            return {"cached": cache[key]}

        df = pd.DataFrame([data.dict()])

        processed = preprocessor.transform(df)
        processed = processed.toarray() if hasattr(processed, "toarray") else processed

        start = time.time()
        prediction = await asyncio.to_thread(model.predict, processed)
        result = float(prediction.flatten()[0] >= 0.5)
        end = time.time()

        cache[key] = result

        log_event("prediction", data.dict())

        save_prediction(data.dict(), result)

        return {
            "prediction": result,
            "latency": end - start
        }

    except Exception as e:
        return {"error": str(e)}

# ================= BATCH =================
@app.post("/v2/predict")
@limiter.limit("10/minute")
async def predict_batch(request: Request, data: List[UserInput], api_key: str = Depends(verify_api_key)):
    try:
        load_model()

        df = pd.DataFrame([d.dict() for d in data])

        processed = preprocessor.transform(df)
        processed = processed.toarray() if hasattr(processed, "toarray") else processed

        prediction = await asyncio.to_thread(model.predict, processed)
        results = [float(p >= 0.5) for p in prediction.flatten()]

        for i, item in enumerate(data):
            save_prediction(item.dict(), results[i])

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}