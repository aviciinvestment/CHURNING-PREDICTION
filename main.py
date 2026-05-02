from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import pandas as pd
import io
import joblib
import tensorflow as tf
from pydantic import BaseModel, Field
import logging
import json
from datetime import datetime
import os
import time
import asyncio
from typing import List


import psycopg2

conn = psycopg2.connect(os.getenv("postgresql://churningapp_user:Xs0sb21PWtcQFIXAK4SmEhzZ7VLEOLVf@dpg-d7qr5mugvqtc73b26qtg-a/churningapp"))
cursor = conn.cursor()


print(os.listdir())
# ===================== APP INIT =====================
app = FastAPI()



limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ===================== CONFIG =====================
API_KEY = os.getenv("API_KEY")

BASE_DIR = os.getcwd()
model_path = os.path.join(BASE_DIR, "tf_model.h5")
preprocessor_path = os.path.join(BASE_DIR, "preprocessing.pkl")

# ===================== LOAD MODEL =====================
model = tf.keras.models.load_model(model_path)
preprocessor = joblib.load(preprocessor_path)

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO)

def log_event(event_type, data):
    log_data = {
        "event": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    logging.info(json.dumps(log_data))

# ===================== CACHE =====================
cache = {}

# ===================== AUTH =====================
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ===================== DATA SAVE =====================
def save_prediction(data, result):
    cursor.execute("""
        INSERT INTO predictions (input, prediction, timestamp)
        VALUES (?, ?, ?)
    """, (
        str(data),
        result,
        datetime.utcnow().isoformat()
    ))
    conn.commit()

# ===================== MODEL INPUT =====================
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

# ===================== ROOT =====================
@app.get("/")
def home():
    return {"message": "API is working"}

# ===================== RATE LIMIT HANDLER =====================
@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Slow down."}
    )

# ===================== SINGLE PREDICTION =====================
@app.post("/v1/predict")
@limiter.limit("10/minute")
async def predict_score(request: Request, data: UserInput, api_key: str = Depends(verify_api_key)):
    try:
        log_event("request_received", data.dict())

        input_key = str(data.dict())

        # CACHE CHECK
        if input_key in cache:
            log_event("cache_hit", data.dict())
            return {"predicted score": cache[input_key], "message": "cached result"}

        # PREPARE DATA
        new_data = pd.DataFrame({
            "CreditScore": [data.CreditScore],
            "Age": [data.Age],
            "Tenure": [data.Tenure],
            "Balance": [data.Balance],
            "NumOfProducts": [data.NumOfProducts],
            "HasCrCard": [data.HasCrCard],
            "IsActiveMember": [data.IsActiveMember],
            "EstimatedSalary": [data.EstimatedSalary],
            "Geography": [data.Geography],
            "Gender": [data.Gender]
        })

        new_processed = preprocessor.transform(new_data)
        new_processed = new_processed.toarray() if hasattr(new_processed, "toarray") else new_processed

        # PERFORMANCE TRACKING
        start_time = time.time()

        prediction = await asyncio.to_thread(model.predict, new_processed)

        result = float(prediction.flatten()[0] >= 0.5)

        cache[input_key] = result

        end_time = time.time()

        log_event("performance", {"latency_seconds": end_time - start_time})

        log_event("prediction_made", {
            "input": data.dict(),
            "result": result
        })

        save_prediction(data.dict(), result)

        cursor.execute(
            "INSERT INTO predictions (input, prediction, timestamp) VALUES (%s, %s, %s)",
            (str(data), result, datetime.utcnow().isoformat())
        )
        conn.commit()

        return {
            "predicted score": result,
            "message": "model version v1"
        }

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}

# ===================== BATCH PREDICTION =====================
@app.post("/v2/predict")
@limiter.limit("10/minute")
async def predict_batch(request: Request, data: List[UserInput], api_key: str = Depends(verify_api_key)):
    try:
        rows = []

        for item in data:
            rows.append({
                "CreditScore": item.CreditScore,
                "Age": item.Age,
                "Tenure": item.Tenure,
                "Balance": item.Balance,
                "NumOfProducts": item.NumOfProducts,
                "HasCrCard": item.HasCrCard,
                "IsActiveMember": item.IsActiveMember,
                "EstimatedSalary": item.EstimatedSalary,
                "Geography": item.Geography,
                "Gender": item.Gender
            })

        df = pd.DataFrame(rows)

        processed = preprocessor.transform(df)
        processed = processed.toarray() if hasattr(processed, "toarray") else processed

        prediction = await asyncio.to_thread(model.predict, processed)

        results = [float(p >= 0.5) for p in prediction.flatten()]

        for i, item in enumerate(data):
            save_prediction(item.dict(), results[i])

        log_event("batch_prediction", {"count": len(results)})

        cursor.execute(
            "INSERT INTO predictions (input, prediction, timestamp) VALUES (%s, %s, %s)",
            (str(data), results, datetime.utcnow().isoformat())
        )
        conn.commit()

        return {
            "predictions": results,
            "message": "model version v2"
        }

    except Exception as e:
        return {"error": str(e)}