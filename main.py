from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
import joblib
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
import logging
import json
from datetime import datetime
import os
import time
import asyncio

cache = {}

BASE_DIR = os.getcwd()
model_path = os.path.join(BASE_DIR, "tf_model.h5")
preprocessor_path = os.path.join(BASE_DIR, "preprocessing.pkl")

logging.basicConfig(level = logging.INFO)


def log_event(event_type, data):
    log_data = {
        "event": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    logging.info(json.dumps(log_data))










app = FastAPI()
model = tf.keras.models.load_model("tf_model.h5")
preprocessor = joblib.load("preprocessing.pkl")
class UserInput(BaseModel):
    CreditScore:float = Field(..., ge = 300,le = 900)
    Age:int = Field(..., ge = 18,le = 100)
    Tenure:float = Field(..., ge = 0,le = 50)
    Balance:float = Field(..., ge = 0)
    NumOfProducts:float = Field(..., ge = 1,le = 10)
    HasCrCard:int = Field(..., ge = 0,le = 1)
    IsActiveMember:int = Field(..., ge = 0,le = 1)
    EstimatedSalary:float = Field(..., ge = 0)
    Geography:str
    Gender:str
@app.get("/")
def home():
    return {"message": "API is working"}
@app.post("/student")
def process_student(name:str,score:int):
    if score >= 90:
        remark = "Excellent"
    elif score >= 70:
        remark = "Good"
    elif score >= 50:
        remark = "Average"
    else:
        remark = "Fail"

    return {
        "name":name,
        "score":score,
        "remark":remark
    }

@app.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            return {"error": "file is empty"}
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if "score" not in df.columns:
            return {"error": "csv must contain a 'score' column"}
        if not pd.api.types.is_numeric_dtype(df['score']):
            return{"error": "'score' column must contain numbers"}
        df = df.dropna(subset=["score"])
        if len(df) == 0:
            return {"error": "Novalid score data found"}

        average_score = df["score"].mean()
        max_score = df["score"].max()
        min_score = df["score"].min()


        passed = len(df[df["score"]>=50])
        failed = len(df[df["score"]<50])

        return {
            "average_score": round(float(average_score), 2),
            "highest_score": int(max_score),
            "lowest_score": int(min_score),
            "total_students": int(len(df)),
            "passed":int(passed),
            "failed":int(failed)
        }
    
    except Exception as e:
        return {"error":str(e)}
    
@app.post("/predict")
async def predict_score(data: UserInput):
    try:
        log_event("Request_receied",data.dict())
        new_data = pd.DataFrame({
            "CreditScore":[data.CreditScore],"Age":[data.Age],
            "Tenure":[data.Tenure],"Balance":[data.Balance],
            "NumOfProducts":[data.NumOfProducts],"HasCrCard":[data.HasCrCard],
            "IsActiveMember":[data.IsActiveMember],"EstimatedSalary":[data.EstimatedSalary],
            "Geography":[data.Geography],"Gender":[data.Gender]						
        })

        input_key = str(data.dict())

        if input_key in cache:
            log_event("cache_hit", data.dict())
            return {"predicted score": cache[input_key]}


        new_processed = preprocessor.transform(new_data)
        new_processed = new_processed.toarray() if hasattr(new_processed,"toarray") else new_processed
        start_time = time.time()
        prediction = await asyncio.to_thread(model.predict(new_processed))

        cache[input_key] = result 

        result = float(prediction.flatten()[0] >= 0.5)
        end_time = time.time()

        log_event("performance", {
           "latency_seconds":end_time-start_time})
        log_event("prediction made", {
            "input": data.dict(),
            "result":result
        })        
        return {
            "predicted score": result                                                            
        }
    
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}