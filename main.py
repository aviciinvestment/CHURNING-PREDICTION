from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
import joblib
import numpy as np
import tensorflow as tf
from pydantic import BaseModel





app = FastAPI()
model = tf.keras.models.load_model("tf_model.h5")
preprocessor = joblib.load("preprocessing.pkl")
class UserInput(BaseModel):
    CreditScore:float
    Age:int
    Tenure:float
    Balance:float
    NumOfProducts:float
    HasCrCard:int
    IsActiveMember:int
    EstimatedSalary:float
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
def predict_score(data: UserInput):
    new_data = pd.DataFrame({
        "CreditScore":[data.CreditScore],"Age":[data.Age],
        "Tenure":[data.Tenure],"Balance":[data.Balance],
        "NumOfProducts":[data.NumOfProducts],"HasCrCard":[data.HasCrCard],
        "IsActiveMember":[data.IsActiveMember],"EstimatedSalary":[data.EstimatedSalary],
        "Geography":[data.Geography],"Gender":[data.Gender]						
    })
    
    new_processed = preprocessor.transform(new_data)
    new_processed = new_processed.toarray() if hasattr(new_processed,"toarray") else new_processed

    prediction = model.predict(new_processed)
    return {
        "predicted score": float(prediction.flatten()[0] >= 0.5)                                                                
    }