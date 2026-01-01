from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Copra Multi-Model Prediction API")

# Load preprocessing
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Load models
models = {
    "Random Forest": joblib.load("rf_model.joblib"),
    "SVM": joblib.load("svm_model.joblib"),
    "KNN": joblib.load("knn_model.joblib"),
    "Logistic Regression": joblib.load("lr_model.joblib")
}

class CopraInput(BaseModel):
    moisture: float
    temperature: float
    color: int

@app.get("/")
def root():
    return {"message": "Copra FastAPI is running"}

@app.post("/predict-all")
def predict_all(input: CopraInput):

    input_df = pd.DataFrame(
        [[input.moisture, input.temperature, input.color]],
        columns=["moisture", "temperature", "color"]
    )

    input_scaled = scaler.transform(input_df)

    results = {}

    for name, model in models.items():
        pred = model.predict(input_scaled)
        quality = label_encoder.inverse_transform(pred)[0]
        results[name] = quality

    return {
        "predictions": results
    }
