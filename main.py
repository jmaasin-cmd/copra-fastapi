from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Copra Quality Prediction API")

# Load model and preprocessing objects
model = joblib.load("copra_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Request body schema
class CopraInput(BaseModel):
    moisture: float
    temperature: float
    color: int

@app.get("/")
def root():
    return {"message": "Copra FastAPI is running successfully"}

@app.post("/predict")
def predict(input: CopraInput):
    # Convert input to DataFrame with correct feature names
    input_df = pd.DataFrame(
        [[input.moisture, input.temperature, input.color]],
        columns=["moisture", "temperature", "color"]
    )

    # Apply same preprocessing
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Convert numeric label back to text
    quality = label_encoder.inverse_transform(prediction)[0]

    return {
        "copra_quality": quality
    }
