from fastapi import FastAPI
from pydantic import BaseModel

from models.pollution_aqi.predict import predict_aqi

app = FastAPI(
    title="CitySense360 – AQI Prediction API",
    description="Predict Air Quality Index for Smart Cities",
    version="1.0"
)

class AQIRequest(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float

@app.post("/predict-aqi")
def predict_air_quality(data: AQIRequest):
    result = predict_aqi(
        pm25=data.pm25,
        pm10=data.pm10,
        no2=data.no2,
        so2=data.so2,
        co=data.co,
        o3=data.o3
    )

    return {
        "status": "success",
        "data": result
    }
