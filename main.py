import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")
try:
    # make features JSON-serializable if needed
    features_list = list(features)
except Exception:
    features_list = features


# input schema
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income of the neighborhood")
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=32, le=42)
    Longitude: float = Field(..., ge=-125, le=-114)


@app.get("/")
def home():
    return {
        "message": "California House Prediction API",
        "status": "Running",
        "endpoint": "send POST request to /predict",
    }


@app.get("/health")
def health():
    return {
        "status": "Running",
        "model": type(model).__name__,
        "features": features_list,
        "avg_error": "$32754",
    }


@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame(
            [
                {
                    "MedInc": house.MedInc,
                    "HouseAge": house.HouseAge,
                    "AveRooms": house.AveRooms,
                    "AveBedrms": house.AveBedrms,
                    "Population": house.Population,
                    "AveOccup": house.AveOccup,
                    "Latitude": house.Latitude,
                    "Longitude": house.Longitude,
                }
            ]
        )

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousand",
            "confidence_range": f"${price_usd - 32754:,.0f} to ${price_usd + 32754:,.0f}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failure: {e}")