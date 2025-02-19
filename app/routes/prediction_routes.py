from fastapi import APIRouter, Depends
from app.models.prediction_models import InputData
from app.services.model_handler import ModelHandler
from app.core.config import Config
from app.database.dependencis import get_db
from app.models.database_models import Prediction
from datetime import datetime, timezone
from sqlalchemy.orm import Session

# FastAPI application setup
router = APIRouter()

# Initialize the model handler
model_handler = ModelHandler(
    model_path=Config.MODEL_PATH,
    one_hot_encoder_path=Config.ONE_HOT_ENCODER_PATH,
    label_encoder_path=Config.LABEL_ENCODER_PATH,
)

# Healthcheck endpoint to verify application status
@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}

# Prediction endpoint
@router.post('/predict')
def predict(input_data: InputData, db: Session = Depends(get_db)):
    prediction = model_handler.predict(input_data.model_dump())
    db_prediction = Prediction(
        brand=input_data.Brand,
        production_year=input_data.Year,
        used_or_new=input_data.UsedOrNew,
        transmission=input_data.Transmission,
        drive_type=input_data.DriveType,
        fuel_type=input_data.FuelType,
        fuel_consumption=input_data.FuelConsumption,
        kilometres=input_data.Kilometres,
        cylinder_in_engine=input_data.CylindersinEngine,
        body_type=input_data.BodyType,
        doors=input_data.Doors,
        seats=input_data.Seats,
        prediction_price=round(prediction[0], 2),
        timestamp=datetime.now(timezone.utc) 
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return {"prediction": prediction}